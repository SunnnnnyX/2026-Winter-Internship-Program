import os
import re
import time
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ===================== 你要改的配置 =====================
DB_PATH = Path(r"D:\marketdata\ashare_ifind.sqlite")

# ===== 选择抓取日期 =====
# 1) 单日：["20260205"]
# 2) 多日：["20260205","20260204"]
# 3) 区间：用 DATE_RANGE_START/END（闭区间），TARGET_DATES 置空
TARGET_DATES = [20260206]  # e.g. ["20260205"]

DATE_RANGE_START = ""  # e.g. "20260115"
DATE_RANGE_END   = ""  # e.g. "20260205"

# 如果上面都没填：默认抓今天
DEFAULT_TODAY = True

# 批大小：建议 200~500。越大请求越少，但更容易超时/限流
BATCH = 300

# 请求节奏
SLEEP = 0.08
RETRY = 3

# iFinD HTTP
BASE = "https://quantapi.51ifind.com/api/v1"
ACCESS_TOKEN = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
if not ACCESS_TOKEN:
    raise RuntimeError("请先设置环境变量 IFIND_ACCESS_TOKEN（setx 后重启 Spyder/终端）")
HEADERS = {"Content-Type": "application/json", "access_token": ACCESS_TOKEN}

# 全A代码缓存（避免每次都去拉全市场列表）
CACHE_DIR = Path(r"D:\ifind_daily_ingest_cache")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
# 改成缓存 code+name，避免 ST 过滤没法做
ALLCODES_CACHE = CACHE_DIR / "all_a_codes_with_name.csv.gz"

# ===================== 过滤规则（你要的） =====================
ENABLE_EXCLUDE_PREFIX = True
EXCLUDE_PREFIXES = ("688", "300", "9")  # 688/300/9 开头不要

ENABLE_EXCLUDE_ST = True  # ST/*ST 不要


# ===================== 建表（自动） =====================
DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS daily_bar (
  code TEXT NOT NULL,
  trade_date TEXT NOT NULL,
  open REAL,
  high REAL,
  low REAL,
  close REAL,
  volume REAL,
  amount REAL,
  source TEXT DEFAULT 'ifind',
  PRIMARY KEY(code, trade_date)
);

CREATE INDEX IF NOT EXISTS idx_daily_trade_date ON daily_bar(trade_date);
CREATE INDEX IF NOT EXISTS idx_daily_code ON daily_bar(code);
"""

SESSION = requests.Session()
SESSION.headers.update(HEADERS)


# ===================== 工具函数 =====================
def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def to_ths_code(code6_str: str) -> str:
    c = code6(code6_str)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"

def ymd_to_dash(ymd: str) -> str:
    s = str(ymd).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def is_excluded_prefix(code6_str: str) -> bool:
    c = code6(code6_str)
    return c.startswith(EXCLUDE_PREFIXES)

def is_st_name(name: str) -> bool:
    """
    名称里含 ST/*ST/ＳＴ 都判定为 ST
    """
    if not name or str(name).strip().lower() == "nan":
        return False
    s = str(name).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("＊", "*")      # 全角星号
    s = s.replace("ＳＴ", "ST")   # 全角ST
    return ("ST" in s)            # 覆盖 ST / *ST

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    last_err = None
    for _ in range(RETRY):
        try:
            r = SESSION.post(url, json=payload, timeout=60)
            r.raise_for_status()
            js = r.json()
            ec = js.get("errorcode", 0)
            if ec != 0:
                raise RuntimeError(f"errorcode={ec} errmsg={js.get('errmsg')}")
            return js
        except Exception as e:
            last_err = repr(e)
            time.sleep(0.8)
    raise RuntimeError(f"[{endpoint}] failed: {last_err}")

def tables_to_long_df(js: dict) -> pd.DataFrame:
    tables = js.get("tables") or []
    out = []
    for t in tables:
        thscode = t.get("thscode", "")
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None and "time" not in df.columns:
            df.insert(0, "time", times)
        if thscode:
            df.insert(0, "thscode", thscode)
        out.append(df)
    if not out:
        return pd.DataFrame()

    df = pd.concat(out, ignore_index=True)
    df = df.rename(columns={"time": "trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")

    for c in ["open", "high", "low", "close", "volume", "amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")

    df = df.dropna(subset=["thscode", "trade_date", "open", "high", "low", "close"])
    return df[["thscode", "trade_date", "open", "high", "low", "close", "volume", "amount"]]

def _pick_col(df: pd.DataFrame, patterns) -> str | None:
    cols = list(df.columns)
    for pat in patterns:
        for c in cols:
            if re.search(pat, str(c), flags=re.IGNORECASE):
                return c
    return None

def fetch_all_a_codes_with_name() -> pd.DataFrame:
    """
    返回 DataFrame: code(6位), name(可空)
    并写入 ALLCODES_CACHE（gzip）
    """
    # 先读缓存
    if ALLCODES_CACHE.exists():
        try:
            d = pd.read_csv(ALLCODES_CACHE, compression="gzip")
            if not d.empty and "code" in d.columns:
                d["code"] = d["code"].astype(str).map(code6)
                if "name" not in d.columns:
                    d["name"] = ""
                return d[["code", "name"]]
        except Exception:
            pass

    # 用 smart_stock_picking 拿全A列表（尽量拿到 code+name）
    queries = [("沪深A股 股票 列表", "stock"), ("全部A股", "stock")]
    got = {}  # code -> name(可空)
    last_note = ""

    for q, stype in queries:
        try:
            js = post_json("smart_stock_picking", {"searchstring": q, "searchtype": stype})
            tables = js.get("tables") or []
            for t in tables:
                tab = t.get("table") or {}
                d = pd.DataFrame(tab)
                if d.empty:
                    continue

                # 找 code 列
                code_col = _pick_col(d, [r"\bcode\b", r"证券代码", r"股票代码", r"代码"])
                name_col = _pick_col(d, [r"\bname\b", r"证券名称", r"名称", r"简称"])

                # 如果找不到列，就退化为：在所有文本里正则挖 code
                if not code_col:
                    for col in d.columns:
                        for s in d[col].astype(str).tolist():
                            for x in re.findall(r"\b\d{6}\b", s):
                                got.setdefault(x, "")
                    continue

                # 正常：按行读 code / name
                for _, r in d.iterrows():
                    c = r.get(code_col, "")
                    c6 = code6(c)
                    if not re.fullmatch(r"\d{6}", c6):
                        continue
                    nm = ""
                    if name_col:
                        nm = str(r.get(name_col, "")).strip()
                        if nm.lower() == "nan":
                            nm = ""
                    if c6 not in got or (not got[c6] and nm):
                        got[c6] = nm

            last_note = f"{q}: codes={len(got)}"
            time.sleep(SLEEP)
            if len(got) > 2000:
                break
        except Exception as e:
            last_note = f"{q}: err={repr(e)}"

    if not got:
        raise RuntimeError(f"无法获取全A代码列表。last={last_note}")

    out = pd.DataFrame({"code": sorted(got.keys()), "name": [got[k] for k in sorted(got.keys())]})
    out["code"] = out["code"].astype(str).map(code6)
    out.to_csv(ALLCODES_CACHE, index=False, encoding="utf-8-sig", compression="gzip")
    return out[["code", "name"]]

def build_universe() -> pd.DataFrame:
    """
    返回 universe: code, thscode, name
    并按你要求过滤：ST / 前缀 688/300/9
    """
    d = fetch_all_a_codes_with_name().copy()
    d["code"] = d["code"].astype(str).map(code6)
    d["name"] = d.get("name", "").astype(str)

    # 过滤前缀
    if ENABLE_EXCLUDE_PREFIX:
        m = ~d["code"].astype(str).str.startswith(EXCLUDE_PREFIXES)
        d = d.loc[m].copy()

    # 过滤 ST（仅当 name 有效）
    if ENABLE_EXCLUDE_ST:
        d = d.loc[~d["name"].apply(is_st_name)].copy()

    d["thscode"] = d["code"].apply(to_ths_code)
    d = d.drop_duplicates(subset=["code"]).reset_index(drop=True)
    return d[["code", "thscode", "name"]]

def fetch_daily_1bar_batch(ths_codes: list[str], ymd: str) -> pd.DataFrame:
    d = ymd_to_dash(ymd)
    payload = {
        "codes": ",".join(ths_codes),
        "indicators": "open,high,low,close,volume,amount",
        "startdate": d,
        "enddate": d,
        "functionpara": {"Fill": "Blank"},
    }
    js = post_json("cmd_history_quotation", payload)
    df = tables_to_long_df(js)
    time.sleep(SLEEP)
    return df[df["trade_date"] == ymd].copy()

def upsert_daily(con: sqlite3.Connection, df: pd.DataFrame) -> int:
    if df is None or df.empty:
        return 0

    df = df.copy()
    df["code"] = df["thscode"].astype(str).str.replace(r"\D", "", regex=True).str[:6].map(code6)

    sql = """
    INSERT INTO daily_bar(code, trade_date, open, high, low, close, volume, amount, source)
    VALUES(?,?,?,?,?,?,?,?,?)
    ON CONFLICT(code, trade_date) DO UPDATE SET
      open=excluded.open,
      high=excluded.high,
      low=excluded.low,
      close=excluded.close,
      volume=excluded.volume,
      amount=excluded.amount,
      source=excluded.source
    """
    data = []
    for _, r in df.iterrows():
        data.append((
            r["code"], r["trade_date"],
            float(r["open"]), float(r["high"]), float(r["low"]), float(r["close"]),
            float(r["volume"]) if pd.notna(r["volume"]) else None,
            float(r["amount"]) if pd.notna(r["amount"]) else None,
            "ifind",
        ))
    con.executemany(sql, data)
    return len(data)

def _parse_ymd(s: str) -> pd.Timestamp:
    return pd.to_datetime(str(s).strip(), format="%Y%m%d", errors="coerce")

def build_date_list() -> list[str]:
    """
    优先级：
    1) TARGET_DATES 非空：用它（按输入顺序）
    2) DATE_RANGE_START/END 非空：生成闭区间
    3) DEFAULT_TODAY：抓今天
    """
    if TARGET_DATES:
        out = []
        for x in TARGET_DATES:
            ts = _parse_ymd(x)
            if pd.isna(ts):
                raise RuntimeError(f"TARGET_DATES 里有非法日期: {x}")
            out.append(ts.strftime("%Y%m%d"))
        return out

    if DATE_RANGE_START and DATE_RANGE_END:
        s = _parse_ymd(DATE_RANGE_START)
        e = _parse_ymd(DATE_RANGE_END)
        if pd.isna(s) or pd.isna(e):
            raise RuntimeError(f"DATE_RANGE_START/END 非法: {DATE_RANGE_START}, {DATE_RANGE_END}")
        if s > e:
            s, e = e, s
        days = pd.date_range(s, e, freq="D")
        return [d.strftime("%Y%m%d") for d in days]

    if DEFAULT_TODAY:
        return [pd.Timestamp.now().strftime("%Y%m%d")]

    raise RuntimeError("没有指定日期：请设置 TARGET_DATES 或 DATE_RANGE_START/END 或 DEFAULT_TODAY=True")


# ===================== 主流程 =====================
def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    date_list = build_date_list()
    # 注意：这里是自然日列表，交易日/非交易日都可能出现
    print(f"[Dates] {date_list}")

    # 连接DB并确保表存在
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(DDL)
        con.commit()

        uni = build_universe()
        ths_codes = uni["thscode"].astype(str).tolist()

        print(f"[Universe] total={len(uni)} batch={BATCH} "
              f"(exclude_prefix={ENABLE_EXCLUDE_PREFIX} {EXCLUDE_PREFIXES}, exclude_ST={ENABLE_EXCLUDE_ST})")

        total_upsert = 0
        total_errors = 0
        total_empty_batches = 0

        for ymd in date_list:
            print(f"\n=== Ingest trade_date={ymd} ===")
            day_upsert = 0
            day_empty = 0

            for i in tqdm(range(0, len(ths_codes), BATCH), desc=f"Ingest {ymd}"):
                batch = ths_codes[i:i + BATCH]
                try:
                    df = fetch_daily_1bar_batch(batch, ymd)
                    n = upsert_daily(con, df)
                    day_upsert += n
                    if n == 0:
                        day_empty += 1
                except Exception as e:
                    total_errors += 1
                    print(f"[{ymd} batch {i}] err={repr(e)}")
                    continue

            con.commit()
            total_upsert += day_upsert
            total_empty_batches += day_empty
            print(f"[{ymd}] upserted={day_upsert} empty_batches={day_empty}")

        print(f"\n[Done] total_upserted={total_upsert} errors={total_errors} empty_batches={total_empty_batches}")
        print("DB:", DB_PATH)

    finally:
        con.close()


if __name__ == "__main__":
    main()
