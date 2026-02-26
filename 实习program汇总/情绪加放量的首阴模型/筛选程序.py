import os
import re
import time
import json
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm


# ===================== 你要改的配置 =====================
OUTPUT_XLSX = Path(r"D:\candidates_scan_today.xlsx")   # 每天生成候选表

# iFinD HTTP
BASE = "https://quantapi.51ifind.com/api/v1"
ACCESS_TOKEN = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
if not ACCESS_TOKEN:
    raise RuntimeError("请先设置环境变量 IFIND_ACCESS_TOKEN（setx 后重启 Spyder/终端）")
HEADERS = {"Content-Type": "application/json", "access_token": ACCESS_TOKEN}

# 缓存目录（第一次慢，后面快）
CACHE_DIR = Path(r"D:\ifind_cache_daily_scan")
CACHE_DIR.mkdir(parents=True, exist_ok=True)
DAILY_CACHE_DIR = CACHE_DIR / "daily"
NAME_CACHE_JSON = CACHE_DIR / "name_cache.json"
IND_CACHE_JSON  = CACHE_DIR / "industry_cache.json"
ALLCODES_CACHE  = CACHE_DIR / "all_a_codes.csv.gz"

# 请求节奏
SLEEP = 0.12
RETRY = 3

# ===================== 扫描参数 =====================
SCAN_LAST_N_DAYS =3          # 回扫最近 N 个 t（交易日）
DAILY_LOOKBACK_DAYS = 80      # 拉日线窗口冗余
MAX_CANDIDATES = 800          # 候选上限（0=不截断）

# ===================== 大入场条件 =====================
LIMITUP_CLOSE_EQ_HIGH = True
BEAR_BODY_TH = -0.03          # t实体跌：close/open - 1 <= -3%
VOL_MULT = 3.0                # t成交量 >= t-1 * 3

# 过热过滤（三者满足其一就不入）
USE_CALENDAR_LOOKBACK = True
LU_WINDOW_DAYS = 5
LU_MIN_CNT = 3
RUNUP_WINDOW_DAYS = 10
RUNUP_TH = 0.30
T_BODY_DROP_FILTER_TH = -0.09  # t实体<=-9% 不入（t=大阴线那天）

# 输出给实时用的关键价
INIT_STOP_MULT = 0.90          # 初始止损 = 0.9*L

# ===================== 你要的“硬过滤” =====================
ENABLE_EXCLUDE_PREFIX = True
EXCLUDE_PREFIXES = ("688", "300", "9")   # 688/300/9 直接跳过

ENABLE_EXCLUDE_ST = True                # ST/*ST 直接跳过


# ===================== 工具函数 =====================
def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def ymd_to_dash(ymd8: str) -> str:
    s = str(ymd8).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def to_ths_code(code6_str: str) -> str:
    c = code6(code6_str)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"

def is_excluded_prefix(code6_str: str) -> bool:
    c = code6(code6_str)
    return c.startswith(EXCLUDE_PREFIXES)

def is_st_name(name: str) -> bool:
    """
    名称里含 ST/*ST/ＳＴ 都判定为 ST
    """
    if not name:
        return False
    s = str(name).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("＊", "*")   # 全角星号
    s = s.replace("ＳＴ", "ST")  # 全角ST
    return ("ST" in s)  # 覆盖 ST / *ST

def safe_excel_path(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "ab"):
            pass
        return path
    except PermissionError:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")

def cache_read_json(fp: Path) -> dict:
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}

def cache_write_json(obj: dict, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    last_err = None
    for _ in range(RETRY):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
            r.raise_for_status()
            js = r.json()
            if js.get("errorcode", 0) != 0:
                raise RuntimeError(f"errorcode={js.get('errorcode')} errmsg={js.get('errmsg')}")
            return js
        except Exception as e:
            last_err = repr(e)
            time.sleep(0.8)
    raise RuntimeError(f"[{endpoint}] failed: {last_err}")

def tables_to_df(js: dict) -> pd.DataFrame:
    tables = js.get("tables") or []
    out = []
    for t in tables:
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None and "time" not in df.columns:
            df.insert(0, "time", times)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ===================== iFinD 拉日线（带缓存） =====================
def fetch_daily(ths_code: str, startdate: str, enddate: str) -> pd.DataFrame:
    fp = DAILY_CACHE_DIR / f"{ths_code.replace('.','_')}_{startdate}_{enddate}.csv.gz"
    if fp.exists():
        try:
            df = pd.read_csv(fp, compression="gzip")
            if not df.empty and "trade_date" in df.columns:
                return df
        except Exception:
            pass

    payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "startdate": startdate,
        "enddate": enddate,
        "functionpara": {"Fill": "Blank"},
    }
    js = post_json("cmd_history_quotation", payload)
    df = tables_to_df(js)
    time.sleep(SLEEP)

    if df.empty:
        out = pd.DataFrame(columns=["trade_date","open","high","low","close","volume","amount"])
        fp.parent.mkdir(parents=True, exist_ok=True)
        out.to_csv(fp, index=False, encoding="utf-8-sig", compression="gzip")
        return out

    df = df.rename(columns={"time":"trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["trade_date","open","high","low","close"]).sort_values("trade_date").reset_index(drop=True)
    df = df[["trade_date","open","high","low","close","volume","amount"]]

    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False, encoding="utf-8-sig", compression="gzip")
    return df


# ===================== 名称/行业（缓存） =====================
def fetch_stock_name(ths_code: str) -> str:
    cache = cache_read_json(NAME_CACHE_JSON)
    if ths_code in cache and cache[ths_code]:
        return str(cache[ths_code])

    q = f"{ths_code} 名称"
    try:
        js = post_json("smart_stock_picking", {"searchstring": q, "searchtype": "stock"})
        df = tables_to_df(js)
        time.sleep(SLEEP)
        if not df.empty:
            cols = list(df.columns)
            name_col = None
            for pat in [r"名称", r"证券名称", r"name", r"简称"]:
                for c in cols:
                    if re.search(pat, str(c), flags=re.IGNORECASE):
                        name_col = c
                        break
                if name_col:
                    break
            if name_col:
                v = str(df.iloc[0][name_col]).strip()
                if v and v.lower() != "nan":
                    cache[ths_code] = v
                    cache_write_json(cache, NAME_CACHE_JSON)
                    return v
    except Exception:
        pass

    cache[ths_code] = ""
    cache_write_json(cache, NAME_CACHE_JSON)
    return ""

def fetch_industry_name(ths_code: str) -> str:
    cache = cache_read_json(IND_CACHE_JSON)
    if ths_code in cache and cache[ths_code]:
        return str(cache[ths_code])

    queries = [
        f"{ths_code} 申万一级行业",
        f"{ths_code} 所属行业",
        f"{ths_code} 行业",
    ]
    for q in queries:
        try:
            js = post_json("smart_stock_picking", {"searchstring": q, "searchtype": "stock"})
            df = tables_to_df(js)
            time.sleep(SLEEP)
            if df.empty:
                continue
            cols = list(df.columns)
            pick = None
            for pat in [r"申万.*一级.*行业", r"一级行业", r"所属行业", r"行业名称", r"行业"]:
                for c in cols:
                    if re.search(pat, str(c)):
                        pick = c
                        break
                if pick:
                    break
            if pick:
                v = str(df.iloc[0][pick]).strip()
                v = re.sub(r"\s+", " ", v)
                if v and v.lower() != "nan" and len(v) <= 40:
                    cache[ths_code] = v
                    cache_write_json(cache, IND_CACHE_JSON)
                    return v
        except Exception:
            continue

    cache[ths_code] = ""
    cache_write_json(cache, IND_CACHE_JSON)
    return ""


# ===================== 过热过滤 =====================
def _slice_by_calendar_days_include_end(daily: pd.DataFrame, end_trade_date: str, window_days: int) -> pd.DataFrame:
    if daily.empty:
        return daily.iloc[0:0]
    end_dt = pd.to_datetime(end_trade_date, format="%Y%m%d", errors="coerce")
    if pd.isna(end_dt):
        return daily.iloc[0:0]
    win_end = end_dt
    win_start = end_dt - pd.Timedelta(days=window_days - 1)
    dts = pd.to_datetime(daily["trade_date"], format="%Y%m%d", errors="coerce")
    m = (dts >= win_start) & (dts <= win_end)
    return daily.loc[m].copy()

def overheat_filter(daily: pd.DataFrame, idx_t: int, t: str, tminus1: str) -> tuple[bool, str]:
    row_t = daily.iloc[idx_t]
    ot = float(row_t["open"])
    ct = float(row_t["close"])
    body_t = (ct / ot - 1.0) if ot > 0 else np.nan
    hit3 = (np.isfinite(body_t) and body_t <= T_BODY_DROP_FILTER_TH)

    w_lu = _slice_by_calendar_days_include_end(daily, tminus1, LU_WINDOW_DAYS) if USE_CALENDAR_LOOKBACK else daily.iloc[max(0, idx_t-6):idx_t].copy()
    w_ru = _slice_by_calendar_days_include_end(daily, tminus1, RUNUP_WINDOW_DAYS) if USE_CALENDAR_LOOKBACK else daily.iloc[max(0, idx_t-11):idx_t].copy()

    hit1 = False
    lu_cnt = np.nan
    if not w_lu.empty:
        lu_cnt = int((w_lu["close"].astype(float) == w_lu["high"].astype(float)).sum())
        hit1 = (lu_cnt >= LU_MIN_CNT)

    hit2 = False
    runup = np.nan
    if len(w_ru) >= 2:
        c0 = float(w_ru.iloc[0]["close"])
        c1 = float(w_ru.iloc[-1]["close"])
        if c0 > 0:
            runup = c1 / c0 - 1.0
            hit2 = (np.isfinite(runup) and runup > RUNUP_TH)

    if hit1 or hit2 or hit3:
        parts = []
        if hit1:
            parts.append(f"limitup_cnt={lu_cnt}>=3")
        if hit2:
            parts.append(f"runup={runup:.2%}>30%")
        if hit3:
            parts.append(f"t_body={body_t:.2%}<=-9%")
        return True, " | ".join(parts)
    return False, f"ok(t_body={body_t:.2%})"


# ===================== 获取全A代码（缓存） =====================
def fetch_all_a_share_codes() -> list[str]:
    if ALLCODES_CACHE.exists():
        try:
            df = pd.read_csv(ALLCODES_CACHE, compression="gzip")
            if not df.empty and "code" in df.columns:
                return df["code"].astype(str).tolist()
        except Exception:
            pass

    queries = [
        ("A股 股票 列表", "stock"),
        ("沪深A股 股票 列表", "stock"),
        ("全部A股", "stock"),
    ]
    got = set()
    last_note = ""
    for q, stype in queries:
        try:
            js = post_json("smart_stock_picking", {"searchstring": q, "searchtype": stype})
            df = tables_to_df(js)
            time.sleep(SLEEP)
            if df is None or df.empty:
                last_note = f"{q}/{stype}: empty"
                continue
            for col in df.columns:
                ser = df[col].astype(str)
                for s in ser.tolist():
                    m = re.findall(r"\b\d{6}\b", s)
                    for x in m:
                        got.add(x)
            last_note = f"{q}/{stype}: rows={len(df)} codes={len(got)}"
            if len(got) > 2000:
                break
        except Exception as e:
            last_note = f"{q}/{stype}: err={repr(e)}"

    if not got:
        raise RuntimeError(f"无法获取全市场A股列表（smart_stock_picking）。last={last_note}")

    out = pd.DataFrame({"code": sorted(got)})
    out.to_csv(ALLCODES_CACHE, index=False, encoding="utf-8-sig", compression="gzip")
    return out["code"].tolist()


# ===================== 扫描输出 candidates_scan =====================
@dataclass
class CandidateRow:
    code: str
    ths_code: str
    name: str
    industry: str
    tminus1: str
    t: str
    tplus1: str
    L: float
    stop: float
    body_t: float
    vol_mult: float
    note: str

def scan_candidates(codes: list[str], asof_ymd: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    asof_dt = pd.to_datetime(asof_ymd, format="%Y%m%d", errors="coerce")
    startdate = ymd_to_dash((asof_dt - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d"))
    enddate   = ymd_to_dash(asof_dt.strftime("%Y%m%d"))

    rows = []
    errs = []

    for code in tqdm(codes, desc=f"Scan last {SCAN_LAST_N_DAYS} t-days"):
        try:
            if ENABLE_EXCLUDE_PREFIX and is_excluded_prefix(code):
                continue

            ths = to_ths_code(code)
            daily = fetch_daily(ths, startdate, enddate)
            if daily.empty or len(daily) < 5:
                continue

            dates = daily["trade_date"].astype(str).tolist()
            tail = dates[-(SCAN_LAST_N_DAYS + 2):]  # 留 t-1、t+1
            dmap = daily.set_index("trade_date")

            for j in range(1, len(tail)-1):
                t = tail[j]
                tminus1 = tail[j-1]
                tplus1 = tail[j+1]

                if t not in dmap.index or tminus1 not in dmap.index:
                    continue
                row_t = dmap.loc[t]
                row_tm1 = dmap.loc[tminus1]

                # 条件1：t-1 涨停（收盘=最高）
                if LIMITUP_CLOSE_EQ_HIGH:
                    if not (float(row_tm1["close"]) == float(row_tm1["high"])):
                        continue

                # 条件2：t 大阴实体跌
                o = float(row_t["open"])
                c = float(row_t["close"])
                if o <= 0:
                    continue
                body = c / o - 1.0
                if not (body <= BEAR_BODY_TH):
                    continue

                # 条件3：t 放量
                v_t = float(row_t["volume"]) if pd.notna(row_t["volume"]) else np.nan
                v_tm1 = float(row_tm1["volume"]) if pd.notna(row_tm1["volume"]) else np.nan
                if not (np.isfinite(v_t) and np.isfinite(v_tm1) and v_tm1 > 0 and v_t >= VOL_MULT * v_tm1):
                    continue

                # 过热过滤
                idx_t = daily.index[daily["trade_date"] == t][0]
                hit, why = overheat_filter(daily, idx_t, t, tminus1)
                if hit:
                    continue

                # 到这一步才取 name（减少请求）；取到后做 ST 硬过滤
                name = fetch_stock_name(ths) or ""
                if ENABLE_EXCLUDE_ST and is_st_name(name):
                    continue

                industry = fetch_industry_name(ths) or ""

                L = float(row_t["low"])
                stop = INIT_STOP_MULT * L
                rows.append(CandidateRow(
                    code=code, ths_code=ths, name=name, industry=industry,
                    tminus1=tminus1, t=t, tplus1=tplus1,
                    L=L, stop=stop,
                    body_t=body,
                    vol_mult=(v_t / v_tm1),
                    note=f"ok | body={body:.2%} vol_mult={v_t/v_tm1:.2f} | overheat={why}"
                ).__dict__)
        except Exception as e:
            errs.append({"code": code, "stage": "scan", "err": repr(e)})

    df = pd.DataFrame(rows)
    if not df.empty:
        df["_score"] = (-df["body_t"].astype(float)) * 1.0 + (df["vol_mult"].astype(float)) * 0.2
        df = df.sort_values(["t", "_score"], ascending=[False, False]).drop(columns=["_score"]).reset_index(drop=True)
        if MAX_CANDIDATES and len(df) > MAX_CANDIDATES:
            df = df.head(MAX_CANDIDATES).copy()

    return df, pd.DataFrame(errs)


def main():
    asof_ymd = pd.Timestamp.now().strftime("%Y%m%d")
    codes = fetch_all_a_share_codes()
    print(f"[Universe] codes={len(codes)} asof={asof_ymd}")

    df_cand, df_err = scan_candidates(codes, asof_ymd)
    print(f"[Done] candidates={len(df_cand)} errors={len(df_err)}")

    out_path = safe_excel_path(OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_cand.to_excel(w, index=False, sheet_name="candidates_scan")
        df_err.to_excel(w, index=False, sheet_name="errors")
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
