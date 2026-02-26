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
INPUT_XLSX = Path(r"D:\a_share_daily\无st涨停重设后_纯数字代码.xlsx")  # 你的信号表
OUTPUT_XLSX = Path(r"D:\batch_backtest_ifind_results5.xlsx")

# iFinD HTTP
BASE = "https://quantapi.51ifind.com/api/v1"
ACCESS_TOKEN = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
if not ACCESS_TOKEN:
    raise RuntimeError("请先设置环境变量 IFIND_ACCESS_TOKEN（setx 后重启 Spyder/终端）")
HEADERS = {"Content-Type": "application/json", "access_token": ACCESS_TOKEN}

# 资金与参数
INIT_CASH = 100_000.0
FEE_RATE = 0.0           # 单边费率（先0）
ATR_N = 14
CH_K = 3.0
INIT_STOP_MULT = 0.9
HOLD_DAYS = 7            # 从 t+1 开始（含 t+1 当天）数 7 根交易日日K
REQUIRE_FULL_HOLD_WINDOW = False

# ===== 新增：大盘过滤（同花顺全A）=====
# 满足其一不做：1) close < MA20  2) 含t的3根累计跌幅 < -2.5%
MARKET_INDEX_CANDIDATES = [
    "883957",       # 常见写法
    "883957.SH",
    "883957.SZ",
    "883957.TI",
]
MARKET_MA_N = 20
MARKET_3BAR_DROP_TH = -0.025

# ===== 新增：不做这些代码前缀 =====
EXCLUDE_PREFIXES = ("9", "688", "300")

# ===== 过滤器（3 条满足其一就不入）=====
# 1) t-1 结束的 5 天窗口：涨停(收盘=最高)根数 >=3
# 2) t-1 结束的 10 天窗口：涨幅 >30%
# 3) t 当天实体跌幅（close/open-1）<= -9% 直接不入
USE_CALENDAR_LOOKBACK = True  # 按自然日窗口（节假日不影响“天数”定义），并且包含 t-1 当天

LU_WINDOW_DAYS = 5
LU_MIN_CNT = 3

RUNUP_WINDOW_DAYS = 10
RUNUP_TH = 0.30

# 若切换成“按交易日根数窗口”，用下面两个
LU_WINDOW_BARS = 5
RUNUP_WINDOW_BARS = 10

# 第3条：t 日实体跌幅阈值
T_BODY_DROP_TH = -0.08  # <= -9% 过滤

# 行业汇总参数（保留，不影响策略）
INDUSTRY_MIN_SAMPLES = 1

# 缓存（强烈建议开，否则很慢）
CACHE_DIR = Path(r"D:\ifind_cache_batch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 请求节奏
SLEEP = 0.15             # 防止被限流，太快就加大
RETRY = 3


# ===================== 工具函数 =====================
def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def ymd8(x) -> str:
    s = str(x).strip().replace("-", "")
    s = re.sub(r"\D", "", s)
    return s[:8]

def ymd_to_dash(ymd) -> str:
    ymd_str = str(ymd).strip()
    return f"{ymd_str[:4]}-{ymd_str[4:6]}-{ymd_str[6:8]}"

def to_ths_code(code6_str: str) -> str:
    c = code6(code6_str)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"

def is_excluded_code(code6_str: str) -> bool:
    c = code6(code6_str)
    return c.startswith(EXCLUDE_PREFIXES)

def lot_shares(cash: float, price: float) -> int:
    if price <= 0:
        return 0
    return int(cash // (price * 100.0)) * 100

def wilder_atr(high, low, close, n=14) -> pd.Series:
    high = pd.Series(high).astype(float)
    low  = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def ensure_30m(df: pd.DataFrame) -> pd.DataFrame:
    """如果返回不是30m，自动重采样到30m（保守处理）。"""
    if df.empty:
        return df
    df = df.sort_values("dt").reset_index(drop=True)
    deltas = df["dt"].diff().dropna().dt.total_seconds() / 60.0
    med = float(deltas.median()) if len(deltas) else 30.0
    if med >= 25:
        return df

    tmp = df.set_index("dt")
    ohlc = tmp[["open","high","low","close"]].resample("30T", label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    )
    out = ohlc.copy()
    if "volume" in tmp.columns:
        out["volume"] = tmp["volume"].resample("30T", label="left", closed="left").sum()
    else:
        out["volume"] = np.nan
    if "amount" in tmp.columns:
        out["amount"] = tmp["amount"].resample("30T", label="left", closed="left").sum()
    else:
        out["amount"] = np.nan

    out = out.dropna(subset=["open","high","low","close"]).reset_index()
    return out

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
        thscode = t.get("thscode") or t.get("thsCode") or t.get("code")
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None and "time" not in df.columns:
            df.insert(0, "time", times)
        if thscode is not None and "ts_code" not in df.columns:
            df.insert(0, "ts_code", thscode)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def cache_read(fp: Path) -> pd.DataFrame | None:
    if fp.exists():
        try:
            return pd.read_csv(fp, compression="gzip")
        except Exception:
            return None
    return None

def cache_write(df: pd.DataFrame, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False, encoding="utf-8-sig", compression="gzip")

def cache_read_text(fp: Path) -> dict | None:
    if fp.exists():
        try:
            return json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            return None
    return None

def cache_write_text(obj: dict, fp: Path):
    fp.parent.mkdir(parents=True, exist_ok=True)
    fp.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")

def safe_excel_writer_path(path: Path) -> Path:
    """
    如果目标xlsx被占用/无权限，自动改成新文件名避免 PermissionError。
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "ab"):
            pass
        return path
    except PermissionError:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")


# ===================== iFinD 拉数（带缓存） =====================
def fetch_daily(ths_code: str, startdate: str, enddate: str) -> pd.DataFrame:
    fp = CACHE_DIR / "daily" / f"{ths_code.replace('.','_')}_{startdate}_{enddate}.csv.gz"
    cached = cache_read(fp)
    if cached is not None and not cached.empty and "trade_date" in cached.columns:
        return cached

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
        cache_write(out, fp)
        return out

    df = df.rename(columns={"time":"trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["trade_date","open","high","low","close"]).sort_values("trade_date").reset_index(drop=True)
    df = df[["trade_date","open","high","low","close","volume","amount"]]
    cache_write(df, fp)
    return df

def fetch_hf_30m(ths_code: str, starttime: str, endtime: str) -> tuple[pd.DataFrame, dict]:
    sday = starttime[:10].replace("-","")
    eday = endtime[:10].replace("-","")
    fp = CACHE_DIR / "hf30m" / f"{ths_code.replace('.','_')}_{sday}_{eday}.csv.gz"
    cached = cache_read(fp)
    if cached is not None and not cached.empty and ("dt" in cached.columns):
        df = cached.copy()
        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        df = df.dropna(subset=["dt"]).sort_values("dt").reset_index(drop=True)
        df["d"] = df["dt"].dt.strftime("%Y%m%d")
        df["atr"] = wilder_atr(df["high"], df["low"], df["close"], n=ATR_N)
        return df, {"ok_payload": "cache"}

    base_payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "starttime": starttime,
        "endtime": endtime,
    }
    candidates = [
        {"functionpara": {"Interval": "30", "Fill": "Blank"}},
        {"functionpara": {"interval": "30", "fill": "Blank"}},
        {"functionpara": {"Interval": 30}},
        {},
    ]
    debug = {"ok_payload": None, "last_err": None}

    for extra in candidates:
        payload = dict(base_payload)
        payload.update(extra)
        try:
            js = post_json("high_frequency", payload)
            df = tables_to_df(js)
            time.sleep(SLEEP)
            if df.empty:
                continue
            debug["ok_payload"] = payload

            df = df.rename(columns={"time":"dt"})
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df = df.dropna(subset=["dt"]).copy()
            for c in ["open","high","low","close","volume","amount"]:
                df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
            df = df.dropna(subset=["open","high","low","close"]).sort_values("dt").reset_index(drop=True)
            df = df[["dt","open","high","low","close","volume","amount"]]
            df = ensure_30m(df)

            df["d"] = df["dt"].dt.strftime("%Y%m%d")
            df["atr"] = wilder_atr(df["high"], df["low"], df["close"], n=ATR_N)

            tmp = df.copy()
            tmp["dt"] = tmp["dt"].astype(str)
            cache_write(tmp, fp)

            return df, debug
        except Exception as e:
            debug["last_err"] = repr(e)

    return pd.DataFrame(), debug


# ===================== 行业（保留原样：不影响策略过滤） =====================
def _guess_industry_from_df(df: pd.DataFrame) -> tuple[str, str]:
    if df is None or df.empty:
        return "", "industry_df_empty"

    col_patterns = [
        r"申万.*一级.*行业",
        r"申万.*行业.*一级",
        r"一级行业",
        r"所属行业",
        r"行业名称",
        r"行业",
        r"板块",
    ]

    cols = list(df.columns)
    pick_col = None
    for pat in col_patterns:
        for c in cols:
            if re.search(pat, str(c)):
                pick_col = c
                break
        if pick_col is not None:
            break

    def clean_val(v) -> str:
        s = str(v).strip()
        s = re.sub(r"\s+", " ", s)
        if s == "" or re.fullmatch(r"[\d\.\-]+", s):
            return ""
        if len(s) > 40:
            return ""
        return s

    if pick_col is not None:
        v = clean_val(df.iloc[0][pick_col])
        if v:
            return v, f"hit_col={pick_col}"

    for c in cols:
        v = clean_val(df.iloc[0][c])
        if v and any(k in v for k in ["申万", "行业", "板块", "一级", "二级", "三级"]):
            return v, f"scan_col={c}"

    for c in cols:
        v = clean_val(df.iloc[0][c])
        if v and re.search(r"[\u4e00-\u9fff]", v):
            return v, f"fallback_col={c}"

    return "", "industry_not_found"

def fetch_industry(ths_code: str) -> tuple[str, str]:
    fp = CACHE_DIR / "industry" / f"{ths_code.replace('.','_')}.json"
    cached = cache_read_text(fp)
    if cached and isinstance(cached, dict) and "industry" in cached:
        return str(cached.get("industry") or ""), f"cache:{cached.get('note','')}"

    queries = [
        f"{ths_code} 申万一级行业",
        f"{ths_code} 所属申万一级行业",
        f"{ths_code} 所属行业",
        f"{ths_code} 行业",
        f"{ths_code} 属于什么行业",
    ]

    last_note = ""
    for q in queries:
        try:
            payload = {"searchstring": q, "searchtype": "stock"}
            js = post_json("smart_stock_picking", payload)
            df = tables_to_df(js)
            time.sleep(SLEEP)

            ind, note = _guess_industry_from_df(df)
            last_note = f"q={q} | {note} | cols={list(df.columns)[:12]}"
            if ind:
                cache_write_text({"industry": ind, "note": last_note}, fp)
                return ind, last_note
        except Exception as e:
            last_note = f"q={q} | err={repr(e)}"

    cache_write_text({"industry": "", "note": last_note}, fp)
    return "", last_note


# ===================== 大盘（全A）过滤 =====================
def resolve_market_index_daily(startdate: str, enddate: str) -> tuple[str, pd.DataFrame, str]:
    """
    尝试多个候选 code，返回 (ok_code, df, note)
    df 必须包含 trade_date/close，且非空
    """
    last_note = ""
    for c in MARKET_INDEX_CANDIDATES:
        try:
            df = fetch_daily(c, startdate, enddate)
            if df is not None and (not df.empty) and ("trade_date" in df.columns) and ("close" in df.columns):
                return c, df, "ok"
            last_note = f"try={c} got_empty_or_missing_cols"
        except Exception as e:
            last_note = f"try={c} err={repr(e)}"
            continue
    return "", pd.DataFrame(), last_note

def market_filter(index_daily: pd.DataFrame, t_trade: str) -> tuple[bool, str]:
    """
    满足其一不做：
      1) close_t < MA20_t
      2) close_t / close_{t-2} - 1 < -2.5%   （包括t日的3根K累计跌幅）
    """
    if index_daily is None or index_daily.empty:
        return False, "index_daily_empty"

    d = index_daily.copy()
    d = d.dropna(subset=["trade_date", "close"]).sort_values("trade_date").reset_index(drop=True)
    closes = pd.to_numeric(d["close"], errors="coerce")
    d["close"] = closes
    d = d.dropna(subset=["close"]).reset_index(drop=True)
    if d.empty:
        return False, "index_close_all_nan"

    dates = d["trade_date"].astype(str).tolist()
    arr = np.array(dates)
    idx = np.searchsorted(arr, t_trade, side="right") - 1
    if idx < 0 or idx >= len(dates) or dates[idx] != t_trade:
        return False, f"t_not_in_index:{t_trade}"

    c_t = float(d.loc[idx, "close"])

    # 条件1：MA20
    if idx >= MARKET_MA_N - 1:
        ma20 = float(d["close"].rolling(MARKET_MA_N).mean().iloc[idx])
        if np.isfinite(ma20) and c_t < ma20:
            return True, f"index_close<{MARKET_MA_N}MA ({c_t:.3f}<{ma20:.3f})"
    else:
        ma20 = np.nan

    # 条件2：3根累计跌幅（t-2 到 t）
    if idx >= 2:
        c_base = float(d.loc[idx - 2, "close"])
        if c_base > 0 and np.isfinite(c_base):
            cum3 = c_t / c_base - 1.0
            if np.isfinite(cum3) and cum3 < MARKET_3BAR_DROP_TH:
                return True, f"index_3bar_drop={cum3:.2%}<{MARKET_3BAR_DROP_TH:.2%}"
    else:
        cum3 = np.nan

    return False, f"index_ok ma20={ma20 if np.isfinite(ma20) else 'NA'} cum3={cum3 if np.isfinite(cum3) else 'NA'}"


# ===================== 过滤器（自然日 / 交易日根数，均包含 t-1 当天） =====================
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

def _slice_by_bars_include_end(daily: pd.DataFrame, idx_t: int, bars: int) -> pd.DataFrame:
    end_incl = idx_t - 1  # t-1
    start = end_incl - bars + 1
    if start < 0:
        return daily.iloc[0:0]
    return daily.iloc[start:end_incl + 1].copy()

def overheat_filter(daily: pd.DataFrame, idx_t: int, t_trade: str, tminus1: str) -> tuple[bool, str]:
    """
    三条满足其一就过滤：
    A: 以 t-1 为窗口末端（含 t-1）: 5天窗口涨停根数 >=3
    B: 以 t-1 为窗口末端（含 t-1）: 10天窗口涨幅 >30%
    C: t 当天实体跌幅 close/open-1 <= -9%
    """
    if daily.empty or idx_t < 0 or idx_t >= len(daily):
        return False, "daily/idx invalid"

    # C：t 当天实体跌幅
    row_t = daily.iloc[idx_t]
    ot = float(row_t["open"])
    ct = float(row_t["close"])
    body_drop_t = (ct / ot - 1.0) if ot > 0 else np.nan
    c_hit = (np.isfinite(body_drop_t) and body_drop_t <= T_BODY_DROP_TH)

    # A/B：以 t-1 为窗口末端
    if USE_CALENDAR_LOOKBACK:
        w_lu = _slice_by_calendar_days_include_end(daily, tminus1, LU_WINDOW_DAYS)
        w_ru = _slice_by_calendar_days_include_end(daily, tminus1, RUNUP_WINDOW_DAYS)
        mode_note = f"calendar_inc_tminus1(lu={LU_WINDOW_DAYS}d, runup={RUNUP_WINDOW_DAYS}d)"
    else:
        w_lu = _slice_by_bars_include_end(daily, idx_t, LU_WINDOW_BARS)
        w_ru = _slice_by_bars_include_end(daily, idx_t, RUNUP_WINDOW_BARS)
        mode_note = f"bars_inc_tminus1(lu={LU_WINDOW_BARS}, runup={RUNUP_WINDOW_BARS})"

    a_hit = False
    lu_cnt = np.nan
    if not w_lu.empty:
        lu_cnt = int((w_lu["close"].astype(float) == w_lu["high"].astype(float)).sum())
        if lu_cnt >= LU_MIN_CNT:
            a_hit = True

    b_hit = False
    runup = np.nan
    if len(w_ru) >= 2:
        c0 = float(w_ru.iloc[0]["close"])
        c1 = float(w_ru.iloc[-1]["close"])
        if np.isfinite(c0) and c0 > 0:
            runup = c1 / c0 - 1.0
            if np.isfinite(runup) and runup > RUNUP_TH:
                b_hit = True

    if a_hit or b_hit or c_hit:
        parts = [mode_note]
        if a_hit:
            parts.append(f"limitup_cnt={lu_cnt}>=3")
        if b_hit:
            parts.append(f"runup={runup:.2%}>30%")
        if c_hit:
            parts.append(f"t_body_drop={body_drop_t:.2%}<=-9%")
        return True, " | ".join(parts)

    return False, f"{mode_note} | t_body_drop={body_drop_t:.2%}"


# ===================== 回测单条信号 =====================
@dataclass
class ResultRow:
    code: str
    name: str
    ths_code: str
    industry: str
    t: str
    tminus1: str
    tplus1: str
    last_day: str
    entry_dt: str | None
    entry_px: float | None
    exit_dt: str | None
    exit_px: float | None
    shares: int
    initial_cash: float
    final_cash: float | None
    pnl: float | None
    ret_pct: float | None
    max_dd_pct: float | None
    mfe_pct: float | None
    mae_pct: float | None
    reason: str
    note: str

def backtest_one_signal(code: str, name: str, industry: str, t_raw: str,
                        daily: pd.DataFrame, m30_all: pd.DataFrame,
                        index_daily: pd.DataFrame) -> ResultRow:
    code = code6(code)
    t = ymd8(t_raw)
    ths_code = to_ths_code(code)

    if daily.empty:
        return ResultRow(code, name, ths_code, industry, t, "", "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_daily", "daily empty")

    dates = daily["trade_date"].tolist()
    arr = np.array(dates)

    idx = np.searchsorted(arr, t, side="right") - 1
    if idx < 1:
        return ResultRow(code, name, ths_code, industry, t, "", "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_t_or_tminus1", "t not found in daily range")

    t_trade = dates[idx]
    tminus1 = dates[idx - 1]

    if idx + 1 >= len(dates):
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_tplus1", "no t+1 in daily")

    tplus1 = dates[idx + 1]

    # last_day：从 t+1 开始含 t+1，当作 HOLD_DAYS 根交易日
    target_end_idx = idx + 1 + HOLD_DAYS - 1
    if REQUIRE_FULL_HOLD_WINDOW and target_end_idx >= len(dates):
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, tplus1, "",
                         None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None,
                         "no_enough_future_daily",
                         f"need_end_idx={target_end_idx}, but len(dates)={len(dates)} (daily ends at {dates[-1]})")

    end_idx = min(target_end_idx, len(dates) - 1)
    last_day = dates[end_idx]

    # 过滤1：你的三合一过滤（不改）
    hit, why = overheat_filter(daily, idx, t_trade, tminus1)
    if hit:
        return ResultRow(
            code, str(name), ths_code, str(industry),
            t_trade, tminus1, tplus1, last_day,
            None, None, None, None, 0, INIT_CASH,
            None, None, None, None, None, None,
            "filtered_overheat", why
        )

    # 过滤2：大盘（同花顺全A）过滤（新增）
    m_hit, m_note = market_filter(index_daily, t_trade)
    if m_hit:
        return ResultRow(
            code, str(name), ths_code, str(industry),
            t_trade, tminus1, tplus1, last_day,
            None, None, None, None, 0, INIT_CASH,
            None, None, None, None, None, None,
            "filtered_market", m_note
        )

    dmap = daily.set_index("trade_date")
    L = float(dmap.loc[t_trade, "low"])
    O = float(dmap.loc[t_trade, "open"])
    H_up = float(dmap.loc[tminus1, "high"])

    # 30m窗口切片
    start_dt = pd.to_datetime(f"{ymd_to_dash(tplus1)} 09:30:00")
    end_dt   = pd.to_datetime(f"{ymd_to_dash(last_day)} 15:00:00")
    m30 = m30_all[(m30_all["dt"] >= start_dt) & (m30_all["dt"] <= end_dt)].copy()
    if m30.empty:
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_30m_range",
                         f"30m empty in range {start_dt}~{end_dt}")

    # 入场：t+1 09:30~10:30 触L
    win = m30[(m30["dt"] >= pd.to_datetime(f"{ymd_to_dash(tplus1)} 09:30:00")) &
              (m30["dt"] <  pd.to_datetime(f"{ymd_to_dash(tplus1)} 10:30:00"))].copy()
    if win.empty:
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_entry_window_empty", "t+1 60m window empty")

    entry_i = None
    for ridx in win.index:
        lo, hi = float(m30.loc[ridx, "low"]), float(m30.loc[ridx, "high"])
        if lo <= L <= hi:
            entry_i = int(np.where(m30.index == ridx)[0][0])
            break

    if entry_i is None:
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_entry", f"t+1 60m not touch L={L:.3f}")

    entry_dt = m30.iloc[entry_i]["dt"]
    entry_px = L

    buy_px = entry_px * (1 + FEE_RATE)
    shares = lot_shares(INIT_CASH, buy_px)
    if shares <= 0:
        return ResultRow(code, name, ths_code, industry, t_trade, tminus1, tplus1, last_day, str(entry_dt), float(entry_px),
                         None, None, 0, INIT_CASH, None, None, None, None, None, None, "no_cash", "cash not enough for 100 shares")

    cash_left = INIT_CASH - shares * buy_px

    stop = INIT_STOP_MULT * L
    entry_day = pd.to_datetime(entry_dt).strftime("%Y%m%d")

    chandelier_on = False
    hh = -np.inf

    touchedH_intraday = False
    touchedO_intraday = False
    pending_apply_H = False
    pending_apply_O = False
    pending_turn_on_chandelier = False

    pending_exit = False
    pending_reason = ""

    exit_dt = None
    exit_px = None
    reason = "time_exit"

    eq = []
    eq_t = []
    prev_day = None

    def apply_pending_updates(cur_day):
        nonlocal stop, pending_apply_H, pending_apply_O, pending_turn_on_chandelier, chandelier_on, hh
        if cur_day == entry_day:
            pending_apply_H = False
            pending_apply_O = False
            pending_turn_on_chandelier = False
            return
        if pending_apply_H:
            stop = max(stop, H_up)
            pending_apply_H = False
        if pending_apply_O:
            stop = max(stop, O)
            pending_apply_O = False
        if pending_turn_on_chandelier:
            chandelier_on = True
            hh = -np.inf
            pending_turn_on_chandelier = False

    for i in range(entry_i, len(m30)):
        bar = m30.iloc[i]
        dt = bar["dt"]
        day = bar["d"]
        o30 = float(bar["open"])
        h30 = float(bar["high"])
        l30 = float(bar["low"])
        c30 = float(bar["close"])
        atrv = float(bar["atr"]) if pd.notna(bar["atr"]) else np.nan

        if prev_day is None or day != prev_day:
            apply_pending_updates(day)
            prev_day = day

        eq.append(cash_left + shares * c30 * (1 - FEE_RATE))
        eq_t.append(dt)

        if pending_exit and day != entry_day:
            exit_dt = dt
            exit_px = o30 * (1 - FEE_RATE)
            reason = pending_reason + "_Tplus1_forced"
            break

        if l30 <= stop:
            if day == entry_day:
                pending_exit = True
                pending_reason = "stop_hit"
            else:
                exec_px = o30 if o30 <= stop else stop
                exit_dt = dt
                exit_px = exec_px * (1 - FEE_RATE)
                reason = "stop_hit"
                break

        if day == entry_day:
            if h30 >= H_up:
                touchedH_intraday = True
            if h30 >= O:
                touchedO_intraday = True
            continue

        if h30 >= H_up:
            touchedH_intraday = True
        if h30 >= O:
            touchedO_intraday = True

        if chandelier_on:
            hh = max(hh, h30)
            if np.isfinite(atrv) and atrv > 0:
                stop = max(stop, hh - CH_K * atrv)

        is_day_end = (i == len(m30) - 1) or (m30.iloc[i + 1]["d"] != day)
        if is_day_end:
            day_close = float(dmap.loc[day, "close"]) if day in dmap.index else c30
            if touchedH_intraday and (day_close >= H_up):
                pending_apply_H = True
            if touchedO_intraday and (day_close >= O):
                pending_apply_O = True
                pending_turn_on_chandelier = True
            touchedH_intraday = False
            touchedO_intraday = False

    if exit_dt is None:
        last_bar = m30.iloc[-1]
        exit_dt = last_bar["dt"]
        exit_px = float(last_bar["close"]) * (1 - FEE_RATE)
        reason = "time_exit"

    final_cash = cash_left + shares * float(exit_px)
    pnl = final_cash - INIT_CASH
    ret_pct = pnl / INIT_CASH * 100.0

    eqs = pd.Series(eq, index=pd.to_datetime(eq_t))
    eqs = eqs[eqs.index <= pd.to_datetime(exit_dt)]
    peak = eqs.cummax()
    max_dd_pct = float((eqs / peak - 1.0).min() * 100.0) if len(eqs) else np.nan

    hold = m30.iloc[entry_i:].copy()
    hold = hold[hold["dt"] <= pd.to_datetime(exit_dt)]
    mfe_pct = (float(hold["high"].max()) / entry_px - 1.0) * 100.0 if not hold.empty else np.nan
    mae_pct = (float(hold["low"].min()) / entry_px - 1.0) * 100.0 if not hold.empty else np.nan

    note = f"L={L:.3f}, O={O:.3f}, H_up={H_up:.3f}, stop0={INIT_STOP_MULT*L:.3f}"
    return ResultRow(
        code=code, name=str(name), ths_code=ths_code, industry=str(industry),
        t=t_trade, tminus1=tminus1, tplus1=tplus1, last_day=last_day,
        entry_dt=str(entry_dt), entry_px=float(entry_px),
        exit_dt=str(exit_dt), exit_px=float(exit_px),
        shares=int(shares),
        initial_cash=float(INIT_CASH),
        final_cash=float(final_cash),
        pnl=float(pnl),
        ret_pct=float(ret_pct),
        max_dd_pct=float(max_dd_pct) if np.isfinite(max_dd_pct) else np.nan,
        mfe_pct=float(mfe_pct) if np.isfinite(mfe_pct) else np.nan,
        mae_pct=float(mae_pct) if np.isfinite(mae_pct) else np.nan,
        reason=reason,
        note=note,
    )


# ===================== 主流程 =====================
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_XLSX}")

    xls = pd.read_excel(INPUT_XLSX, sheet_name=None)
    df_all = []
    for sname, df in xls.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["__sheet"] = sname
        df_all.append(tmp)
    sig = pd.concat(df_all, ignore_index=True)

    if "ts_code" not in sig.columns or "date_t" not in sig.columns:
        raise RuntimeError("Excel里必须有 ts_code 和 date_t 两列")
    if "stock_name" not in sig.columns:
        sig["stock_name"] = ""

    sig["code"] = sig["ts_code"].apply(code6)
    sig["t"] = sig["date_t"].apply(ymd8)
    sig["name"] = sig["stock_name"].astype(str)

    # ===== 先准备大盘指数日线（一次拉够整个区间）=====
    all_t = sig["t"].dropna().astype(str)
    if all_t.empty:
        raise RuntimeError("信号表 date_t 解析为空")

    tmin = pd.to_datetime(all_t.min(), format="%Y%m%d", errors="coerce")
    tmax = pd.to_datetime(all_t.max(), format="%Y%m%d", errors="coerce")
    if pd.isna(tmin) or pd.isna(tmax):
        raise RuntimeError("无法解析信号表最小/最大日期")

    # 多留些历史用于MA20/3bar
    idx_start = ymd_to_dash((tmin - pd.Timedelta(days=260)).strftime("%Y%m%d"))
    idx_end   = ymd_to_dash((tmax + pd.Timedelta(days=40)).strftime("%Y%m%d"))

    market_code, index_daily, idx_note = resolve_market_index_daily(idx_start, idx_end)
    if index_daily.empty:
        raise RuntimeError(f"全A指数日线拉取失败，请检查 MARKET_INDEX_CANDIDATES。note={idx_note}")
    print(f"[Market] Using index code: {market_code}  (range {idx_start}~{idx_end})")

    results = []
    errors = []

    # ===== 先把不做的票直接落结果（避免你看不到为什么没了）=====
    sig["excluded"] = sig["code"].apply(is_excluded_code)
    excluded_rows = sig[sig["excluded"]].copy()
    sig_run = sig[~sig["excluded"]].copy()

    for _, row in excluded_rows.iterrows():
        results.append(ResultRow(
            code=row["code"],
            name=str(row["name"]),
            ths_code=to_ths_code(row["code"]),
            industry="",
            t=str(row["t"]),
            tminus1="",
            tplus1="",
            last_day="",
            entry_dt=None,
            entry_px=None,
            exit_dt=None,
            exit_px=None,
            shares=0,
            initial_cash=float(INIT_CASH),
            final_cash=None,
            pnl=None,
            ret_pct=None,
            max_dd_pct=None,
            mfe_pct=None,
            mae_pct=None,
            reason="excluded_prefix",
            note=f"excluded because code startswith {EXCLUDE_PREFIXES}"
        ).__dict__)

    # ===== 按股票分组：每只股票拉一次日线+一次30m区间 =====
    for code, g in tqdm(list(sig_run.groupby("code")), desc="By stock"):
        ths_code = to_ths_code(code)
        t_list = sorted(g["t"].astype(str).tolist())

        # 行业：每只股票查一次（保留）
        try:
            industry, ind_note = fetch_industry(ths_code)
        except Exception as e:
            industry, ind_note = "", f"industry_err={repr(e)}"

        # 日线范围
        tmin_s = pd.to_datetime(t_list[0], format="%Y%m%d", errors="coerce")
        tmax_s = pd.to_datetime(t_list[-1], format="%Y%m%d", errors="coerce")
        if pd.isna(tmin_s) or pd.isna(tmax_s):
            for _, row in g.iterrows():
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "date_parse", "err": "bad t", "industry": industry})
            continue

        startdate = ymd_to_dash((tmin_s - pd.Timedelta(days=260)).strftime("%Y%m%d"))
        enddate   = ymd_to_dash((tmax_s + pd.Timedelta(days=120)).strftime("%Y%m%d"))

        try:
            daily = fetch_daily(ths_code, startdate, enddate)
        except Exception as e:
            for _, row in g.iterrows():
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "daily", "err": repr(e), "industry": industry})
            continue

        if daily.empty:
            for _, row in g.iterrows():
                d = ResultRow(code, row["name"], ths_code, industry, row["t"], "", "", "", None, None, None, None,
                              0, INIT_CASH, None, None, None, None, None, None, "no_daily", "daily empty").__dict__
                d["industry_note"] = ind_note
                results.append(d)
            continue

        dates = daily["trade_date"].tolist()
        arr = np.array(dates)

        # 预估30m拉取范围
        tplus1_list = []
        lastday_list = []
        for t in t_list:
            idx = np.searchsorted(arr, t, side="right") - 1
            if idx < 1 or idx + 1 >= len(dates):
                continue
            tplus1 = dates[idx + 1]
            target_end_idx = idx + 1 + HOLD_DAYS - 1
            end_idx = min(target_end_idx, len(dates) - 1)
            last_day = dates[end_idx]
            tplus1_list.append(tplus1)
            lastday_list.append(last_day)

        if not tplus1_list:
            for _, row in g.iterrows():
                d = ResultRow(code, row["name"], ths_code, industry, row["t"], "", "", "", None, None, None, None,
                              0, INIT_CASH, None, None, None, None, None, None, "no_tplus1", "no t+1 in daily").__dict__
                d["industry_note"] = ind_note
                results.append(d)
            continue

        hf_start = f"{ymd_to_dash(str(min(tplus1_list)))} 09:30:00"
        hf_end   = f"{ymd_to_dash(str(max(lastday_list)))} 15:00:00"

        try:
            m30_all, debug = fetch_hf_30m(ths_code, hf_start, hf_end)
        except Exception as e:
            for _, row in g.iterrows():
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "hf", "err": repr(e), "industry": industry})
            continue

        if m30_all.empty:
            for _, row in g.iterrows():
                d = ResultRow(code, row["name"], ths_code, industry, row["t"], "", "", "", None, None, None, None,
                              0, INIT_CASH, None, None, None, None, None, None, "no_30m", f"hf empty; debug={debug}").__dict__
                d["industry_note"] = ind_note
                results.append(d)
            continue

        # 对每条信号回测（新增：传 index_daily 进去做大盘过滤）
        for _, row in g.iterrows():
            try:
                r = backtest_one_signal(
                    code=row["code"],
                    name=row["name"],
                    industry=industry,
                    t_raw=row["t"],
                    daily=daily,
                    m30_all=m30_all,
                    index_daily=index_daily
                )
                d = r.__dict__.copy()
                d["industry_note"] = ind_note
                d["market_index_code"] = market_code
                results.append(d)
            except Exception as e:
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "backtest", "err": repr(e), "industry": industry})

    df_res = pd.DataFrame(results)
    df_err = pd.DataFrame(errors)

    if not df_res.empty:
        df_res = df_res.sort_values(["t", "code"]).reset_index(drop=True)

    # 行业汇总（保留）
    df_ind = pd.DataFrame()
    if not df_res.empty and "industry" in df_res.columns:
        tmp = df_res.copy()
        tmp["industry"] = tmp["industry"].fillna("").astype(str).replace({"": "未知/未取到"})
        tmp["is_win"] = tmp["ret_pct"].fillna(0) > 0

        df_ind = (
            tmp.groupby("industry", dropna=False)
               .agg(
                   samples=("ret_pct", "count"),
                   win_rate=("is_win", "mean"),
                   avg_ret_pct=("ret_pct", "mean"),
                   med_ret_pct=("ret_pct", "median"),
                   total_pnl=("pnl", "sum"),
                   avg_pnl=("pnl", "mean"),
                   avg_max_dd_pct=("max_dd_pct", "mean"),
                   filtered_overheat_cnt=("reason", lambda s: int((s == "filtered_overheat").sum())),
                   filtered_market_cnt=("reason", lambda s: int((s == "filtered_market").sum())),
                   excluded_cnt=("reason", lambda s: int((s == "excluded_prefix").sum()))
               )
               .reset_index()
        )
        df_ind["win_rate"] = df_ind["win_rate"] * 100.0
        df_ind = df_ind[df_ind["samples"] >= INDUSTRY_MIN_SAMPLES].sort_values(
            ["avg_ret_pct", "win_rate", "samples"], ascending=[False, False, False]
        ).reset_index(drop=True)

    out_path = safe_excel_writer_path(OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_res.to_excel(w, index=False, sheet_name="results")
        if not df_ind.empty:
            df_ind.to_excel(w, index=False, sheet_name="industry_summary")
        df_err.to_excel(w, index=False, sheet_name="errors")

    print("Saved:", out_path)
    if not df_ind.empty:
        print("\n=== Top industries by avg_ret_pct ===")
        print(df_ind.head(20).to_string(index=False))
    if not df_res.empty:
        print("\n=== Results head ===")
        print(df_res.head(10).to_string(index=False))
    if not df_err.empty:
        print("\nErrors head:")
        print(df_err.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
