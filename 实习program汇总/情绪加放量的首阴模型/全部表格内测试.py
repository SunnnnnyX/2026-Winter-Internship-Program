
import os
import re
import time
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests
from tqdm import tqdm

# ===================== 你要改的配置 =====================
INPUT_XLSX = Path(r"D:\a_share_daily\无st涨停重设后_纯数字代码.xlsx")  # 你的信号表
OUTPUT_XLSX = Path(r"D:\batch_backtest_ifind_results.xlsx")

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
HOLD_DAYS = 7

# 缓存（强烈建议开，否则534条会很慢）
CACHE_DIR = Path(r"D:\ifind_cache_batch")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 请求节奏
SLEEP = 0.15             # 防止被限流，太快就加大
RETRY = 3



def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def ymd8(x) -> str:
    s = str(x).strip().replace("-", "")
    s = re.sub(r"\D", "", s)
    return s[:8]

def ymd_to_dash(ymd) -> str:
    # 核心修复：兼容int/str类型的8位日期，彻底解决TypeError
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
        thscode = t.get("thscode")
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None:
            df.insert(0, "time", times)
        if thscode is not None:
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
    # 缓存按日期范围（精确到天，避免文件爆炸）
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

            # 写缓存（dt转字符串存）
            tmp = df.copy()
            tmp["dt"] = tmp["dt"].astype(str)
            cache_write(tmp, fp)

            return df, debug
        except Exception as e:
            debug["last_err"] = repr(e)

    return pd.DataFrame(), debug


# ===================== 回测单条信号（复用同一套逻辑） =====================
@dataclass
class ResultRow:
    code: str
    name: str
    ths_code: str
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

def backtest_one_signal(code: str, name: str, t_raw: str,
                        daily: pd.DataFrame, m30_all: pd.DataFrame) -> ResultRow:
    code = code6(code)
    t = ymd8(t_raw)
    ths_code = to_ths_code(code)

    if daily.empty:
        return ResultRow(code, name, ths_code, t, "", "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_daily", "daily empty")

    dates = daily["trade_date"].tolist()
    arr = np.array(dates)

    idx = np.searchsorted(arr, t, side="right") - 1
    if idx < 1:
        return ResultRow(code, name, ths_code, t, "", "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_t_or_tminus1", "t not found in daily range")

    t_trade = dates[idx]
    tminus1 = dates[idx - 1]
    if idx + 1 >= len(dates):
        return ResultRow(code, name, ths_code, t_trade, tminus1, "", "", None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_tplus1", "no t+1 in daily")

    tplus1 = dates[idx + 1]
    end_idx = min(idx + 1 + HOLD_DAYS - 1, len(dates) - 1)
    last_day = dates[end_idx]

    dmap = daily.set_index("trade_date")
    L = float(dmap.loc[t_trade, "low"])
    O = float(dmap.loc[t_trade, "open"])
    H_up = float(dmap.loc[tminus1, "high"])

    # 30m窗口切片
    start_dt = pd.to_datetime(f"{ymd_to_dash(tplus1)} 09:30:00")
    end_dt   = pd.to_datetime(f"{ymd_to_dash(last_day)} 15:00:00")
    m30 = m30_all[(m30_all["dt"] >= start_dt) & (m30_all["dt"] <= end_dt)].copy()
    if m30.empty:
        return ResultRow(code, name, ths_code, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_30m_range",
                         f"30m empty in range {start_dt}~{end_dt}")

    # 入场：t+1 09:30~10:30 触L
    win = m30[(m30["dt"] >= pd.to_datetime(f"{ymd_to_dash(tplus1)} 09:30:00")) &
              (m30["dt"] <  pd.to_datetime(f"{ymd_to_dash(tplus1)} 10:30:00"))].copy()
    if win.empty:
        return ResultRow(code, name, ths_code, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_entry_window_empty", "t+1 60m window empty")

    entry_i = None
    for ridx in win.index:
        lo, hi = float(m30.loc[ridx, "low"]), float(m30.loc[ridx, "high"])
        if lo <= L <= hi:
            entry_i = int(np.where(m30.index == ridx)[0][0])
            break

    if entry_i is None:
        return ResultRow(code, name, ths_code, t_trade, tminus1, tplus1, last_day, None, None, None, None, 0, INIT_CASH,
                         None, None, None, None, None, None, "no_entry", f"t+1 60m not touch L={L:.3f}")

    entry_dt = m30.iloc[entry_i]["dt"]
    entry_px = L

    buy_px = entry_px * (1 + FEE_RATE)
    shares = lot_shares(INIT_CASH, buy_px)
    if shares <= 0:
        return ResultRow(code, name, ths_code, t_trade, tminus1, tplus1, last_day, str(entry_dt), float(entry_px),
                         None, None, 0, INIT_CASH, None, None, None, None, None, None, "no_cash", "cash not enough for 100 shares")

    cash_left = INIT_CASH - shares * buy_px

    # 状态（方案1&2）
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

    # 指标
    eq = []
    eq_t = []
    prev_day = None

    def apply_pending_updates(cur_dt, cur_day):
        nonlocal stop, pending_apply_H, pending_apply_O, pending_turn_on_chandelier, chandelier_on, hh
        # 方案2：入场当日不抬、不启吊灯
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

        # 日切换：应用“昨日确认、今日生效”
        if prev_day is None or day != prev_day:
            apply_pending_updates(dt, day)
            prev_day = day

        eq.append(cash_left + shares * c30 * (1 - FEE_RATE))
        eq_t.append(dt)

        # T+1：入场当日触发卖出意图，次日第一根bar开盘强制出
        if pending_exit and day != entry_day:
            exit_dt = dt
            exit_px = o30 * (1 - FEE_RATE)
            reason = pending_reason + "_Tplus1_forced"
            break

        # 止损
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

        # 方案2：入场当日不抬、不启吊灯
        if day == entry_day:
            if h30 >= H_up:
                touchedH_intraday = True
            if h30 >= O:
                touchedO_intraday = True
            continue

        # 方案1：盘中触及只记录
        if h30 >= H_up:
            touchedH_intraday = True
        if h30 >= O:
            touchedO_intraday = True

        # 吊灯
        if chandelier_on:
            hh = max(hh, h30)
            if np.isfinite(atrv) and atrv > 0:
                stop = max(stop, hh - CH_K * atrv)

        # 日终：收盘确认 => 次日生效
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
        code=code, name=str(name), ths_code=ths_code,
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


# ===================== 主流程：按代码分组，减少HTTP调用 =====================
def main():
    if not INPUT_XLSX.exists():
        raise FileNotFoundError(f"找不到输入文件：{INPUT_XLSX}")

    xls = pd.read_excel(INPUT_XLSX, sheet_name=None)
    # 你这份表一般有两个sheet，直接合并
    df_all = []
    for sname, df in xls.items():
        if df is None or df.empty:
            continue
        tmp = df.copy()
        tmp["__sheet"] = sname
        df_all.append(tmp)
    sig = pd.concat(df_all, ignore_index=True)

    # 必要列
    if "ts_code" not in sig.columns or "date_t" not in sig.columns:
        raise RuntimeError("Excel里必须有 ts_code 和 date_t 两列（你这份表本来就有）")
    if "stock_name" not in sig.columns:
        sig["stock_name"] = ""

    # 标准化
    sig["code"] = sig["ts_code"].apply(code6)
    sig["t"] = sig["date_t"].apply(ymd8)
    sig["name"] = sig["stock_name"].astype(str)

    results = []
    errors = []

    # 按股票分组：每只股票只拉一次日线+一次30m大区间，然后切片跑每条信号
    for code, g in tqdm(list(sig.groupby("code")), desc="By stock"):
        ths_code = to_ths_code(code)
        t_list = sorted(g["t"].tolist())

        # 日线范围：覆盖这只股票所有信号
        tmin = pd.to_datetime(t_list[0])
        tmax = pd.to_datetime(t_list[-1])
        startdate = ymd_to_dash((tmin - pd.Timedelta(days=260)).strftime("%Y%m%d"))
        enddate   = ymd_to_dash((tmax + pd.Timedelta(days=120)).strftime("%Y%m%d"))

        try:
            daily = fetch_daily(ths_code, startdate, enddate)
        except Exception as e:
            for _, row in g.iterrows():
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "daily", "err": repr(e)})
            continue

        # 先用日线把每条信号的 tplus1/last_day 预估出来，用来决定30m拉取范围
        if daily.empty:
            for _, row in g.iterrows():
                results.append(ResultRow(code, row["name"], ths_code, row["t"], "", "", "", None, None, None, None,
                                         0, INIT_CASH, None, None, None, None, None, None, "no_daily", "daily empty").__dict__)
            continue

        dates = daily["trade_date"].tolist()
        arr = np.array(dates)

        tplus1_list = []
        lastday_list = []
        for t in t_list:
            idx = np.searchsorted(arr, t, side="right") - 1
            if idx < 1 or idx + 1 >= len(dates):
                continue
            tplus1 = dates[idx + 1]
            end_idx = min(idx + 1 + HOLD_DAYS - 1, len(dates) - 1)
            last_day = dates[end_idx]
            tplus1_list.append(tplus1)
            lastday_list.append(last_day)

        if not tplus1_list:
            for _, row in g.iterrows():
                results.append(ResultRow(code, row["name"], ths_code, row["t"], "", "", "", None, None, None, None,
                                         0, INIT_CASH, None, None, None, None, None, None, "no_tplus1", "no t+1 in daily").__dict__)
            continue

        # 核心修复：强转str，双重保险避免min/max隐式转int
        hf_start = f"{ymd_to_dash(str(min(tplus1_list)))} 09:30:00"
        hf_end   = f"{ymd_to_dash(str(max(lastday_list)))} 15:00:00"

        try:
            m30_all, debug = fetch_hf_30m(ths_code, hf_start, hf_end)
        except Exception as e:
            for _, row in g.iterrows():
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "hf", "err": repr(e)})
            continue

        if m30_all.empty:
            for _, row in g.iterrows():
                results.append(ResultRow(code, row["name"], ths_code, row["t"], "", "", "", None, None, None, None,
                                         0, INIT_CASH, None, None, None, None, None, None, "no_30m", f"hf empty; debug={debug}").__dict__)
            continue

        # 对每条信号跑回测
        for _, row in g.iterrows():
            try:
                r = backtest_one_signal(code=row["code"], name=row["name"], t_raw=row["t"], daily=daily, m30_all=m30_all)
                results.append(r.__dict__)
            except Exception as e:
                errors.append({"code": code, "name": row["name"], "t": row["t"], "stage": "backtest", "err": repr(e)})

    df_res = pd.DataFrame(results)
    df_err = pd.DataFrame(errors)

    # 排序更好看
    if not df_res.empty:
        df_res = df_res.sort_values(["t", "code"]).reset_index(drop=True)

    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        df_res.to_excel(w, index=False, sheet_name="results")
        df_err.to_excel(w, index=False, sheet_name="errors")

    print("Saved:", OUTPUT_XLSX)
    if not df_res.empty:
        print(df_res.head(10).to_string(index=False))
    if not df_err.empty:
        print("\nErrors head:")
        print(df_err.head(10).to_string(index=False))


if __name__ == "__main__":
    main()