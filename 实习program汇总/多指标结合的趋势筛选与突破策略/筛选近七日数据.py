# -*- coding: utf-8 -*-
"""
02_scan_last7td_vA.py

从 SQLite 读取日线，计算指标，扫描“最近7个已收盘交易日”内的版本A信号并导出Excel
- 不下载数据
- 扫描窗口按 trade_cal 最近N个交易日
- 白天运行不会扫描“今天未收盘”的数据

依赖：
pip install tushare pandas openpyxl
环境变量：
TUSHARE_TOKEN=你的token
"""

import os
import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd
import tushare as ts


# ===================== 配置 =====================
DB_PATH = Path(r"D:\a_share_daily_tushare_new.sqlite")   # <- 指向 01 脚本生成的库
OUT_XLSX = Path(r"D:\scan_signals_last7td.xlsx")

SCAN_LAST_N_TRADING_DAYS = 7

# 收盘缓冲：15:10后才认为“今日已收盘”
CLOSE_CUTOFF_HHMM = "15:10"

# 版本A参数
BPS_BREAK = 0.003
VOL_MULT  = 1.8

# 新增：单K涨幅硬条件
MIN_PCT = 0.08

# 涨停标签阈值（严格）
LIMIT_TAG_PCT = 0.099
# ===============================================


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def compute_and_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = df["close"]; h = df["high"]; l = df["low"]; v = df["volume"]

    df["pct"] = c / c.shift(1) - 1.0

    df["ma5"]  = c.rolling(5,  min_periods=5).mean()
    df["ma20"] = c.rolling(20, min_periods=20).mean()

    mb = df["ma20"]
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    df["mb20"] = mb
    df["ub20"] = mb + 2 * sd
    df["lb20"] = mb - 2 * sd
    df["bw"]   = (df["ub20"] - df["lb20"]) / df["mb20"]

    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    df["dif"] = dif
    df["dea"] = dea
    df["hist"] = dif - dea

    df["res20"] = h.rolling(20, min_periods=20).max().shift(1)
    df["vma20"] = v.rolling(20, min_periods=20).mean().shift(1)

    df["break_pct"] = df["close"] / df["res20"] - 1.0
    denom = (h - l).replace(0, np.nan)
    df["close_pos"] = (df["close"] - l) / denom
    df["vol_mult"] = df["volume"] / df["vma20"]

    c1 = df["close"] > df["res20"] * (1 + BPS_BREAK)
    c2 = (df["close"] > df["mb20"]) & (df["mb20"] > df["mb20"].shift(1))
    c3 = df["dif"] > df["dea"]
    c4 = (df["close"] > df["ma5"]) & (df["ma5"] > df["ma5"].shift(1))
    c5 = df["volume"] > df["vma20"] * VOL_MULT
    c6 = df["pct"] >= MIN_PCT

    df["signal_vA"] = c1 & c2 & c3 & c4 & c5 & c6
    df["is_limit_up"] = df["pct"] >= LIMIT_TAG_PCT
    return df


def last_closed_trading_day(pro) -> pd.Timestamp:
    now = pd.Timestamp.now()
    today = now.normalize()

    end = today.strftime("%Y%m%d")
    start = (today - pd.Timedelta(days=60)).strftime("%Y%m%d")

    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, fields="cal_date,is_open")
    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal = cal.sort_values("cal_date")
    if cal.empty:
        raise RuntimeError("trade_cal取不到交易日")

    last_td = cal["cal_date"].iloc[-1]

    hh, mm = map(int, CLOSE_CUTOFF_HHMM.split(":"))
    close_cutoff = today + pd.Timedelta(hours=hh, minutes=mm)

    if last_td == today and now < close_cutoff:
        if len(cal) < 2:
            return last_td
        return cal["cal_date"].iloc[-2]

    return last_td


def get_last_n_trading_days_ending_at(pro, end_td: pd.Timestamp, n: int) -> list[pd.Timestamp]:
    """
    取以 end_td 为终点的最近 n 个交易日（开市日）
    """
    end = end_td.strftime("%Y%m%d")
    start = (end_td - pd.Timedelta(days=120)).strftime("%Y%m%d")  # 留足够缓冲
    cal = pro.trade_cal(exchange="SSE", start_date=start, end_date=end, fields="cal_date,is_open")
    cal = cal[cal["is_open"] == 1].copy()
    cal["cal_date"] = pd.to_datetime(cal["cal_date"])
    cal = cal.sort_values("cal_date")

    days = cal["cal_date"].tolist()
    if len(days) <= n:
        return days
    return days[-n:]


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到数据库：{DB_PATH}")

    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN（setx 后重启 Spyder/终端）")

    ts.set_token(token)
    pro = ts.pro_api()

    # ===== 关键：扫描只到最后一个已收盘交易日 =====
    end_td = last_closed_trading_day(pro)
    last_days = get_last_n_trading_days_ending_at(pro, end_td, SCAN_LAST_N_TRADING_DAYS)
    if not last_days:
        raise RuntimeError("trade_cal 没取到交易日")

    scan_start = last_days[0].strftime("%Y-%m-%d")
    scan_end   = last_days[-1].strftime("%Y-%m-%d")
    start_dt = pd.to_datetime(scan_start)
    end_dt   = pd.to_datetime(scan_end)

    print(f"[ScanWindow] last {SCAN_LAST_N_TRADING_DAYS} closed trading days: {scan_start} ~ {scan_end}")

    conn = sqlite3.connect(DB_PATH.as_posix())
    uni = pd.read_sql_query("SELECT ts_code, symbol, name FROM universe", conn)
    if uni.empty:
        conn.close()
        raise RuntimeError("universe 表为空：先跑 01 脚本建库")

    all_records = []
    latest_records = []

    for _, r in uni.iterrows():
        ts_code = r["ts_code"]
        symbol = r.get("symbol", "")
        name = r.get("name", "")

        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM daily WHERE ts_code=? ORDER BY date",
            conn, params=(ts_code,)
        )
        if df.empty or len(df) < 80:
            continue

        df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()
        if len(df) < 80:
            continue

        out = compute_and_signal(df)

        sig = out.loc[
            (out["date"] >= start_dt) &
            (out["date"] <= end_dt) &
            (out["signal_vA"])
        ].copy()
        if sig.empty:
            continue

        sig.insert(0, "ts_code", ts_code)
        sig.insert(1, "symbol", symbol)
        sig.insert(2, "name", name)

        all_records.append(sig)
        latest_records.append(sig.sort_values("date").iloc[-1:].copy())

    conn.close()

    if not all_records:
        print("No signals found in scan window.")
        return

    all_df = pd.concat(all_records, ignore_index=True)
    latest_df = pd.concat(latest_records, ignore_index=True)

    sort_cols = ["date", "is_limit_up", "vol_mult", "break_pct", "hist"]
    all_df = all_df.sort_values(sort_cols, ascending=[False, False, False, False, False])
    latest_df = latest_df.sort_values(sort_cols, ascending=[False, False, False, False, False])

    keep_cols = [
        "ts_code", "symbol", "name", "date",
        "open","high","low","close","volume",
        "pct", "is_limit_up",
        "res20","break_pct",
        "mb20","bw",
        "ma5",
        "dif","dea","hist",
        "vma20","vol_mult",
        "close_pos",
    ]

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        latest_df[keep_cols].to_excel(writer, sheet_name="latest_per_stock", index=False)
        all_df[keep_cols].to_excel(writer, sheet_name="all_signals", index=False)

    print(f"Saved: {OUT_XLSX}")
    print(f"Signals rows: {len(all_df)} ; Latest rows: {len(latest_df)}")


if __name__ == "__main__":
    main()
