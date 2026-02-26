# -*- coding: utf-8 -*-
"""
Step 2: 从 SQLite 读取日线，按版本A筛选，并加入你的新条件：
- 硬条件：信号日 单K收盘涨幅 >= 8%
- 涨停仅做标签：只用涨幅判定（严格：收盘涨幅接近10%，炸板不算）
- 不额外过滤一字板（不加 high==low 之类）

输出：
- latest_per_stock：每只股票区间内最近一次触发
- all_signals：区间内所有触发日明细
"""

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


# ===================== 配置 =====================
DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")
OUT_XLSX = Path(r"D:\breakout_vA_tushare_20251201_to_now.xlsx")

SIGNAL_START = "2025-12-01"
SIGNAL_END   = "2026-02-06"

# 版本A原参数
BPS_BREAK = 0.003
VOL_MULT  = 1.8

# 新增：单K涨幅硬条件
MIN_PCT = 0.08

# 涨停标签：只用涨幅（严格，炸板不算）
# 你若更严格可改 0.0995
LIMIT_TAG_PCT = 0.099


# ===================== 指标 =====================
def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()

def compute_and_signal(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = df["close"]
    h = df["high"]
    l = df["low"]
    v = df["volume"]

    # === 新增：单日收盘涨幅 ===
    df["pct"] = c / c.shift(1) - 1.0

    # === MA ===
    df["ma5"]  = c.rolling(5,  min_periods=5).mean()
    df["ma20"] = c.rolling(20, min_periods=20).mean()

    # === BOLL(20,2) ===
    mb = df["ma20"]
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    df["mb20"] = mb
    df["ub20"] = mb + 2 * sd
    df["lb20"] = mb - 2 * sd
    df["bw"]   = (df["ub20"] - df["lb20"]) / df["mb20"]

    # === MACD(12,26,9) ===
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    df["dif"] = dif
    df["dea"] = dea
    df["hist"] = dif - dea

    # === Res20 & VMA20：用 t-1 ===
    df["res20"] = h.rolling(20, min_periods=20).max().shift(1)
    df["vma20"] = v.rolling(20, min_periods=20).mean().shift(1)

    # === 辅助 ===
    df["break_pct"] = df["close"] / df["res20"] - 1.0
    denom = (h - l).replace(0, np.nan)
    df["close_pos"] = (df["close"] - l) / denom
    df["vol_mult"] = df["volume"] / df["vma20"]

    # === 版本A条件 ===
    c1 = df["close"] > df["res20"] * (1 + BPS_BREAK)
    c2 = (df["close"] > df["mb20"]) & (df["mb20"] > df["mb20"].shift(1))
    c3 = df["dif"] > df["dea"]
    c4 = (df["close"] > df["ma5"]) & (df["ma5"] > df["ma5"].shift(1))
    c5 = df["volume"] > df["vma20"] * VOL_MULT

    # === 新增硬条件：单K涨幅>=8% ===
    c6 = df["pct"] >= MIN_PCT

    df["signal_vA"] = c1 & c2 & c3 & c4 & c5 & c6

    # === 涨停标签：只用涨幅（严格），不参与筛选 ===
    df["is_limit_up"] = df["pct"] >= LIMIT_TAG_PCT

    return df


def main():
    conn = sqlite3.connect(DB_PATH.as_posix())

    # 取股票列表（ts_code + symbol + name）
    uni = pd.read_sql_query("SELECT ts_code, symbol, name FROM universe", conn)
    if uni.empty:
        raise RuntimeError("universe 表为空：请先运行 01_download_to_sqlite_tushare.py")

    start_dt = pd.to_datetime(SIGNAL_START)
    end_dt   = pd.to_datetime(SIGNAL_END)

    all_records = []
    latest_records = []

    for _, r in uni.iterrows():
        ts_code = r["ts_code"]
        symbol = r.get("symbol", "")
        name = r.get("name", "")

        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM daily WHERE ts_code=? ORDER BY date",
            conn,
            params=(ts_code,),
        )
        if df.empty or len(df) < 80:
            continue

        df["date"] = pd.to_datetime(df["date"])
        for c in ["open","high","low","close","volume"]:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        df = df.dropna()

        out = compute_and_signal(df)

        sig = out.loc[(out["date"] >= start_dt) & (out["date"] <= end_dt) & (out["signal_vA"])].copy()
        if sig.empty:
            continue

        sig.insert(0, "ts_code", ts_code)
        sig.insert(1, "symbol", symbol)
        sig.insert(2, "name", name)

        all_records.append(sig)
        latest_records.append(sig.sort_values("date").iloc[-1:].copy())

    conn.close()

    if not all_records:
        print("No signals found in the given period.")
        return

    all_df = pd.concat(all_records, ignore_index=True)
    latest_df = pd.concat(latest_records, ignore_index=True)

    # 排序：涨停标签优先（但不影响筛选），再按放量/突破/动能
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

    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as writer:
        latest_df[keep_cols].to_excel(writer, sheet_name="latest_per_stock", index=False)
        all_df[keep_cols].to_excel(writer, sheet_name="all_signals", index=False)

    print(f"Done. Saved: {OUT_XLSX}")


if __name__ == "__main__":
    main()
