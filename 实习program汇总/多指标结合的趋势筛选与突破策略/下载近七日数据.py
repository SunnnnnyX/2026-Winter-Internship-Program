# -*- coding: utf-8 -*-
"""
01_build_or_update_sqlite_tushare.py

新建/增量更新 SQLite（日线qfq）
- universe：上市股票列表
- daily：qfq OHLC + volume
- meta：每只股票 last_date 断点续跑

关键改动：
- END_DATE 不再是“今天”，而是“最后一个已收盘交易日”
  -> 白天运行不会下载当天未收盘的数据

依赖：
pip install tushare pandas tqdm
环境变量：
TUSHARE_TOKEN=你的token
"""

import os
import time
import random
import sqlite3
from pathlib import Path

import pandas as pd
from tqdm import tqdm
import tushare as ts


# ===================== 配置 =====================
DB_PATH = Path(r"D:\a_share_daily_tushare_new.sqlite")

# 第一次建库时建议设一个更早的日期；以后增量会从 meta.last_date+1 自动续跑
DEFAULT_START = "2026-02-10"   # <-- 你可以改更早（如 2020-01-01）

# 节流与重试
SLEEP_BASE = 0.18
SLEEP_JITTER = 0.25
MAX_RETRY = 6

# 收盘缓冲：15:10后才认为“今日已收盘”
CLOSE_CUTOFF_HHMM = "15:10"
# ===============================================


def ymd_to_yyyymmdd(s: str) -> str:
    return s.replace("-", "")


def next_start_date(last_date: str | None, default_start: str) -> str:
    if not last_date:
        return default_start
    dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


def last_closed_trading_day(pro) -> pd.Timestamp:
    """
    返回“最后一个已收盘的交易日”（SSE）
    - 如果今天是交易日但未到15:10，则返回上一个交易日
    """
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

    # 最近交易日是今天，但未过收盘缓冲 -> 回退一个交易日
    if last_td == today and now < close_cutoff:
        if len(cal) < 2:
            return last_td
        return cal["cal_date"].iloc[-2]

    return last_td


def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    cur.execute("""
    CREATE TABLE IF NOT EXISTS universe (
        ts_code   TEXT PRIMARY KEY,
        symbol    TEXT,
        name      TEXT,
        market    TEXT,
        list_date TEXT
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily (
        ts_code TEXT NOT NULL,
        date    TEXT NOT NULL,   -- YYYY-MM-DD
        open    REAL,
        high    REAL,
        low     REAL,
        close   REAL,
        volume  REAL,
        PRIMARY KEY (ts_code, date)
    )
    """)

    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        ts_code   TEXT PRIMARY KEY,
        last_date TEXT
    )
    """)

    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_date ON daily(date)")
    conn.commit()


def get_last_date(conn: sqlite3.Connection, ts_code: str) -> str | None:
    cur = conn.cursor()
    cur.execute("SELECT last_date FROM meta WHERE ts_code=?", (ts_code,))
    row = cur.fetchone()
    return row[0] if row else None


def upsert_last_date(conn: sqlite3.Connection, ts_code: str, last_date: str):
    cur = conn.cursor()
    cur.execute("""
    INSERT INTO meta(ts_code, last_date) VALUES(?,?)
    ON CONFLICT(ts_code) DO UPDATE SET last_date=excluded.last_date
    """, (ts_code, last_date))
    conn.commit()


def save_universe(conn: sqlite3.Connection, uni: pd.DataFrame):
    cur = conn.cursor()
    rows = [
        (r["ts_code"], r.get("symbol"), r.get("name"), r.get("market"), r.get("list_date"))
        for _, r in uni.iterrows()
    ]
    cur.executemany("""
    INSERT OR REPLACE INTO universe(ts_code, symbol, name, market, list_date)
    VALUES (?,?,?,?,?)
    """, rows)
    conn.commit()


def insert_daily(conn: sqlite3.Connection, ts_code: str, df: pd.DataFrame):
    cur = conn.cursor()
    rows = []
    for _, r in df.iterrows():
        rows.append((
            ts_code,
            r["date"],
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        ))
    cur.executemany("""
    INSERT OR REPLACE INTO daily(ts_code, date, open, high, low, close, volume)
    VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


def fetch_daily_and_qfq(pro, ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    daily + adj_factor 自算 qfq
    qfq = price * adj_factor / last_adj_factor(区间内最后一天)
    """
    start_ = ymd_to_yyyymmdd(start)
    end_ = ymd_to_yyyymmdd(end)

    for k in range(1, MAX_RETRY + 1):
        try:
            d = pro.daily(ts_code=ts_code, start_date=start_, end_date=end_)
            if d is None or d.empty:
                return None

            af = pro.adj_factor(ts_code=ts_code, start_date=start_, end_date=end_)
            if af is None or af.empty:
                return None

            d = d[["trade_date", "open", "high", "low", "close", "vol"]].copy()
            af = af[["trade_date", "adj_factor"]].copy()
            m = pd.merge(d, af, on="trade_date", how="inner")
            if m.empty:
                return None

            m = m.sort_values("trade_date")
            last_adj = float(m["adj_factor"].iloc[-1])
            ratio = m["adj_factor"].astype(float) / last_adj

            out = pd.DataFrame({
                "date": pd.to_datetime(m["trade_date"]).dt.strftime("%Y-%m-%d"),
                "open": m["open"].astype(float) * ratio,
                "high": m["high"].astype(float) * ratio,
                "low":  m["low"].astype(float) * ratio,
                "close": m["close"].astype(float) * ratio,
                "volume": m["vol"].astype(float),
            }).dropna()

            return None if out.empty else out

        except Exception:
            if k == MAX_RETRY:
                return None
            time.sleep(0.9 * k + random.random() * 0.6)

    return None


def main():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN（setx 后重启 Spyder/终端）")

    ts.set_token(token)
    pro = ts.pro_api()

    # ===== 关键：只更新到最后一个已收盘交易日 =====
    end_date_dt = last_closed_trading_day(pro)
    END_DATE = end_date_dt.strftime("%Y-%m-%d")
    print(f"[END_DATE] last closed trading day = {END_DATE}")

    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH.as_posix())
    init_db(conn)

    # 1) 更新 universe（仅上市）
    uni = pro.stock_basic(exchange="", list_status="L",
                          fields="ts_code,symbol,name,market,list_date")
    if uni is None or uni.empty:
        conn.close()
        raise RuntimeError("stock_basic 返回为空，请检查token/权限。")

    save_universe(conn, uni)

    end_dt = pd.to_datetime(END_DATE)

    # 2) 逐只增量更新
    uni2 = pd.read_sql_query("SELECT ts_code FROM universe", conn)

    for _, r in tqdm(uni2.iterrows(), total=len(uni2), desc="Update daily(qfq)"):
        ts_code = r["ts_code"]

        time.sleep(SLEEP_BASE + random.random() * SLEEP_JITTER)

        last = get_last_date(conn, ts_code)
        start = next_start_date(last, DEFAULT_START)
        if pd.to_datetime(start) > end_dt:
            continue

        df = fetch_daily_and_qfq(pro, ts_code, start, END_DATE)
        if df is None or df.empty:
            continue

        insert_daily(conn, ts_code, df)
        upsert_last_date(conn, ts_code, df["date"].max())

    conn.close()
    print(f"Done. DB saved at: {DB_PATH}")


if __name__ == "__main__":
    main()
