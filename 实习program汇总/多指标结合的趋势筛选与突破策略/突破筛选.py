# -*- coding: utf-8 -*-
"""
Step 1: 用 Tushare 拉取A股日线（前复权 qfq 价格）并写入 SQLite
并新增：daily_basic（市值/股本/换手率等）落库

特性：
- 支持断点续跑（meta.last_date 仍用于 daily；daily_basic 以同一日期更新）
- 支持重试、节流
- END_DATE 自动为今天（本地日期）

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
DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")

# 为了算MA/MACD/BOLL等，需要更早的历史（别太短）
DEFAULT_START = "2023-01-01"   # 建议别用“今天”，否则你指标都算不出来
END_DATE = pd.Timestamp.today().strftime("%Y-%m-%d")  # 自动到今天（本地）

# 节流与重试（TuShare有频控，建议别太激进）
SLEEP_BASE = 0.18
SLEEP_JITTER = 0.25
MAX_RETRY = 6

# daily_basic字段（你可以按需加减）
DAILY_BASIC_FIELDS = (
    "ts_code,trade_date,"
    "turnover_rate,turnover_rate_f,volume_ratio,"
    "total_share,float_share,free_share,"
    "total_mv,circ_mv"
)

# ===================== 工具 =====================
def ymd_to_yyyymmdd(s: str) -> str:
    return s.replace("-", "")

def next_start_date(last_date: str | None, default_start: str) -> str:
    if not last_date:
        return default_start
    dt = pd.to_datetime(last_date) + pd.Timedelta(days=1)
    return dt.strftime("%Y-%m-%d")


# ===================== DB 初始化 =====================
def init_db(conn: sqlite3.Connection):
    cur = conn.cursor()

    # 股票列表缓存
    cur.execute("""
    CREATE TABLE IF NOT EXISTS universe (
        ts_code  TEXT PRIMARY KEY,
        symbol   TEXT,
        name     TEXT,
        market   TEXT,
        list_date TEXT
    )
    """)

    # 日线数据（qfq价格，volume用TuShare daily的 vol）
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

    # 新增：市值/股本/换手等（Tushare daily_basic）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS daily_basic (
        ts_code TEXT NOT NULL,
        date    TEXT NOT NULL,   -- YYYY-MM-DD
        turnover_rate   REAL,
        turnover_rate_f REAL,
        volume_ratio    REAL,
        total_share     REAL,
        float_share     REAL,
        free_share      REAL,
        total_mv        REAL,
        circ_mv         REAL,
        PRIMARY KEY (ts_code, date)
    )
    """)

    # 断点续跑（用于 daily 的更新起点）
    cur.execute("""
    CREATE TABLE IF NOT EXISTS meta (
        ts_code   TEXT PRIMARY KEY,
        last_date TEXT
    )
    """)

    # 给 daily_basic 建索引（可选，但后面JOIN/筛选会快）
    cur.execute("CREATE INDEX IF NOT EXISTS idx_daily_basic_date ON daily_basic(date)")
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


def load_universe(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT ts_code, symbol, name, market, list_date FROM universe", conn)


def insert_daily(conn: sqlite3.Connection, ts_code: str, df: pd.DataFrame):
    cur = conn.cursor()
    rows = [
        (
            ts_code,
            r["date"],
            float(r["open"]),
            float(r["high"]),
            float(r["low"]),
            float(r["close"]),
            float(r["volume"]),
        )
        for _, r in df.iterrows()
    ]
    cur.executemany("""
    INSERT OR REPLACE INTO daily(ts_code, date, open, high, low, close, volume)
    VALUES (?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


def insert_daily_basic(conn: sqlite3.Connection, ts_code: str, df: pd.DataFrame):
    """
    df 字段：date + daily_basic字段
    """
    cur = conn.cursor()

    # 统一列名存在性
    for col in ["turnover_rate", "turnover_rate_f", "volume_ratio",
                "total_share", "float_share", "free_share",
                "total_mv", "circ_mv"]:
        if col not in df.columns:
            df[col] = None

    rows = []
    for _, r in df.iterrows():
        rows.append((
            ts_code,
            r["date"],
            None if pd.isna(r["turnover_rate"]) else float(r["turnover_rate"]),
            None if pd.isna(r["turnover_rate_f"]) else float(r["turnover_rate_f"]),
            None if pd.isna(r["volume_ratio"]) else float(r["volume_ratio"]),
            None if pd.isna(r["total_share"]) else float(r["total_share"]),
            None if pd.isna(r["float_share"]) else float(r["float_share"]),
            None if pd.isna(r["free_share"]) else float(r["free_share"]),
            None if pd.isna(r["total_mv"]) else float(r["total_mv"]),
            None if pd.isna(r["circ_mv"]) else float(r["circ_mv"]),
        ))

    cur.executemany("""
    INSERT OR REPLACE INTO daily_basic(
        ts_code, date,
        turnover_rate, turnover_rate_f, volume_ratio,
        total_share, float_share, free_share,
        total_mv, circ_mv
    )
    VALUES (?,?,?,?,?,?,?,?,?,?)
    """, rows)
    conn.commit()


# ===================== TuShare 拉取（daily + adj_factor 自算qfq） =====================
def fetch_daily_and_qfq(pro, ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    返回字段：date(open/high/low/close为qfq), volume(原vol)
    - daily: open/high/low/close/vol/trade_date
    - adj_factor: adj_factor/trade_date
    - qfq: price * adj_factor / last_adj_factor
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
            })

            out = out.dropna()
            return None if out.empty else out

        except Exception:
            if k == MAX_RETRY:
                return None
            time.sleep(0.9 * k + random.random() * 0.6)

    return None


def fetch_daily_basic(pro, ts_code: str, start: str, end: str) -> pd.DataFrame | None:
    """
    拉 daily_basic：返回 date + 你需要的字段
    注意：TuShare 返回 trade_date=YYYYMMDD
    """
    start_ = ymd_to_yyyymmdd(start)
    end_ = ymd_to_yyyymmdd(end)

    for k in range(1, MAX_RETRY + 1):
        try:
            b = pro.daily_basic(
                ts_code=ts_code,
                start_date=start_,
                end_date=end_,
                fields=DAILY_BASIC_FIELDS
            )
            if b is None or b.empty:
                return None

            b = b.copy()
            b["date"] = pd.to_datetime(b["trade_date"]).dt.strftime("%Y-%m-%d")
            b = b.drop(columns=["trade_date"], errors="ignore")

            # TuShare通常倒序，转升序
            b = b.sort_values("date").reset_index(drop=True)
            return None if b.empty else b

        except Exception:
            if k == MAX_RETRY:
                return None
            time.sleep(0.9 * k + random.random() * 0.6)

    return None


def main():
    token = os.environ.get("TUSHARE_TOKEN", "").strip()
    if not token:
        raise RuntimeError("请先设置环境变量 TUSHARE_TOKEN（setx 后重启Spyder/终端）")

    ts.set_token(token)
    pro = ts.pro_api()

    conn = sqlite3.connect(DB_PATH.as_posix())
    init_db(conn)

    # 1) 拉取股票列表（缓存到 universe）
    uni = pro.stock_basic(exchange="", list_status="L",
                          fields="ts_code,symbol,name,market,list_date")
    if uni is None or uni.empty:
        raise RuntimeError("TuShare stock_basic 返回为空，请检查token/网络/权限。")

    save_universe(conn, uni)
    uni = load_universe(conn)

    end_dt = pd.to_datetime(END_DATE)

    # 2) 逐只增量更新（daily + daily_basic）
    for _, row in tqdm(uni.iterrows(), total=len(uni), desc="Download to SQLite (TuShare)"):
        ts_code = row["ts_code"]

        time.sleep(SLEEP_BASE + random.random() * SLEEP_JITTER)

        last = get_last_date(conn, ts_code)
        start = next_start_date(last, DEFAULT_START)
        if pd.to_datetime(start) > end_dt:
            continue

        # --- daily(qfq) ---
        df_d = fetch_daily_and_qfq(pro, ts_code, start, END_DATE)
        if df_d is not None and not df_d.empty:
            insert_daily(conn, ts_code, df_d)
            upsert_last_date(conn, ts_code, df_d["date"].max())

        # --- daily_basic(市值/股本/换手等) ---
        # 用同一区间（start~END_DATE），与 daily 对齐
        df_b = fetch_daily_basic(pro, ts_code, start, END_DATE)
        if df_b is not None and not df_b.empty:
            insert_daily_basic(conn, ts_code, df_b)

    conn.close()
    print(f"Done. DB saved at: {DB_PATH}")
    print("Tables: universe, daily(qfq), daily_basic(mv/shares/turnover), meta")


if __name__ == "__main__":
    main()
