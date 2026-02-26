import sqlite3
from pathlib import Path

DB_PATH = Path(r"D:\marketdata\ashare_ifind.sqlite")

DDL = """
PRAGMA journal_mode=WAL;
PRAGMA synchronous=NORMAL;

CREATE TABLE IF NOT EXISTS daily_bar (
  code TEXT NOT NULL,         -- 6位
  trade_date TEXT NOT NULL,   -- YYYYMMDD
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

CREATE TABLE IF NOT EXISTS stock_meta (
  code TEXT PRIMARY KEY,
  name TEXT,
  industry TEXT,
  updated_at TEXT
);

CREATE TABLE IF NOT EXISTS candidates (
  asof_date TEXT NOT NULL,    -- 你跑扫描那天（通常=trade_date当日）
  code TEXT NOT NULL,
  name TEXT,
  industry TEXT,
  tminus1 TEXT,
  t TEXT,
  tplus1 TEXT,
  L REAL,
  stop REAL,
  body_t REAL,
  vol_mult REAL,
  note TEXT,
  PRIMARY KEY(asof_date, code, t)
);

CREATE INDEX IF NOT EXISTS idx_cand_asof ON candidates(asof_date);
"""

def main():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        con.executescript(DDL)
        con.commit()
        print("DB initialized:", DB_PATH)
    finally:
        con.close()

if __name__ == "__main__":
    main()
