# -*- coding: utf-8 -*-
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque


# ===================== 配置 =====================
INPUT_PATH = Path(r"C:\Users\24106\Desktop\breakout_vA_filtered_no688")  # 可错
DESKTOP_DIR = Path(r"C:\Users\24106\Desktop")

DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")
DATE_START = "2025-12-01"
DATE_END   = "2026-02-06"
OUT_XLSX = Path(r"D:\batch_exec_backtest_limitup_MA5 2.xlsx")

INIT_CASH = 1_000_000.0

# ==== 只做涨停信号日 ====
LIMIT_TAG_PCT = 0.099
ONLY_TRADE_IF_SIGNAL_LIMITUP = True

# ==== 一字板过滤（只影响入场，不影响信号计算）====
DISALLOW_YIZIBAN_ENTRY = True
YIZIBAN_STRICT = True
# True: open==high==low==close 且 high==low
# False: 只要 high==low 就算（一字板更宽松）

# ==== 交易成本（先 0）====
COMM_RATE = 0.0
SLIPPAGE_BP = 0.0

# ==== 期末强平 ====
FORCE_EXIT_AT_END = True

# ==== 止损成交方式 ====
USE_CONSERVATIVE_STOP_FILL = True  # True: min(next_open, stop)；False: next_open

# ==== 批量限制 ====
MAX_CODES = None
# ===============================================


# ===================== 仓位 preset（你切这个） =====================
PRESET_NAME = "L1"   # L1 / L2 / L3

PRESETS = {
    "L1": dict(MAX_EXPOSURE=0.50, TRANCHE_PCT=0.25, ADD_ATR=1.0),
    "L2": dict(MAX_EXPOSURE=0.60, TRANCHE_PCT=0.30, ADD_ATR=0.8),
    "L3": dict(MAX_EXPOSURE=0.75, TRANCHE_PCT=0.25, ADD_ATR=0.6),
}

P = PRESETS[PRESET_NAME]
MAX_EXPOSURE = float(P["MAX_EXPOSURE"])
TRANCHE_PCT  = float(P["TRANCHE_PCT"])
ADD_ATR      = float(P["ADD_ATR"])
MAX_TRANCHES = int(np.floor(MAX_EXPOSURE / TRANCHE_PCT + 1e-9))
# ==================================================================


# ===================== 指标/风控参数 =====================
ATR_N = 14
K_ATR = 3.0

# 入场确认：entry_day 收盘必须 >= MA5（你没说改，这条保留）
MA_CONFIRM_CLOSE_ABOVE = True

# 你 MA5 新规则：
# 1) 入场后3天内：2天收盘 < MA5 => 次日开盘退出
FAST_WIN_DAYS = 3
FAST_WIN_BREACH = 2

# 2) 持仓期间滚动7天：3天收盘 < MA5 => 次日开盘退出（你刚要求的修改）
ROLL_WIN_DAYS = 7
ROLL_WIN_BREACH = 3
# ========================================================


# ===== 信号（版本A + pct>=8%）=====
BPS_BREAK = 0.003
VOL_MULT  = 1.8
MIN_PCT   = 0.08


def ema(s: pd.Series, span: int) -> pd.Series:
    return s.ewm(span=span, adjust=False).mean()


def atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    prev_c = c.shift(1)
    tr = pd.concat([(h - l), (h - prev_c).abs(), (l - prev_c).abs()], axis=1).max(axis=1)
    return tr.rolling(n, min_periods=n).mean()


def compute_signal_vA(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_values("date").copy()
    c = df["close"]; h = df["high"]; v = df["volume"]

    df["pct"] = c / c.shift(1) - 1.0
    df["ma5"]  = c.rolling(5,  min_periods=5).mean()
    df["ma20"] = c.rolling(20, min_periods=20).mean()

    mb = df["ma20"]
    sd = c.rolling(20, min_periods=20).std(ddof=0)
    df["mb20"] = mb

    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    df["dif"] = dif
    df["dea"] = dea

    df["res20"] = h.rolling(20, min_periods=20).max().shift(1)
    df["vma20"] = v.rolling(20, min_periods=20).mean().shift(1)

    c1 = df["close"] > df["res20"] * (1 + BPS_BREAK)
    c2 = (df["close"] > df["mb20"]) & (df["mb20"] > df["mb20"].shift(1))
    c3 = df["dif"] > df["dea"]
    c4 = (df["close"] > df["ma5"]) & (df["ma5"] > df["ma5"].shift(1))
    c5 = df["volume"] > df["vma20"] * VOL_MULT
    c6 = df["pct"] >= MIN_PCT

    df["signal_vA"] = (c1 & c2 & c3 & c4 & c5 & c6)
    df["is_limit_up"] = df["pct"] >= LIMIT_TAG_PCT
    return df


def is_yiziban_signal_day(row: pd.Series) -> bool:
    """
    判定“信号日是否一字板”。
    - strict: open==high==low==close (且 high==low)
    - loose : high==low
    """
    try:
        o = float(row["open"])
        h = float(row["high"])
        l = float(row["low"])
        c = float(row["close"])
    except Exception:
        return False

    if YIZIBAN_STRICT:
        return (h == l) and (o == h) and (c == h)
    else:
        return (h == l)


def apply_cost(price: float, side: str) -> float:
    slip = SLIPPAGE_BP / 10000.0
    if side == "buy":
        p = price * (1 + slip)
        p = p * (1 + COMM_RATE)
    else:
        p = price * (1 - slip)
        p = p * (1 - COMM_RATE)
    return p


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame) -> dict:
    ec = equity_curve.sort_values("date").copy()
    ec["ret"] = ec["equity"].pct_change()
    rets = ec["ret"].dropna()

    start_eq = float(ec["equity"].iloc[0])
    end_eq   = float(ec["equity"].iloc[-1])

    ec["peak"] = ec["equity"].cummax()
    ec["dd"] = ec["equity"] / ec["peak"] - 1.0
    max_drawdown = float(ec["dd"].min())

    if len(rets) > 1 and rets.std(ddof=0) > 0:
        sharpe = float((rets.mean() / rets.std(ddof=0)) * np.sqrt(252))
    else:
        sharpe = np.nan

    sum_pnl = end_eq - start_eq

    if trades is None or len(trades) == 0:
        num_trades = 0
        win_rate = np.nan
        profit_factor = np.nan
    else:
        pnl = pd.to_numeric(trades["pnl"], errors="coerce").dropna()
        num_trades = int(len(trades))
        win_rate = float((pnl > 0).mean()) if len(pnl) else np.nan
        gross_win = float(pnl[pnl > 0].sum()) if len(pnl) else 0.0
        gross_loss = float(-pnl[pnl < 0].sum()) if len(pnl) else 0.0
        profit_factor = float(gross_win / gross_loss) if gross_loss > 0 else np.inf

    return {
        "sum_pnl": float(sum_pnl),
        "total_return": end_eq / start_eq - 1.0,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def backtest_one_stock(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["atr"] = atr(df, ATR_N)
    df = compute_signal_vA(df)

    start_dt = pd.to_datetime(DATE_START)
    end_dt   = pd.to_datetime(DATE_END)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
    if len(df) < 30:
        return pd.DataFrame(), pd.DataFrame()

    cash = INIT_CASH
    shares = 0.0
    avg_cost = np.nan

    in_pos = False
    tranches = 0
    peak_high = np.nan
    stop = np.nan

    entry_date = None
    entry_avg = np.nan
    entry_idx = None

    # MA5 计数器/窗口
    fast_below_cnt = 0
    roll_below = deque(maxlen=ROLL_WIN_DAYS)

    trades = []
    eq_rows = []

    def do_exit(next_row, exit_price, reason):
        nonlocal cash, shares, avg_cost, in_pos, tranches, peak_high, stop
        nonlocal entry_date, entry_avg, entry_idx, fast_below_cnt, roll_below

        fill = apply_cost(exit_price, "sell")
        cash += shares * fill

        trades.append({
            "entry_date": entry_date,
            "exit_date": next_row["date"],
            "entry_avg": float(entry_avg),
            "exit_price": float(fill),
            "shares": float(shares),
            "pnl": float((fill - entry_avg) * shares),
            "ret": float(fill / entry_avg - 1.0),
            "reason": reason,
            "tranches": int(tranches),
        })

        # 清仓并重置
        shares = 0.0
        avg_cost = np.nan
        in_pos = False
        tranches = 0
        peak_high = np.nan
        stop = np.nan
        entry_date = None
        entry_avg = np.nan
        entry_idx = None
        fast_below_cnt = 0
        roll_below.clear()

    for i in range(len(df)):
        row = df.loc[i]
        date = row["date"]

        equity = cash + shares * float(row["close"])
        eq_rows.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "shares": shares,
            "close": float(row["close"]),
            "ma5": float(row["ma5"]) if pd.notna(row["ma5"]) else np.nan,
            "avg_cost": avg_cost,
            "tranches": tranches,
            "stop": stop,
            "signal_vA": bool(row.get("signal_vA", False)),
            "is_limit_up": bool(row.get("is_limit_up", False)) if pd.notna(row.get("is_limit_up", np.nan)) else False,
        })

        if i == len(df) - 1:
            break

        next_row = df.loc[i + 1]
        next_open = float(next_row["open"])

        # ================= 持仓 =================
        if in_pos:
            below_ma5 = False
            if pd.notna(row["ma5"]) and pd.notna(row["close"]):
                below_ma5 = float(row["close"]) < float(row["ma5"])

            # 入场后3天内：2次 below -> 退出
            if entry_idx is not None:
                day_in_pos = i - entry_idx + 1
                if day_in_pos <= FAST_WIN_DAYS:
                    if below_ma5:
                        fast_below_cnt += 1
                    if fast_below_cnt >= FAST_WIN_BREACH:
                        do_exit(next_row, next_open, f"ma5_fast_{FAST_WIN_DAYS}d_{FAST_WIN_BREACH}breach")
                        continue

            # 滚动7天：3次 below -> 退出
            roll_below.append(1 if below_ma5 else 0)
            if len(roll_below) == ROLL_WIN_DAYS and sum(roll_below) >= ROLL_WIN_BREACH:
                do_exit(next_row, next_open, f"ma5_roll_{ROLL_WIN_DAYS}d_{ROLL_WIN_BREACH}breach")
                continue

            # Chandelier
            peak_high = float(row["high"]) if not pd.notna(peak_high) else max(float(peak_high), float(row["high"]))
            if pd.notna(row["atr"]) and pd.notna(peak_high):
                stop = peak_high - K_ATR * float(row["atr"])

            if pd.notna(stop) and float(row["low"]) <= float(stop):
                fill0 = min(next_open, float(stop)) if USE_CONSERVATIVE_STOP_FILL else next_open
                do_exit(next_row, fill0, "chandelier_stop")
                continue

            # 加仓（不变）
            if tranches < MAX_TRANCHES and pd.notna(row["atr"]) and pd.notna(avg_cost):
                add_trigger = float(avg_cost) + ADD_ATR * float(row["atr"])
                if float(row["close"]) >= add_trigger:
                    target_add_value = INIT_CASH * TRANCHE_PCT
                    max_total_value = INIT_CASH * MAX_EXPOSURE
                    current_value = shares * next_open
                    remain = max(0.0, max_total_value - current_value)
                    add_value = min(target_add_value, remain, cash)

                    if add_value > 0:
                        buy_px = apply_cost(next_open, "buy")
                        add_shares = add_value / buy_px
                        new_cost = (avg_cost * shares + buy_px * add_shares) / (shares + add_shares)

                        shares += add_shares
                        cash -= add_shares * buy_px
                        avg_cost = new_cost
                        tranches += 1

            continue

        # ================= 空仓：入场 =================
        if bool(row.get("signal_vA", False)):
            # 涨停信号日过滤
            if ONLY_TRADE_IF_SIGNAL_LIMITUP and (not bool(row.get("is_limit_up", False))):
                continue

            # 一字板：信号日不给进（你新要求）
            if DISALLOW_YIZIBAN_ENTRY and is_yiziban_signal_day(row):
                continue

            buy_value = min(INIT_CASH * TRANCHE_PCT, INIT_CASH * MAX_EXPOSURE, cash)
            if buy_value <= 0:
                continue

            buy_px = apply_cost(next_open, "buy")
            buy_shares = buy_value / buy_px

            # 入场确认：entry_day 收盘 >= MA5
            if MA_CONFIRM_CLOSE_ABOVE:
                if pd.notna(next_row["ma5"]) and pd.notna(next_row["close"]):
                    if float(next_row["close"]) < float(next_row["ma5"]):
                        continue

            shares = buy_shares
            cash -= buy_shares * buy_px
            avg_cost = buy_px

            in_pos = True
            tranches = 1
            entry_date = next_row["date"]
            entry_avg = avg_cost
            entry_idx = i + 1

            peak_high = float(next_row["high"])
            stop = peak_high - K_ATR * float(next_row["atr"]) if pd.notna(next_row["atr"]) else np.nan

            fast_below_cnt = 0
            roll_below.clear()

    # 期末强平
    if FORCE_EXIT_AT_END and in_pos and shares > 0:
        last_row = df.iloc[-1]
        fill = apply_cost(float(last_row["close"]), "sell")
        cash += shares * fill
        trades.append({
            "entry_date": entry_date,
            "exit_date": last_row["date"],
            "entry_avg": float(entry_avg),
            "exit_price": float(fill),
            "shares": float(shares),
            "pnl": float((fill - entry_avg) * shares),
            "ret": float(fill / entry_avg - 1.0),
            "reason": "end_of_period",
            "tranches": int(tranches),
        })
        eq_rows.append({
            "date": last_row["date"],
            "equity": cash,
            "cash": cash,
            "shares": 0.0,
            "close": float(last_row["close"]),
            "ma5": float(last_row["ma5"]) if pd.notna(last_row["ma5"]) else np.nan,
            "avg_cost": np.nan,
            "tranches": 0,
            "stop": np.nan,
            "signal_vA": False,
            "is_limit_up": False,
        })

    return pd.DataFrame(eq_rows), pd.DataFrame(trades)


# ============== 桌面抓代码 + 映射 ts_code（不变） ==============
def normalize_symbol6(x) -> str:
    if pd.isna(x):
        return ""
    s = str(x).strip()
    s = s.split(".")[0]
    return s.zfill(6) if s.isdigit() else s


def extract_codes_from_any_df(df: pd.DataFrame) -> list[str]:
    if df is None or df.empty:
        return []
    cols = {str(c).lower(): c for c in df.columns}
    cand_cols = []
    for key in ["ts_code", "symbol", "code", "股票代码", "证券代码"]:
        if key in cols:
            cand_cols.append(cols[key])
    codes = []
    for c in cand_cols:
        codes.extend(df[c].dropna().astype(str).tolist())
    return codes


def extract_codes_from_file(fp: Path) -> list[str]:
    codes = []
    try:
        if fp.suffix.lower() in (".xlsx", ".xls"):
            sheets = pd.read_excel(fp, sheet_name=None)
            for _, dfi in sheets.items():
                codes.extend(extract_codes_from_any_df(dfi))
        elif fp.suffix.lower() == ".csv":
            try:
                df = pd.read_csv(fp, encoding="utf-8-sig")
            except Exception:
                df = pd.read_csv(fp, encoding="gbk", errors="ignore")
            codes.extend(extract_codes_from_any_df(df))
    except Exception:
        pass

    name = fp.name
    codes.extend(re.findall(r"\b\d{6}\.(?:SZ|SH)\b", name, flags=re.I))
    codes.extend(re.findall(r"\b\d{6}\b", name))
    out, seen = [], set()
    for x in codes:
        x = str(x).strip().replace(" ", "")
        if not x:
            continue
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


def gather_codes(input_path: Path) -> list[str]:
    files = []
    if input_path.is_dir():
        for ext in ("*.xlsx", "*.xls", "*.csv"):
            files.extend(input_path.rglob(ext))
    elif input_path.is_file():
        files = [input_path]
    else:
        raise FileNotFoundError(f"找不到路径：{input_path}")

    raw_codes = []
    for fp in files:
        raw_codes.extend(extract_codes_from_file(fp))

    cleaned = []
    for x in raw_codes:
        x = x.strip().replace(" ", "")
        if re.fullmatch(r"\d{6}\.(SZ|SH)", x, flags=re.I):
            cleaned.append(x.upper())
        elif re.fullmatch(r"\d{6}", x):
            cleaned.append(x)
    return list(dict.fromkeys(cleaned))


def auto_find_input_path(desktop_dir: Path) -> Path:
    candidates = []
    for p in desktop_dir.iterdir():
        try:
            name = p.name.lower()
            if "breakout_va" in name or "breakout" in name:
                candidates.append(p)
        except Exception:
            pass

    for ext in ("*.xlsx", "*.xls", "*.csv"):
        for p in desktop_dir.rglob(ext):
            name = p.name.lower()
            if "breakout_va" in name or "breakout" in name:
                candidates.append(p)

    if not candidates:
        raise FileNotFoundError("桌面里没找到包含 breakout/breakout_va 的文件或文件夹。")

    candidates = list(dict.fromkeys(candidates))
    candidates.sort(key=lambda x: x.stat().st_mtime, reverse=True)
    return candidates[0]


def map_to_ts_codes(raw_codes: list[str], uni: pd.DataFrame) -> list[str]:
    sym2ts = {normalize_symbol6(s): t for s, t in zip(uni["symbol"], uni["ts_code"])}
    ts_codes = []
    for x in raw_codes:
        if re.fullmatch(r"\d{6}\.(SZ|SH)", x, flags=re.I):
            ts_codes.append(x.upper())
        elif re.fullmatch(r"\d{6}", x):
            s6 = normalize_symbol6(x)
            if s6 in sym2ts:
                ts_codes.append(sym2ts[s6])
    return list(dict.fromkeys(ts_codes))


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到数据库：{DB_PATH}")

    input_path = INPUT_PATH
    if not input_path.exists():
        input_path = auto_find_input_path(DESKTOP_DIR)
        print(f"[0] INPUT_PATH 不存在，自动找到：{input_path}")

    raw_codes = gather_codes(input_path)
    print(f"[1] Found raw codes: {len(raw_codes)}")

    conn = sqlite3.connect(DB_PATH.as_posix())
    try:
        uni = pd.read_sql_query("SELECT ts_code, symbol, name FROM universe", conn)
        if uni.empty:
            raise RuntimeError("universe 表为空：请先跑下载脚本建库")
    finally:
        conn.close()

    ts_codes = map_to_ts_codes(raw_codes, uni)
    print(f"[2] Mapped ts_codes: {len(ts_codes)}")
    if MAX_CODES is not None:
        ts_codes = ts_codes[:MAX_CODES]

    if not ts_codes:
        raise RuntimeError("没提取到任何可回测 ts_code。")

    conn = sqlite3.connect(DB_PATH.as_posix())
    per_stock_rows = []
    all_trades = []

    try:
        for idx, ts in enumerate(ts_codes, 1):
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM daily WHERE ts_code=? ORDER BY date",
                conn, params=(ts,)
            )
            if df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()
            if len(df) < 80:
                continue

            equity, trades = backtest_one_stock(df)
            if equity.empty:
                continue

            metrics = compute_metrics(equity, trades)

            row_uni = uni.loc[uni["ts_code"] == ts]
            name = row_uni["name"].iloc[0] if len(row_uni) else ""
            symbol = normalize_symbol6(row_uni["symbol"].iloc[0]) if len(row_uni) else ""

            per_stock_rows.append({
                "ts_code": ts, "symbol": symbol, "name": name,
                "preset": PRESET_NAME,
                "max_exposure": MAX_EXPOSURE,
                "tranche_pct": TRANCHE_PCT,
                "add_atr": ADD_ATR,
                "ma_fast_rule": f"{FAST_WIN_DAYS}d_{FAST_WIN_BREACH}breach",
                "ma_roll_rule": f"{ROLL_WIN_DAYS}d_{ROLL_WIN_BREACH}breach",
                "only_limitup_signal": ONLY_TRADE_IF_SIGNAL_LIMITUP,
                "disallow_yiziban_entry": DISALLOW_YIZIBAN_ENTRY,
                "yiziban_strict": YIZIBAN_STRICT,
                **metrics
            })

            if trades is not None and len(trades):
                trades2 = trades.copy()
                trades2.insert(0, "ts_code", ts)
                trades2.insert(1, "symbol", symbol)
                trades2.insert(2, "name", name)
                trades2.insert(3, "preset", PRESET_NAME)
                all_trades.append(trades2)

            if idx % 50 == 0:
                print(f"[3] Done {idx}/{len(ts_codes)}")

    finally:
        conn.close()

    if not per_stock_rows:
        raise RuntimeError("回测完没有任何结果（可能区间内无信号/无交易）。")

    per_stock = pd.DataFrame(per_stock_rows).sort_values(["total_return", "sum_pnl"], ascending=[False, False])
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        per_stock.to_excel(w, sheet_name="per_stock_metrics", index=False)
        trades_df.to_excel(w, sheet_name="trades", index=False)

    print(f"\nSaved: {OUT_XLSX}")
    print(f"Stocks: {len(per_stock)}; Trades rows: {len(trades_df)}")


if __name__ == "__main__":
    main()
