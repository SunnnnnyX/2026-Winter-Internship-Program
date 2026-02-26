# -*- coding: utf-8 -*-
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


# ===================== 配置 =====================
INPUT_PATH = Path(r"C:\Users\24106\Desktop\breakout_vA_filtered_no688")  # 可错
DESKTOP_DIR = Path(r"C:\Users\24106\Desktop")

DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")  # 数据库源不变
DATE_START = "2025-12-01"
DATE_END   = "2026-02-06"
OUT_XLSX = Path(r"D:\batch_exec_backtest_results.xlsx")

INIT_CASH = 1_000_000.0
MAX_EXPOSURE = 0.50
TRANCHE_PCT  = 0.25
MAX_TRANCHES = int(np.floor(MAX_EXPOSURE / TRANCHE_PCT + 1e-9))

FORCE_EXIT_AT_END = True   # 回测期末强制平仓（避免“浮盈但收益=0”）

ATR_N = 14
K_ATR = 3.0
ADD_ATR = 1.0

COMM_RATE = 0.0
SLIPPAGE_BP = 0.0
USE_CONSERVATIVE_STOP_FILL = True
RF_ANNUAL = 0.0

# 新增：开仓后窗口（交易日数）
RUNUP_WINDOW_DAYS = 20

MAX_CODES = None
# ===============================================


# ===== 信号（版本A + pct>=8%）=====
BPS_BREAK = 0.003
VOL_MULT  = 1.8
MIN_PCT   = 0.08
LIMIT_TAG_PCT = 0.099


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


def apply_cost(price: float, side: str) -> float:
    slip = SLIPPAGE_BP / 10000.0
    if side == "buy":
        p = price * (1 + slip)
        p = p * (1 + COMM_RATE)
    else:
        p = price * (1 - slip)
        p = p * (1 - COMM_RATE)
    return p


def max_runup_20d(df_slice: pd.DataFrame, entry_idx: int, entry_price: float, window: int = 20):
    """
    从 entry_idx（该次入场日所在行）起，往后 window 个交易日（含入场日）：
    - 找最高 high
    - 返回：abs / pct / high / high_date
    """
    n = len(df_slice)
    if entry_idx is None or entry_idx < 0 or entry_idx >= n:
        return (np.nan, np.nan, np.nan, pd.NaT)

    end_i = min(n, entry_idx + window)
    win = df_slice.iloc[entry_idx:end_i]
    if win.empty:
        return (np.nan, np.nan, np.nan, pd.NaT)

    hi = win["high"].astype(float)
    max_high = float(hi.max())
    max_pos = int(hi.idxmax())
    max_date = pd.to_datetime(df_slice.loc[max_pos, "date"])

    abs_runup = max_high - float(entry_price)
    pct_runup = abs_runup / float(entry_price) if entry_price and entry_price > 0 else np.nan
    return (float(abs_runup), float(pct_runup), float(max_high), max_date)


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """
    sum_pnl 用 “最终权益 - 初始权益”（包含浮盈/已实现）
    其它按你要的：回报率、盈亏比、交易数、最大回撤、夏普、胜率
    """
    ec = equity_curve.sort_values("date").copy()
    ec["ret"] = ec["equity"].pct_change()
    rets = ec["ret"].dropna()

    start_eq = float(ec["equity"].iloc[0])
    end_eq   = float(ec["equity"].iloc[-1])

    ec["peak"] = ec["equity"].cummax()
    ec["dd"] = ec["equity"] / ec["peak"] - 1.0
    max_drawdown = float(ec["dd"].min())

    rf_daily = rf_annual / 252.0
    if len(rets) > 1 and rets.std(ddof=0) > 0:
        sharpe = float(((rets.mean() - rf_daily) / rets.std(ddof=0)) * np.sqrt(252))
    else:
        sharpe = np.nan

    sum_pnl = end_eq - start_eq

    if trades is None or len(trades) == 0:
        num_trades = 0
        win_rate = np.nan
        profit_factor = np.nan
    else:
        pnl = trades["pnl"].astype(float)
        num_trades = int(len(trades))
        win_rate = float((pnl > 0).mean())
        gross_win = float(pnl[pnl > 0].sum())
        gross_loss = float(-pnl[pnl < 0].sum())
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


def backtest_one_stock(df: pd.DataFrame, trade_id_start: int = 1):
    """
    返回：
    - equity_curve
    - trades（每笔交易一行，附带 best_entry_runup_20d_*）
    - entry_runups（每次入场一行：开仓/加仓都有）
    - next_trade_id（用于外部累加）
    """
    df = df.sort_values("date").reset_index(drop=True).copy()
    df["atr"] = atr(df, ATR_N)
    df = compute_signal_vA(df)

    start_dt = pd.to_datetime(DATE_START)
    end_dt   = pd.to_datetime(DATE_END)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
    if len(df) < 5:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), trade_id_start

    cash = INIT_CASH
    shares = 0.0
    avg_cost = np.nan

    in_pos = False
    tranches = 0
    peak_high = np.nan
    stop = np.nan

    entry_date = None
    entry_avg = np.nan
    trade_id = trade_id_start

    # ✅ 新增：记录“每次入场事件”（开仓/加仓）
    # 每个元素：{"trade_id", "k", "entry_date", "entry_idx", "entry_px", "entry_shares"}
    entry_events = []

    trades = []
    eq_rows = []

    for i in range(len(df)):
        row = df.loc[i]
        date = row["date"]

        # 收盘盯市
        mkt_value = shares * float(row["close"])
        equity = cash + mkt_value

        eq_rows.append({
            "date": date,
            "equity": equity,
            "cash": cash,
            "shares": shares,
            "close": float(row["close"]),
            "avg_cost": avg_cost,
            "tranches": tranches,
            "stop": stop,
            "signal_vA": bool(row.get("signal_vA", False)),
        })

        if i == len(df) - 1:
            break

        next_row = df.loc[i + 1]
        next_open = float(next_row["open"])

        # ===== 持仓：止损优先 =====
        if in_pos and pd.notna(stop):
            if float(row["low"]) <= float(stop):
                fill = min(next_open, float(stop)) if USE_CONSERVATIVE_STOP_FILL else next_open
                fill = apply_cost(fill, "sell")
                cash += shares * fill

                # ===== 退出时：对本 trade_id 的每次入场分别计算 20D runup =====
                this_entries = [e for e in entry_events if e["trade_id"] == trade_id]
                entry_runup_rows = []
                best_abs = -np.inf
                best_pct = -np.inf
                entry_prices_list = []

                for e in this_entries:
                    run_abs, run_pct, run_high, run_high_date = max_runup_20d(
                        df_slice=df,
                        entry_idx=int(e["entry_idx"]),
                        entry_price=float(e["entry_px"]),
                        window=RUNUP_WINDOW_DAYS
                    )
                    entry_prices_list.append(f"{e['entry_px']:.4f}")
                    entry_runup_rows.append({
                        "trade_id": trade_id,
                        "entry_k": int(e["k"]),
                        "entry_date": e["entry_date"],
                        "entry_px": float(e["entry_px"]),
                        "entry_shares": float(e["entry_shares"]),
                        "max_runup_20d_abs": run_abs,
                        "max_runup_20d_pct": run_pct,
                        "max_runup_20d_high": run_high,
                        "max_runup_20d_high_date": run_high_date,
                    })
                    if pd.notna(run_abs) and run_abs > best_abs:
                        best_abs = run_abs
                    if pd.notna(run_pct) and run_pct > best_pct:
                        best_pct = run_pct

                # 写 trades（每笔交易一行）
                trades.append({
                    "trade_id": trade_id,
                    "entry_date": entry_date,
                    "exit_date": next_row["date"],
                    "entry_avg": float(entry_avg),
                    "exit_price": float(fill),
                    "shares": float(shares),
                    "pnl": float((fill - entry_avg) * shares),
                    "ret": float(fill / entry_avg - 1.0),
                    "reason": "chandelier_stop",
                    "tranches": int(tranches),

                    # ✅ 新增：本笔交易中“各次入场价”（开仓/加仓）
                    "entry_prices": ",".join(entry_prices_list),

                    # ✅ 新增：本笔交易里，按“每次入场价”算出来的最佳 20D 最大空间（取最大那口）
                    "best_entry_runup_20d_abs": float(best_abs) if best_abs != -np.inf else np.nan,
                    "best_entry_runup_20d_pct": float(best_pct) if best_pct != -np.inf else np.nan,
                })

                # 把每次入场的 runup 行暂存到一个临时字段里（函数末尾统一返回）
                # 这里用一个小技巧：塞进 trades[-1] 的隐藏字段不太干净，所以用局部累加列表
                if "___entry_runups_acc" not in locals():
                    ___entry_runups_acc = []
                for rrr in entry_runup_rows:
                    rrr["exit_date"] = next_row["date"]
                    rrr["exit_reason"] = "chandelier_stop"
                    ___entry_runups_acc.append(rrr)

                # 清仓 & 准备下一笔 trade_id
                shares = 0.0
                avg_cost = np.nan
                in_pos = False
                tranches = 0
                peak_high = np.nan
                stop = np.nan
                entry_date = None
                entry_avg = np.nan

                trade_id += 1
                continue

        # ===== 持仓：更新止损 + 加仓 =====
        if in_pos:
            peak_high = float(row["high"]) if not pd.notna(peak_high) else max(float(peak_high), float(row["high"]))
            if pd.notna(row["atr"]) and pd.notna(peak_high):
                stop = peak_high - K_ATR * float(row["atr"])

            # 加仓：收盘确认 -> 次日开盘执行（逻辑不变）
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

                        # ✅ 新增：记录这次加仓的“入场价/入场日/入场索引”
                        entry_events.append({
                            "trade_id": trade_id,
                            "k": tranches,                 # 第几口（2/3/..）
                            "entry_date": next_row["date"],# 次日开盘成交日
                            "entry_idx": i + 1,            # next_row 对应的索引
                            "entry_px": float(buy_px),
                            "entry_shares": float(add_shares),
                        })
            continue

        # ===== 空仓：开仓（信号日 -> 次日开盘）=====
        if bool(row.get("signal_vA", False)):
            buy_value = min(INIT_CASH * TRANCHE_PCT, INIT_CASH * MAX_EXPOSURE, cash)
            if buy_value <= 0:
                continue

            buy_px = apply_cost(next_open, "buy")
            buy_shares = buy_value / buy_px

            shares = buy_shares
            cash -= buy_shares * buy_px
            avg_cost = buy_px

            in_pos = True
            tranches = 1
            entry_date = next_row["date"]
            entry_avg = avg_cost

            # ✅ 新增：记录开仓那一口的入场价/入场日/入场索引
            entry_events.append({
                "trade_id": trade_id,
                "k": 1,
                "entry_date": next_row["date"],
                "entry_idx": i + 1,
                "entry_px": float(buy_px),
                "entry_shares": float(buy_shares),
            })

            peak_high = float(next_row["high"])
            stop = peak_high - K_ATR * float(next_row["atr"]) if pd.notna(next_row["atr"]) else np.nan

    # ===== 期末强制平仓 =====
    if FORCE_EXIT_AT_END and in_pos and shares > 0:
        last_row = df.iloc[-1]
        last_date = last_row["date"]
        last_close = float(last_row["close"])

        fill = apply_cost(last_close, "sell")
        cash += shares * fill

        # 对本 trade_id 的每次入场分别计算 20D runup
        this_entries = [e for e in entry_events if e["trade_id"] == trade_id]
        entry_runup_rows = []
        best_abs = -np.inf
        best_pct = -np.inf
        entry_prices_list = []

        for e in this_entries:
            run_abs, run_pct, run_high, run_high_date = max_runup_20d(
                df_slice=df,
                entry_idx=int(e["entry_idx"]),
                entry_price=float(e["entry_px"]),
                window=RUNUP_WINDOW_DAYS
            )
            entry_prices_list.append(f"{e['entry_px']:.4f}")
            entry_runup_rows.append({
                "trade_id": trade_id,
                "entry_k": int(e["k"]),
                "entry_date": e["entry_date"],
                "entry_px": float(e["entry_px"]),
                "entry_shares": float(e["entry_shares"]),
                "max_runup_20d_abs": run_abs,
                "max_runup_20d_pct": run_pct,
                "max_runup_20d_high": run_high,
                "max_runup_20d_high_date": run_high_date,
            })
            if pd.notna(run_abs) and run_abs > best_abs:
                best_abs = run_abs
            if pd.notna(run_pct) and run_pct > best_pct:
                best_pct = run_pct

        trades.append({
            "trade_id": trade_id,
            "entry_date": entry_date,
            "exit_date": last_date,
            "entry_avg": float(entry_avg),
            "exit_price": float(fill),
            "shares": float(shares),
            "pnl": float((fill - entry_avg) * shares),
            "ret": float(fill / entry_avg - 1.0),
            "reason": "end_of_period",
            "tranches": int(tranches),
            "entry_prices": ",".join(entry_prices_list),
            "best_entry_runup_20d_abs": float(best_abs) if best_abs != -np.inf else np.nan,
            "best_entry_runup_20d_pct": float(best_pct) if best_pct != -np.inf else np.nan,
        })

        if "___entry_runups_acc" not in locals():
            ___entry_runups_acc = []
        for rrr in entry_runup_rows:
            rrr["exit_date"] = last_date
            rrr["exit_reason"] = "end_of_period"
            ___entry_runups_acc.append(rrr)

        eq_rows.append({
            "date": last_date,
            "equity": cash,
            "cash": cash,
            "shares": 0.0,
            "close": last_close,
            "avg_cost": np.nan,
            "tranches": 0,
            "stop": np.nan,
            "signal_vA": False,
        })

        trade_id += 1

    equity_df = pd.DataFrame(eq_rows)
    trades_df = pd.DataFrame(trades)

    entry_runups_df = pd.DataFrame(locals().get("___entry_runups_acc", []))
    return equity_df, trades_df, entry_runups_df, trade_id


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
            if "breakout_va_filtered" in name or "breakout" in name:
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


def summarize_entry_runups(entry_runups_df: pd.DataFrame) -> dict:
    """
    按“每次入场（开仓/加仓）”的 runup 汇总：
    - 平均/最好/最后一次入场 的 20D runup（pct/abs）
    """
    if entry_runups_df is None or entry_runups_df.empty:
        return {
            "avg_entry_runup_20d_pct": np.nan,
            "best_entry_runup_20d_pct": np.nan,
            "last_entry_runup_20d_pct": np.nan,
            "avg_entry_runup_20d_abs": np.nan,
            "best_entry_runup_20d_abs": np.nan,
            "last_entry_runup_20d_abs": np.nan,
        }

    t = entry_runups_df.sort_values(["trade_id", "entry_date", "entry_k"]).copy()
    pct = pd.to_numeric(t["max_runup_20d_pct"], errors="coerce")
    abs_ = pd.to_numeric(t["max_runup_20d_abs"], errors="coerce")

    last_pct = float(pct.iloc[-1]) if len(pct) else np.nan
    last_abs = float(abs_.iloc[-1]) if len(abs_) else np.nan

    best_idx = int(pct.idxmax()) if pct.notna().any() else None
    best_pct = float(pct.loc[best_idx]) if best_idx is not None else np.nan
    best_abs = float(abs_.loc[best_idx]) if best_idx is not None else np.nan

    return {
        "avg_entry_runup_20d_pct": float(pct.mean()) if pct.notna().any() else np.nan,
        "best_entry_runup_20d_pct": best_pct,
        "last_entry_runup_20d_pct": last_pct,
        "avg_entry_runup_20d_abs": float(abs_.mean()) if abs_.notna().any() else np.nan,
        "best_entry_runup_20d_abs": best_abs,
        "last_entry_runup_20d_abs": last_abs,
    }


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
        print(f"[2.1] Apply MAX_CODES => {len(ts_codes)}")

    if not ts_codes:
        raise RuntimeError("没提取到任何可回测 ts_code。")

    conn = sqlite3.connect(DB_PATH.as_posix())
    per_stock_rows = []
    all_trades = []
    all_entry_runups = []

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

            # 回测（返回 trades + entry_runups）
            equity, trades, entry_runups, _ = backtest_one_stock(df, trade_id_start=1)
            if equity.empty:
                continue

            metrics = compute_metrics(equity, trades, rf_annual=RF_ANNUAL)

            row_uni = uni.loc[uni["ts_code"] == ts]
            name = row_uni["name"].iloc[0] if len(row_uni) else ""
            symbol = normalize_symbol6(row_uni["symbol"].iloc[0]) if len(row_uni) else ""

            # per-stock 汇总：基于“每次入场”的 runup
            runup_summary = summarize_entry_runups(entry_runups)

            per_stock_rows.append({
                "ts_code": ts,
                "symbol": symbol,
                "name": name,
                **metrics,
                **runup_summary
            })

            if trades is not None and len(trades):
                trades2 = trades.copy()
                trades2.insert(0, "ts_code", ts)
                trades2.insert(1, "symbol", symbol)
                trades2.insert(2, "name", name)
                all_trades.append(trades2)

            if entry_runups is not None and len(entry_runups):
                e2 = entry_runups.copy()
                e2.insert(0, "ts_code", ts)
                e2.insert(1, "symbol", symbol)
                e2.insert(2, "name", name)
                all_entry_runups.append(e2)

            if idx % 50 == 0:
                print(f"[3] Done {idx}/{len(ts_codes)}")

    finally:
        conn.close()

    if not per_stock_rows:
        raise RuntimeError("回测完没有任何结果（可能区间内无信号/无交易）。")

    per_stock = pd.DataFrame(per_stock_rows).sort_values(["total_return", "sum_pnl"], ascending=[False, False])
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    entry_runups_df = pd.concat(all_entry_runups, ignore_index=True) if all_entry_runups else pd.DataFrame()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        per_stock.to_excel(w, sheet_name="per_stock_metrics", index=False)
        trades_df.to_excel(w, sheet_name="trades", index=False)
        entry_runups_df.to_excel(w, sheet_name="entry_runup_20d", index=False)

    print(f"\nSaved: {OUT_XLSX}")
    print(f"Stocks: {len(per_stock)}; Trades rows: {len(trades_df)}; Entry-runup rows: {len(entry_runups_df)}")


if __name__ == "__main__":
    main()
