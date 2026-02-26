# -*- coding: utf-8 -*-
import re
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd
from collections import deque

# ===================== 配置 =====================
# ✅ 股票池：近7日信号表（只是股票池来源）
INPUT_PATH = Path(r"D:\scan_signals_last7td.xlsx")

DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")

# 你想“看近7天交易表现”，所以这里 DATE_START 不再用于“限制开仓”
# 它只用于“最少保留多少历史数据”（可以很早，算指标用）
HISTORY_START = "2024-01-01"     # 给指标留足历史，随便早点
DATE_END      = "2026-02-10"     # 回测截止日（必须是已收盘交易日）

OUT_XLSX = Path(r"D:\batch_exec_backtest_results_with_TP_T_last7.xlsx")
INIT_CASH = 1_000_000.0

# ✅ 只允许最近 N 个交易日内开仓（你要的“近7天交易回测”）
ENTRY_LAST_N_TD = 7

# 只做涨停信号日
LIMIT_TAG_PCT = 0.099
ONLY_TRADE_IF_SIGNAL_LIMITUP = True

# 含信号日的过去5个交易日有2个板不进
HOT5_WIN = 5
HOT5_LU  = 2

# 一字板不给进
DISALLOW_YIZIBAN_ENTRY = True
YIZIBAN_STRICT = True  # True: open=high=low=close 且 high==low; False: 只要 high==low

FORCE_EXIT_AT_END = True

ATR_N = 14
K_ATR = 3.0

# 加仓触发（苛刻条件：触发价 + MA5上行 + MA10上行）
ADD_ATR = 1.0

COMM_RATE = 0.0
SLIPPAGE_BP = 0.0
USE_CONSERVATIVE_STOP_FILL = True

# MA5 风控（3天2次 + 7天3次）
FAST_WIN_DAYS = 3
FAST_WIN_BREACH = 2
ROLL_WIN_DAYS = 7
ROLL_WIN_BREACH = 3

# ========= 仓位计划：试仓15% -> 加10% -> 剩余补满到50% =========
MAX_EXPOSURE = 0.50
ENTRY_PCT = 0.15
ADD_PCTS = [0.10, None]  # None 表示“把剩余补满到 MAX_EXPOSURE”
# =============================================================

# ============= MA10 支撑位 TP-A（止盈一半 + 做T回补）=============
USE_TP_T = True
TP_SELL_FRAC = 0.50  # 止盈卖出一半

# TP-A：从MA10上方跌破MA10，且当前仍盈利（避免亏着卖一半）
TP_REQUIRE_PROFIT = True

# 做T回补（三条件）：close>MA10、MA10上行、close>昨日high；收盘确认->次日开盘回补
# =================================================================

MA_CONFIRM_CLOSE_ABOVE = False
MAX_CODES = None
# ===============================================


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
    df["ma10"] = c.rolling(10, min_periods=10).mean()
    df["ma20"] = c.rolling(20, min_periods=20).mean()

    mb = df["ma20"]
    dif = ema(c, 12) - ema(c, 26)
    dea = ema(dif, 9)
    df["dif"] = dif
    df["dea"] = dea

    df["res20"] = h.rolling(20, min_periods=20).max().shift(1)
    df["vma20"] = v.rolling(20, min_periods=20).mean().shift(1)

    c1 = df["close"] > df["res20"] * (1 + BPS_BREAK)
    c2 = (df["close"] > mb) & (mb > mb.shift(1))
    c3 = df["dif"] > df["dea"]
    c4 = (df["close"] > df["ma5"]) & (df["ma5"] > df["ma5"].shift(1))
    c5 = df["volume"] > df["vma20"] * VOL_MULT
    c6 = df["pct"] >= MIN_PCT

    df["signal_vA"] = (c1 & c2 & c3 & c4 & c5 & c6)

    df["is_limit_up"] = df["pct"] >= LIMIT_TAG_PCT

    df["lu_cnt_5_incl"] = df["is_limit_up"].rolling(HOT5_WIN, min_periods=HOT5_WIN).sum()
    df["hot_5d_2lu_incl"] = df["lu_cnt_5_incl"] >= HOT5_LU
    df["signal_vA"] = df["signal_vA"] & (~df["hot_5d_2lu_incl"])
    return df

def is_yiziban_signal_day(row: pd.Series) -> bool:
    try:
        o = float(row["open"]); h = float(row["high"]); l = float(row["low"]); c = float(row["close"])
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

def tranche_pct_for_index(tr_idx: int) -> float:
    plan = [ENTRY_PCT] + ADD_PCTS
    if tr_idx <= 0 or tr_idx > len(plan):
        return 0.0
    pct = plan[tr_idx - 1]
    if pct is not None:
        return float(pct)
    used = 0.0
    for p in plan[:tr_idx - 1]:
        if p is None:
            break
        used += float(p)
    return max(0.0, float(MAX_EXPOSURE) - used)

MAX_TRANCHES = 1 + len(ADD_PCTS)

def backtest_one_stock(df: pd.DataFrame, ts_code: str):
    # 先用长历史算指标
    df = df.sort_values("date").reset_index(drop=True).copy()
    df = df[df["date"] >= pd.to_datetime(HISTORY_START)].copy()

    end_dt = pd.to_datetime(DATE_END)
    df = df[df["date"] <= end_dt].reset_index(drop=True)
    if len(df) < 60:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    df["atr"] = atr(df, ATR_N)
    df = compute_signal_vA(df)

    # ✅ 关键：只允许最近 ENTRY_LAST_N_TD 个交易日内的信号开仓
    unique_dates = df["date"].drop_duplicates().sort_values().tolist()
    if len(unique_dates) >= ENTRY_LAST_N_TD:
        entry_window_start = unique_dates[-ENTRY_LAST_N_TD]
    else:
        entry_window_start = unique_dates[0]

    cash = INIT_CASH
    shares = 0.0
    avg_cost = np.nan

    in_pos = False
    tranches = 0
    peak_high = np.nan
    stop = np.nan

    entry_date = None
    entry_idx = None
    trade_id = 0
    trade_start_equity = None

    fast_below_cnt = 0
    roll_below = deque(maxlen=ROLL_WIN_DAYS)

    pending_rebuy_qty = 0.0

    trades = []
    eq_rows = []
    inflows = []
    events = []

    def record_inflow(kind: str, inflow_date, price, value, qty, tranches_after: int):
        inflows.append({
            "ts_code": ts_code,
            "trade_id": trade_id,
            "inflow_type": kind,
            "inflow_date": pd.to_datetime(inflow_date),
            "inflow_price": float(price),
            "inflow_value": float(value),
            "inflow_shares": float(qty),
            "tranches_after": int(tranches_after),
        })

    def record_event(evt_type: str, evt_date, side: str, price: float, qty: float, reason: str, extra=None):
        nonlocal tranches, pending_rebuy_qty
        d = {
            "ts_code": ts_code,
            "trade_id": trade_id,
            "event_type": evt_type,
            "event_date": pd.to_datetime(evt_date),
            "side": side,
            "price": float(price),
            "qty": float(qty),
            "value": float(price * qty),
            "reason": reason,
            "tranches_after": int(tranches),
            "pending_rebuy_qty_after": float(pending_rebuy_qty),
        }
        if extra:
            d.update(extra)
        events.append(d)

    def do_full_exit(next_row, exit_price, reason):
        nonlocal cash, shares, avg_cost, in_pos, tranches, peak_high, stop
        nonlocal entry_date, entry_idx, fast_below_cnt, roll_below
        nonlocal pending_rebuy_qty, trade_start_equity

        fill = apply_cost(exit_price, "sell")
        cash += shares * fill
        record_event("exit", next_row["date"], "SELL", float(fill), float(shares), reason)

        end_equity = cash
        pnl = float(end_equity - (trade_start_equity if trade_start_equity is not None else INIT_CASH))
        trades.append({
            "ts_code": ts_code,
            "trade_id": int(trade_id),
            "entry_date": entry_date,
            "exit_date": next_row["date"],
            "reason": reason,
            "pnl": pnl,
        })

        shares = 0.0
        avg_cost = np.nan
        in_pos = False
        tranches = 0
        peak_high = np.nan
        stop = np.nan
        entry_date = None
        entry_idx = None
        fast_below_cnt = 0
        roll_below.clear()
        pending_rebuy_qty = 0.0
        trade_start_equity = None

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
            "ma10": float(row["ma10"]) if pd.notna(row["ma10"]) else np.nan,
            "avg_cost": avg_cost,
            "tranches": tranches,
            "stop": stop,
            "signal_vA": bool(row.get("signal_vA", False)),
            "is_limit_up": bool(row.get("is_limit_up", False)),
            "pending_rebuy_qty": pending_rebuy_qty,
            "entry_window_start": entry_window_start,  # 方便你调试看
        })

        if i == len(df) - 1:
            break

        next_row = df.loc[i + 1]
        next_open = float(next_row["open"])

        if in_pos:
            # MA5 风控
            below_ma5 = False
            if pd.notna(row["ma5"]) and pd.notna(row["close"]):
                below_ma5 = float(row["close"]) < float(row["ma5"])

            if entry_idx is not None:
                day_in_pos = i - entry_idx + 1
                if day_in_pos <= FAST_WIN_DAYS:
                    if below_ma5:
                        fast_below_cnt += 1
                    if fast_below_cnt >= FAST_WIN_BREACH:
                        do_full_exit(next_row, next_open, f"ma5_fast_{FAST_WIN_DAYS}d_{FAST_WIN_BREACH}breach")
                        continue

            roll_below.append(1 if below_ma5 else 0)
            if len(roll_below) == ROLL_WIN_DAYS and sum(roll_below) >= ROLL_WIN_BREACH:
                do_full_exit(next_row, next_open, f"ma5_roll_{ROLL_WIN_DAYS}d_{ROLL_WIN_BREACH}breach")
                continue

            # 吊灯止损
            peak_high = float(row["high"]) if not pd.notna(peak_high) else max(float(peak_high), float(row["high"]))
            if pd.notna(row["atr"]) and pd.notna(peak_high):
                stop = peak_high - K_ATR * float(row["atr"])

            if pd.notna(stop) and float(row["low"]) <= float(stop):
                fill0 = min(next_open, float(stop)) if USE_CONSERVATIVE_STOP_FILL else next_open
                do_full_exit(next_row, fill0, "chandelier_stop")
                continue

            # TP-A & 做T回补
            if USE_TP_T and pd.notna(row.get("ma10", np.nan)):
                ma10 = float(row["ma10"])
                prev_ma10 = float(df.loc[i-1, "ma10"]) if i-1 >= 0 and pd.notna(df.loc[i-1, "ma10"]) else np.nan
                prev_close = float(df.loc[i-1, "close"]) if i-1 >= 0 else np.nan
                close = float(row["close"])

                # 回补
                if pending_rebuy_qty > 0:
                    cond1 = close > ma10
                    cond2 = pd.notna(prev_ma10) and (ma10 > prev_ma10)
                    cond3 = (i-1 >= 0) and (close > float(df.loc[i-1, "high"]))
                    if cond1 and cond2 and cond3:
                        buy_px = apply_cost(next_open, "buy")

                        max_total_value = INIT_CASH * MAX_EXPOSURE
                        current_value = shares * next_open
                        remain_value = max(0.0, max_total_value - current_value)

                        want_value = pending_rebuy_qty * buy_px
                        can_value = min(want_value, remain_value, cash)

                        if can_value > 0:
                            buy_qty = can_value / buy_px
                            cost = buy_qty * buy_px

                            shares += buy_qty
                            cash -= cost
                            avg_cost = (avg_cost * (shares - buy_qty) + buy_px * buy_qty) / shares

                            pending_rebuy_qty = max(0.0, pending_rebuy_qty - buy_qty)
                            record_inflow("t_rebuy", next_row["date"], buy_px, cost, buy_qty, tranches)
                            record_event("t_rebuy", next_row["date"], "BUY", buy_px, buy_qty,
                                         "t_rebuy(close>ma10 & ma10_up & close>yhigh)",
                                         {"cond1_close_gt_ma10": cond1, "cond2_ma10_up": cond2, "cond3_close_gt_yhigh": cond3})
                    continue

                # 卖一半
                prev_above = pd.notna(prev_ma10) and (prev_close >= prev_ma10)
                now_below  = close < ma10
                profitable = (not TP_REQUIRE_PROFIT) or (pd.notna(avg_cost) and close > float(avg_cost))

                if prev_above and now_below and profitable and shares > 0:
                    sell_qty = shares * TP_SELL_FRAC
                    if sell_qty > 0:
                        sell_px = apply_cost(next_open, "sell")
                        cash += sell_qty * sell_px
                        shares -= sell_qty
                        pending_rebuy_qty = sell_qty

                        record_event("tp_sell_half", next_row["date"], "SELL", sell_px, sell_qty,
                                     "tpA_break_ma10_sell_half",
                                     {"prev_above_ma10": prev_above, "now_below_ma10": now_below, "profitable": profitable})
                        continue

            # 计划加仓：15% -> 10% -> 补满到50%
            if tranches < MAX_TRANCHES and pd.notna(row["atr"]) and pd.notna(avg_cost):
                add_trigger = float(avg_cost) + ADD_ATR * float(row["atr"])
                cond_price = float(row["close"]) >= add_trigger

                cond_ma5 = pd.notna(row["ma5"]) and (float(row["close"]) > float(row["ma5"])) and \
                           (float(row["ma5"]) > float(df.loc[i-1, "ma5"]) if i-1 >= 0 and pd.notna(df.loc[i-1, "ma5"]) else True)
                cond_ma10 = pd.notna(row["ma10"]) and (float(row["close"]) > float(row["ma10"])) and \
                            (float(row["ma10"]) > float(df.loc[i-1, "ma10"]) if i-1 >= 0 and pd.notna(df.loc[i-1, "ma10"]) else True)

                if cond_price and cond_ma5 and cond_ma10:
                    next_tr_idx = tranches + 1
                    pct = tranche_pct_for_index(next_tr_idx)
                    if pct <= 0:
                        continue

                    target_add_value = INIT_CASH * pct
                    max_total_value = INIT_CASH * MAX_EXPOSURE

                    current_value = shares * next_open
                    remain = max(0.0, max_total_value - current_value)
                    add_value = min(target_add_value, remain, cash)

                    if add_value > 0:
                        buy_px = apply_cost(next_open, "buy")
                        add_qty = add_value / buy_px
                        cost = add_qty * buy_px

                        avg_cost = (avg_cost * shares + buy_px * add_qty) / (shares + add_qty)
                        shares += add_qty
                        cash -= cost
                        tranches += 1

                        kind = "add_1" if tranches == 2 else ("add_2" if tranches == 3 else f"add_{tranches-1}")
                        record_inflow(kind, next_row["date"], buy_px, cost, add_qty, tranches)
                        record_event("add", next_row["date"], "BUY", buy_px, add_qty,
                                     "add_strict(cond_price & ma5_up & ma10_up)",
                                     {"cond_price": cond_price, "cond_ma5": bool(cond_ma5), "cond_ma10": bool(cond_ma10)})
            continue

        # 空仓：入场（✅ 关键限制：只允许最近7个交易日内开仓）
        if date < entry_window_start:
            continue

        if bool(row.get("signal_vA", False)):
            if ONLY_TRADE_IF_SIGNAL_LIMITUP and (not bool(row.get("is_limit_up", False))):
                continue
            if DISALLOW_YIZIBAN_ENTRY and is_yiziban_signal_day(row):
                continue

            if MA_CONFIRM_CLOSE_ABOVE:
                if pd.notna(next_row["ma5"]) and pd.notna(next_row["close"]):
                    if float(next_row["close"]) < float(next_row["ma5"]):
                        continue

            pct = tranche_pct_for_index(1)  # 15%
            buy_value = min(INIT_CASH * pct, INIT_CASH * MAX_EXPOSURE, cash)
            if buy_value <= 0:
                continue

            buy_px = apply_cost(next_open, "buy")
            buy_qty = buy_value / buy_px
            cost = buy_qty * buy_px

            trade_id += 1
            trade_start_equity = cash + shares * float(row["close"])

            shares = buy_qty
            cash -= cost
            avg_cost = buy_px

            in_pos = True
            tranches = 1
            entry_date = next_row["date"]
            entry_idx = i + 1

            peak_high = float(next_row["high"])
            stop = peak_high - K_ATR * float(next_row["atr"]) if pd.notna(next_row["atr"]) else np.nan

            fast_below_cnt = 0
            roll_below.clear()
            pending_rebuy_qty = 0.0

            record_inflow("entry", next_row["date"], buy_px, cost, buy_qty, tranches)
            record_event("entry", next_row["date"], "BUY", buy_px, buy_qty, "entry_15pct")

    # 期末强平（到 DATE_END）
    if FORCE_EXIT_AT_END and in_pos and shares > 0:
        last_row = df.iloc[-1]
        do_full_exit(last_row, float(last_row["close"]), "end_of_period")

        eq_rows.append({
            "date": last_row["date"],
            "equity": cash,
            "cash": cash,
            "shares": 0.0,
            "close": float(last_row["close"]),
            "ma5": float(last_row["ma5"]) if pd.notna(last_row["ma5"]) else np.nan,
            "ma10": float(last_row["ma10"]) if pd.notna(last_row["ma10"]) else np.nan,
            "avg_cost": np.nan,
            "tranches": 0,
            "stop": np.nan,
            "signal_vA": False,
            "is_limit_up": False,
            "pending_rebuy_qty": 0.0,
        })

    # ✅ 为了让输出更“干净”：只保留最近7个交易日窗口内发生的事件/交易/流入
    eq_df = pd.DataFrame(eq_rows)
    trades_df = pd.DataFrame(trades)
    inflows_df = pd.DataFrame(inflows)
    events_df = pd.DataFrame(events)

    # trades 用 entry_date / exit_date 过滤（只看近7日开仓产生的交易）
    if not trades_df.empty:
        trades_df["entry_date"] = pd.to_datetime(trades_df["entry_date"])
        trades_df = trades_df[trades_df["entry_date"] >= entry_window_start].copy()

    if not inflows_df.empty:
        inflows_df["inflow_date"] = pd.to_datetime(inflows_df["inflow_date"])
        inflows_df = inflows_df[inflows_df["inflow_date"] >= entry_window_start].copy()

    if not events_df.empty:
        events_df["event_date"] = pd.to_datetime(events_df["event_date"])
        events_df = events_df[events_df["event_date"] >= entry_window_start].copy()

    # equity 你要只看近7日也可以（这里我也裁一下，避免你看到历史）
    if not eq_df.empty:
        eq_df["date"] = pd.to_datetime(eq_df["date"])
        eq_df = eq_df[eq_df["date"] >= entry_window_start].copy()

    return eq_df, trades_df, inflows_df, events_df


# ===================== 股票池读取（来自 scan_signals_last7td.xlsx）=====================
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
    if not input_path.exists():
        raise FileNotFoundError(f"找不到路径：{input_path}")
    raw_codes = extract_codes_from_file(input_path)

    cleaned = []
    for x in raw_codes:
        x = x.strip().replace(" ", "")
        if re.fullmatch(r"\d{6}\.(SZ|SH)", x, flags=re.I):
            cleaned.append(x.upper())
        elif re.fullmatch(r"\d{6}", x):
            cleaned.append(x)
    return list(dict.fromkeys(cleaned))

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

    if not INPUT_PATH.exists():
        raise FileNotFoundError(f"找不到股票池文件：{INPUT_PATH}")

    raw_codes = gather_codes(INPUT_PATH)
    print(f"[1] Found raw codes (from last7 xlsx): {len(raw_codes)}")

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
        raise RuntimeError("没提取到任何可回测 ts_code。请检查 scan_signals_last7td.xlsx 的代码列。")

    conn = sqlite3.connect(DB_PATH.as_posix())
    per_stock_rows = []
    all_trades = []
    all_inflows = []
    all_events = []

    try:
        for idx, ts_code in enumerate(ts_codes, 1):
            df = pd.read_sql_query(
                "SELECT date, open, high, low, close, volume FROM daily WHERE ts_code=? ORDER BY date",
                conn, params=(ts_code,)
            )
            if df.empty:
                continue

            df["date"] = pd.to_datetime(df["date"])
            for c in ["open", "high", "low", "close", "volume"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            df = df.dropna()
            if len(df) < 80:
                continue

            equity, trades, inflows, events = backtest_one_stock(df, ts_code=ts_code)
            if equity.empty and trades.empty and inflows.empty and events.empty:
                continue

            metrics = compute_metrics(equity if not equity.empty else pd.DataFrame([{"date": pd.to_datetime(DATE_END), "equity": INIT_CASH}]),
                                     trades)

            row_uni = uni.loc[uni["ts_code"] == ts_code]
            name = row_uni["name"].iloc[0] if len(row_uni) else ""
            symbol = normalize_symbol6(row_uni["symbol"].iloc[0]) if len(row_uni) else ""

            per_stock_rows.append({
                "ts_code": ts_code,
                "symbol": symbol,
                "name": name,
                "entry_last_n_td": ENTRY_LAST_N_TD,
                "max_exposure": MAX_EXPOSURE,
                "entry_pct": ENTRY_PCT,
                "add1_pct": ADD_PCTS[0],
                "add2_pct": "fill_to_50%" if ADD_PCTS[1] is None else ADD_PCTS[1],
                "add_atr": ADD_ATR,
                "tp_sell_frac": TP_SELL_FRAC if USE_TP_T else 0.0,
                "only_trade_if_signal_limitup": ONLY_TRADE_IF_SIGNAL_LIMITUP,
                "hot_rule": f"incl_{HOT5_WIN}d_lu>={HOT5_LU}",
                "disallow_yiziban_entry": DISALLOW_YIZIBAN_ENTRY,
                "ma_fast_rule": f"{FAST_WIN_DAYS}d_{FAST_WIN_BREACH}breach",
                "ma_roll_rule": f"{ROLL_WIN_DAYS}d_{ROLL_WIN_BREACH}breach",
                **metrics
            })

            if not trades.empty:
                t2 = trades.copy()
                t2.insert(1, "symbol", symbol)
                t2.insert(2, "name", name)
                all_trades.append(t2)

            if not inflows.empty:
                f2 = inflows.copy()
                f2.insert(1, "symbol", symbol)
                f2.insert(2, "name", name)
                all_inflows.append(f2)

            if not events.empty:
                e2 = events.copy()
                e2.insert(1, "symbol", symbol)
                e2.insert(2, "name", name)
                all_events.append(e2)

            if idx % 50 == 0:
                print(f"[3] Done {idx}/{len(ts_codes)}")

    finally:
        conn.close()

    if not per_stock_rows:
        raise RuntimeError("回测完没有任何结果（近7交易日窗口内可能无开仓）。")

    per_stock = pd.DataFrame(per_stock_rows).sort_values(["total_return", "sum_pnl"], ascending=[False, False])
    trades_df = pd.concat(all_trades, ignore_index=True) if all_trades else pd.DataFrame()
    inflows_df = pd.concat(all_inflows, ignore_index=True) if all_inflows else pd.DataFrame()
    events_df = pd.concat(all_events, ignore_index=True) if all_events else pd.DataFrame()

    OUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUT_XLSX, engine="openpyxl") as w:
        per_stock.to_excel(w, sheet_name="per_stock_metrics", index=False)
        trades_df.to_excel(w, sheet_name="trades", index=False)
        inflows_df.to_excel(w, sheet_name="cash_inflows", index=False)
        events_df.to_excel(w, sheet_name="events", index=False)

    print(f"\nSaved: {OUT_XLSX}")
    print(f"Stocks: {len(per_stock)}; Trades rows: {len(trades_df)}; Inflows rows: {len(inflows_df)}; Events rows: {len(events_df)}")


if __name__ == "__main__":
    main()
