
import sqlite3
from pathlib import Path
import numpy as np
import pandas as pd


# ===================== 你要改的配置 =====================
DB_PATH = Path(r"D:\a_share_daily_tushare.sqlite")
TS_CODE = "605007.SH"
DATE_START = "2025-12-01"
DATE_END   = "2026-02-06"

INIT_CASH = 1_000_000.0

# 分仓参数
MAX_EXPOSURE = 0.50          # 总仓位上限（≤50%）
TRANCHE_PCT  = 0.25          # 每次建仓资金占初始资金比例（25%）
MAX_TRANCHES = int(np.floor(MAX_EXPOSURE / TRANCHE_PCT + 1e-9))  # 默认2

# ATR / 吊灯止损
ATR_N = 14
K_ATR = 3.0

# 加仓触发
ADD_ATR = 1.0

# 成本（先全 0）
COMM_RATE = 0.0              # 单边佣金比例（万3=0.0003）
SLIPPAGE_BP = 0.0            # 单边滑点 bp（1bp=0.01%）

# 止损成交：保守（更差价）
USE_CONSERVATIVE_STOP_FILL = True

# 夏普：无风险利率（年化）
RF_ANNUAL = 0.0

# 关键：期末强制平仓（避免“收益=0但权益上涨”）
FORCE_EXIT_AT_END = True
# =======================================================


# ===== 信号：复刻你 Step2 的版本A + pct>=8% =====
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
    df["is_limit_up"] = df["pct"] >= LIMIT_TAG_PCT  # 标签，不参与筛选
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


def compute_metrics(equity_curve: pd.DataFrame, trades: pd.DataFrame, rf_annual: float = 0.0) -> dict:
    """
    你要的指标：
    - sum_pnl: 最终权益 - 初始权益（包含浮盈/已实现）
    - 其他：按你要求计算
    """
    ec = equity_curve.sort_values("date").copy()
    ec["ret"] = ec["equity"].pct_change()
    rets = ec["ret"].dropna()

    start_eq = float(ec["equity"].iloc[0])
    end_eq   = float(ec["equity"].iloc[-1])

    # 最大回撤
    ec["peak"] = ec["equity"].cummax()
    ec["dd"] = ec["equity"] / ec["peak"] - 1.0
    max_drawdown = float(ec["dd"].min())

    # 夏普（按252交易日年化）
    rf_daily = rf_annual / 252.0
    if len(rets) > 1 and rets.std(ddof=0) > 0:
        sharpe = float(((rets.mean() - rf_daily) / rets.std(ddof=0)) * np.sqrt(252))
    else:
        sharpe = np.nan

    # 赚了多少：最终权益 - 初始权益（避免“没平仓=0”）
    sum_pnl = end_eq - start_eq

    # trades 相关指标
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
        "ts_code": TS_CODE,
        "sum_pnl": float(sum_pnl),
        "total_return": end_eq / start_eq - 1.0,
        "profit_factor": profit_factor,
        "num_trades": num_trades,
        "max_drawdown": max_drawdown,
        "sharpe": sharpe,
        "win_rate": win_rate,
    }


def backtest_one(df: pd.DataFrame):
    df = df.sort_values("date").reset_index(drop=True).copy()

    df["atr"] = atr(df, ATR_N)
    df = compute_signal_vA(df)

    start_dt = pd.to_datetime(DATE_START)
    end_dt   = pd.to_datetime(DATE_END)
    df = df[(df["date"] >= start_dt) & (df["date"] <= end_dt)].reset_index(drop=True)
    if len(df) < 5:
        raise RuntimeError("回测区间数据太少")

    cash = INIT_CASH
    shares = 0.0
    avg_cost = np.nan

    in_pos = False
    tranches = 0
    peak_high = np.nan
    stop = np.nan

    entry_date = None
    entry_avg = np.nan

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

                trades.append({
                    "entry_date": entry_date,
                    "exit_date": next_row["date"],
                    "entry_avg": float(entry_avg),
                    "exit_price": float(fill),
                    "shares": float(shares),
                    "pnl": float((fill - entry_avg) * shares),
                    "ret": float(fill / entry_avg - 1.0),
                    "reason": "chandelier_stop",
                    "tranches": int(tranches),
                })

                # 清仓
                shares = 0.0
                avg_cost = np.nan
                in_pos = False
                tranches = 0
                peak_high = np.nan
                stop = np.nan
                entry_date = None
                entry_avg = np.nan
                continue

        # ===== 持仓：更新止损 + 加仓 =====
        if in_pos:
            peak_high = float(row["high"]) if not pd.notna(peak_high) else max(float(peak_high), float(row["high"]))
            if pd.notna(row["atr"]) and pd.notna(peak_high):
                stop = peak_high - K_ATR * float(row["atr"])

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

            peak_high = float(next_row["high"])
            stop = (peak_high - K_ATR * float(next_row["atr"])) if pd.notna(next_row["atr"]) else np.nan

    # ===== 期末强制平仓：把浮盈变成已实现 + 给 trades 一个退出记录 =====
    if FORCE_EXIT_AT_END and in_pos and shares > 0:
        last_row = df.iloc[-1]
        last_date = last_row["date"]
        last_close = float(last_row["close"])

        fill = apply_cost(last_close, "sell")
        cash += shares * fill

        trades.append({
            "entry_date": entry_date,
            "exit_date": last_date,
            "entry_avg": float(entry_avg),
            "exit_price": float(fill),
            "shares": float(shares),
            "pnl": float((fill - entry_avg) * shares),
            "ret": float(fill / entry_avg - 1.0),
            "reason": "end_of_period",
            "tranches": int(tranches),
        })

        # 再补一行平仓后的权益，确保 metrics 最后一行是“已平仓”
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

    equity_curve = pd.DataFrame(eq_rows)
    trades_df = pd.DataFrame(trades)
    metrics = compute_metrics(equity_curve, trades_df, rf_annual=RF_ANNUAL)
    return equity_curve, trades_df, metrics


def main():
    if not DB_PATH.exists():
        raise FileNotFoundError(f"找不到数据库：{DB_PATH}")

    conn = sqlite3.connect(DB_PATH.as_posix())
    try:
        df = pd.read_sql_query(
            "SELECT date, open, high, low, close, volume FROM daily WHERE ts_code=? ORDER BY date",
            conn, params=(TS_CODE,)
        )
    finally:
        conn.close()

    if df.empty:
        raise RuntimeError(f"daily 表里找不到 ts_code={TS_CODE}")

    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close", "volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna()

    equity, trades, metrics = backtest_one(df)

    print("===== METRICS =====")
    print(f"赚了多少(sum_pnl): {metrics['sum_pnl']:.2f}")
    print(f"回报率(total_return): {metrics['total_return']:.4%}")
    print(f"盈亏比(profit_factor): {metrics['profit_factor']}")
    print(f"交易数(num_trades): {metrics['num_trades']}")
    print(f"最大回撤(max_drawdown): {metrics['max_drawdown']:.4%}")
    print(f"夏普比率(sharpe): {metrics['sharpe']}")
    print(f"胜率(win_rate): {metrics['win_rate']}")

    out_xlsx = Path(f"exec_backtest_{TS_CODE.replace('.','_')}_{DATE_START}_to_{DATE_END}.xlsx")
    with pd.ExcelWriter(out_xlsx, engine="openpyxl") as w:
        equity.to_excel(w, sheet_name="equity_curve", index=False)
        trades.to_excel(w, sheet_name="trades", index=False)
        pd.DataFrame([metrics]).to_excel(w, sheet_name="metrics", index=False)

    print(f"\nSaved: {out_xlsx.resolve()}")


if __name__ == "__main__":
    main()
