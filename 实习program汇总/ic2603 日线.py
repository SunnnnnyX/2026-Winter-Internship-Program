import numpy as np
import pandas as pd
import akshare as ak

# =========================
# 参数区
# =========================
SYMBOL = "IC0"
ATR_N = 14
K_ATR = 2.5

INIT_EQUITY = 1_000_000.0
MULTIPLIER = 200
COMM_PER_SIDE = 0.0
SLIPPAGE_PTS = 0.0

REQUIRE_OPEN_UP = True

# +1R 保本
USE_BREAK_EVEN = True
BE_R = 1.0

# 不用大趋势过滤（按你要求）
USE_TREND_FILTER = False
MA_FAST = 50
MA_SLOW = 200

# 你要的规则：限时达标，否则退出
USE_TIMEOUT_EXIT = True
TIMEOUT_T = 12          # 入场后第 T 天检查（T天内没达标则退出）
TP_XR = 0.8             # 

FORCE_CLOSE_AT_END = True

DO_GRID = False
GRID_K_ATR = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]


# =========================
# 工具函数
# =========================
def wilder_atr(df: pd.DataFrame, n: int) -> pd.Series:
    prev_close = df["close"].shift(1)
    tr = pd.concat([
        (df["high"] - df["low"]).abs(),
        (df["high"] - prev_close).abs(),
        (df["low"] - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def max_drawdown(eq: pd.Series) -> float:
    peak = eq.cummax()
    dd = (eq - peak) / peak
    return float(dd.min())

def sharpe_ratio(daily_ret: pd.Series) -> float:
    x = daily_ret.dropna()
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(252))

def load_daily(symbol: str) -> pd.DataFrame:
    # 方法1：指定IC2606合约（新浪财经合约代码）
    # 新浪财经IC2606的代码是 "IC2606"（而非IC0）
    df = ak.futures_zh_daily_sina(symbol="IC2603").copy()
    
    # 方法2：如果新浪没有IC2606，改用同花顺接口（更全）
    # df = ak.futures_zh_daily_ths(symbol="IC2606").copy()

    # 数据清洗（和原逻辑一致，保证兼容性）
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # 过滤IC2606的有效存续期（可选，避免过期数据）
    df = df[(df["date"] >= "2025-01-01") & (df["date"] <= "2026-06-30")]
    df = df.dropna(subset=["open", "high", "low", "close"]).reset_index(drop=True)
    return df

def build_signal(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    cond = (
        (h.shift(2) < h.shift(1)) & (h.shift(1) < h) &
        (l.shift(2) < l.shift(1)) & (l.shift(1) < l) &
        (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) < o.shift(1)) & (o.shift(1) < o)
    return cond.fillna(False)


# =========================
# 回测
# =========================
def backtest(k_atr: float = 2.5) -> dict:
    df = load_daily(SYMBOL)
    df["atr"] = wilder_atr(df, ATR_N)
    df["signal"] = build_signal(df)

    # 趋势过滤（默认关闭）
    if USE_TREND_FILTER:
        df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
        df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
        df["trend_ok"] = (df["close"] > df["ma_slow"]) & (df["ma_fast"] > df["ma_slow"])
        df["trend_ok"] = df["trend_ok"].fillna(False)
    else:
        df["trend_ok"] = True

    n = len(df)
    cash = INIT_EQUITY
    pos = 0

    entry_px = np.nan
    entry_i = -1
    stop = np.nan
    R = np.nan

    highest_close = -np.inf
    max_high_since_entry = -np.inf

    equity = np.full(n, np.nan, dtype=float)
    pos_flag = np.zeros(n, dtype=int)

    trades = []

    def close_pos(i: int, exit_px: float, reason: str):
        nonlocal cash, pos, entry_px, entry_i, stop, R, highest_close, max_high_since_entry
        pnl = (exit_px - entry_px) * MULTIPLIER - 2 * COMM_PER_SIDE
        cash += pnl
        trades.append({
            "entry_date": df.loc[entry_i, "date"],
            "exit_date": df.loc[i, "date"],
            "entry": float(entry_px),
            "exit": float(exit_px),
            "pnl": float(pnl),
            "R_mult": float((exit_px - entry_px) / R) if R > 0 else np.nan,
            "reason": reason
        })
        pos = 0
        entry_px = np.nan
        entry_i = -1
        stop = np.nan
        R = np.nan
        highest_close = -np.inf
        max_high_since_entry = -np.inf

    for i in range(n):
        # ===== 管理/出场 =====
        if pos == 1:
            pos_flag[i] = 1
            highest_close = max(highest_close, df.loc[i, "close"])
            max_high_since_entry = max(max_high_since_entry, df.loc[i, "high"])

            # ATR追踪止损
            atr_i = df.loc[i, "atr"]
            if np.isfinite(atr_i) and atr_i > 0:
                trail = highest_close - k_atr * atr_i
                stop = max(stop, trail)

            # +1R 保本
            if USE_BREAK_EVEN and np.isfinite(R) and R > 0:
                if df.loc[i, "high"] >= entry_px + BE_R * R:
                    stop = max(stop, entry_px)

            # 1) 先处理“止损触发”（当日 low <= stop）
            if df.loc[i, "low"] <= stop:
                exit_px = stop - SLIPPAGE_PTS
                close_pos(i, exit_px, reason="stop")

            # 2) 再处理“限时达标，否则退出”
            # 解释：如果到第 T 天（含）仍未触达 entry + TP_XR*R，则收盘退出
            if pos == 1 and USE_TIMEOUT_EXIT and R > 0 and (i - entry_i) >= TIMEOUT_T:
                target = entry_px + TP_XR * R
                if max_high_since_entry < target:
                    exit_px = df.loc[i, "close"] - SLIPPAGE_PTS
                    close_pos(i, exit_px, reason="timeout")

        # ===== 入场：信号在 i-1，i 开盘入 =====
        if pos == 0 and i >= 1 and df.loc[i-1, "signal"] and bool(df.loc[i-1, "trend_ok"]):
            first_idx = i - 3
            if first_idx >= 0:
                e = df.loc[i, "open"] + SLIPPAGE_PTS
                s = df.loc[first_idx, "low"]
                r = e - s
                if r > 0 and np.isfinite(df.loc[i, "atr"]):
                    pos = 1
                    entry_px = e
                    entry_i = i
                    stop = s
                    R = r
                    highest_close = df.loc[i, "close"]
                    max_high_since_entry = df.loc[i, "high"]
                    pos_flag[i] = 1

        # ===== 记录权益 =====
        if pos == 1:
            equity[i] = cash + (df.loc[i, "close"] - entry_px) * MULTIPLIER
        else:
            equity[i] = cash

    # ===== 期末强制平仓 =====
    if pos == 1 and FORCE_CLOSE_AT_END:
        exit_px = df.loc[n-1, "close"] - SLIPPAGE_PTS
        close_pos(n-1, exit_px, reason="eod")
        equity[-1] = cash

    eq = pd.Series(equity, index=df["date"])
    ret = eq.pct_change()

    trades_df = pd.DataFrame(trades)
    realized_pnl = float(trades_df["pnl"].sum()) if len(trades_df) else 0.0
    exposure = float(pos_flag.mean())

    # ===== Expectancy =====
    n_trades = len(trades_df)
    if n_trades:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] <= 0, "pnl"]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0  # 负数

        expectancy_trade = win_rate * avg_win + (1 - win_rate) * avg_loss
        expectancy_R = float(trades_df["R_mult"].mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
        payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf
    else:
        win_rate = avg_win = avg_loss = expectancy_trade = expectancy_R = 0.0
        profit_factor = payoff_ratio = 0.0

    expectancy_per_day = realized_pnl / n if n > 0 else 0.0
    hold_days = int(pos_flag.sum())
    expectancy_per_hold_day = realized_pnl / hold_days if hold_days > 0 else 0.0

    out = {
        "symbol": SYMBOL,
        "rows": n,
        "data_range": (df["date"].min(), df["date"].max()),
        "signals": int(df["signal"].sum()),
        "k_atr": float(k_atr),

        "trades": int(n_trades),
        "realized_pnl": realized_pnl,
        "final_equity": float(eq.iloc[-1]),
        "exposure": exposure,

        "win_rate": float(win_rate),
        "avg_R": float(trades_df["R_mult"].mean()) if n_trades else 0.0,
        "max_dd": max_drawdown(eq),

        "sharpe_full": sharpe_ratio(ret),
        "sharpe_active": sharpe_ratio(ret[pos_flag == 1]),

        "expectancy_trade": float(expectancy_trade),
        "expectancy_R": float(expectancy_R),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "payoff_ratio": float(payoff_ratio),
        "expectancy_per_day": float(expectancy_per_day),
        "expectancy_per_hold_day": float(expectancy_per_hold_day),

        "trades_df": trades_df
    }
    return out


def run_once():
    out = backtest(k_atr=K_ATR)
    d0, d1 = out["data_range"]
    print(f"Data range: {d0.date()} -> {d1.date()}, rows={out['rows']}")
    print("signals:", out["signals"])

    print("\n===== Summary =====")
    keys = [
        "k_atr","trades","realized_pnl","final_equity","exposure",
        "win_rate","avg_R","max_dd","sharpe_full","sharpe_active",
        "expectancy_trade","expectancy_R","avg_win","avg_loss",
        "profit_factor","payoff_ratio","expectancy_per_day","expectancy_per_hold_day"
    ]
    for k in keys:
        print(f"{k}: {out[k]}")

    if out["trades"] > 0:
        print("\nLast trades:")
        print(out["trades_df"].tail(12).to_string(index=False))

def grid_search():
    rows = []
    best = None
    for k in GRID_K_ATR:
        out = backtest(k_atr=k)
        rows.append({
            "k_atr": k,
            "trades": out["trades"],
            "expectancy_trade": out["expectancy_trade"],
            "expectancy_R": out["expectancy_R"],
            "profit_factor": out["profit_factor"],
            "max_dd": out["max_dd"],
            "sharpe_active": out["sharpe_active"],
            "exposure": out["exposure"],
            "pnl": out["realized_pnl"],
        })
        if best is None or out["expectancy_trade"] > best["expectancy_trade"]:
            best = out

    res = pd.DataFrame(rows).sort_values(["expectancy_trade", "sharpe_active"], ascending=False)
    print("\n===== Grid Search (sorted by expectancy_trade) =====")
    print(res.to_string(index=False))

    print("\nBEST by expectancy_trade:")
    show = ["k_atr","trades","realized_pnl","max_dd","sharpe_active","expectancy_trade","expectancy_R","profit_factor","exposure"]
    for k in show:
        print(f"{k}: {best[k]}")

if __name__ == "__main__":
    run_once()
    if DO_GRID:
        grid_search()
