import numpy as np
import pandas as pd
from pathlib import Path

# ==========================================================
# 0) 两份Excel数据路径（只改这里）
# ==========================================================
DATA_DIR = Path(r"C:\Users\24106\Desktop")  # 你的Excel放哪就改哪

DAILY_FILE = "中证500.xlsx"
M30_FILE   = "中证500 30分钟线.xlsx"

DAILY_SHEET = 0
M30_SHEET   = 0

DATE_START = None   # 例如 "2023-01-01"
DATE_END   = None   # 例如 "2026-01-21"

# ==========================================================
# 1) 策略参数（保持你的原日线辅助线逻辑不变）
# ==========================================================
ATR_N = 14
K_ATR = 3.0

INIT_EQUITY = 1_000_000.0
MULTIPLIER = 200
COMM_PER_SIDE = 0.0
SLIPPAGE_PTS = 0.0

REQUIRE_OPEN_UP = True

USE_BREAK_EVEN = True
BE_R = 1.0

USE_TREND_FILTER = False
MA_FAST = 50
MA_SLOW = 200

USE_TIMEOUT_EXIT = True
TIMEOUT_T = 12
TP_XR = 0.8

FORCE_CLOSE_AT_END = True

# ==========================================================
# 2) 30min 执行层（过滤假突破）
# ==========================================================
USE_M30_EXECUTION = True

# 注意：你的30分钟数据从 2023-01-30 才开始
# 如果日线更早，早期信号会因为缺分钟数据被 skip 掉
M30_MISSING_POLICY = "skip"   # "skip" 或 "fallback"

# (A) 入场触发：连续3根30min满足“走高规则”
M30_NEED_BARS = 3
M30_MAX_WAIT_BARS = 0          # 0=全天都可触发；建议 10~12 防尾盘才进
ENTRY_AT_NEXT_BAR_OPEN = True  # True=确认后下一根30m开盘进；False=确认bar收盘进

# (B) 接受度概率（Acceptance）
USE_ACCEPTANCE = True
ACCEPT_W = 6
ACCEPT_P = 0.67
ACCEPT_LEVEL = "prev_high"     # 关键位：昨日高点（日线）
ACCEPT_EXIT_AT = "close"

# (C) “明显要止损就止损”：分钟 low 触发止损
CHECK_INTRADAY_STOP_ON_ENTRY_DAY = True
CHECK_M30_STOP_EACH_DAY = True

# (D) 30min ATR 紧急止损：entry - k * ATR30
USE_M30_ATR_STOP = True
M30_ATR_N = 14
M30_STOP_ATR_MULT = 2.5

# 接受度容忍回踩：close >= L - tol
ACCEPT_TOL_ATR = 0.10


# ==========================================================
# 工具函数
# ==========================================================
def wilder_atr_core(high, low, close, n: int):
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low).abs(),
        (high - prev_close).abs(),
        (low - prev_close).abs()
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def wilder_atr_df(df: pd.DataFrame, n: int) -> pd.Series:
    return wilder_atr_core(df["high"], df["low"], df["close"], n)

def max_drawdown(eq: pd.Series) -> float:
    if len(eq) == 0:
        return 0.0
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

def _auto_pick_excel(dir_: Path, hint: str) -> Path:
    candidates = list(dir_.glob("*.xlsx")) + list(dir_.glob("*.xls"))
    msg = (
        f"找不到指定文件：{hint}\n"
        f"目录：{dir_}\n"
        f"当前Excel候选：\n" + "\n".join([f"  - {c.name}" for c in candidates])
    )
    raise FileNotFoundError(msg)

def _pick_col(df: pd.DataFrame, candidates):
    cols = [str(c).strip() for c in df.columns]
    lower_map = {c: c.lower() for c in cols}

    # 1) 精确命中
    for c in candidates:
        if c in cols:
            return c

    # 2) 模糊命中（包含关系）
    cands_lower = [str(x).lower() for x in candidates]
    for col in cols:
        scol = lower_map[col]
        for c in cands_lower:
            if c in scol:
                return col
    return None


# ==========================================================
# 1) 读日线 Excel
# ==========================================================
def load_daily_from_excel() -> pd.DataFrame:
    path = DATA_DIR / DAILY_FILE
    if not path.exists():
        path = _auto_pick_excel(DATA_DIR, str(path))

    print("Using DAILY file:", str(path))

    df = pd.read_excel(path, sheet_name=DAILY_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    date_col  = _pick_col(df, ["日期", "交易日期", "date", "datetime", "时间"])
    open_col  = _pick_col(df, ["开盘价", "open"])
    high_col  = _pick_col(df, ["最高价", "high"])
    low_col   = _pick_col(df, ["最低价", "low"])
    close_col = _pick_col(df, ["收盘价", "close"])

    missing = [k for k, v in [
        ("date", date_col), ("open", open_col), ("high", high_col),
        ("low", low_col), ("close", close_col)
    ] if v is None]
    if missing:
        raise ValueError(f"日线Excel缺列：{missing}\n实际列：{list(df.columns)}")

    df = df.rename(columns={
        date_col: "date",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close"
    })

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["date", "open", "high", "low", "close"]).copy()
    df = df.sort_values("date").drop_duplicates("date", keep="last").reset_index(drop=True)

    if DATE_START is not None:
        df = df[df["date"] >= pd.to_datetime(DATE_START)]
    if DATE_END is not None:
        df = df[df["date"] <= pd.to_datetime(DATE_END)]

    return df.reset_index(drop=True)


# ==========================================================
# 2) 读 30min Excel
# ==========================================================
def load_m30_from_excel() -> pd.DataFrame:
    path = DATA_DIR / M30_FILE
    if not path.exists():
        path = _auto_pick_excel(DATA_DIR, str(path))

    print("Using M30 file:", str(path))

    df = pd.read_excel(path, sheet_name=M30_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    dt_col    = _pick_col(df, ["时间", "日期时间", "datetime", "date", "day", "dt"])
    open_col  = _pick_col(df, ["开盘价", "open"])
    high_col  = _pick_col(df, ["最高价", "high"])
    low_col   = _pick_col(df, ["最低价", "low"])
    close_col = _pick_col(df, ["收盘价", "close"])

    missing = [k for k, v in [
        ("dt", dt_col), ("open", open_col), ("high", high_col),
        ("low", low_col), ("close", close_col)
    ] if v is None]
    if missing:
        raise ValueError(f"30min Excel缺列：{missing}\n实际列：{list(df.columns)}")

    df = df.rename(columns={
        dt_col: "dt",
        open_col: "open",
        high_col: "high",
        low_col: "low",
        close_col: "close"
    })

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["dt", "open", "high", "low", "close"]).copy()
    df = df.sort_values("dt").reset_index(drop=True)

    # 交易日键
    df["d"] = df["dt"].dt.normalize()

    # ATR30（用于接受度容忍 + ATR止损）
    df["atr30"] = wilder_atr_core(df["high"], df["low"], df["close"], M30_ATR_N)

    if DATE_START is not None:
        df = df[df["dt"] >= pd.to_datetime(DATE_START)]
    if DATE_END is not None:
        df = df[df["dt"] <= pd.to_datetime(DATE_END)]

    return df.reset_index(drop=True)

def build_m30_map(m30: pd.DataFrame) -> dict:
    return {d: g.sort_values("dt").reset_index(drop=True) for d, g in m30.groupby("d")}


# ==========================================================
# 3) 日线信号（不改）
# ==========================================================
def build_signal_daily(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    cond = (
        (h.shift(2) < h.shift(1)) & (h.shift(1) < h) &
        (l.shift(2) < l.shift(1)) & (l.shift(1) < l) &
        (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) < o.shift(1)) & (o.shift(1) < o)
    return cond.fillna(False)


# ==========================================================
# 4) 30min 入场确认（三连走高 + 接受度）
# ==========================================================
def find_threebar_rise_idx(day_bars: pd.DataFrame) -> int | None:
    if day_bars is None or len(day_bars) < (M30_NEED_BARS + 1):
        return None

    g = day_bars
    if M30_MAX_WAIT_BARS and M30_MAX_WAIT_BARS > 0:
        g = g.iloc[:M30_MAX_WAIT_BARS].copy()
        if len(g) < (M30_NEED_BARS + 1):
            return None

    h, l, o, c = g["high"], g["low"], g["open"], g["close"]
    cond = (
        (h.shift(2) < h.shift(1)) & (h.shift(1) < h) &
        (l.shift(2) < l.shift(1)) & (l.shift(1) < l) &
        (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) < o.shift(1)) & (o.shift(1) < o)
    cond = cond.fillna(False)

    hit = cond[cond].index
    return None if len(hit) == 0 else int(hit[0])

def pick_entry_price_time(day_bars: pd.DataFrame, third_idx: int):
    if ENTRY_AT_NEXT_BAR_OPEN:
        j = third_idx + 1
        if j >= len(day_bars):
            return None
        return float(day_bars.loc[j, "open"]), pd.Timestamp(day_bars.loc[j, "dt"]), j
    else:
        return float(day_bars.loc[third_idx, "close"]), pd.Timestamp(day_bars.loc[third_idx, "dt"]), third_idx

def acceptance_prob(day_bars: pd.DataFrame, start_bar_index: int, level: float, W: int, tol: float) -> float | None:
    end = start_bar_index + W - 1
    if end >= len(day_bars):
        return None
    w = day_bars.iloc[start_bar_index:end+1]
    return float((w["close"] >= level - tol).mean())

def acceptance_exit_price(day_bars: pd.DataFrame, start_bar_index: int, W: int) -> float | None:
    end = start_bar_index + W - 1
    if end >= len(day_bars):
        return None
    return float(day_bars.iloc[end]["close"])


# ==========================================================
# 5) 回测：日线信号不变；30min负责验证真假突破（不改思路）
# ==========================================================
def backtest(df_daily: pd.DataFrame, m30_map: dict | None) -> dict:
    df = df_daily.copy()
    df["atr"] = wilder_atr_df(df, ATR_N)
    df["signal"] = build_signal_daily(df)

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
    entry_ts = None

    stop = np.nan
    R = np.nan

    highest_close = -np.inf
    max_high_since_entry = -np.inf

    equity = np.full(n, np.nan, dtype=float)
    pos_flag = np.zeros(n, dtype=int)

    trades = []
    skipped_no_m30 = 0
    used_m30_entries = 0
    used_fallback_entries = 0

    def close_pos(i: int, exit_px: float, reason: str):
        nonlocal cash, pos, entry_px, entry_i, entry_ts, stop, R, highest_close, max_high_since_entry
        pnl = (exit_px - entry_px) * MULTIPLIER - 2 * COMM_PER_SIDE
        cash += pnl
        trades.append({
            "entry_date": df.loc[entry_i, "date"],
            "entry_ts": entry_ts,
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
        entry_ts = None
        stop = np.nan
        R = np.nan
        highest_close = -np.inf
        max_high_since_entry = -np.inf

    for i in range(n):
        today = pd.Timestamp(df.loc[i, "date"]).normalize()

        # 1) 持仓管理（日线辅助线逻辑不变）
        if pos == 1:
            pos_flag[i] = 1

            # 分钟止损优先（你说的“分钟掉下去就止损”）
            if CHECK_M30_STOP_EACH_DAY and (m30_map is not None):
                day_bars = m30_map.get(today, None)
                if day_bars is not None and len(day_bars):
                    if float(day_bars["low"].min()) <= stop:
                        exit_px = stop - SLIPPAGE_PTS
                        close_pos(i, exit_px, reason="m30_stop")

            # 还持仓 -> 走日线止损/保本/超时
            if pos == 1:
                highest_close = max(highest_close, float(df.loc[i, "close"]))
                max_high_since_entry = max(max_high_since_entry, float(df.loc[i, "high"]))

                atr_i = float(df.loc[i, "atr"])
                if np.isfinite(atr_i) and atr_i > 0:
                    trail = highest_close - K_ATR * atr_i
                    stop = max(stop, trail)

                if USE_BREAK_EVEN and np.isfinite(R) and R > 0:
                    if float(df.loc[i, "high"]) >= entry_px + BE_R * R:
                        stop = max(stop, entry_px)

                if float(df.loc[i, "low"]) <= stop:
                    exit_px = stop - SLIPPAGE_PTS
                    close_pos(i, exit_px, reason="daily_stop")

                if pos == 1 and USE_TIMEOUT_EXIT and R > 0 and (i - entry_i) >= TIMEOUT_T:
                    target = entry_px + TP_XR * R
                    if max_high_since_entry < target:
                        exit_px = float(df.loc[i, "close"]) - SLIPPAGE_PTS
                        close_pos(i, exit_px, reason="timeout")

        # 2) 入场：日线信号在 i-1，准备在 i 这天开
        if pos == 0 and i >= 1 and bool(df.loc[i-1, "signal"]) and bool(df.loc[i-1, "trend_ok"]):
            first_idx = i - 3
            if first_idx >= 0 and np.isfinite(df.loc[i, "atr"]):
                daily_stop = float(df.loc[first_idx, "low"])
                did_enter = False

                # 30min 执行层：必须三连走高才进
                if USE_M30_EXECUTION and (m30_map is not None):
                    day_bars = m30_map.get(today, None)

                    if day_bars is None or len(day_bars) < (M30_NEED_BARS + 1):
                        if M30_MISSING_POLICY == "skip":
                            skipped_no_m30 += 1
                    else:
                        third_idx = find_threebar_rise_idx(day_bars)
                        if third_idx is not None:
                            ep = pick_entry_price_time(day_bars, third_idx)
                            if ep is not None:
                                entry0, ts, entry_bar_index = ep
                                entry_price = float(entry0) + SLIPPAGE_PTS
                                R0 = entry_price - daily_stop

                                if R0 > 0:
                                    pos = 1
                                    entry_px = entry_price
                                    entry_i = i
                                    entry_ts = ts
                                    stop = daily_stop
                                    R = R0
                                    highest_close = float(df.loc[i, "close"])
                                    max_high_since_entry = float(df.loc[i, "high"])
                                    pos_flag[i] = 1
                                    did_enter = True
                                    used_m30_entries += 1

                                    # 30m ATR 紧急止损：entry - k*ATR30
                                    if USE_M30_ATR_STOP:
                                        atr30_here = float(day_bars.loc[entry_bar_index, "atr30"]) if "atr30" in day_bars.columns else np.nan
                                        if np.isfinite(atr30_here) and atr30_here > 0:
                                            m30_atr_stop = entry_px - M30_STOP_ATR_MULT * atr30_here
                                            stop = max(stop, m30_atr_stop)

                                    # 入场当天：分钟 low 触发止损
                                    if did_enter and CHECK_INTRADAY_STOP_ON_ENTRY_DAY:
                                        after = day_bars[day_bars["dt"] >= ts]
                                        if len(after) and float(after["low"].min()) <= stop:
                                            exit_px = stop - SLIPPAGE_PTS
                                            close_pos(i, exit_px, reason="m30_stop_entryday")
                                            did_enter = False

                                    # 接受度复核：关键位=昨日高点（日线）
                                    if did_enter and USE_ACCEPTANCE:
                                        level = float(df.loc[i-1, "high"])  # prev_high
                                        atr30_here = float(day_bars.loc[entry_bar_index, "atr30"]) if "atr30" in day_bars.columns else np.nan
                                        tol = (ACCEPT_TOL_ATR * atr30_here) if (np.isfinite(atr30_here) and atr30_here > 0) else 0.0

                                        acc = acceptance_prob(day_bars, entry_bar_index, level, ACCEPT_W, tol)
                                        if (acc is not None) and (acc < ACCEPT_P):
                                            px = acceptance_exit_price(day_bars, entry_bar_index, ACCEPT_W)
                                            if px is not None:
                                                exit_px = float(px) - SLIPPAGE_PTS
                                                close_pos(i, exit_px, reason=f"accept_fail({acc:.2f})")
                                                did_enter = False

                # fallback：没分钟数据就按日线开盘进（可选）
                if (not did_enter) and (not USE_M30_EXECUTION or M30_MISSING_POLICY == "fallback"):
                    entry_price = float(df.loc[i, "open"]) + SLIPPAGE_PTS
                    R0 = entry_price - daily_stop
                    if R0 > 0:
                        pos = 1
                        entry_px = entry_price
                        entry_i = i
                        entry_ts = None
                        stop = daily_stop
                        R = R0
                        highest_close = float(df.loc[i, "close"])
                        max_high_since_entry = float(df.loc[i, "high"])
                        pos_flag[i] = 1
                        used_fallback_entries += 1

        # 3) 权益（日线收盘）
        if pos == 1:
            equity[i] = cash + (float(df.loc[i, "close"]) - entry_px) * MULTIPLIER
        else:
            equity[i] = cash

    # 期末强制平仓
    if pos == 1 and FORCE_CLOSE_AT_END:
        exit_px = float(df.loc[n-1, "close"]) - SLIPPAGE_PTS
        close_pos(n-1, exit_px, reason="eod")
        equity[-1] = cash

    eq = pd.Series(equity, index=df["date"])
    ret = eq.pct_change(fill_method=None)

    trades_df = pd.DataFrame(trades)
    realized_pnl = float(trades_df["pnl"].sum()) if len(trades_df) else 0.0
    exposure = float(pos_flag.mean()) if n > 0 else 0.0

    n_trades = len(trades_df)
    if n_trades:
        wins = trades_df.loc[trades_df["pnl"] > 0, "pnl"]
        losses = trades_df.loc[trades_df["pnl"] <= 0, "pnl"]
        win_rate = len(wins) / n_trades
        avg_win = float(wins.mean()) if len(wins) else 0.0
        avg_loss = float(losses.mean()) if len(losses) else 0.0
        expectancy_trade = win_rate * avg_win + (1 - win_rate) * avg_loss
        expectancy_R = float(trades_df["R_mult"].mean())
        profit_factor = float(wins.sum() / abs(losses.sum())) if len(losses) and losses.sum() != 0 else np.inf
        payoff_ratio = float(avg_win / abs(avg_loss)) if avg_loss != 0 else np.inf
    else:
        win_rate = avg_win = avg_loss = expectancy_trade = expectancy_R = 0.0
        profit_factor = payoff_ratio = 0.0

    return {
        "rows": n,
        "data_range": (df["date"].min(), df["date"].max()),
        "signals": int(df["signal"].sum()),
        "trades": int(n_trades),
        "realized_pnl": realized_pnl,
        "final_equity": float(eq.iloc[-1]) if len(eq) else INIT_EQUITY,
        "exposure": exposure,
        "win_rate": float(win_rate),
        "avg_R": float(trades_df["R_mult"].mean()) if n_trades else 0.0,
        "max_dd": max_drawdown(eq),
        "sharpe_full": sharpe_ratio(ret),
        "sharpe_active": sharpe_ratio(ret[pos_flag == 1]) if n > 0 else 0.0,
        "expectancy_trade": float(expectancy_trade),
        "expectancy_R": float(expectancy_R),
        "avg_win": float(avg_win),
        "avg_loss": float(avg_loss),
        "profit_factor": float(profit_factor),
        "payoff_ratio": float(payoff_ratio),
        "skipped_no_m30": int(skipped_no_m30),
        "used_m30_entries": int(used_m30_entries),
        "used_fallback_entries": int(used_fallback_entries),
        "trades_df": trades_df
    }


# ==========================================================
# main
# ==========================================================
if __name__ == "__main__":
    df_daily = load_daily_from_excel()
    print(f"[D] rows={len(df_daily)}, range={df_daily['date'].min().date()} -> {df_daily['date'].max().date()}")

    m30_map = None
    if USE_M30_EXECUTION:
        df_m30 = load_m30_from_excel()
        print(f"[M30] rows={len(df_m30)}, range={df_m30['dt'].min()} -> {df_m30['dt'].max()}, days={df_m30['d'].nunique()}")
        m30_map = build_m30_map(df_m30)

    out = backtest(df_daily, m30_map=m30_map)

    d0, d1 = out["data_range"]
    print(f"\nData range: {d0.date()} -> {d1.date()}, rows={out['rows']}")
    print("signals:", out["signals"])
    print(f"used_m30_entries={out['used_m30_entries']}, used_fallback_entries={out['used_fallback_entries']}, skipped_no_m30={out['skipped_no_m30']}")

    print("\n===== Summary =====")
    keys = [
        "trades","realized_pnl","final_equity","exposure",
        "win_rate","avg_R","max_dd","sharpe_full","sharpe_active",
        "expectancy_trade","expectancy_R","avg_win","avg_loss",
        "profit_factor","payoff_ratio"
    ]
    for k in keys:
        print(f"{k}: {out[k]}")

    if out["trades"] > 0:
        print("\nLast trades:")
        print(out["trades_df"].tail(50).to_string(index=False))

