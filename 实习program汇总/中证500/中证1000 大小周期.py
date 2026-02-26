import numpy as np
import pandas as pd
from pathlib import Path


# ==========================================================
# 0) 数据源设置（只改这里）
# ==========================================================
# 30min：你给的中证1000分钟线 Excel（本机路径）
M30_FILE = Path(r"C:\Users\24106\Desktop\中证1000.xlsx")
M30_SHEET = 0

# 日线：AkShare 拉取中证1000
# 常见：sh000852（中证1000指数）
DAILY_SYMBOL_CANDIDATES = [
    "sh000852",  # 中证1000 常用
    "sz399852",  # 有的系统会用这个写法（不一定有）
    "000852",    # 少数接口可能接受
]

# 对齐区间（建议至少让日线起点 >= 30min 覆盖起点）
DATE_START = None   # 例如 "2025-01-01"
DATE_END   = None   # 例如 "2026-01-21"


# ==========================================================
# 1) 策略参数（逻辑不变）
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
# 2) 30min 执行层（过滤假突破，逻辑不变）
# ==========================================================
USE_M30_EXECUTION = True
M30_MISSING_POLICY = "skip"   # "skip" 或 "fallback"

# (A) 30min 三连走高确认
M30_NEED_BARS = 3
M30_MAX_WAIT_BARS = 0
ENTRY_AT_NEXT_BAR_OPEN = True

# (B) 接受度概率（6根窗口）
USE_ACCEPTANCE = True
ACCEPT_W = 6
ACCEPT_P = 0.67
ACCEPT_LEVEL = "prev_high"     # 关键位：昨日高点（日线）
ACCEPT_EXIT_AT = "close"       # 概率失败后按窗口最后一根 close 出

# (C) 分钟触发止损（low 打到 stop 就出）
CHECK_INTRADAY_STOP_ON_ENTRY_DAY = True
CHECK_M30_STOP_EACH_DAY = True

# (D) 入场瞬间 30min ATR 紧急止损：entry - k * ATR30
USE_M30_ATR_STOP = True
M30_ATR_N = 14
M30_STOP_ATR_MULT = 2.5

# 接受度容忍回踩（close >= level - tol 也算“站得住”）
ACCEPT_TOL_ATR = 0.10


# ==========================================================
# 工具函数
# ==========================================================
def _try_import_akshare():
    try:
        import akshare as ak
        return ak, None
    except Exception as e:
        return None, e

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

def _pick_col(df: pd.DataFrame, candidates):
    cols = [str(c).strip() for c in df.columns]
    lower = {c: c.lower() for c in cols}

    for c in candidates:
        if c in cols:
            return c

    cands_lower = [str(x).lower() for x in candidates]
    for col in cols:
        s = lower[col]
        for cand in cands_lower:
            if cand in s:
                return col
    return None


# ==========================================================
# 1) 日线：AkShare 拉取指数日线（自动尝试多个 symbol）
# ==========================================================
def fetch_daily_index_any(symbol_candidates: list[str]) -> tuple[str, pd.DataFrame]:
    ak, err = _try_import_akshare()
    if ak is None:
        raise RuntimeError(f"无法导入 akshare：{err}（先 pip install akshare）")

    last_err = None
    for sym in symbol_candidates:
        try:
            if DATE_START is None and DATE_END is None:
                df = ak.stock_zh_index_daily_em(symbol=sym).copy()
            else:
                sd = "19900101" if DATE_START is None else pd.to_datetime(DATE_START).strftime("%Y%m%d")
                ed = "20500101" if DATE_END is None else pd.to_datetime(DATE_END).strftime("%Y%m%d")
                df = ak.stock_zh_index_daily_em(symbol=sym, start_date=sd, end_date=ed).copy()

            need = ["date", "open", "high", "low", "close"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                raise ValueError(f"日线数据缺列 {miss}，实际列={list(df.columns)}")

            df["date"] = pd.to_datetime(df["date"], errors="coerce")
            for c in ["open", "high", "low", "close"]:
                df[c] = pd.to_numeric(df[c], errors="coerce")

            df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)

            if DATE_START is not None:
                df = df[df["date"] >= pd.to_datetime(DATE_START)]
            if DATE_END is not None:
                df = df[df["date"] <= pd.to_datetime(DATE_END)]

            if len(df) == 0:
                raise ValueError("返回为空DataFrame")

            return sym, df.reset_index(drop=True)

        except Exception as e:
            last_err = e

    raise RuntimeError(f"所有候选 symbol 都拉不到日线：{symbol_candidates}；最后错误：{repr(last_err)}")


# ==========================================================
# 2) 30min：从 Excel 导入
# ==========================================================
def load_m30_from_excel() -> pd.DataFrame:
    if not M30_FILE.exists():
        raise FileNotFoundError(f"找不到 30min 文件：{M30_FILE}")

    df = pd.read_excel(M30_FILE, sheet_name=M30_SHEET)
    df.columns = [str(c).strip() for c in df.columns]

    dt_col    = _pick_col(df, ["时间", "日期时间", "datetime", "dt", "day", "date"])
    open_col  = _pick_col(df, ["开盘价", "开盘", "open"])
    high_col  = _pick_col(df, ["最高价", "最高", "high"])
    low_col   = _pick_col(df, ["最低价", "最低", "low"])
    close_col = _pick_col(df, ["收盘价", "收盘", "close"])

    missing = [k for k, v in [("dt", dt_col), ("open", open_col), ("high", high_col), ("low", low_col), ("close", close_col)] if v is None]
    if missing:
        raise ValueError(f"30min Excel缺列：{missing}\n实际列：{list(df.columns)}")

    df = df.rename(columns={dt_col: "dt", open_col: "open", high_col: "high", low_col: "low", close_col: "close"})

    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["dt", "open", "high", "low", "close"]).sort_values("dt").reset_index(drop=True)
    df["d"] = df["dt"].dt.normalize()
    df["atr30"] = wilder_atr_core(df["high"], df["low"], df["close"], M30_ATR_N)

    if DATE_START is not None:
        df = df[df["dt"] >= pd.to_datetime(DATE_START)]
    if DATE_END is not None:
        df = df[df["dt"] <= pd.to_datetime(DATE_END)]

    return df.reset_index(drop=True)

def build_m30_map(m30: pd.DataFrame) -> dict:
    return {d: g.sort_values("dt").reset_index(drop=True) for d, g in m30.groupby("d")}


# ==========================================================
# 3) 日线信号（不变）
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
# 4) 30min 规则（不变）
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
    return float(day_bars.iloc[end]["close"])  # 概率失败：窗口最后一根 close


# ==========================================================
# 5) 回测（逻辑不变）
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

            # 分钟 low 触发止损
            if CHECK_M30_STOP_EACH_DAY and (m30_map is not None):
                day_bars = m30_map.get(today, None)
                if day_bars is not None and len(day_bars):
                    if float(day_bars["low"].min()) <= stop:
                        close_pos(i, stop - SLIPPAGE_PTS, reason="m30_stop")

            # 还持仓 -> 日线 ATR 吊灯 / 保本 / 超时
            if pos == 1:
                highest_close = max(highest_close, float(df.loc[i, "close"]))
                max_high_since_entry = max(max_high_since_entry, float(df.loc[i, "high"]))

                atr_i = float(df.loc[i, "atr"])
                if np.isfinite(atr_i) and atr_i > 0:
                    stop = max(stop, highest_close - K_ATR * atr_i)

                if USE_BREAK_EVEN and np.isfinite(R) and R > 0:
                    if float(df.loc[i, "high"]) >= entry_px + BE_R * R:
                        stop = max(stop, entry_px)

                if float(df.loc[i, "low"]) <= stop:
                    close_pos(i, stop - SLIPPAGE_PTS, reason="daily_stop")

                if pos == 1 and USE_TIMEOUT_EXIT and R > 0 and (i - entry_i) >= TIMEOUT_T:
                    target = entry_px + TP_XR * R
                    if max_high_since_entry < target:
                        close_pos(i, float(df.loc[i, "close"]) - SLIPPAGE_PTS, reason="timeout")

        # 2) 入场：日线信号在 i-1，准备在 i 这天开
        if pos == 0 and i >= 1 and bool(df.loc[i-1, "signal"]) and bool(df.loc[i-1, "trend_ok"]):
            first_idx = i - 3
            if first_idx >= 0 and np.isfinite(df.loc[i, "atr"]):
                daily_stop = float(df.loc[first_idx, "low"])
                did_enter = False

                # 30min 过滤：当天必须出现三连走高才进
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

                                    # 入场瞬间：m30_stop = entry - k*ATR30；stop = max(日线止损, m30_stop)
                                    if USE_M30_ATR_STOP:
                                        atr30_here = float(day_bars.loc[entry_bar_index, "atr30"]) if "atr30" in day_bars.columns else np.nan
                                        if np.isfinite(atr30_here) and atr30_here > 0:
                                            m30_stop = entry_px - M30_STOP_ATR_MULT * atr30_here
                                            stop = max(stop, m30_stop)

                                    # 入场当天：分钟 low 触发止损
                                    if did_enter and CHECK_INTRADAY_STOP_ON_ENTRY_DAY:
                                        after = day_bars[day_bars["dt"] >= ts]
                                        if len(after) and float(after["low"].min()) <= stop:
                                            close_pos(i, stop - SLIPPAGE_PTS, reason="m30_stop_entryday")
                                            did_enter = False

                                    # 接受度概率：失败则按窗口最后一根 close 退出
                                    if did_enter and USE_ACCEPTANCE:
                                        level = float(df.loc[i-1, "high"])  # prev_high
                                        atr30_here = float(day_bars.loc[entry_bar_index, "atr30"]) if "atr30" in day_bars.columns else np.nan
                                        tol = (ACCEPT_TOL_ATR * atr30_here) if (np.isfinite(atr30_here) and atr30_here > 0) else 0.0

                                        acc = acceptance_prob(day_bars, entry_bar_index, level, ACCEPT_W, tol)
                                        if (acc is not None) and (acc < ACCEPT_P):
                                            px = acceptance_exit_price(day_bars, entry_bar_index, ACCEPT_W)
                                            if px is not None:
                                                close_pos(i, float(px) - SLIPPAGE_PTS, reason=f"accept_fail({acc:.2f})")
                                                did_enter = False

                # 分钟缺失的 fallback（如果你想保留）
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

        equity[i] = cash + (float(df.loc[i, "close"]) - entry_px) * MULTIPLIER if pos == 1 else cash

    # 期末强制平仓
    if pos == 1 and FORCE_CLOSE_AT_END:
        close_pos(n - 1, float(df.loc[n - 1, "close"]) - SLIPPAGE_PTS, reason="eod")
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
    # 1) 30min：Excel
    m30_map = None
    if USE_M30_EXECUTION:
        m30 = load_m30_from_excel()
        print(f"[M30] file={M30_FILE.name}, rows={len(m30)}, range={m30['dt'].min()} -> {m30['dt'].max()}, days={m30['d'].nunique()}")
        m30_map = build_m30_map(m30)

    # 2) 日线：AkShare（自动尝试候选 symbol）
    sym, df_daily = fetch_daily_index_any(DAILY_SYMBOL_CANDIDATES)
    print(f"[D] symbol={sym}, rows={len(df_daily)}, range={df_daily['date'].min().date()} -> {df_daily['date'].max().date()}")

    # 3) 回测
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
