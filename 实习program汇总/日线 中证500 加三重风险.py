import numpy as np
import pandas as pd
from pathlib import Path

# =========================
# 数据路径（只改这里就行）
# =========================
DESKTOP_DIR = Path(r"C:\Users\24106\Desktop")
DATA_FILE = "中证500.xlsx"   # <- 你的文件名如果不同，就改这里
SHEET_NAME = 0

# 可选：截取日期区间（不想截就设为 None）
DATE_START = None  # 例如 "2017-01-01"
DATE_END   = None  # 例如 "2026-06-30"



SYMBOL = "IC0"      # 占位用（Excel数据不按symbol过滤）
ATR_N = 14
K_ATR = 3

INIT_EQUITY = 1_000_000.0
MULTIPLIER = 200
COMM_PER_SIDE = 0.0
SLIPPAGE_PTS = 0.0

REQUIRE_OPEN_UP = True

# +1R 保本
USE_BREAK_EVEN = True
BE_R = 1.0

# 趋势过滤
USE_TREND_FILTER = False
MA_FAST = 50
MA_SLOW = 200

# 你要的：限时达标，否则退出
# 入场后 T 天内若最高价没到 entry + TP_XR*R，则第T天收盘退出；
# 若到过阈值，则继续看移动止损吃右尾。
USE_TIMEOUT_EXIT = True
TIMEOUT_T = 12
TP_XR = 0.8

FORCE_CLOSE_AT_END = True

# 网格搜索（可选）
DO_GRID = False
GRID_K_ATR = [2.0, 2.25, 2.5, 2.75, 3.0, 3.25, 3.5]


# 杠杆/仓位
BASE_LOTS = 1          # 基础手数
RISK_PCT = 0.03    # 每笔最多亏账户的比例
MAX_LEVERAGE = 3   # 最大名义杠杆：名义头寸/权益
MARGIN_RATE = 0.1 # 初始保证金率
MAX_LOTS_CAP = 20#手数上限


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
    return float(dd.min()) if len(dd) else 0.0

def sharpe_ratio(daily_ret: pd.Series) -> float:
    x = daily_ret.dropna()
    if len(x) < 2:
        return 0.0
    sd = x.std(ddof=0)
    if sd == 0:
        return 0.0
    return float(x.mean() / sd * np.sqrt(252))

def _auto_pick_excel(desktop: Path) -> Path:
    """如果指定文件不存在：尝试桌面仅有一个xlsx/xls就自动选它，否则报错列出候选。"""
    candidates = list(desktop.glob("*.xlsx")) + list(desktop.glob("*.xls"))
    if len(candidates) == 1:
        return candidates[0]
    msg = (
        f"找不到指定文件：{desktop}\n"
        f"你设置的是：{desktop / DATA_FILE}\n"
        f"桌面Excel候选有：\n" + "\n".join([f"  - {c.name}" for c in candidates]) +
        "\n\n请把 DATA_FILE 改成上面其中一个文件名。"
    )
    raise FileNotFoundError(msg)

def load_daily_from_excel() -> pd.DataFrame:
    path = DESKTOP_DIR / DATA_FILE
    if not path.exists():
        path = _auto_pick_excel(DESKTOP_DIR)

    print("Using data file:", str(path))

    df = pd.read_excel(path, sheet_name=SHEET_NAME)
    df.columns = [str(c).strip() for c in df.columns]

    # 列名候选（按你xlsx实际列名可以继续补）
    date_candidates  = ["日期", "交易日期", "datetime", "时间"]
    open_candidates  = ["开盘价"]
    high_candidates  = ["最高价"]
    low_candidates   = ["最低价"]
    close_candidates = ["收盘价"]

    def pick_col(cands):
        for c in cands:
            if c in df.columns:
                return c
        for col in df.columns:
            scol = str(col).lower()
            for c in cands:
                if str(c).lower() in scol:
                    return col
        return None

    date_col  = pick_col(date_candidates)
    open_col  = pick_col(open_candidates)
    high_col  = pick_col(high_candidates)
    low_col   = pick_col(low_candidates)
    close_col = pick_col(close_candidates)

    missing = [k for k, v in [("date",date_col),("open",open_col),("high",high_col),("low",low_col),("close",close_col)] if v is None]
    if missing:
        raise ValueError(f"Excel缺少必要列：{missing}\n当前列：{list(df.columns)}")

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
    df = df.sort_values("date").reset_index(drop=True)
    df = df.drop_duplicates(subset=["date"], keep="last").reset_index(drop=True)

    if DATE_START is not None:
        df = df[df["date"] >= pd.to_datetime(DATE_START)]
    if DATE_END is not None:
        df = df[df["date"] <= pd.to_datetime(DATE_END)]
    df = df.reset_index(drop=True)

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


# 杠杆：计算手数（新增）
def calc_lots(equity: float, entry_px: float, R_points: float) -> int:
    if not np.isfinite(equity) or equity <= 0:
        return 0
    if not np.isfinite(entry_px) or entry_px <= 0:
        return 0
    if not np.isfinite(R_points) or R_points <= 0:
        return 0

    # 1) 风险预算 -> 风险允许的最大手数
    risk_budget = equity * RISK_PCT
    loss_per_lot = R_points * MULTIPLIER  # 到止损每手亏损（点 * 乘数）
    lots_risk = int(risk_budget // loss_per_lot)

    # 2) 名义杠杆上限
    notional_per_lot = entry_px * MULTIPLIER
    lots_lev = int((equity * MAX_LEVERAGE) // notional_per_lot)

    # 3) 保证金上限（不做强平模拟，只限制入场规模）
    margin_per_lot = notional_per_lot * MARGIN_RATE
    lots_margin = int(equity // margin_per_lot) if margin_per_lot > 0 else MAX_LOTS_CAP

    lots_cap = min(lots_lev, lots_margin, MAX_LOTS_CAP)

    # 若连1手都无法满足杠杆/保证金约束，则不允许开仓
    if lots_cap < 1:
        return 0

    # 保持你原来“至少1手”的行为，同时允许在风险预算范围内加手数
    lots = max(BASE_LOTS, lots_risk)
    lots = min(lots, lots_cap)
    return int(max(lots, 0))


# =========================
# 回测
# =========================
def backtest(df: pd.DataFrame, k_atr: float = 2.5) -> dict:
    df = df.copy()
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
    pos_lots = 0  # ✅ 新增：持仓手数（杠杆）

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
        nonlocal cash, pos, pos_lots, entry_px, entry_i, stop, R, highest_close, max_high_since_entry
        # ✅ 改：PnL 乘手数
        pnl = (exit_px - entry_px) * MULTIPLIER * pos_lots - 2 * COMM_PER_SIDE * pos_lots
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
        pos_lots = 0
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

            # ATR追踪止损（Chandelier）
            atr_i = df.loc[i, "atr"]
            if np.isfinite(atr_i) and atr_i > 0:
                trail = highest_close - k_atr * atr_i
                stop = max(stop, trail)

            # +1R 保本
            if USE_BREAK_EVEN and np.isfinite(R) and R > 0:
                if df.loc[i, "high"] >= entry_px + BE_R * R:
                    stop = max(stop, entry_px)

            # 1) 先处理止损触发（当日 low <= stop）
            if df.loc[i, "low"] <= stop:
                exit_px = stop - SLIPPAGE_PTS
                close_pos(i, exit_px, reason="stop")

            # 2) 再处理限时达标，否则退出（只有“从未达标”的单子会触发）
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
                    # ✅ 改：计算手数（杠杆）
                    lots = calc_lots(cash, e, r)
                    if lots >= 1:
                        pos = 1
                        pos_lots = lots
                        entry_px = e
                        entry_i = i
                        stop = s
                        R = r
                        highest_close = df.loc[i, "close"]
                        max_high_since_entry = df.loc[i, "high"]
                        pos_flag[i] = 1

        # ===== 记录权益 =====
        if pos == 1:
            # ✅ 改：持仓盈亏乘手数
            equity[i] = cash + (df.loc[i, "close"] - entry_px) * MULTIPLIER * pos_lots
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

        # 新增：风控参数存入结果字典
        "base_lots": BASE_LOTS,
        "risk_pct": RISK_PCT,
        "max_leverage": MAX_LEVERAGE,
        "margin_rate": MARGIN_RATE,
        "max_lots_cap": MAX_LOTS_CAP,

        "trades_df": trades_df
    }
    return out


def run_once(df: pd.DataFrame):
    out = backtest(df, k_atr=K_ATR)
    d0, d1 = out["data_range"]
    print(f"Data range: {d0.date()} -> {d1.date()}, rows={out['rows']}")
    print("signals:", out["signals"])

    # 新增：输出风控参数
    print("\n===== 风控参数配置 =====")
    print(f"基础手数 (BASE_LOTS): {out['base_lots']}")
    print(f"单笔最大风险比例 (RISK_PCT): {out['risk_pct']*100:.2f}%")
    print(f"最大名义杠杆 (MAX_LEVERAGE): {out['max_leverage']} 倍")
    print(f"初始保证金率 (MARGIN_RATE): {out['margin_rate']*100:.2f}%")
    print(f"最大手数上限 (MAX_LOTS_CAP): {out['max_lots_cap']}")

    print("\n===== 回测结果汇总 =====")
    keys = [
        "k_atr","trades","realized_pnl","final_equity","exposure",
        "win_rate","avg_R","max_dd","sharpe_full","sharpe_active",
        "expectancy_trade","expectancy_R","avg_win","avg_loss",
        "profit_factor","payoff_ratio","expectancy_per_day","expectancy_per_hold_day"
    ]
    for k in keys:
        # 格式化输出，让数值更易读
        if k in ["win_rate", "max_dd", "exposure"]:
            print(f"{k}: {out[k]:.4f}")
        elif k in ["realized_pnl", "final_equity", "avg_win", "avg_loss", "expectancy_trade", "expectancy_per_day", "expectancy_per_hold_day"]:
            print(f"{k}: {out[k]:.2f}")
        elif k in ["trades", "k_atr", "avg_R", "sharpe_full", "sharpe_active", "profit_factor", "payoff_ratio", "expectancy_R"]:
            print(f"{k}: {out[k]:.3f}")
        else:
            print(f"{k}: {out[k]}")

    if out["trades"] > 0:
        print("\n===== 最近交易记录 =====")
        print(out["trades_df"].tail(10).to_string(index=False))  # 改为显示最近10笔，避免输出过长

def grid_search(df: pd.DataFrame):
    rows = []
    best = None
    for k in GRID_K_ATR:
        out = backtest(df, k_atr=k)
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
            # 网格搜索中也包含风控参数
            "base_lots": out["base_lots"],
            "risk_pct": out["risk_pct"],
            "max_leverage": out["max_leverage"],
            "margin_rate": out["margin_rate"]
        })
        if best is None or out["expectancy_trade"] > best["expectancy_trade"]:
            best = out

    # 输出网格搜索的风控参数
    print("\n=====风控参数配置 =====")
    print(f"基础手数 (BASE_LOTS): {best['base_lots']}")
    print(f"单笔最大风险比例 (RISK_PCT): {best['risk_pct']*100:.2f}%")
    print(f"最大名义杠杆 (MAX_LEVERAGE): {best['max_leverage']} 倍")
    print(f"初始保证金率 (MARGIN_RATE): {best['margin_rate']*100:.2f}%")

    print("\n===== 网格搜索结果（按期望收益排序） =====")
    res = pd.DataFrame(rows).sort_values(["expectancy_trade", "sharpe_active"], ascending=False)
    # 格式化数值显示
    res["risk_pct"] = res["risk_pct"] * 100
    res["max_dd"] = res["max_dd"].round(4)
    res["sharpe_active"] = res["sharpe_active"].round(3)
    res["expectancy_trade"] = res["expectancy_trade"].round(2)
    res["pnl"] = res["pnl"].round(2)
    print(res.to_string(index=False))

    print("\n===== 最优参数（按期望收益） =====")
    show = ["k_atr","trades","realized_pnl","max_dd","sharpe_active","expectancy_trade","expectancy_R","profit_factor","exposure"]
    for kk in show:
        if kk in ["max_dd", "exposure"]:
            print(f"{kk}: {best[kk]:.4f}")
        elif kk in ["realized_pnl", "expectancy_trade"]:
            print(f"{kk}: {best[kk]:.2f}")
        else:
            print(f"{kk}: {best[kk]:.3f}")

if __name__ == "__main__":
    df = load_daily_from_excel()
    run_once(df)
    if DO_GRID:
        grid_search(df)