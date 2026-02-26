import numpy as np
import pandas as pd
import akshare as ak
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
import os


mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
mpl.rcParams['axes.unicode_minus'] = False

# 交易参数（集中定义，便于调整）
CONFIG = {
    "INIT_EQUITY": 1_000_000,       # 初始资金 Initial Capital
    "RISK_PCT": 0.01,               # 风险比例 Risk Percentage
    "STOP_N": 2,                    # 止损ATR倍数 Stop Loss ATR Multiple
    "TRAIL_N": 2.5,                 # 跟踪止损ATR倍数 Trailing Stop ATR Multiple
    "BE_N": 1.5,                    # 保本止损ATR倍数 Break Even ATR Multiple
    "MULTIPLIER": 10.0,             # 合约乘数 Contract Multiplier
    "TICK_SIZE": 1.0,               # 最小跳动 Minimum Price Fluctuation
    "SLIP_TICKS": 1.0,              # 滑点跳数 Slippage Ticks
    "COMM_PER_LOT": 0.0,            # 单边手续费（每手）Commission Per Lot (One-way)
    "BUF": 0.1,                     # 突破阈值缓冲（ATR倍数）Breakout Buffer (ATR Multiple)
    "USE_TWO_BAR_CONFIRM": False,   # 两根K线确认开关 Two-bar Confirmation Switch
    "USE_EMA_FILTER": True,         # EMA趋势过滤开关 EMA Trend Filter Switch
    "EMA_SPAN": 40,                 # EMA周期 EMA Period
}
CONFIG["SLIP"] = CONFIG["SLIP_TICKS"] * CONFIG["TICK_SIZE"]  # 滑点金额 Slippage Amount


def load_futures_data(symbol: str) -> pd.DataFrame:
    """加载期货日线数据（兼容akshare不同版本）Load Futures Daily Data"""
    try:
        df = ak.futures_zh_daily_sina(symbol=symbol)
    except:
        df = ak.futures_zh_daily_sina(symbol=f"期货_{symbol}")
    
    rename_map = {
        "date": "dt_日期",
        "datetime": "dt_日期",
        "hold": "open_interest_持仓量"
    }
    df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
    
    df["dt_日期"] = pd.to_datetime(df["dt_日期"])
    numeric_cols = ["open_开盘价", "high_最高价", "low_最低价", "close_收盘价", "volume_成交量", "open_interest_持仓量"]
    original_cols = ["open", "high", "low", "close", "volume", "open_interest"]
    for orig_col, new_col in zip(original_cols, numeric_cols):
        if orig_col in df.columns:
            df.rename(columns={orig_col: new_col}, inplace=True)
            df[new_col] = pd.to_numeric(df[new_col], errors="coerce")
        else:
            df[new_col] = np.nan
    
    df = (df.dropna(subset=["dt_日期", "open_开盘价", "high_最高价", "low_最低价", "close_收盘价"])
          .sort_values("dt_日期")
          .drop_duplicates("dt_日期", keep="last")
          .reset_index(drop=True))
    
    print(f"数据加载完成 | Data Load Completed | 总行数 Total Rows: {len(df)} | 时间范围 Time Range: {df['dt_日期'].iloc[0]} -> {df['dt_日期'].iloc[-1]}")
    return df


def calculate_indicators(df: pd.DataFrame, entry_n: int, exit_n: int, atr_n: int) -> pd.DataFrame:
    """计算交易所需指标：ATR、EMA、唐奇安通道、交易信号 Calculate Trading Indicators"""
    df = df.copy()
    
    # 1. 计算TR/ATR (True Range / Average True Range)
    prev_close = df["close_收盘价"].shift(1)
    tr = pd.concat([
        df["high_最高价"] - df["low_最低价"],
        (df["high_最高价"] - prev_close).abs(),
        (df["low_最低价"] - prev_close).abs()
    ], axis=1).max(axis=1)
    df["atr_平均真实波幅"] = tr.rolling(atr_n, min_periods=atr_n).mean()
    
    # 2. 计算EMA (Exponential Moving Average)
    df["ema40_40期指数移动平均"] = df["close_收盘价"].ewm(span=CONFIG["EMA_SPAN"], adjust=False).mean()
    
    # 3. 唐奇安通道（入场/出场）Donchian Channel (Entry/Exit)
    df["entry_high_入场上轨"] = df["high_最高价"].shift(1).rolling(entry_n).max()
    df["entry_low_入场下轨"] = df["low_最低价"].shift(1).rolling(entry_n).min()
    df["exit_high_出场上轨"] = df["high_最高价"].shift(1).rolling(exit_n).max()
    df["exit_low_出场下轨"] = df["low_最低价"].shift(1).rolling(exit_n).min()
    
    # 4. 入场信号条件 Entry Signal Conditions
    cond_long = df["close_收盘价"] > (df["entry_high_入场上轨"] + CONFIG["BUF"] * df["atr_平均真实波幅"])
    cond_short = df["close_收盘价"] < (df["entry_low_入场下轨"] - CONFIG["BUF"] * df["atr_平均真实波幅"])
    
    # 两根K线确认（可选）Two-bar Confirmation (Optional)
    if CONFIG["USE_TWO_BAR_CONFIRM"]:
        df["long_entry_sig_多头入场信号"] = cond_long & cond_long.shift(1)
        df["short_entry_sig_空头入场信号"] = cond_short & cond_short.shift(1)
    else:
        df["long_entry_sig_多头入场信号"] = cond_long
        df["short_entry_sig_空头入场信号"] = cond_short
    
    # 出场信号 Exit Signals
    df["long_exit_sig_多头出场信号"] = df["close_收盘价"] < df["exit_low_出场下轨"]
    df["short_exit_sig_空头出场信号"] = df["close_收盘价"] > df["exit_high_出场上轨"]
    
    # 有效数据过滤 Valid Data Filtering
    valid_cols = ["atr_平均真实波幅", "entry_high_入场上轨", "entry_low_入场下轨", "exit_high_出场上轨", "exit_low_出场下轨"]
    valid_df = df.dropna(subset=valid_cols)
    print(f"指标计算完成 | Indicator Calculation Completed | 有效数据行数 Valid Rows: {len(valid_df)}")
    print(f"入场信号统计 Entry Signal Stats | 多头 Long: {valid_df['long_entry_sig_多头入场信号'].sum()} | 空头 Short: {valid_df['short_entry_sig_空头入场信号'].sum()}")
    
    return df


def run_backtest(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """执行回测主逻辑，返回交易事件和资金曲线 Run Backtest Main Logic"""
    # 初始化回测变量 Initialize Backtest Variables
    equity = CONFIG["INIT_EQUITY"]
    pos = 0  # 持仓方向 Position: 1=多 Long, -1=空 Short, 0=空仓 Flat
    lots = 0  # 持仓手数 Position Lots
    stop_price = None  # 止损价格 Stop Price
    entry_price = None  # 开仓价格 Entry Price
    entry_atr = None  # 开仓时ATR ATR at Entry
    hh, ll = None, None  # 持仓后最高/最低价 High/Low after Position
    
    events = []  # 交易事件记录 Trading Events Record
    equity_curve = [(df.loc[0, "dt_日期"], equity, pos, lots)]  # 资金曲线 Equity Curve
    
    # 逐行回测 Iterative Backtest
    for i in range(1, len(df)):
        bar = df.iloc[i]
        prev_bar = df.iloc[i-1]
        dt, o, h, l, c = bar["dt_日期"], bar["open_开盘价"], bar["high_最高价"], bar["low_最低价"], bar["close_收盘价"]
        
        # 1. 盯市：上一根收盘到本根开盘（跳空处理）Mark to Market: Previous Close to Current Open (Gap Handling)
        if pos != 0:
            equity += pos * lots * CONFIG["MULTIPLIER"] * (o - prev_bar["close_收盘价"])
        
        # 2. 执行出场（上一根信号，本根开盘）Execute Exit (Previous Signal, Current Open)
        if pos == 1 and prev_bar["long_exit_sig_多头出场信号"]:
            events.append([dt, "EXIT_LONG_多头平仓", pos, lots, o, equity])
            pos, lots, stop_price = 0, 0, None
            entry_price, entry_atr, hh, ll = None, None, None, None
        elif pos == -1 and prev_bar["short_exit_sig_空头出场信号"]:
            events.append([dt, "EXIT_SHORT_空头平仓", pos, lots, o, equity])
            pos, lots, stop_price = 0, 0, None
            entry_price, entry_atr, hh, ll = None, None, None, None
        
        # 3. 跟踪止损更新 Trailing Stop Update
        if pos != 0 and not pd.isna(prev_bar["atr_平均真实波幅"]) and prev_bar["atr_平均真实波幅"] > 0:
            prev_atr = float(prev_bar["atr_平均真实波幅"])
            prev_h, prev_l = float(prev_bar["high_最高价"]), float(prev_bar["low_最低价"])
            
            # 更新持仓后最高/最低价 Update High/Low after Position
            hh = prev_h if hh is None else max(hh, prev_h)
            ll = prev_l if ll is None else min(ll, prev_l)
            
            # 计算跟踪止损价格 Calculate Trailing Stop Price
            if pos == 1:
                trail_stop = hh - CONFIG["TRAIL_N"] * prev_atr
                stop_price = max(stop_price, trail_stop) if stop_price else trail_stop
            else:
                trail_stop = ll + CONFIG["TRAIL_N"] * prev_atr
                stop_price = min(stop_price, trail_stop) if stop_price else trail_stop
            
            # 保本止损：盈利达到BE_N*ATR时，止损抬到开仓价 Break Even Stop: Lift Stop to Entry Price when Profit >= BE_N*ATR
            if entry_price and entry_atr and stop_price:
                if pos == 1 and (hh - entry_price) >= CONFIG["BE_N"] * entry_atr:
                    stop_price = max(stop_price, entry_price)
                elif pos == -1 and (entry_price - ll) >= CONFIG["BE_N"] * entry_atr:
                    stop_price = min(stop_price, entry_price)
        
        # 4. 执行止损（含跳空+滑点处理）Execute Stop Loss (Include Gap + Slippage)
        if pos != 0 and stop_price is not None:
            hit_stop = (pos == 1 and l <= stop_price) or (pos == -1 and h >= stop_price)
            if hit_stop:
                # 跳空穿过止损：按开盘价成交 Gap Through Stop: Execute at Open Price
                fill_price = stop_price
                if pos == 1 and o <= stop_price:
                    fill_price = o
                if pos == -1 and o >= stop_price:
                    fill_price = o
                
                # 滑点处理 Slippage Handling
                fill_price = fill_price - CONFIG["SLIP"] if pos == 1 else fill_price + CONFIG["SLIP"]
                
                # 更新资金 Update Equity
                equity += pos * lots * CONFIG["MULTIPLIER"] * (fill_price - o)
                
                # 记录止损类型 Record Stop Type
                action = "STOP_LOSS_止损离场"
                if entry_price:
                    if pos == 1 and fill_price > entry_price:
                        action = "TRAIL_STOP_跟踪止损"
                    if pos == -1 and fill_price < entry_price:
                        action = "TRAIL_STOP_跟踪止损"
                
                events.append([dt, action, pos, lots, fill_price, equity])
                pos, lots, stop_price = 0, 0, None
                entry_price, entry_atr, hh, ll = None, None, None, None
        
        # 5. 持仓到收盘的盯市 Mark to Market until Close
        if pos != 0:
            equity += pos * lots * CONFIG["MULTIPLIER"] * (c - o)
        
        # 6. 开仓逻辑（空仓+有效ATR+信号触发）Entry Logic (Flat + Valid ATR + Signal Trigger)
        if pos == 0 and not pd.isna(prev_bar["atr_平均真实波幅"]) and prev_bar["atr_平均真实波幅"] > 0:
            atr = float(prev_bar["atr_平均真实波幅"])
            risk_budget = equity * CONFIG["RISK_PCT"]
            risk_per_lot = CONFIG["STOP_N"] * atr * CONFIG["MULTIPLIER"]
            new_lots = int(math.floor(risk_budget / risk_per_lot))
            
            # EMA趋势过滤 EMA Trend Filter
            long_filter = (prev_bar["close_收盘价"] > prev_bar["ema40_40期指数移动平均"]) if CONFIG["USE_EMA_FILTER"] else True
            short_filter = (prev_bar["close_收盘价"] < prev_bar["ema40_40期指数移动平均"]) if CONFIG["USE_EMA_FILTER"] else True
            
            # 多头开仓 Long Entry
            if new_lots >= 1 and prev_bar["long_entry_sig_多头入场信号"] and long_filter:
                pos, lots = 1, new_lots
                stop_price = o - CONFIG["STOP_N"] * atr
                entry_price, entry_atr = o, atr
                hh, ll = o, o
                events.append([dt, "ENTER_LONG_开仓做多", pos, lots, o, equity])
                equity += pos * lots * CONFIG["MULTIPLIER"] * (c - o)
            
            # 空头开仓 Short Entry
            elif new_lots >= 1 and prev_bar["short_entry_sig_空头入场信号"] and short_filter:
                pos, lots = -1, new_lots
                stop_price = o + CONFIG["STOP_N"] * atr
                entry_price, entry_atr = o, atr
                hh, ll = o, o
                events.append([dt, "ENTER_SHORT_开仓做空", pos, lots, o, equity])
                equity += pos * lots * CONFIG["MULTIPLIER"] * (c - o)
        
        # 记录资金曲线 Record Equity Curve
        equity_curve.append((dt, equity, pos, lots))
    
    # 转换为DataFrame Convert to DataFrame
    events_df = pd.DataFrame(events, columns=["dt_日期", "action_交易动作", "dir_持仓方向", "lots_持仓手数", "price_成交价格", "equity_账户权益"])
    equity_df = pd.DataFrame(equity_curve, columns=["dt_日期", "equity_账户权益", "pos_持仓方向", "lots_持仓手数"])
    
    # 回测结果统计 Backtest Result Statistics
    total_return = (equity_df["equity_账户权益"].iloc[-1] / equity_df["equity_账户权益"].iloc[0]) - 1
    dd_series = equity_df["equity_账户权益"] / equity_df["equity_账户权益"].cummax() - 1.0
    max_dd = dd_series.min()
    
    print("\n=== 回测核心统计 | Backtest Core Statistics ===")
    print(f"交易事件数 Trading Events Count: {len(events_df)}")
    print(f"事件类型分布 Event Type Distribution:\n{events_df['action_交易动作'].value_counts()}")
    print(f"总收益率 Total Return: {total_return:.2%}")
    print(f"最大回撤 Maximum Drawdown: {max_dd:.2%}")
    
    return events_df, equity_df


def plot_results(df: pd.DataFrame, events_df: pd.DataFrame, equity_df: pd.DataFrame, symbol: str, entry_n: int, exit_n: int):
    """绘制回测结果可视化图表 Plot Backtest Results"""
    # 数据预处理 Data Preprocessing
    df = df.sort_values("dt_日期").reset_index(drop=True)
    events_df = events_df.sort_values("dt_日期").reset_index(drop=True)
    equity_df = equity_df.sort_values("dt_日期").reset_index(drop=True)
    
    # 持仓序列（用于填充持仓区间）Position Series (for Position Interval Filling)
    pos_series = equity_df.set_index("dt_日期")["pos_持仓方向"].reindex(df["dt_日期"]).ffill().fillna(0)
    
    # 提取交易信号点 Extract Trading Signal Points
    enter_long = events_df[events_df["action_交易动作"] == "ENTER_LONG_开仓做多"]
    enter_short = events_df[events_df["action_交易动作"] == "ENTER_SHORT_开仓做空"]
    exit_long = events_df[events_df["action_交易动作"] == "EXIT_LONG_多头平仓"]
    exit_short = events_df[events_df["action_交易动作"] == "EXIT_SHORT_空头平仓"]
    stop_evt = events_df[events_df["action_交易动作"].isin(["STOP_LOSS_止损离场", "TRAIL_STOP_跟踪止损"])]
    
    # 图1：价格+信号+持仓区间 Price + Signal + Position Interval
    fig, ax = plt.subplots(1, 1, figsize=(14, 6))
    ax.plot(df["dt_日期"], df["close_收盘价"], lw=1, label="close_收盘价")
    ax.plot(df["dt_日期"], df["ema40_40期指数移动平均"], lw=1, label="ema40_40期指数移动平均")
    ax.plot(df["dt_日期"], df["entry_high_入场上轨"], lw=0.8, label="entry_high_入场上轨")
    ax.plot(df["dt_日期"], df["entry_low_入场下轨"], lw=0.8, label="entry_low_入场下轨")
    
    ax.scatter(enter_long["dt_日期"], enter_long["price_成交价格"], marker="^", s=70, c="green", label="ENTER_LONG_开仓做多")
    ax.scatter(enter_short["dt_日期"], enter_short["price_成交价格"], marker="v", s=70, c="red", label="ENTER_SHORT_开仓做空")
    ax.scatter(exit_long["dt_日期"], exit_long["price_成交价格"], marker="x", s=70, c="darkgreen", label="EXIT_LONG_多头平仓")
    ax.scatter(exit_short["dt_日期"], exit_short["price_成交价格"], marker="x", s=70, c="darkred", label="EXIT_SHORT_空头平仓")
    ax.scatter(stop_evt["dt_日期"], stop_evt["price_成交价格"], marker="*", s=90, c="orange", label="STOP_止损触发")
    
    # 持仓区间填充 Position Interval Filling
    ymin, ymax = ax.get_ylim()
    ax.fill_between(df["dt_日期"], ymin, ymax, where=(pos_series == 1), alpha=0.08, color="green", label="Long Window_多头持仓区间")
    ax.fill_between(df["dt_日期"], ymin, ymax, where=(pos_series == -1), alpha=0.08, color="red", label="Short Window_空头持仓区间")
    
    ax.set_title(f"期货 Futures {symbol} 海龟交易信号 Turtle Trading Signal（入场周期 Entry Period {entry_n} | 出场周期 Exit Period {exit_n}）", fontsize=12)
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()
    
    # 图2：资金曲线 Equity Curve
    fig, ax = plt.subplots(1, 1, figsize=(14, 4))
    ax.plot(equity_df["dt_日期"], equity_df["equity_账户权益"], lw=1, label="equity_账户净值")
    ax.set_title("账户净值曲线 Equity Curve", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.show()


def generate_trade_metrics(df: pd.DataFrame, events_df: pd.DataFrame):
    """生成交易明细指标（持仓天数、盈利、MFE等）Generate Trade Metrics"""
    df = df.sort_values("dt_日期").reset_index(drop=True)
    events_df = events_df.sort_values("dt_日期").reset_index(drop=True).copy()
    
    # 构建时间到索引的映射 Build Date to Index Mapping
    dt_to_idx = {pd.Timestamp(t): i for i, t in enumerate(df["dt_日期"])}
    
    # 定义信号类型判断函数 Define Signal Type Judgment Function
    is_enter = lambda a: "ENTER_LONG" in a or "ENTER_SHORT" in a
    is_exit = lambda a: "EXIT_LONG" in a or "EXIT_SHORT" in a or "STOP_LOSS" in a or "TRAIL_STOP" in a
    
    # 匹配开平仓对 Match Entry-Exit Pairs
    trades = []
    current_trade = None
    for _, row in events_df.iterrows():
        action = row["action_交易动作"]
        dt = pd.Timestamp(row["dt_日期"])
        price = float(row["price_成交价格"])
        direction = int(np.sign(row.get("dir_持仓方向", 0))) or (1 if "ENTER_LONG" in action else -1)
        
        # 开仓 Entry
        if is_enter(action) and current_trade is None:
            entry_idx = dt_to_idx.get(dt)
            if entry_idx is None:
                continue
            current_trade = {
                "direction_方向": direction,
                "lots_手数": float(row["lots_持仓手数"]),
                "entry_dt_开仓日期": dt,
                "entry_px_开仓价格": price,
                "entry_idx_开仓索引": entry_idx
            }
        
        # 平仓 Exit
        elif is_exit(action) and current_trade is not None:
            exit_idx = dt_to_idx.get(dt)
            if exit_idx is None:
                current_trade = None
                continue
            
            current_trade.update({
                "exit_dt_平仓日期": dt,
                "exit_px_平仓价格": price,
                "exit_idx_平仓索引": exit_idx,
                "exit_action_平仓动作": action
            })
            trades.append(current_trade)
            current_trade = None
    
    # 无有效交易时直接返回 Return if No Valid Trades
    if not trades:
        print("无完整交易记录，跳过交易指标生成 | No Complete Trade Records, Skip Trade Metrics Generation")
        return
    
    # 计算交易指标 Calculate Trade Metrics
    trade_metrics = []
    for trade in trades:
        dir_ = trade["direction_方向"]
        lots_ = trade["lots_手数"]
        e_idx, x_idx = trade["entry_idx_开仓索引"], trade["exit_idx_平仓索引"]
        e_px, x_px = trade["entry_px_开仓价格"], trade["exit_px_平仓价格"]
        
        # 盈利计算 Profit Calculation
        pnl_points = dir_ * (x_px - e_px)
        pnl_money = pnl_points * lots_ * CONFIG["MULTIPLIER"]
        
        # MFE（最大有利波动）Maximum Favorable Excursion
        trade_bar = df.iloc[e_idx: x_idx + 1]
        if dir_ == 1:
            mfe = (trade_bar["high_最高价"].max() - e_px) if len(trade_bar) else 0.0
        else:
            mfe = (e_px - trade_bar["low_最低价"].min()) if len(trade_bar) else 0.0
        
        # 盈利捕获率/回吐率 Profit Capture Rate / Giveback Rate
        capture = (pnl_points / mfe) if mfe > 0 else np.nan
        giveback = (1 - capture) if pd.notna(capture) else np.nan
        
        # 持仓天数 Holding Days
        hold_days = (trade["exit_dt_平仓日期"].normalize() - trade["entry_dt_开仓日期"].normalize()).days
        
        trade_metrics.append({
            "side_方向": "LONG_多头" if dir_ == 1 else "SHORT_空头",
            "lots_手数": lots_,
            "entry_dt_开仓日期": trade["entry_dt_开仓日期"],
            "entry_px_开仓价格": e_px,
            "exit_dt_平仓日期": trade["exit_dt_平仓日期"],
            "exit_px_平仓价格": x_px,
            "exit_action_平仓动作": trade["exit_action_平仓动作"],
            "hold_days_持仓天数": hold_days,
            "pnl_points_盈利点数": pnl_points,
            "pnl_money_盈利金额": pnl_money,
            "MFE_最大有利波动": mfe,
            "Realized_实际盈利": pnl_points,
            "Capture_盈利捕获率": capture,
            "Giveback_盈利回吐率": giveback
        })
    
    # 转换为DataFrame并保存 Convert to DataFrame and Save
    tm_df = pd.DataFrame(trade_metrics).sort_values("entry_dt_开仓日期").reset_index(drop=True)
    tm_df.to_csv("trade_metrics_daily.csv", index=False, encoding="utf-8-sig")
    with pd.ExcelWriter("trade_metrics_daily.xlsx", engine="openpyxl") as writer:
        tm_df.to_excel(writer, sheet_name="trades_交易明细", index=False)
    print(f"\n交易指标已保存 | Trade Metrics Saved | 有效交易数 Valid Trades: {len(tm_df)}")
    print("\n=== 交易明细预览（前20条）| Trade Details Preview (Top 20) ===")
    preview_cols = ["entry_dt_开仓日期", "entry_px_开仓价格", "exit_dt_平仓日期", "exit_px_平仓价格", "side_方向", "lots_手数", "pnl_money_盈利金额", "exit_action_平仓动作"]
    print(tm_df[preview_cols].head(20).to_string(index=False))
    
    # 绘制Capture分布/CDF图 Plot Capture Distribution/CDF
    cap_series = tm_df["Capture_盈利捕获率"].dropna()
    if cap_series.empty:
        print("无有效Capture数据，跳过绘图 | No Valid Capture Data, Skip Plotting")
        return
    
    # Capture分布直方图 Capture Distribution Histogram
    plt.figure(figsize=(10, 4))
    plt.hist(cap_series, bins=30, alpha=0.7)
    plt.title("盈利捕获率分布 Capture Distribution（Realized/MFE）", fontsize=12)
    plt.xlabel("Capture_盈利捕获率")
    plt.ylabel("Count_交易次数")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("capture_distribution_daily.png", dpi=200)
    plt.show()
    
    # Capture统计 Capture Statistics
    print("\n=== 盈利捕获率统计 | Capture Statistics ===")
    print(f"有效样本数 Valid Samples: {len(cap_series)}")
    print(f"均值 Mean: {cap_series.mean():.3f}")
    print(f"中位数 Median: {cap_series.median():.3f}")
    print(f"捕获率<0占比 Capture < 0 Ratio: {(cap_series < 0).mean():.2%}")
    print(f"捕获率<0.3占比 Capture < 0.3 Ratio: {(cap_series < 0.3).mean():.2%}")
    print(f"捕获率<0.5占比 Capture < 0.5 Ratio: {(cap_series < 0.5).mean():.2%}")
    
    # Capture累计分布曲线 Capture CDF Curve
    cap_sorted = np.sort(cap_series.values)
    plt.figure(figsize=(8, 4))
    plt.plot(cap_sorted, np.linspace(0, 1, len(cap_sorted)), lw=1)
    plt.title("盈利捕获率累计分布 Capture CDF（日线 Daily）", fontsize=12)
    plt.xlabel("Capture_盈利捕获率")
    plt.ylabel("CDF_累计概率")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("capture_cdf_daily.png", dpi=200)
    plt.show()



if __name__ == "__main__":
    # 输入参数 Input Parameters
    symbol = input("请输入合约代码（主力合约或单个合约均可）Please enter contract code: ")
    entry_n = int(input("请输入入场突破周期（天）Please enter entry breakout period (days): "))
    exit_n = int(input("请输入出场突破周期（天）Please enter exit breakout period (days): "))
    atr_n = int(input("请输入ATR测算周期（天）Please enter ATR calculation period (days): "))
    
    # 流程执行 Process Execution
    raw_df = load_futures_data(symbol)
    indicator_df = calculate_indicators(raw_df, entry_n, exit_n, atr_n)
    events_df, equity_df = run_backtest(indicator_df)
    
    # 保存基础结果 Save Basic Results
    events_df.to_csv("daily_turtle_events.csv", index=False, encoding="utf-8-sig")
    equity_df.to_csv("daily_turtle_equity.csv", index=False, encoding="utf-8-sig")
    print("\n基础结果文件已保存 | Basic Result Files Saved：")
    print(f"- 交易事件 Trading Events: {os.path.abspath('daily_turtle_events.csv')}")
    print(f"- 资金曲线 Equity Curve: {os.path.abspath('daily_turtle_equity.csv')}")
    
    # 可视化+交易指标 Visualization + Trade Metrics
    plot_results(indicator_df, events_df, equity_df, symbol, entry_n, exit_n)
    generate_trade_metrics(indicator_df, events_df)