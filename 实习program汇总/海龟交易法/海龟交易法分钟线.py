import numpy as np
import pandas as pd
import akshare as ak  
import math
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.rcParams['font.sans-serif'] = ['Microsoft YaHei', 'SimHei', 'Noto Sans CJK SC']
mpl.rcParams['axes.unicode_minus'] = False  

symbol=input("请输入代码：")
period=input("请输入想构建的分钟线间隔数（间隔分钟数）：")

ENTRY_N = int(input("请输入您需要入场突破周期（天）："))#周期
EXIT_N  = int(input("请输入您需要出场突破周期（天）："))#出场突破
ATR_N   = int(input("请输入您需要ATR测算周期（天）："))

TAKE_PROFIT_N = 0

INIT_EQUITY = 1_000_000
RISK_PCT = 0.01
STOP_N  = 2.0
TRAIL_N = 3.0 
"""
MAX_LOTS = 40      # 手数上限：建议10~20
MIN_LOTS = 1       # 手数下限：一般就1；算不够就不交易"""

MULTIPLIER = 10.0 

TICK_SIZE = 1.0     # 最小跳动
SLIP_TICKS = 1.0   # 滑点：多少跳
COMM_PER_LOT = 0.0  # 单边手续费：每手
slip = SLIP_TICKS * TICK_SIZE
df = ak.futures_zh_minute_sina(symbol=symbol, period=period)


#  改列名 + 类型处理
df = df.rename(columns={"datetime": "dt", "hold": "open_interest"})
df["dt"] = pd.to_datetime(df["dt"])
for c in ["open","high","low","close","volume","open_interest"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
print(df)

# 排序去重
df = (df.dropna(subset=["dt","open","high","low","close"])
        .sort_values("dt")
        .drop_duplicates("dt", keep="last")
        .reset_index(drop=True))

print("rows:", len(df), "range:", df["dt"].iloc[0], "->", df["dt"].iloc[-1])

prev_close = df["close"].shift(1)
tr = pd.concat([
    (df["high"] - df["low"]),
    (df["high"] - prev_close).abs(),
    (df["low"] - prev_close).abs()
], axis=1).max(axis=1)


df["atr"] = tr.rolling(ATR_N, min_periods=ATR_N).mean()

df["ema80"] = df["close"].ewm(span=80, adjust=False).mean()
BUF = 0.1

df["entry_high"] = df["high"].shift(1).rolling(ENTRY_N).max()
df["entry_low"]  = df["low"].shift(1).rolling(ENTRY_N).min()
df["exit_high"]  = df["high"].shift(1).rolling(EXIT_N).max()
df["exit_low"]   = df["low"].shift(1).rolling(EXIT_N).min()

cond_long  = df["close"] > (df["entry_high"] + BUF * df["atr"])
cond_short = df["close"] < (df["entry_low"]  - BUF * df["atr"])

# 两根确认：连续两根收盘都在突破阈值之外
df["long_entry_sig"]  = cond_long  & cond_long.shift(1)
df["short_entry_sig"] = cond_short & cond_short.shift(1)

df["long_exit_sig"]   = df["close"] < df["exit_low"]
df["short_exit_sig"]  = df["close"] > df["exit_high"]

valid = df.dropna(subset=["atr","entry_high","entry_low","exit_high","exit_low"])
print("valid rows:", len(valid))
print("entries:", int(valid["long_entry_sig"].sum()), int(valid["short_entry_sig"].sum()))

equity = INIT_EQUITY
pos = 0
lots = 0
stop_price = None
tp_price = None

entry_price = None
entry_atr = None
hh = None   # 持仓后最高价（多头）
ll = None   # 持仓后最低价（空头）

events = []
equity_curve = []

equity_curve.append([df.loc[0, "dt"], equity, pos, lots])

for i in range(1, len(df)):
    bar = df.iloc[i]
    prev = df.iloc[i-1]   # 上一根：提供信号/ATR

    dt = bar["dt"]
    o, h, l, c = bar["open"], bar["high"], bar["low"], bar["close"]

    # ① 先盯市：上一根收盘 -> 本根开盘（跳空）
    if pos != 0:
        equity += pos * lots * MULTIPLIER * (o - prev["close"])

    # ② 先执行退出（上一根信号，本根开盘）
    if pos == 1 and prev["long_exit_sig"]:
        events.append([dt, "EXIT_LONG", pos, lots, o, equity])
        pos, lots, stop_price, tp_price = 0, 0, None, None   # 清空tp_price
        entry_price, entry_atr, hh, ll = None, None, None, None
    elif pos == -1 and prev["short_exit_sig"]:
        events.append([dt, "EXIT_SHORT", pos, lots, o, equity])
        pos, lots, stop_price, tp_price = 0, 0, None, None   # 清空tp_price
        entry_price, entry_atr, hh, ll = None, None, None, None

    # === 跟踪止损：用上一根ATR做跟踪（更贴合波动），hh/ll 用上一根高低更新 ===
    if pos != 0 and (not pd.isna(prev["atr"])) and float(prev["atr"]) > 0:
        prev_atr = float(prev["atr"])
    
        prev_h = float(prev["high"])
        prev_l = float(prev["low"])
    
        hh = prev_h if hh is None else max(hh, prev_h)
        ll = prev_l if ll is None else min(ll, prev_l)
    
        if pos == 1:
            trail_stop = hh - TRAIL_N * prev_atr
            stop_price = max(stop_price, trail_stop)
        else:
            trail_stop = ll + TRAIL_N * prev_atr
            stop_price = min(stop_price, trail_stop)

        # ③ 再执行止损（含跳空处理）
    if pos != 0 and stop_price is not None:
        hit = (pos == 1 and l <= stop_price) or (pos == -1 and h >= stop_price)
        if hit:
            fill = stop_price
    
            # 跳空穿过止损：按开盘成交（更差的价）
            if pos == 1 and o <= stop_price:      # 多头止损，开盘已经在止损价下方
                fill = o
            if pos == -1 and o >= stop_price:     # 空头止损，开盘已经在止损价上方
                fill = o
    
            # 可选：滑点（你现在 COMM=0，不影响）
            if pos == 1:   fill = fill - slip   # 多头平仓卖出更差
            else:          fill = fill + slip   # 空头平仓买入更差
    
            equity += pos * lots * MULTIPLIER * (fill - o)
            action = "STOP_LOSS"
            if entry_price is not None:
                if pos == 1 and fill > entry_price:
                    action = "TRAIL_STOP"
                if pos == -1 and fill < entry_price:
                    action = "TRAIL_STOP"
            
            events.append([dt, action, pos, lots, fill, equity])
    
            pos, lots, stop_price, tp_price = 0, 0, None, None
            entry_price, entry_atr, hh, ll = None, None, None, None

    # ③.5 止盈（先止损 再止盈）
    if pos != 0 and tp_price is not None:
        hit_tp = (pos == 1 and h >= tp_price) or (pos == -1 and l <= tp_price)
        if hit_tp:
            equity += pos * lots * MULTIPLIER * (tp_price - o)  # 从开盘到止盈价
            events.append([dt, "TAKE_PROFIT", pos, lots, tp_price, equity])
            pos, lots, stop_price, tp_price = 0, 0, None, None
            entry_price, entry_atr, hh, ll = None, None, None, None

    # ④ 持仓到收盘的盯市
    if pos != 0:
        equity += pos * lots * MULTIPLIER * (c - o)

    # ⑤ 空仓才考虑开仓（上一根信号，本根开盘）
    if pos == 0 and (not pd.isna(prev["atr"])) and prev["atr"] > 0:
        atr = float(prev["atr"])
        risk_budget = equity * RISK_PCT
        risk_per_lot = STOP_N * atr * MULTIPLIER
        new_lots = int(math.floor(risk_budget / risk_per_lot)) 
        """
        # 先上限
       new_lots = min(new_lots, MAX_LOTS)
        # 再检查下限：不够就跳过，但要把权益曲线记下来
        if new_lots < MIN_LOTS:
            equity_curve.append([dt, equity, pos, lots])
            continue"""
            
        if new_lots >= 1 and prev["long_entry_sig"] and (prev["close"] > prev["ema80"]):
            pos, lots = 1, new_lots
            stop_price = o - STOP_N * atr
            entry_price = o
            entry_atr = atr
            hh = o
            ll = o
            
            # 多头止盈价必须在入场价上方
            tp_price = (o + TAKE_PROFIT_N * atr) if TAKE_PROFIT_N > 0 else None
            if tp_price is not None and tp_price <= o:
                raise ValueError(f"多头止盈价错误 tp={tp_price} entry={o} TAKE_PROFIT_N={TAKE_PROFIT_N} atr={atr}")

            events.append([dt, "ENTER_LONG", pos, lots, o, equity])
            equity += pos * lots * MULTIPLIER * (c - o)

        elif new_lots >= 1 and prev["short_entry_sig"] and (prev["close"] < prev["ema80"]):
            pos, lots = -1, new_lots
            stop_price = o + STOP_N * atr
            entry_price = o
            entry_atr = atr
            hh = o
            ll = o
            
            # 空头止盈价必须在入场价下方
            tp_price = (o - TAKE_PROFIT_N * atr) if TAKE_PROFIT_N > 0 else None
            if tp_price is not None and tp_price >= o:
                raise ValueError(f"空头止盈价错误 tp={tp_price} entry={o} TAKE_PROFIT_N={TAKE_PROFIT_N} atr={atr}")

            events.append([dt, "ENTER_SHORT", pos, lots, o, equity])
            equity += pos * lots * MULTIPLIER * (c - o)

    equity_curve.append([dt, equity, pos, lots])

ev = pd.DataFrame(events, columns=["dt","action","dir","lots","price","equity"])
eq = pd.DataFrame(equity_curve, columns=["dt","equity","pos","lots"])

print(ev["action"].value_counts())
print("events:", len(ev))
print("total return:", eq["equity"].iloc[-1] / eq["equity"].iloc[0] - 1)

e = eq["equity"].astype(float)
dd = e / e.cummax() - 1.0
print("max drawdown:", dd.min())

ev.to_csv("minute_turtle_events.csv", index=False, encoding="utf-8-sig")
print("saved -> minute_turtle_events.csv")

import os


eq = pd.DataFrame(equity_curve, columns=["dt", "equity", "pos", "lots"])
eq["dt"] = pd.to_datetime(eq["dt"], errors="coerce")

eq.to_csv("minute_turtle_equity.csv", index=False, encoding="utf-8-sig")
print("saved ->", os.path.abspath("minute_turtle_equity.csv"))

df["dt"] = pd.to_datetime(df["dt"])
ev["dt"] = pd.to_datetime(ev["dt"])
eq["dt"] = pd.to_datetime(eq["dt"])

df = df.sort_values("dt").reset_index(drop=True)
ev = ev.sort_values("dt").reset_index(drop=True)
eq = eq.sort_values("dt").reset_index(drop=True)

pos_series = eq.set_index("dt")["pos"].reindex(df["dt"]).ffill().fillna(0)

enter_long = ev[ev["action"] == "ENTER_LONG"]
enter_short = ev[ev["action"] == "ENTER_SHORT"]
exit_long = ev[ev["action"] == "EXIT_LONG"]
exit_short = ev[ev["action"] == "EXIT_SHORT"]
stop_evt = ev[ev["action"].isin(["STOP", "STOP_LOSS", "TRAIL_STOP"])]
tp_evt = ev[ev["action"] == "TAKE_PROFIT"]


fig, ax = plt.subplots(1, 1, figsize=(14, 6))
ax.plot(df["dt"], df["close"], linewidth=1, label="close 价格")
ax.plot(df["dt"], df["ema80"], linewidth=1, label="ema80均线")

if "entry_high" in df.columns:
    ax.plot(df["dt"], df["entry_high"], linewidth=0.8, label="entry_high 唐奇安上线")
if "entry_low" in df.columns:
    ax.plot(df["dt"], df["entry_low"], linewidth=0.8, label="entry_low 唐奇安下线")

ax.scatter(enter_long["dt"], enter_long["price"], marker="^", s=70, label="ENTER_LONG 开仓做多")
ax.scatter(enter_short["dt"], enter_short["price"], marker="v", s=70, label="ENTER_SHORT 开仓做空")
ax.scatter(exit_long["dt"], exit_long["price"], marker="x", s=70, label="EXIT_LONG 多头平仓")
ax.scatter(exit_short["dt"], exit_short["price"], marker="x", s=70, label="EXIT_SHORT 空头平仓")
ax.scatter(stop_evt["dt"], stop_evt["price"], marker="*", s=90, label="STOP 止损位")
ax.scatter(tp_evt["dt"], tp_evt["price"], marker="o", s=60, label="TAKE_PROFIT 止盈位")

ymin, ymax = ax.get_ylim()
ax.fill_between(df["dt"], ymin, ymax, where=(pos_series.values == 1), alpha=0.08, color="green", label="Long Window 做多持仓区间")
ax.fill_between(df["dt"], ymin, ymax, where=(pos_series.values == -1), alpha=0.08, color="red", label="Short Window 做空持仓区间")

ax.set_title(f"期货代码为{symbol} {period} m 分钟线的Turtle Signals")
ax.legend(loc="best")
ax.grid(True)
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

fig, ax = plt.subplots(1, 1, figsize=(14, 4))
ax.plot(eq["dt"], eq["equity"], linewidth=1, label="equity")
ax.set_title("Equity Curve 最终收益曲线")
ax.grid(True)
ax.legend(loc="best")
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()


#  交易区间表（开平仓时间/价格/区间盈利，输出成表格
ev2 = ev.sort_values("dt").reset_index(drop=True).copy()
df2 = df.sort_values("dt").reset_index(drop=True).copy()

dt_to_idx = {pd.Timestamp(t): i for i, t in enumerate(df2["dt"])}

def is_enter(a: str) -> bool:
    return a in ["ENTER_LONG", "ENTER_SHORT"]

def is_exit(a: str) -> bool:
    return a in ["EXIT_LONG", "EXIT_SHORT", "STOP", "STOP_LOSS", "TRAIL_STOP", "TAKE_PROFIT"]


trades = []
cur = None

for _, r in ev2.iterrows():
    a = str(r["action"])
    t = pd.Timestamp(r["dt"])
    px = float(r["price"])
    d = int(np.sign(float(r.get("dir", 0)))) or (1 if "LONG" in a else -1)

    if is_enter(a) and cur is None:
        e_idx = dt_to_idx.get(t, None)
        if e_idx is None:
            continue
        cur = {
            "direction": d,
            "lots": float(r["lots"]),
            "entry_dt": t,
            "entry_px": px,
            "entry_idx": e_idx,
        }

    elif is_exit(a) and cur is not None:
        x_idx = dt_to_idx.get(t, None)
        if x_idx is None:
            cur = None
            continue

        cur.update(
            {
                "exit_dt": t,
                "exit_px": px,
                "exit_idx": x_idx,
                "exit_action": a,
            }
        )
        trades.append(cur)
        cur = None

trades = pd.DataFrame(trades)
cap = pd.Series(dtype=float)

if trades.empty:
    print("没有完整交易，无法生成 trade_metrics 表。")
else:
    rows = []
    for _, trd in trades.iterrows():
        d = int(trd["direction"])
        lots_ = float(trd["lots"])
        e_i, x_i = int(trd["entry_idx"]), int(trd["exit_idx"])
        e_px, x_px = float(trd["entry_px"]), float(trd["exit_px"])

        # 区间盈利（点数/金额）
        pnl_points = d * (x_px - e_px)
        pnl_money = pnl_points * lots_ * MULTIPLIER

        # 指标：MFE / Realized / Capture
        seg = df2.iloc[e_i : max(e_i, x_i - 1) + 1]
        if d == 1:
            mfe = (seg["high"].max() - e_px) if len(seg) else 0.0
        else:
            mfe = (e_px - seg["low"].min()) if len(seg) else 0.0

        realized = pnl_points
        capture = (realized / mfe) if mfe > 0 else np.nan
        giveback = (1 - capture) if pd.notna(capture) else np.nan

        hold_minutes = (trd["exit_dt"] - trd["entry_dt"]).total_seconds() / 60.0

        rows.append(
            {
                "side": "LONG" if d == 1 else "SHORT",
                "lots": lots_,
                "entry_dt": trd["entry_dt"],
                "entry_px": e_px,
                "exit_dt": trd["exit_dt"],
                "exit_px": x_px,
                "exit_action": trd["exit_action"],
                "hold_minutes": hold_minutes,
                "pnl_points": pnl_points,
                "pnl_money": pnl_money,
                "MFE": mfe,
                "Realized": realized,
                "Capture": capture,
                "Giveback": giveback,
            }
        )

    tm = pd.DataFrame(rows).sort_values("entry_dt").reset_index(drop=True)

    tm.to_csv("trade_metrics.csv", index=False, encoding="utf-8-sig")
    print("saved -> trade_metrics.csv")
    
    bad_tp = tm[(tm["exit_action"]=="TAKE_PROFIT") & (
        ((tm["side"]=="LONG")  & (tm["exit_px"] <= tm["entry_px"])) |
        ((tm["side"]=="SHORT") & (tm["exit_px"] >= tm["entry_px"]))
    )]
    print("TAKE_PROFIT 异常笔数:", len(bad_tp))
    print(bad_tp[["entry_dt","side","entry_px","exit_px","pnl_points"]].head(30))
    # 给 Excel
    with pd.ExcelWriter("trade_metrics.xlsx", engine="openpyxl") as w:
        tm.to_excel(w, sheet_name="trades", index=False)
    print("saved -> trade_metrics.xlsx")

    # 控制台快速检查：开平仓时间价格盈利（前几笔）
    print("\n=== trades preview (开平仓时间/价格/区间盈利) ===")
    print(tm[["entry_dt", "entry_px", "exit_dt", "exit_px", "side", "lots", "pnl_points", "pnl_money", "exit_action"]].head(20).to_string(index=False))
    cap = tm["Capture"].dropna().astype(float)
    
if cap.empty:
    print("cap 为空（没有可用Capture），跳过 capture 分布/CDF 绘图。")
else:
    plt.figure(figsize=(10,4))
    plt.hist(cap, bins=30)
    plt.title("Capture Distribution (Realized / MFE)")
    plt.xlabel("Capture")
    plt.ylabel("Count")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("capture_distribution.png", dpi=200)
    plt.show()

    print("saved -> capture_distribution.png")

    print("\n=== Capture stats ===")
    print("count:", int(cap.shape[0]))
    print("mean:", float(cap.mean()))
    print("median:", float(cap.median()))
    print("P(Capture < 0):", float((cap < 0).mean()))
    print("P(Capture < 0.3):", float((cap < 0.3).mean()))
    print("P(Capture < 0.5):", float((cap < 0.5).mean()))

    cap_sorted = np.sort(cap.values)
    plt.figure(figsize=(8,4))
    plt.plot(cap_sorted, np.linspace(0, 1, len(cap_sorted), endpoint=True))
    plt.title("Capture累计分布图")
    plt.xlabel("Capture")
    plt.ylabel("CDF")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("capture_cdf.png", dpi=200)
    plt.show()