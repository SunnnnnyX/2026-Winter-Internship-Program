# -*- coding: utf-8 -*-
"""
中证500（000905）实时监测：日线信号 + 30min 执行过滤
- 数据：AkShare 东方财富指数接口（免费、可能有延迟/缺口）
- 频率：每 60 秒刷新一次
- 输出：当前行情 + 建议动作（开仓/平仓/持有/等待）

重要提醒：
1) 这是“监测/建议”脚本，不是自动下单；你需要自己在券商/期货软件执行。
2) 仅用 30min K 线做盘中判断，会错过 30min 内的极端波动；如果你要更细的止损触发，应改用 1min 数据。
"""

from __future__ import annotations

import time
import math
import traceback
from dataclasses import dataclass
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import akshare as ak

# ==========================================================
# 0) 基本参数：指数代码/刷新频率
# ==========================================================
INDEX_CODE = "000905"        # 中证500指数代码（无市场前缀）
DAILY_SYMBOL = "csi000905"   # AkShare: stock_zh_index_daily_em 支持 csi + 代码

REFRESH_SEC = 60             # 延时 1min 刷新
LOOKBACK_DAYS_DAILY = 800    # 日线拉取回看天数（足够算ATR/均线等）
LOOKBACK_DAYS_M30 = 20       # 30min 拉取回看天数（足够当日/近期）

# 是否“假定你已执行建议动作”，用于更新内部持仓状态（否则会每分钟重复提示同一动作）
ASSUME_EXECUTED = True

# ==========================================================
# 1) 策略参数（沿用你之前脚本的逻辑与命名）
# ==========================================================
ATR_N = 14
K_ATR = 3.0

MULTIPLIER = 1.0  # 指数点位本身没有合约乘数；你如果拿它映射 IC 期货，可改成 200

REQUIRE_OPEN_UP = True  # True=做多要求 open 递增；做空要求 open 递减（镜像）

USE_BREAK_EVEN = True
BE_R = 1.0

USE_TREND_FILTER = False
MA_FAST = 50
MA_SLOW = 200

USE_TIMEOUT_EXIT = True
TIMEOUT_T = 12
TP_XR = 0.8

# 30min 执行层
USE_M30_EXECUTION = True
M30_NEED_BARS = 3
ENTRY_AT_NEXT_BAR_OPEN = True

USE_ACCEPTANCE = True
ACCEPT_W = 6
ACCEPT_P = 0.67
ACCEPT_TOL_ATR = 0.10

USE_M30_ATR_STOP = True
M30_ATR_N = 14
M30_STOP_ATR_MULT = 2.5


# ==========================================================
# 工具函数（复用你原脚本思路）
# ==========================================================
def wilder_atr_core(high, low, close, n: int) -> pd.Series:
    prev_close = close.shift(1)
    tr = pd.concat(
        [(high - low).abs(), (high - prev_close).abs(), (low - prev_close).abs()],
        axis=1
    ).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()

def build_signal_long(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    cond = (
        (h.shift(2) < h.shift(1)) & (h.shift(1) < h) &
        (l.shift(2) < l.shift(1)) & (l.shift(1) < l) &
        (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) < o.shift(1)) & (o.shift(1) < o)
    return cond.fillna(False)

def build_signal_short(df: pd.DataFrame) -> pd.Series:
    h, l, o, c = df["high"], df["low"], df["open"], df["close"]
    cond = (
        (h.shift(2) > h.shift(1)) & (h.shift(1) > h) &
        (l.shift(2) > l.shift(1)) & (l.shift(1) > l) &
        (c.shift(2) > c.shift(1)) & (c.shift(1) > c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) > o.shift(1)) & (o.shift(1) > o)
    return cond.fillna(False)

def find_threebar_rise_idx(day_bars: pd.DataFrame) -> int | None:
    if day_bars is None or len(day_bars) < (M30_NEED_BARS + 1):
        return None
    g = day_bars.reset_index(drop=True)
    h, l, o, c = g["high"], g["low"], g["open"], g["close"]
    cond = (
        (h.shift(2) < h.shift(1)) & (h.shift(1) < h) &
        (l.shift(2) < l.shift(1)) & (l.shift(1) < l) &
        (c.shift(2) < c.shift(1)) & (c.shift(1) < c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) < o.shift(1)) & (o.shift(1) < o)
    hit = cond[cond.fillna(False)].index
    return None if len(hit) == 0 else int(hit[0])

def find_threebar_fall_idx(day_bars: pd.DataFrame) -> int | None:
    if day_bars is None or len(day_bars) < (M30_NEED_BARS + 1):
        return None
    g = day_bars.reset_index(drop=True)
    h, l, o, c = g["high"], g["low"], g["open"], g["close"]
    cond = (
        (h.shift(2) > h.shift(1)) & (h.shift(1) > h) &
        (l.shift(2) > l.shift(1)) & (l.shift(1) > l) &
        (c.shift(2) > c.shift(1)) & (c.shift(1) > c)
    )
    if REQUIRE_OPEN_UP:
        cond = cond & (o.shift(2) > o.shift(1)) & (o.shift(1) > o)
    hit = cond[cond.fillna(False)].index
    return None if len(hit) == 0 else int(hit[0])

def pick_entry_price_time(day_bars: pd.DataFrame, third_idx: int):
    g = day_bars.reset_index(drop=True)
    if ENTRY_AT_NEXT_BAR_OPEN:
        j = third_idx + 1
        if j >= len(g):
            return None
        return float(g.loc[j, "open"]), pd.Timestamp(g.loc[j, "dt"]), j
    else:
        return float(g.loc[third_idx, "close"]), pd.Timestamp(g.loc[third_idx, "dt"]), third_idx

def acceptance_prob(day_bars: pd.DataFrame, start_bar_index: int, level: float, W: int, tol: float, side: int) -> float | None:
    end = start_bar_index + W - 1
    g = day_bars.reset_index(drop=True)
    if end >= len(g):
        return None
    w = g.iloc[start_bar_index:end + 1]
    if side == 1:
        return float((w["close"] >= level - tol).mean())
    else:
        return float((w["close"] <= level + tol).mean())

def acceptance_exit_price(day_bars: pd.DataFrame, start_bar_index: int, W: int) -> float | None:
    end = start_bar_index + W - 1
    g = day_bars.reset_index(drop=True)
    if end >= len(g):
        return None
    return float(g.iloc[end]["close"])


# ==========================================================
# 数据拉取与清洗
# ==========================================================
def _fmt_ymd(dt: datetime) -> str:
    return dt.strftime("%Y%m%d")

def _fmt_dt(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S")

def fetch_daily(now_cn: datetime) -> pd.DataFrame:
    start = now_cn - timedelta(days=LOOKBACK_DAYS_DAILY)
    df = ak.stock_zh_index_daily_em(symbol=DAILY_SYMBOL, start_date=_fmt_ymd(start), end_date=_fmt_ymd(now_cn))
    df = df.rename(columns={"date": "date"})
    # 标准化列名
    df = df.rename(columns={"date": "date", "open": "open", "high": "high", "low": "low", "close": "close"})
    df["date"] = pd.to_datetime(df["date"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["date", "open", "high", "low", "close"]).sort_values("date").reset_index(drop=True)
    return df

def fetch_m30(now_cn: datetime) -> pd.DataFrame:
    start = now_cn - timedelta(days=LOOKBACK_DAYS_M30)
    df = ak.index_zh_a_hist_min_em(
        symbol=INDEX_CODE,
        period="30",
        start_date=_fmt_dt(start.replace(hour=9, minute=30, second=0)),
        end_date=_fmt_dt(now_cn + timedelta(days=1)),
    )
    # 输出字段：时间/开盘/收盘/最高/最低/成交量/成交额/均价
    df = df.rename(columns={"时间": "dt", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low"})
    df["dt"] = pd.to_datetime(df["dt"])
    for c in ["open", "high", "low", "close"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["dt", "open", "high", "low", "close"]).sort_values("dt").reset_index(drop=True)
    df["d"] = df["dt"].dt.normalize()
    # 30min ATR（给 m30 紧急止损/接受度容忍）
    df["atr30"] = wilder_atr_core(df["high"], df["low"], df["close"], M30_ATR_N)
    return df

def fetch_spot() -> dict | None:
    # 取“中证系列指数”全量列表，然后过滤出 000905
    df = ak.stock_zh_index_spot_em(symbol="中证系列指数")
    row = None
    if "代码" in df.columns:
        hit = df[df["代码"].astype(str) == INDEX_CODE]
        if len(hit) > 0:
            row = hit.iloc[0]
    if row is None and "名称" in df.columns:
        hit = df[df["名称"].astype(str).str.contains("中证500", na=False)]
        if len(hit) > 0:
            row = hit.iloc[0]
    if row is None:
        return None

    def _get(k, default=np.nan):
        return float(row[k]) if (k in row and pd.notna(row[k])) else default

    return {
        "name": str(row.get("名称", "中证500")),
        "last": _get("最新价"),
        "open": _get("今开"),
        "high": _get("最高"),
        "low": _get("最低"),
        "prev_close": _get("昨收"),
        "pct": _get("涨跌幅"),
        "chg": _get("涨跌额"),
        "ts": datetime.now(tz=ZoneInfo("Asia/Shanghai")),
    }


# ==========================================================
# 实盘状态机
# ==========================================================
@dataclass
class State:
    pos: int = 0  # +1=LONG, -1=SHORT, 0=FLAT
    entry_px: float = math.nan
    entry_time: pd.Timestamp | None = None
    entry_date: pd.Timestamp | None = None

    stop: float = math.nan
    R: float = math.nan

    # 追踪用
    highest_close: float = -math.inf
    lowest_close: float = math.inf
    max_high_since_entry: float = -math.inf
    min_low_since_entry: float = math.inf

    # 接受度用（需要等到 W 根 30m 出完才能判断）
    acc_level: float = math.nan
    acc_tol: float = 0.0
    entry_bar_index: int | None = None
    acc_checked: bool = False

    # 防止同一天重复触发入场
    last_enter_key: tuple | None = None  # (date, side)

def reset_state(s: State):
    s.pos = 0
    s.entry_px = math.nan
    s.entry_time = None
    s.entry_date = None
    s.stop = math.nan
    s.R = math.nan
    s.highest_close = -math.inf
    s.lowest_close = math.inf
    s.max_high_since_entry = -math.inf
    s.min_low_since_entry = math.inf
    s.acc_level = math.nan
    s.acc_tol = 0.0
    s.entry_bar_index = None
    s.acc_checked = False


def compute_signals_and_atr(daily: pd.DataFrame) -> pd.DataFrame:
    df = daily.copy()
    df["atr"] = wilder_atr_core(df["high"], df["low"], df["close"], ATR_N)
    df["sig_long"] = build_signal_long(df)
    df["sig_short"] = build_signal_short(df)

    if USE_TREND_FILTER:
        df["ma_fast"] = df["close"].rolling(MA_FAST).mean()
        df["ma_slow"] = df["close"].rolling(MA_SLOW).mean()
        df["trend_long"] = (df["close"] > df["ma_slow"]) & (df["ma_fast"] > df["ma_slow"])
        df["trend_short"] = (df["close"] < df["ma_slow"]) & (df["ma_fast"] < df["ma_slow"])
        df["trend_long"] = df["trend_long"].fillna(False)
        df["trend_short"] = df["trend_short"].fillna(False)
    else:
        df["trend_long"] = True
        df["trend_short"] = True

    return df


def decide_action(now_cn: datetime, daily_sig: pd.DataFrame, m30: pd.DataFrame, spot: dict | None, s: State):
    """
    返回 (action, detail_dict)
    action:
      - WAIT
      - ENTER_LONG / ENTER_SHORT
      - EXIT_LONG / EXIT_SHORT
      - HOLD_LONG / HOLD_SHORT
    """
    today = pd.Timestamp(now_cn.date())
    detail = {}

    # 现价/当日高低
    if spot is not None and np.isfinite(spot.get("last", np.nan)):
        cur = float(spot["last"])
        day_high = float(spot.get("high", np.nan))
        day_low = float(spot.get("low", np.nan))
    else:
        # 退化：用最后一根 30m close
        cur = float(m30["close"].iloc[-1]) if len(m30) else math.nan
        day_high = float(m30[m30["d"] == today]["high"].max()) if len(m30) else math.nan
        day_low = float(m30[m30["d"] == today]["low"].min()) if len(m30) else math.nan

    detail["cur"] = cur
    detail["day_high"] = day_high
    detail["day_low"] = day_low

    # 取“上一交易日”（日线里 < today 的最后一行）
    hist = daily_sig[daily_sig["date"] < today]
    if len(hist) < 3:
        detail["msg"] = "日线数据不足（<3个交易日），无法给信号"
        return "WAIT", detail

    prev = hist.iloc[-1]
    prev_idx = hist.index[-1]  # 在 daily_sig 里的真实 index
    prev_day = pd.Timestamp(prev["date"]).normalize()

    want_long = bool(prev["sig_long"]) and bool(prev["trend_long"])
    want_short = bool(prev["sig_short"]) and bool(prev["trend_short"])

    detail["prev_day"] = str(prev_day.date())
    detail["want_long"] = want_long
    detail["want_short"] = want_short

    # 供入场用：三日趋势的第一天（对应你原脚本 first_idx = i-3）
    first_idx = prev_idx - 2
    if first_idx < 0:
        want_long = want_short = False

    # ======================================================
    # A) 持仓管理：先判是否该平仓
    # ======================================================
    if s.pos != 0:
        side = s.pos

        # 用“日线 ATR（上一交易日ATR）”更新吊灯止损（盘中用现价近似）
        atr_now = float(prev.get("atr", np.nan))
        if np.isfinite(atr_now) and atr_now > 0 and np.isfinite(cur):
            if side == 1:
                s.highest_close = max(s.highest_close, cur)
                trail = s.highest_close - K_ATR * atr_now
                s.stop = max(s.stop, trail)
            else:
                s.lowest_close = min(s.lowest_close, cur)
                trail = s.lowest_close + K_ATR * atr_now
                s.stop = min(s.stop, trail)

        # 更新极值（用于 timeout 判定）
        if np.isfinite(day_high):
            s.max_high_since_entry = max(s.max_high_since_entry, day_high)
        if np.isfinite(day_low):
            s.min_low_since_entry = min(s.min_low_since_entry, day_low)

        # 保本
        if USE_BREAK_EVEN and np.isfinite(s.R) and s.R > 0:
            if side == 1 and np.isfinite(day_high) and day_high >= s.entry_px + BE_R * s.R:
                s.stop = max(s.stop, s.entry_px)
            if side == -1 and np.isfinite(day_low) and day_low <= s.entry_px - BE_R * s.R:
                s.stop = min(s.stop, s.entry_px)

        # 1) 止损触发（用当日低/高来近似）
        if np.isfinite(s.stop):
            if side == 1 and np.isfinite(day_low) and day_low <= s.stop:
                detail["exit_px"] = float(s.stop)
                detail["reason"] = "STOP"
                return "EXIT_LONG", detail
            if side == -1 and np.isfinite(day_high) and day_high >= s.stop:
                detail["exit_px"] = float(s.stop)
                detail["reason"] = "STOP"
                return "EXIT_SHORT", detail

        # 2) 接受度：等到 entry_bar_index+W-1 的 30m bar 出现再判
        if USE_ACCEPTANCE and (s.entry_bar_index is not None) and (not s.acc_checked):
            today_bars = m30[m30["d"] == today].reset_index(drop=True)
            if len(today_bars) >= s.entry_bar_index + ACCEPT_W:
                acc = acceptance_prob(today_bars, s.entry_bar_index, s.acc_level, ACCEPT_W, s.acc_tol, side=side)
                if acc is not None and acc < ACCEPT_P:
                    px = acceptance_exit_price(today_bars, s.entry_bar_index, ACCEPT_W)
                    if px is not None:
                        detail["exit_px"] = float(px)
                        detail["reason"] = f"ACCEPT_FAIL({acc:.2f})"
                        return "EXIT_LONG" if side == 1 else "EXIT_SHORT", detail
                s.acc_checked = True

        # 3) 超时退出（用“交易日差”近似）
        if USE_TIMEOUT_EXIT and (s.entry_date is not None) and np.isfinite(s.R) and s.R > 0:
            # 用日线数据里的交易日来计数（更接近真实交易日）
            traded_days = daily_sig["date"].dt.normalize().tolist()
            if s.entry_date.normalize() in traded_days and prev_day.normalize() in traded_days:
                entry_pos = traded_days.index(s.entry_date.normalize())
                prev_pos = traded_days.index(prev_day.normalize())
                if (prev_pos - entry_pos) >= TIMEOUT_T:
                    if side == 1:
                        target = s.entry_px + TP_XR * s.R
                        if s.max_high_since_entry < target:
                            detail["exit_px"] = float(cur)
                            detail["reason"] = "TIMEOUT"
                            return "EXIT_LONG", detail
                    else:
                        target = s.entry_px - TP_XR * s.R
                        if s.min_low_since_entry > target:
                            detail["exit_px"] = float(cur)
                            detail["reason"] = "TIMEOUT"
                            return "EXIT_SHORT", detail

        return "HOLD_LONG" if side == 1 else "HOLD_SHORT", detail

    # ======================================================
    # B) 空仓：判断是否该开仓（今日为“信号后第1天”）
    # ======================================================
    if want_long == want_short:
        # 同时为真或同时为假 -> 不做
        return "WAIT", detail

    side_to_try = 1 if want_long else -1
    enter_key = (today.date(), side_to_try)
    if s.last_enter_key == enter_key:
        # 今天这个方向已经触发过（避免你没下单时刷屏）
        return "WAIT", detail

    # 日线初始止损：多=第一根low；空=第一根high（沿用你原脚本）
    daily_stop = float(daily_sig.loc[first_idx, "low"]) if side_to_try == 1 else float(daily_sig.loc[first_idx, "high"])

    # 30min 执行过滤：当日出现“三连走高/走低”，在下一根 30m 开盘进
    if USE_M30_EXECUTION:
        today_bars = m30[m30["d"] == today].reset_index(drop=True)
        third_idx = find_threebar_rise_idx(today_bars) if side_to_try == 1 else find_threebar_fall_idx(today_bars)

        if third_idx is None:
            detail["msg"] = "等待30min三连条件触发"
            return "WAIT", detail

        ep = pick_entry_price_time(today_bars, third_idx)
        if ep is None:
            detail["msg"] = "三连已出现，但下一根30m尚未生成（等下一根开盘）"
            return "WAIT", detail

        entry_price, entry_ts, entry_bar_index = ep
    else:
        # 不用执行过滤：用现价近似（不推荐）
        entry_price, entry_ts, entry_bar_index = cur, pd.Timestamp(now_cn), None

    # R 必须为正
    R0 = (entry_price - daily_stop) if side_to_try == 1 else (daily_stop - entry_price)
    if not (np.isfinite(R0) and R0 > 0):
        detail["msg"] = "R<=0（入场价与初始止损关系不成立），跳过"
        return "WAIT", detail

    # 准备输出入场建议
    detail.update({
        "entry_px": float(entry_price),
        "entry_ts": str(entry_ts),
        "daily_stop": float(daily_stop),
        "R": float(R0),
        "side": "LONG" if side_to_try == 1 else "SHORT",
    })
    return "ENTER_LONG" if side_to_try == 1 else "ENTER_SHORT", detail


def apply_action(action: str, detail: dict, s: State, now_cn: datetime, m30: pd.DataFrame, daily_sig: pd.DataFrame):
    """假定你已执行建议，用于更新状态"""
    today = pd.Timestamp(now_cn.date())

    if action in ("ENTER_LONG", "ENTER_SHORT"):
        side = 1 if action == "ENTER_LONG" else -1
        s.pos = side
        s.entry_px = float(detail["entry_px"])
        s.entry_time = pd.Timestamp(detail["entry_ts"])
        s.entry_date = today  # 用“入场日”占位；EOD后日线会补齐

        s.stop = float(detail["daily_stop"])
        s.R = float(detail["R"])

        # 初始化追踪
        s.highest_close = s.entry_px if side == 1 else -math.inf
        s.lowest_close = s.entry_px if side == -1 else math.inf
        s.max_high_since_entry = float(detail.get("entry_px", s.entry_px))
        s.min_low_since_entry = float(detail.get("entry_px", s.entry_px))

        # m30 ATR 紧急止损 + 接受度参数
        today_bars = m30[m30["d"] == today].reset_index(drop=True)
        entry_bar_index = None
        # 尽力匹配 entry_ts 的 bar index（不保证完全一致）
        if len(today_bars) and "dt" in today_bars.columns:
            match = today_bars.index[today_bars["dt"] == pd.Timestamp(detail["entry_ts"])]
            if len(match) > 0:
                entry_bar_index = int(match[0])
        # 如果上面没匹配到，就用 detail 里 pick_entry_price_time 返回的 index（如果你想更严格，可自己改）
        # 这里为了稳健：直接用“最接近 entry_ts 的 bar”
        if entry_bar_index is None and len(today_bars):
            entry_bar_index = int((today_bars["dt"] - pd.Timestamp(detail["entry_ts"])).abs().idxmin())

        s.entry_bar_index = entry_bar_index

        if USE_M30_ATR_STOP and (entry_bar_index is not None) and ("atr30" in today_bars.columns):
            atr30_here = float(today_bars.loc[entry_bar_index, "atr30"])
            if np.isfinite(atr30_here) and atr30_here > 0:
                if side == 1:
                    m30_stop = s.entry_px - M30_STOP_ATR_MULT * atr30_here
                    s.stop = max(s.stop, m30_stop)
                else:
                    m30_stop = s.entry_px + M30_STOP_ATR_MULT * atr30_here
                    s.stop = min(s.stop, m30_stop)

        if USE_ACCEPTANCE and (entry_bar_index is not None) and ("atr30" in today_bars.columns):
            atr30_here = float(today_bars.loc[entry_bar_index, "atr30"])
            s.acc_tol = (ACCEPT_TOL_ATR * atr30_here) if (np.isfinite(atr30_here) and atr30_here > 0) else 0.0

            # level：多=prev_day高点；空=prev_day低点
            prev = daily_sig[daily_sig["date"] < today].iloc[-1]
            s.acc_level = float(prev["high"]) if side == 1 else float(prev["low"])
            s.acc_checked = False

        s.last_enter_key = (today.date(), side)

    if action in ("EXIT_LONG", "EXIT_SHORT"):
        reset_state(s)


def print_status(now_cn: datetime, spot: dict | None, daily_sig: pd.DataFrame, m30: pd.DataFrame, s: State, action: str, detail: dict):
    ts = now_cn.strftime("%Y-%m-%d %H:%M:%S")
    name = spot["name"] if spot else "中证500"
    last = spot["last"] if spot else (detail.get("cur", np.nan))
    pct = spot["pct"] if spot else np.nan
    day_high = spot["high"] if spot else detail.get("day_high", np.nan)
    day_low = spot["low"] if spot else detail.get("day_low", np.nan)

    prev_day = detail.get("prev_day", "?")
    want_long = detail.get("want_long", False)
    want_short = detail.get("want_short", False)

    pos_txt = "FLAT" if s.pos == 0 else ("LONG" if s.pos == 1 else "SHORT")

    print("\n" + "=" * 72)
    print(f"[{ts} CN] {name}({INDEX_CODE})  最新: {last:.2f}  涨跌幅: {pct:.2f}%  高/低: {day_high:.2f}/{day_low:.2f}")
    print(f"日线(上一交易日={prev_day}) 信号: long={want_long} short={want_short}")

    if s.pos != 0:
        upnl = (last - s.entry_px) * MULTIPLIER if s.pos == 1 else (s.entry_px - last) * MULTIPLIER
        print(f"持仓: {pos_txt}  entry={s.entry_px:.2f}  stop={s.stop:.2f}  R={s.R:.2f}  未实现PnL(按乘数)={upnl:.2f}")
    else:
        print("持仓: FLAT")

    # 建议动作
    if action.startswith("ENTER"):
        print(f"==> 建议动作：{action}  entry≈{detail.get('entry_px'):.2f}  init_stop≈{detail.get('daily_stop'):.2f}  R≈{detail.get('R'):.2f}")
    elif action.startswith("EXIT"):
        print(f"==> 建议动作：{action}  exit≈{detail.get('exit_px', last):.2f}  reason={detail.get('reason','')}")
    else:
        msg = detail.get("msg", "")
        print(f"==> 建议动作：{action} {msg}")

    # 显示最后一根30m（帮助你对齐）
    today = pd.Timestamp(now_cn.date())
    today_bars = m30[m30["d"] == today]
    if len(today_bars):
        b = today_bars.iloc[-1]
        print(f"30m最近一根: {pd.Timestamp(b['dt'])}  O/H/L/C={b['open']:.2f}/{b['high']:.2f}/{b['low']:.2f}/{b['close']:.2f}")


def main():
    tz = ZoneInfo("Asia/Shanghai")
    state = State()

    print("启动：中证500 实时监测（每 60 秒刷新）")
    print("按 Ctrl+C 退出。\n")

    last_refresh_minute = None

    while True:
        try:
            now_cn = datetime.now(tz=tz)

            # 对齐到每分钟运行一次（避免 drift）
            this_minute = now_cn.replace(second=0, microsecond=0)
            if last_refresh_minute is not None and this_minute == last_refresh_minute:
                time.sleep(0.2)
                continue
            last_refresh_minute = this_minute

            # 拉取数据
            daily = fetch_daily(now_cn)
            daily_sig = compute_signals_and_atr(daily)
            m30 = fetch_m30(now_cn)
            spot = fetch_spot()

            # 决策
            action, detail = decide_action(now_cn, daily_sig, m30, spot, state)

            # 输出
            print_status(now_cn, spot, daily_sig, m30, state, action, detail)

            # 状态更新（默认假定你执行了建议）
            if ASSUME_EXECUTED and action in ("ENTER_LONG", "ENTER_SHORT", "EXIT_LONG", "EXIT_SHORT"):
                apply_action(action, detail, state, now_cn, m30, daily_sig)

            # sleep 到下一分钟
            time.sleep(REFRESH_SEC)

        except KeyboardInterrupt:
            print("\n退出。")
            break
        except Exception as e:
            print("\n[ERROR]", repr(e))
            traceback.print_exc()
            time.sleep(REFRESH_SEC)


if __name__ == "__main__":
    main()
