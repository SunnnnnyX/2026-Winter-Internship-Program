# -*- coding: utf-8 -*-
"""
单股回测（iFinD QuantAPI HTTP + access_token）
改动：采用 方案1 + 方案2（其余思路不变）

方案2（T+1当日不抬紧止损）：
- 入场当日（t+1当天）不允许把 stop 抬到 H_up / O，也不允许开启吊灯止损
- 当天若触发止损 => 只能 pending，次日开盘强制出（仍保留你的T+1约束）

方案1（收盘确认，次日生效）：
- 从 t+2 开始：
  - 若“某日盘中曾触及 H_up”，且该日【日线收盘 >= H_up】 => 次日开盘后生效：stop=max(stop, H_up)
  - 若“某日盘中曾触及 O”，且该日【日线收盘 >= O】 => 次日开盘后生效：stop=max(stop, O)
  - 若“触及O且收盘>=O” => 从次日开始允许开启吊灯止损；吊灯止损按30m ATR 推进

入场（不变）：
- t+1 09:30~10:30 时间窗内，任一bar触到 L=low(t) => 以 L 买入

退出（不变）：
- 触发止损：t+1当日不能卖，次日开盘出；其它日按 stop 当根处理（开盘跳空按开盘价）
- 最多持有7个交易日（从t+1起算），最后一根30m收盘强平

输出：Excel（trade / metrics / daily_used / m30_used / equity_curve / events / debug）
"""

import os
import re
from pathlib import Path
from dataclasses import dataclass

import numpy as np
import pandas as pd
import requests

# ===================== 配置 =====================
BASE = "https://quantapi.51ifind.com/api/v1"
ACCESS_TOKEN = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
if not ACCESS_TOKEN:
    raise RuntimeError("请先设置环境变量 IFIND_ACCESS_TOKEN（setx 后重开 Spyder）")

HEADERS = {"Content-Type": "application/json", "access_token": ACCESS_TOKEN}

OUT_DIR = Path(r"D:\single_stock_backtest_ifind_http")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 资金 / 参数
INIT_CASH = 100_000.0
FEE_RATE = 0.0          # 单边费率（先0）

ATR_N = 14
CH_K = 3.0
INIT_STOP_MULT = 0.9

HOLD_DAYS = 7           # 从 t+1 起算
# ==============================================


# ===================== 通用工具 =====================
def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
    r.raise_for_status()
    js = r.json()
    if js.get("errorcode", 0) != 0:
        raise RuntimeError(f"[{endpoint}] errorcode={js.get('errorcode')} errmsg={js.get('errmsg')} js={js}")
    return js

def tables_to_df(js: dict) -> pd.DataFrame:
    tables = js.get("tables") or []
    out = []
    for t in tables:
        thscode = t.get("thscode")
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None:
            df.insert(0, "time", times)
        if thscode is not None:
            df.insert(0, "ts_code", thscode)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def ymd8(x) -> str:
    s = str(x).strip().replace("-", "")
    s = re.sub(r"\D", "", s)
    return s[:8]

def ymd_to_dash(ymd: str) -> str:
    return f"{ymd[:4]}-{ymd[4:6]}-{ymd[6:8]}"

def to_ths_code(code6_str: str) -> str:
    return f"{code6_str}.SH" if code6_str.startswith(("6", "9")) else f"{code6_str}.SZ"

def lot_shares(cash: float, price: float) -> int:
    if price <= 0:
        return 0
    return int(cash // (price * 100.0)) * 100

def wilder_atr(high, low, close, n=14) -> pd.Series:
    high = pd.Series(high).astype(float)
    low  = pd.Series(low).astype(float)
    close = pd.Series(close).astype(float)
    prev_close = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-prev_close).abs(), (low-prev_close).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False).mean()

def ensure_30m(df: pd.DataFrame) -> pd.DataFrame:
    """如果返回的是1m/5m等，自动重采样到30m；若已是30m则原样返回。"""
    if df.empty:
        return df
    df = df.sort_values("dt").reset_index(drop=True)
    deltas = df["dt"].diff().dropna().dt.total_seconds() / 60.0
    med = float(deltas.median()) if len(deltas) else 30.0
    if med >= 25:
        return df

    tmp = df.set_index("dt")
    ohlc = tmp[["open","high","low","close"]].resample("30T", label="left", closed="left").agg(
        {"open":"first","high":"max","low":"min","close":"last"}
    )
    out = ohlc.copy()
    out["volume"] = tmp["volume"].resample("30T", label="left", closed="left").sum() if "volume" in tmp.columns else np.nan
    out["amount"] = tmp["amount"].resample("30T", label="left", closed="left").sum() if "amount" in tmp.columns else np.nan
    out = out.dropna(subset=["open","high","low","close"]).reset_index()
    return out


# ===================== 拉数 =====================
def fetch_daily(ths_code: str, startdate: str, enddate: str) -> pd.DataFrame:
    payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "startdate": startdate,
        "enddate": enddate,
        "functionpara": {"Fill": "Blank"},
    }
    js = post_json("cmd_history_quotation", payload)
    df = tables_to_df(js)
    if df.empty:
        return pd.DataFrame(columns=["trade_date","open","high","low","close","volume","amount"])
    df = df.rename(columns={"time":"trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["trade_date","open","high","low","close"]).sort_values("trade_date").reset_index(drop=True)
    return df[["trade_date","open","high","low","close","volume","amount"]]

def fetch_hf(ths_code: str, starttime: str, endtime: str) -> tuple[pd.DataFrame, dict]:
    base_payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "starttime": starttime,
        "endtime": endtime,
    }
    candidates = [
        {"functionpara": {"Interval": "30", "Fill": "Blank"}},
        {"functionpara": {"interval": "30", "fill": "Blank"}},
        {"functionpara": {"Interval": 30}},
        {},
    ]
    debug = {"ok_payload": None, "last_err": None}

    for extra in candidates:
        payload = dict(base_payload)
        payload.update(extra)
        try:
            js = post_json("high_frequency", payload)
            df = tables_to_df(js)
            if df.empty:
                continue
            debug["ok_payload"] = payload

            df = df.rename(columns={"time":"dt"})
            df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
            df = df.dropna(subset=["dt"]).copy()
            for c in ["open","high","low","close","volume","amount"]:
                df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
            df = df.dropna(subset=["open","high","low","close"]).sort_values("dt").reset_index(drop=True)
            df = df[["dt","open","high","low","close","volume","amount"]]

            df = ensure_30m(df)
            df["d"] = df["dt"].dt.strftime("%Y%m%d")
            df["atr"] = wilder_atr(df["high"], df["low"], df["close"], n=ATR_N)
            return df, debug
        except Exception as e:
            debug["last_err"] = repr(e)

    return pd.DataFrame(), debug


# ===================== 回测 =====================
@dataclass
class Trade:
    ths_code: str
    t: str
    tminus1: str
    tplus1: str
    entry_dt: str | None
    entry_px: float | None
    exit_dt: str | None
    exit_px: float | None
    shares: int
    final_cash: float | None
    pnl: float | None
    ret_pct: float | None
    reason: str
    note: str

def main():
    code_in = input("输入股票代码（纯数字，如 301171 / 600519）：").strip()
    t_in = input("输入 t 日期（YYYYMMDD，如 20260113）：").strip()

    code = code6(code_in)
    t = ymd8(t_in)
    ths_code = to_ths_code(code)

    # ---- 日线窗口 ----
    t_dt = pd.to_datetime(t)
    d0 = ymd_to_dash((t_dt - pd.Timedelta(days=260)).strftime("%Y%m%d"))
    d1 = ymd_to_dash((t_dt + pd.Timedelta(days=120)).strftime("%Y%m%d"))
    daily = fetch_daily(ths_code, d0, d1)
    if daily.empty:
        raise RuntimeError("日线为空（权限/代码/日期有问题）")

    dates = daily["trade_date"].tolist()
    arr = np.array(dates)

    # t_trade：<=t 的最近交易日
    idx = np.searchsorted(arr, t, side="right") - 1
    if idx < 1:
        raise RuntimeError("找不到 t 或 t-1（日期范围不够）")
    t_trade = dates[idx]
    tminus1 = dates[idx - 1]
    if idx + 1 >= len(dates):
        raise RuntimeError("找不到 t+1（t太靠后或范围不够）")
    tplus1 = dates[idx + 1]

    # 持有窗口结束日（7个交易日）
    end_idx = min(idx + 1 + HOLD_DAYS - 1, len(dates) - 1)
    last_day = dates[end_idx]

    dmap = daily.set_index("trade_date")
    L = float(dmap.loc[t_trade, "low"])
    O = float(dmap.loc[t_trade, "open"])
    H_up = float(dmap.loc[tminus1, "high"])

    # ---- 高频数据（t+1~last_day）----
    starttime = f"{ymd_to_dash(tplus1)} 09:30:00"
    endtime   = f"{ymd_to_dash(last_day)} 15:00:00"
    m30, debug = fetch_hf(ths_code, starttime, endtime)
    if m30.empty:
        out = OUT_DIR / f"single_{code}_{t}_ifind_http.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            pd.DataFrame([{"ths_code": ths_code, "t": t, "error": "hf empty", "debug": str(debug)}]).to_excel(
                w, index=False, sheet_name="debug"
            )
            daily.to_excel(w, index=False, sheet_name="daily_used")
        raise RuntimeError(f"高频为空（看debug）：{out}")

    # ---- 入场：时间窗取 t+1 09:30~10:30 ----
    t1_start = pd.to_datetime(f"{ymd_to_dash(tplus1)} 09:30:00")
    t1_end   = pd.to_datetime(f"{ymd_to_dash(tplus1)} 10:30:00")
    win = m30[(m30["dt"] >= t1_start) & (m30["dt"] < t1_end)].sort_values("dt").copy()

    events = []

    if win.empty:
        tr = Trade(ths_code, t_trade, tminus1, tplus1, None, None, None, None, 0,
                   INIT_CASH, 0.0, 0.0, "no_entry_window_empty",
                   f"win empty; m30_range={m30['dt'].min()}~{m30['dt'].max()}; ok_payload={debug.get('ok_payload')}")
        out = OUT_DIR / f"single_{code}_{t}_ifind_http.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            pd.DataFrame([tr.__dict__]).to_excel(w, index=False, sheet_name="trade")
            daily.to_excel(w, index=False, sheet_name="daily_used")
            m30.to_excel(w, index=False, sheet_name="m30_used")
            win.to_excel(w, index=False, sheet_name="tplus1_60m_window")
            pd.DataFrame(events).to_excel(w, index=False, sheet_name="events")
            pd.DataFrame([{"debug": str(debug)}]).to_excel(w, index=False, sheet_name="debug")
        print("时间窗无数据，已输出：", out)
        return

    entry_loc = None
    for i in win.index:
        lo, hi = float(m30.loc[i, "low"]), float(m30.loc[i, "high"])
        if lo <= L <= hi:
            entry_loc = int(m30.index.get_loc(i))
            break

    if entry_loc is None:
        tr = Trade(ths_code, t_trade, tminus1, tplus1, None, None, None, None, 0,
                   INIT_CASH, 0.0, 0.0, "no_entry",
                   f"L={L}; win_lowmin={float(win['low'].min())}; win_highmax={float(win['high'].max())}")
        out = OUT_DIR / f"single_{code}_{t}_ifind_http.xlsx"
        with pd.ExcelWriter(out, engine="openpyxl") as w:
            pd.DataFrame([tr.__dict__]).to_excel(w, index=False, sheet_name="trade")
            daily.to_excel(w, index=False, sheet_name="daily_used")
            m30.to_excel(w, index=False, sheet_name="m30_used")
            win.to_excel(w, index=False, sheet_name="tplus1_60m_window")
            pd.DataFrame(events).to_excel(w, index=False, sheet_name="events")
            pd.DataFrame([{"debug": str(debug)}]).to_excel(w, index=False, sheet_name="debug")
        print("未入场（时间窗内未触L），已输出：", out)
        return

    # ---- 建仓：以 L 买入（整仓100股）----
    entry_dt = m30.iloc[entry_loc]["dt"]
    entry_px = L
    buy_px = entry_px * (1 + FEE_RATE)
    shares = lot_shares(INIT_CASH, buy_px)
    if shares <= 0:
        raise RuntimeError("资金不足以买100股")

    cash_left = INIT_CASH - shares * buy_px

    # ---- 状态机 ----
    stop = INIT_STOP_MULT * L

    chandelier_on = False
    hh = -np.inf

    # 方案1：盘中触及只记录，收盘确认后“次日生效”
    touchedH_intraday = False
    touchedO_intraday = False
    pending_apply_H = False
    pending_apply_O = False
    pending_turn_on_chandelier = False

    entry_day = pd.to_datetime(entry_dt).strftime("%Y%m%d")
    pending_exit = False
    pending_reason = ""

    exit_dt = None
    exit_px = None
    reason = "time_exit"

    eq = []
    eq_t = []

    def apply_pending_updates(cur_dt, cur_day):
        """在每个交易日开始时应用昨日确认的更新（只在 t+2 及以后允许生效）"""
        nonlocal stop, pending_apply_H, pending_apply_O, pending_turn_on_chandelier, chandelier_on, hh

        # 方案2：入场当日（t+1）不抬 stop，不开启吊灯
        if cur_day == entry_day:
            pending_apply_H = False
            pending_apply_O = False
            pending_turn_on_chandelier = False
            return

        if pending_apply_H:
            old = stop
            stop = max(stop, H_up)
            events.append({"dt": cur_dt, "event": "apply_stop_H_up_next_day", "old_stop": old, "new_stop": stop, "H_up": H_up})
            pending_apply_H = False

        if pending_apply_O:
            old = stop
            stop = max(stop, O)
            events.append({"dt": cur_dt, "event": "apply_stop_O_next_day", "old_stop": old, "new_stop": stop, "O": O})
            pending_apply_O = False

        if pending_turn_on_chandelier:
            chandelier_on = True
            hh = -np.inf  # 将在当天首根bar更新
            events.append({"dt": cur_dt, "event": "turn_on_chandelier_next_day"})
            pending_turn_on_chandelier = False

    prev_day = None

    for j in range(entry_loc, len(m30)):
        bar = m30.iloc[j]
        dt = bar["dt"]
        day = bar["d"]
        o30 = float(bar["open"])
        h30 = float(bar["high"])
        l30 = float(bar["low"])
        c30 = float(bar["close"])
        atrv = float(bar["atr"]) if pd.notna(bar["atr"]) else np.nan

        # 交易日切换：先应用“昨日确认、今日生效”的更新
        if prev_day is None or day != prev_day:
            apply_pending_updates(dt, day)
            prev_day = day

        # 资金曲线（收盘标记）
        eq.append(cash_left + shares * c30 * (1 - FEE_RATE))
        eq_t.append(dt)

        # T+1：若入场当日触发过卖出意图，下一交易日第一根bar开盘强制出
        if pending_exit and day != entry_day:
            exit_dt = dt
            exit_px = o30 * (1 - FEE_RATE)
            reason = pending_reason + "_Tplus1_forced"
            events.append({"dt": dt, "event": "forced_exit_next_day_open", "px": exit_px, "reason": reason})
            break

        # 止损检查（长仓）
        if l30 <= stop:
            events.append({"dt": dt, "event": "stop_hit", "stop": stop, "o": o30, "h": h30, "l": l30, "c": c30, "day": day})
            if day == entry_day:
                pending_exit = True
                pending_reason = "stop_hit"
            else:
                exec_px = o30 if o30 <= stop else stop
                exit_dt = dt
                exit_px = exec_px * (1 - FEE_RATE)
                reason = "stop_hit"
                break

        # -------- 方案2：入场当日不抬 stop / 不开启吊灯 --------
        if day == entry_day:
            # 只允许继续跑，记录盘中是否触及关键价（但不生效）
            if h30 >= H_up:
                touchedH_intraday = True
            if h30 >= O:
                touchedO_intraday = True
            # 吊灯不更新
            continue

        # 方案1：盘中触及只记录（不立刻抬 stop）
        if h30 >= H_up:
            touchedH_intraday = True
        if h30 >= O:
            touchedO_intraday = True

        # 吊灯：只有开启后才更新
        if chandelier_on:
            hh = max(hh, h30)
            if np.isfinite(atrv) and atrv > 0:
                old = stop
                stop = max(stop, hh - CH_K * atrv)
                if stop != old:
                    events.append({"dt": dt, "event": "chandelier_update", "old_stop": old, "new_stop": stop, "hh": hh, "atr": atrv})

        # 日终：用日线收盘做确认，确认后设置“次日生效”
        is_day_end = (j == len(m30) - 1) or (m30.iloc[j + 1]["d"] != day)
        if is_day_end:
            day_close = float(dmap.loc[day, "close"]) if day in dmap.index else c30

            if touchedH_intraday and (day_close >= H_up):
                pending_apply_H = True
                events.append({"dt": dt, "event": "confirm_H_up_close_ge_H_up", "close": day_close, "H_up": H_up})

            if touchedO_intraday and (day_close >= O):
                pending_apply_O = True
                pending_turn_on_chandelier = True
                events.append({"dt": dt, "event": "confirm_O_close_ge_O", "close": day_close, "O": O})

            # 重置当日触及标记（进入下一交易日重新统计）
            touchedH_intraday = False
            touchedO_intraday = False

    # 时间到强平
    if exit_dt is None:
        last_bar = m30.iloc[-1]
        exit_dt = last_bar["dt"]
        exit_px = float(last_bar["close"]) * (1 - FEE_RATE)
        reason = "time_exit"
        events.append({"dt": exit_dt, "event": "time_exit", "px": exit_px})

    final_cash = cash_left + shares * float(exit_px)
    pnl = final_cash - INIT_CASH
    ret_pct = pnl / INIT_CASH * 100.0

    # 指标
    eqs = pd.Series(eq, index=pd.to_datetime(eq_t))
    eqs = eqs[eqs.index <= pd.to_datetime(exit_dt)]
    peak = eqs.cummax()
    mdd_pct = float((eqs / peak - 1.0).min() * 100.0) if len(eqs) else np.nan

    hold = m30.iloc[entry_loc:].copy()
    hold = hold[hold["dt"] <= pd.to_datetime(exit_dt)]
    mfe_pct = (float(hold["high"].max()) / entry_px - 1.0) * 100.0 if not hold.empty else np.nan
    mae_pct = (float(hold["low"].min()) / entry_px - 1.0) * 100.0 if not hold.empty else np.nan

    tr = Trade(
        ths_code=ths_code,
        t=t_trade,
        tminus1=tminus1,
        tplus1=tplus1,
        entry_dt=str(entry_dt),
        entry_px=float(entry_px),
        exit_dt=str(exit_dt),
        exit_px=float(exit_px),
        shares=int(shares),
        final_cash=float(final_cash),
        pnl=float(pnl),
        ret_pct=float(ret_pct),
        reason=reason,
        note=f"L={L:.3f}, O={O:.3f}, H_up={H_up:.3f}, stop0={INIT_STOP_MULT*L:.3f}, "
             f"scheme=confirm_next_day + no_raise_on_entry_day, ok_payload={debug.get('ok_payload')}"
    )

    metrics = {
        "initial_cash": INIT_CASH,
        "final_cash": float(final_cash),
        "pnl": float(pnl),
        "ret_pct": float(ret_pct),
        "shares": int(shares),
        "fee_rate": float(FEE_RATE),
        "max_dd_pct": float(mdd_pct) if np.isfinite(mdd_pct) else np.nan,
        "mfe_pct": float(mfe_pct) if np.isfinite(mfe_pct) else np.nan,
        "mae_pct": float(mae_pct) if np.isfinite(mae_pct) else np.nan,
        "t_trade": t_trade,
        "tplus1": tplus1,
        "last_day": last_day,
        "hf_ok_payload": str(debug.get("ok_payload")),
        "hf_last_err": str(debug.get("last_err")),
    }

    out = OUT_DIR / f"single_{code}_{t}_ifind_http.xlsx"
    with pd.ExcelWriter(out, engine="openpyxl") as w:
        pd.DataFrame([tr.__dict__]).to_excel(w, index=False, sheet_name="trade")
        pd.DataFrame([metrics]).to_excel(w, index=False, sheet_name="metrics")
        daily.to_excel(w, index=False, sheet_name="daily_used")
        m30.to_excel(w, index=False, sheet_name="m30_used")
        eqs.rename("equity").reset_index().rename(columns={"index":"dt"}).to_excel(w, index=False, sheet_name="equity_curve")
        pd.DataFrame(events).to_excel(w, index=False, sheet_name="events")
        pd.DataFrame([{"debug": str(debug)}]).to_excel(w, index=False, sheet_name="debug")

    print("Done:", out)
    print(pd.DataFrame([metrics]).to_string(index=False))
    print(pd.DataFrame([tr.__dict__]).to_string(index=False))


if __name__ == "__main__":
    main()
