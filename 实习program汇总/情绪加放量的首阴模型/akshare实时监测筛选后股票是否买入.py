import re
import time
from pathlib import Path

import numpy as np
import pandas as pd

import akshare as ak


# ===================== 你要改的配置 =====================
OUTPUT_XLSX = Path(r"D:\realtime_manual_monitor_akshare.xlsx")

# 刷新间隔（秒）
POLL_SEC = 60

# 分时粒度（分钟）：1 / 5 / 15 / 30 / 60
MIN_INTERVAL = 1

# 监测窗口（触L入场）
ENTRY_WINDOW_START = "09:30:00"
ENTRY_WINDOW_END   = "10:30:00"

# 初始止损
INIT_STOP_MULT = 0.90

# 收盘停止
MARKET_CLOSE_TIME = "15:00:00"

# 只在 “今天==t+1” 时才做分时监测（严格一致逻辑）。不想限制就改 False
ONLY_MONITOR_IF_TPLUS1_IS_TODAY = True

# 过滤：688/300/9 + ST（名称含 ST/*ST/ＳＴ 都过滤）
EXCLUDE_PREFIXES = ("688", "300", "9")
ENABLE_EXCLUDE_ST = True

# 日线拉取冗余窗口（保证覆盖 t 前后）
DAILY_LOOKBACK_DAYS = 160

# ============ AkShare 分钟数据源优先级（建议不动） ============
# 优先东方财富(股票分钟K) -> 失败再试新浪(股票分钟)
# 注意：不同版本 AkShare 函数名可能略有差异；我做了兼容判断
PREFER_EASTMONEY = True


# ===================== 控制台实时显示（新增） =====================
PRINT_CONSOLE = True          # 是否在控制台打印实时结果
PRINT_TOPN = 30               # 每次最多打印多少行（避免太长）
PRINT_ONLY_IMPORTANT = True   # True=只显示 BUY/SELL/ERR/NO_DATA；False=全量
PRINT_EVERY_N_TICKS = 1       # 每隔N个tick打印一次（1=每次都打印）


# ===================== 基础工具 =====================
def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def ymd8(x) -> str:
    s = str(x).strip().replace("-", "")
    s = re.sub(r"\D", "", s)
    return s[:8]

def ymd_to_dash(ymd: str) -> str:
    s = str(ymd).strip()
    return f"{s[:4]}-{s[4:6]}-{s[6:8]}"

def safe_excel_path(path: Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with open(path, "ab"):
            pass
        return path
    except PermissionError:
        ts = pd.Timestamp.now().strftime("%Y%m%d_%H%M%S")
        return path.with_name(f"{path.stem}_{ts}{path.suffix}")

def is_excluded_prefix(code6_str: str) -> bool:
    c = code6(code6_str)
    return c.startswith(EXCLUDE_PREFIXES)

def is_st_name(name: str) -> bool:
    if not name:
        return False
    s = str(name).strip().upper().replace(" ", "")
    s = s.replace("＊", "*").replace("ＳＴ", "ST")
    return ("ST" in s)


# ===================== 控制台打印（新增） =====================
def print_console_snapshot(df_rt: pd.DataFrame, today_ymd: str, now: pd.Timestamp):
    """
    控制台打印精简实时表：按你已有优先级排序后的 df_rt 直接展示。
    """
    if df_rt is None or df_rt.empty:
        print("[Console] empty watchlist")
        return

    show = df_rt.copy()

    if PRINT_ONLY_IMPORTANT:
        important = {"SELL(stop)", "BUY@L / HOLD", "ERR", "NO_DATA"}
        show = show[show["action"].isin(important)].copy()

    cols = ["code", "name", "industry", "t", "tplus1", "L", "stop", "action", "touch_time", "last_px", "note"]
    cols = [c for c in cols if c in show.columns]
    show = show[cols].copy()

    if "L" in show.columns:
        show["L"] = pd.to_numeric(show["L"], errors="coerce").round(3)
    if "stop" in show.columns:
        show["stop"] = pd.to_numeric(show["stop"], errors="coerce").round(3)
    if "last_px" in show.columns:
        show["last_px"] = pd.to_numeric(show["last_px"], errors="coerce").round(3)

    show = show.head(int(PRINT_TOPN))

    print("\n" + "=" * 110)
    print(f"[Console] {now.strftime('%Y-%m-%d %H:%M:%S')}  today={today_ymd}  showing={len(show)}")
    if show.empty:
        print("(no important rows right now)")
    else:
        print(show.to_string(index=False))
    print("=" * 110 + "\n")


# ===================== AkShare：名称/行业（尽量拿到） =====================
def fetch_name_industry(code: str) -> tuple[str, str]:
    """
    用 ak.stock_info_a_code_name() 拿名称；行业尽量用 ak.stock_individual_info_em
    行业拿不到就空。
    """
    code = code6(code)
    name = ""
    industry = ""

    # 名称（最稳）
    try:
        info = ak.stock_info_a_code_name()
        if isinstance(info, pd.DataFrame) and not info.empty:
            col_code = None
            col_name = None
            for c in info.columns:
                if str(c).lower() in ("code", "代码"):
                    col_code = c
                if str(c).lower() in ("name", "名称"):
                    col_name = c
            if col_code and col_name:
                m = info[info[col_code].astype(str).str.zfill(6) == code]
                if not m.empty:
                    name = str(m.iloc[0][col_name]).strip()
    except Exception:
        pass

    # 行业：尽力从 Eastmoney 个股信息里拿；失败则留空
    try:
        if hasattr(ak, "stock_individual_info_em"):
            df = ak.stock_individual_info_em(symbol=code)
            if isinstance(df, pd.DataFrame) and not df.empty:
                item_col = None
                val_col = None
                for c in df.columns:
                    lc = str(c).lower()
                    if lc in ("item", "指标", "项目", "name"):
                        item_col = c
                    if lc in ("value", "值", "data"):
                        val_col = c
                if item_col and val_col:
                    for kw in ["行业", "申万", "所属行业"]:
                        mm = df[df[item_col].astype(str).str.contains(kw, na=False)]
                        if not mm.empty:
                            industry = str(mm.iloc[0][val_col]).strip()
                            break
    except Exception:
        pass

    return name, industry


# ===================== AkShare：日线 =====================
def fetch_daily_ak(code: str, start: str, end: str) -> pd.DataFrame:
    """
    使用东财A股日线：ak.stock_zh_a_hist(symbol=, period='daily', start_date=, end_date=, adjust='')
    返回统一列：trade_date open high low close volume amount
    """
    code = code6(code)
    df = ak.stock_zh_a_hist(symbol=code, period="daily", start_date=start, end_date=end, adjust="")
    if df is None or df.empty:
        return pd.DataFrame(columns=["trade_date","open","high","low","close","volume","amount"])

    colmap = {}
    for c in df.columns:
        if str(c) in ("日期", "date"):
            colmap[c] = "trade_date"
        elif str(c) in ("开盘", "open"):
            colmap[c] = "open"
        elif str(c) in ("最高", "high"):
            colmap[c] = "high"
        elif str(c) in ("最低", "low"):
            colmap[c] = "low"
        elif str(c) in ("收盘", "close"):
            colmap[c] = "close"
        elif str(c) in ("成交量", "volume"):
            colmap[c] = "volume"
        elif str(c) in ("成交额", "amount"):
            colmap[c] = "amount"
    df = df.rename(columns=colmap)

    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    for c in ["open","high","low","close","volume","amount"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
        else:
            df[c] = np.nan

    df = df.dropna(subset=["trade_date","open","high","low","close"]).sort_values("trade_date").reset_index(drop=True)
    return df[["trade_date","open","high","low","close","volume","amount"]]


def resolve_t_triplet(daily: pd.DataFrame, t_raw: str) -> tuple[str, str, str]:
    """
    给一个 t（你输入的大阴线那天），在日线里按交易日序列定位 t-1/t/t+1
    逻辑：searchsorted 右侧回退
    """
    t = ymd8(t_raw)
    dates = daily["trade_date"].astype(str).tolist()
    arr = np.array(dates)
    idx = np.searchsorted(arr, t, side="right") - 1
    if idx < 1 or idx >= len(dates):
        raise RuntimeError(f"t={t} 不在日线范围内")
    t_trade = dates[idx]
    if idx + 1 >= len(dates):
        raise RuntimeError(f"t={t_trade} 没有 t+1（日线末尾）")
    tminus1 = dates[idx - 1]
    tplus1  = dates[idx + 1]
    return tminus1, t_trade, tplus1


# ===================== AkShare：分钟K（多路 fallback） =====================
def _fetch_minute_eastmoney(code: str, start_dt: pd.Timestamp, end_dt: pd.Timestamp, k: int) -> pd.DataFrame:
    """
    东财分钟K：不同 AkShare 版本函数名可能是 stock_zh_a_hist_min_em
    目标列：dt open high low close volume amount
    """
    code = code6(code)

    if hasattr(ak, "stock_zh_a_hist_min_em"):
        df = ak.stock_zh_a_hist_min_em(
            symbol=code,
            period=str(k),
            start_date=start_dt.strftime("%Y-%m-%d %H:%M:%S"),
            end_date=end_dt.strftime("%Y-%m-%d %H:%M:%S"),
            adjust=""
        )
        if df is None or df.empty:
            return pd.DataFrame()

        colmap = {}
        for c in df.columns:
            if str(c) in ("时间", "datetime", "日期", "time"):
                colmap[c] = "dt"
            elif str(c) in ("开盘", "open"):
                colmap[c] = "open"
            elif str(c) in ("最高", "high"):
                colmap[c] = "high"
            elif str(c) in ("最低", "low"):
                colmap[c] = "low"
            elif str(c) in ("收盘", "close"):
                colmap[c] = "close"
            elif str(c) in ("成交量", "volume"):
                colmap[c] = "volume"
            elif str(c) in ("成交额", "amount"):
                colmap[c] = "amount"
        df = df.rename(columns=colmap)

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan

        df = df.dropna(subset=["dt","open","high","low","close"]).sort_values("dt").reset_index(drop=True)
        return df[["dt","open","high","low","close","volume","amount"]]

    return pd.DataFrame()


def _fetch_minute_sina(code: str, k: int) -> pd.DataFrame:
    """
    新浪分钟：stock_zh_a_minute（若存在）
    通常返回全量当天/近期分钟数据，之后我们再裁剪时间范围。
    """
    code = code6(code)
    if hasattr(ak, "stock_zh_a_minute"):
        df = ak.stock_zh_a_minute(symbol=code, period=str(k))
        if df is None or df.empty:
            return pd.DataFrame()

        colmap = {}
        for c in df.columns:
            lc = str(c).lower()
            if lc in ("datetime", "time", "日期时间", "时间"):
                colmap[c] = "dt"
            elif lc in ("open", "开盘"):
                colmap[c] = "open"
            elif lc in ("high", "最高"):
                colmap[c] = "high"
            elif lc in ("low", "最低"):
                colmap[c] = "low"
            elif lc in ("close", "收盘"):
                colmap[c] = "close"
            elif lc in ("volume", "成交量"):
                colmap[c] = "volume"
            elif lc in ("amount", "成交额"):
                colmap[c] = "amount"
        df = df.rename(columns=colmap)

        df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
        for c in ["open","high","low","close","volume","amount"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
            else:
                df[c] = np.nan

        df = df.dropna(subset=["dt","open","high","low","close"]).sort_values("dt").reset_index(drop=True)
        return df[["dt","open","high","low","close","volume","amount"]]

    return pd.DataFrame()


def fetch_intraday_minutes(code: str, today_ymd: str, start_time: str, end_time: str, k: int) -> pd.DataFrame:
    """
    返回今天 [start_time, end_time] 的分钟K
    """
    start_dt = pd.to_datetime(f"{ymd_to_dash(today_ymd)} {start_time}")
    end_dt = pd.to_datetime(f"{ymd_to_dash(today_ymd)} {end_time}")

    df = pd.DataFrame()
    errs = []

    if PREFER_EASTMONEY:
        try:
            df = _fetch_minute_eastmoney(code, start_dt, end_dt, k)
        except Exception as e:
            errs.append(f"eastmoney_err={repr(e)}")

    if df is None or df.empty:
        try:
            df = _fetch_minute_sina(code, k)
        except Exception as e:
            errs.append(f"sina_err={repr(e)}")

    if df is None or df.empty:
        raise RuntimeError("minute empty; " + ("; ".join(errs) if errs else "minute empty"))

    df = df[(df["dt"] >= start_dt) & (df["dt"] <= end_dt)].copy()
    df = df.sort_values("dt").reset_index(drop=True)
    return df


# ===================== 策略：分钟监控逻辑 =====================
def intraday_decision(code: str, today_ymd: str, L: float, stop: float) -> tuple[str, dict]:
    now = pd.Timestamp.now()
    end = now.strftime("%H:%M:%S")

    df = fetch_intraday_minutes(code, today_ymd, "09:30:00", end, MIN_INTERVAL)
    if df.empty:
        return "NO_DATA", {"note": "intraday empty"}

    last_px = float(df.iloc[-1]["close"])
    last_dt = str(df.iloc[-1]["dt"])

    win_start = pd.to_datetime(f"{ymd_to_dash(today_ymd)} {ENTRY_WINDOW_START}")
    win_end   = pd.to_datetime(f"{ymd_to_dash(today_ymd)} {ENTRY_WINDOW_END}")
    win = df[(df["dt"] >= win_start) & (df["dt"] < win_end)].copy()

    touched = False
    touch_time = ""
    if not win.empty:
        m = (win["low"].astype(float) <= L) & (win["high"].astype(float) >= L)
        if m.any():
            touched = True
            touch_time = str(win.loc[m].iloc[0]["dt"])

    if not touched:
        return "WAIT", {"last_dt": last_dt, "last_px": last_px, "note": f"not touch L={L:.3f} in 09:30-10:30"}

    # 止损：保持你之前逻辑（用 last close）
    if last_px <= stop:
        return "SELL(stop)", {"touch_time": touch_time, "last_dt": last_dt, "last_px": last_px, "stop": stop}

    return "BUY@L / HOLD", {"touch_time": touch_time, "last_dt": last_dt, "last_px": last_px, "stop": stop}


def read_manual_inputs() -> list[tuple[str, str]]:
    print("请输入要监测的股票（每行: 6位代码 空格 t日yyyyMMdd），输入空行结束。")
    print("例如：600481 20260126")
    pairs = []
    while True:
        line = input("> ").strip()
        if not line:
            break
        parts = line.split()
        if len(parts) < 2:
            print("格式不对：必须是 '代码 空格 t日'")
            continue
        pairs.append((code6(parts[0]), ymd8(parts[1])))
    return pairs


def main():
    today_ymd = pd.Timestamp.now().strftime("%Y%m%d")
    close_dt = pd.to_datetime(f"{ymd_to_dash(today_ymd)} {MARKET_CLOSE_TIME}")

    pairs = read_manual_inputs()
    if not pairs:
        print("没有输入任何股票，退出。")
        return

    rows = []
    errors = []

    # ====== 准备阶段：算 t-1/t/t+1 + L/stop + name/industry，并做过滤 ======
    for code, t_in in pairs:
        try:
            if is_excluded_prefix(code):
                raise RuntimeError(f"code前缀被过滤（688/300/9）：{code}")

            t_dt = pd.to_datetime(t_in, format="%Y%m%d", errors="coerce")
            if pd.isna(t_dt):
                raise RuntimeError(f"t格式错误：{t_in}")

            start = (t_dt - pd.Timedelta(days=DAILY_LOOKBACK_DAYS)).strftime("%Y%m%d")
            end = (t_dt + pd.Timedelta(days=60)).strftime("%Y%m%d")
            daily = fetch_daily_ak(code, start, end)
            if daily.empty:
                raise RuntimeError("daily empty (ak.stock_zh_a_hist)")

            tminus1, t_trade, tplus1 = resolve_t_triplet(daily, t_in)
            dmap = daily.set_index("trade_date")

            L = float(dmap.loc[t_trade, "low"])
            stop = INIT_STOP_MULT * L

            name, industry = fetch_name_industry(code)
            if ENABLE_EXCLUDE_ST and is_st_name(name):
                raise RuntimeError(f"ST股票被过滤：name={name}")

            rows.append({
                "code": code,
                "name": name,
                "industry": industry,
                "tminus1": tminus1,
                "t": t_trade,
                "tplus1": tplus1,
                "L": L,
                "stop": stop,
                "monitor_today": (tplus1 == today_ymd),
                "action": "INIT",
                "touch_time": "",
                "last_dt": "",
                "last_px": np.nan,
                "note": ""
            })
        except Exception as e:
            errors.append({"code": code, "t_in": t_in, "stage": "prep", "err": repr(e)})

    df_rt = pd.DataFrame(rows)
    df_err = pd.DataFrame(errors)

    out_path = safe_excel_path(OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_rt.to_excel(w, index=False, sheet_name="realtime")
        df_err.to_excel(w, index=False, sheet_name="errors")
    print(f"[Init saved] {out_path}  watchlist={len(df_rt)}  errors={len(df_err)}")

    if df_rt.empty:
        print("全部都被过滤/或准备失败，退出。请看 errors sheet。")
        return

    tick_cnt = 0

    # ====== 实时循环 ======
    while True:
        now = pd.Timestamp.now()
        if now >= close_dt + pd.Timedelta(minutes=1):
            print("[EOD] market closed, stop.")
            break

        tick_cnt += 1

        actions, touches, last_dts, last_pxs, notes = [], [], [], [], []
        for _, r in df_rt.iterrows():
            try:
                if ONLY_MONITOR_IF_TPLUS1_IS_TODAY and (not bool(r["monitor_today"])):
                    actions.append("NOT_TPLUS1")
                    touches.append("")
                    last_dts.append("")
                    last_pxs.append(np.nan)
                    notes.append(f"today={today_ymd} != t+1={r['tplus1']}")
                    continue

                action, info = intraday_decision(r["code"], today_ymd, float(r["L"]), float(r["stop"]))
                actions.append(action)
                touches.append(info.get("touch_time",""))
                last_dts.append(info.get("last_dt",""))
                last_pxs.append(info.get("last_px", np.nan))
                notes.append(info.get("note",""))
            except Exception as e:
                actions.append("ERR")
                touches.append("")
                last_dts.append("")
                last_pxs.append(np.nan)
                notes.append("")
                errors.append({"code": r.get("code",""), "stage": "intraday", "err": repr(e)})

        df_rt["action"] = actions
        df_rt["touch_time"] = touches
        df_rt["last_dt"] = last_dts
        df_rt["last_px"] = last_pxs
        df_rt["note"] = notes

        # 你的排序逻辑保留
        pri = {"SELL(stop)": 0, "BUY@L / HOLD": 1, "WAIT": 2, "NOT_TPLUS1": 3, "NO_DATA": 4, "ERR": 5}
        df_rt["_pri"] = df_rt["action"].map(pri).fillna(9)
        df_rt = df_rt.sort_values(["_pri","code"]).drop(columns=["_pri"]).reset_index(drop=True)

        out_path = safe_excel_path(OUTPUT_XLSX)
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df_rt.to_excel(w, index=False, sheet_name="realtime")
            pd.DataFrame(errors).to_excel(w, index=False, sheet_name="errors")

        print(f"[Tick] {now.strftime('%H:%M:%S')} saved={out_path}")

        # ====== 新增：控制台直接看结果（保留Excel输出） ======
        if PRINT_CONSOLE and (tick_cnt % PRINT_EVERY_N_TICKS == 0):
            print_console_snapshot(df_rt, today_ymd, now)

        time.sleep(POLL_SEC)

    out_path = safe_excel_path(OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_rt.to_excel(w, index=False, sheet_name="realtime")
        pd.DataFrame(errors).to_excel(w, index=False, sheet_name="errors")
    print("Saved final:", out_path)


if __name__ == "__main__":
    main()
