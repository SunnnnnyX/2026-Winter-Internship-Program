import os
import re
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests


# ===================== 你要改的配置 =====================
OUTPUT_XLSX = Path(r"D:\realtime_manual_monitor.xlsx")

# iFinD HTTP
BASE = "https://quantapi.51ifind.com/api/v1"
ACCESS_TOKEN = os.environ.get("IFIND_ACCESS_TOKEN", "").strip()
if not ACCESS_TOKEN:
    raise RuntimeError("请先设置环境变量 IFIND_ACCESS_TOKEN（setx 后重启 Spyder/终端）")
HEADERS = {"Content-Type": "application/json", "access_token": ACCESS_TOKEN}

# 缓存（日线缓存，避免反复拉）
CACHE_DIR = Path(r"D:\ifind_cache_manual_realtime")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

# 分时粒度（没1分钟权限就改成 "5" 或 "15"）
HF_INTERVAL_MIN = "1"

# 刷新间隔（秒）
POLL_SEC = 60

# 监测窗口（触L入场）
ENTRY_WINDOW_START = "09:30:00"
ENTRY_WINDOW_END   = "10:30:00"

# 初始止损
INIT_STOP_MULT = 0.90

# 收盘停止
MARKET_CLOSE_TIME = "15:00:00"

# 只在 “今天==t+1” 时才做分时监测（强一致逻辑）。不想限制就改 False
ONLY_MONITOR_IF_TPLUS1_IS_TODAY = True

# 过滤：688/300/9 + ST（手动模式也照样帮你过滤）
EXCLUDE_PREFIXES = ("688", "300", "9")
ENABLE_EXCLUDE_ST = True

# 请求节奏
SLEEP = 0.12
RETRY = 3


# ===================== 工具函数 =====================
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

def to_ths_code(code6_str: str) -> str:
    c = code6(code6_str)
    if c.startswith(("6", "9")):
        return f"{c}.SH"
    if c.startswith(("0", "2", "3")):
        return f"{c}.SZ"
    if c.startswith(("4", "8")):
        return f"{c}.BJ"
    return f"{c}.SZ"

def is_excluded_prefix(code6_str: str) -> bool:
    c = code6(code6_str)
    return c.startswith(EXCLUDE_PREFIXES)

def is_st_name(name: str) -> bool:
    if not name:
        return False
    s = str(name).strip().upper().replace(" ", "")
    s = s.replace("＊", "*").replace("ＳＴ", "ST")
    return ("ST" in s)

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

def post_json(endpoint: str, payload: dict) -> dict:
    url = f"{BASE}/{endpoint.lstrip('/')}"
    last_err = None
    for _ in range(RETRY):
        try:
            r = requests.post(url, json=payload, headers=HEADERS, timeout=60)
            r.raise_for_status()
            js = r.json()
            if js.get("errorcode", 0) != 0:
                raise RuntimeError(f"errorcode={js.get('errorcode')} errmsg={js.get('errmsg')}")
            return js
        except Exception as e:
            last_err = repr(e)
            time.sleep(0.8)
    raise RuntimeError(f"[{endpoint}] failed: {last_err}")

def tables_to_df(js: dict) -> pd.DataFrame:
    tables = js.get("tables") or []
    out = []
    for t in tables:
        thscode = t.get("thscode")
        times = t.get("time")
        table = t.get("table") or {}
        df = pd.DataFrame(table)
        if times is not None and "time" not in df.columns:
            df.insert(0, "time", times)
        if thscode is not None and "ts_code" not in df.columns:
            df.insert(0, "ts_code", thscode)
        out.append(df)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()

def fetch_daily(ths_code: str, startdate: str, enddate: str) -> pd.DataFrame:
    fp = CACHE_DIR / "daily" / f"{ths_code.replace('.','_')}_{startdate}_{enddate}.csv.gz"
    if fp.exists():
        try:
            df = pd.read_csv(fp, compression="gzip")
            if not df.empty and "trade_date" in df.columns:
                return df
        except Exception:
            pass

    payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "startdate": startdate,
        "enddate": enddate,
        "functionpara": {"Fill": "Blank"},
    }
    js = post_json("cmd_history_quotation", payload)
    df = tables_to_df(js)
    time.sleep(SLEEP)

    if df.empty:
        return pd.DataFrame(columns=["trade_date","open","high","low","close","volume","amount"])

    df = df.rename(columns={"time":"trade_date"})
    df["trade_date"] = pd.to_datetime(df["trade_date"], errors="coerce").dt.strftime("%Y%m%d")
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["trade_date","open","high","low","close"]).sort_values("trade_date").reset_index(drop=True)
    df = df[["trade_date","open","high","low","close","volume","amount"]]

    fp.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(fp, index=False, encoding="utf-8-sig", compression="gzip")
    return df

def fetch_name_industry(ths_code: str) -> tuple[str, str]:
    # 尽量用 smart_stock_picking 一次拿到信息（字段不稳定，做容错）
    name = ""
    industry = ""
    try:
        js = post_json("smart_stock_picking", {"searchstring": f"{ths_code} 名称 所属行业", "searchtype": "stock"})
        df = tables_to_df(js)
        time.sleep(SLEEP)
        if df is not None and not df.empty:
            cols = list(df.columns)
            # name
            for pat in [r"名称", r"证券名称", r"name", r"简称"]:
                for c in cols:
                    if re.search(pat, str(c), flags=re.IGNORECASE):
                        v = str(df.iloc[0][c]).strip()
                        if v and v.lower() != "nan":
                            name = v
                            break
                if name:
                    break
            # industry
            for pat in [r"申万.*一级.*行业", r"一级行业", r"所属行业", r"行业名称", r"行业"]:
                for c in cols:
                    if re.search(pat, str(c)):
                        v = str(df.iloc[0][c]).strip()
                        if v and v.lower() != "nan" and len(v) <= 40:
                            industry = re.sub(r"\s+", " ", v)
                            break
                if industry:
                    break
    except Exception:
        pass
    return name, industry

def fetch_hf_min(ths_code: str, starttime: str, endtime: str, interval_min: str = "1") -> pd.DataFrame:
    payload = {
        "codes": ths_code,
        "indicators": "open,high,low,close,volume,amount",
        "starttime": starttime,
        "endtime": endtime,
        "functionpara": {"Interval": str(interval_min), "Fill": "Blank"},
    }
    js = post_json("high_frequency", payload)
    df = tables_to_df(js)
    time.sleep(SLEEP)

    if df.empty:
        return pd.DataFrame(columns=["dt","open","high","low","close","volume","amount"])

    df = df.rename(columns={"time":"dt"})
    df["dt"] = pd.to_datetime(df["dt"], errors="coerce")
    df = df.dropna(subset=["dt"]).copy()
    for c in ["open","high","low","close","volume","amount"]:
        df[c] = pd.to_numeric(df.get(c, np.nan), errors="coerce")
    df = df.dropna(subset=["open","high","low","close"]).sort_values("dt").reset_index(drop=True)
    return df[["dt","open","high","low","close","volume","amount"]]


def resolve_t_triplet(daily: pd.DataFrame, t_raw: str) -> tuple[str, str, str]:
    """
    给一个 t（你输入的大阴线那天），在日线里找到对应交易日，并返回 t-1、t、t+1（交易日序列）
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


def intraday_decision(ths_code: str, today_ymd: str, L: float, stop: float) -> tuple[str, dict]:
    start = f"{ymd_to_dash(today_ymd)} 09:30:00"
    end = pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")

    df = fetch_hf_min(ths_code, start, end, interval_min=HF_INTERVAL_MIN)
    if df.empty:
        return "NO_DATA", {"note": "intraday empty"}

    df = df.sort_values("dt").reset_index(drop=True)
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

    if last_px <= stop:
        return "SELL(stop)", {"touch_time": touch_time, "last_dt": last_dt, "last_px": last_px, "stop": stop}
    return "BUY@L / HOLD", {"touch_time": touch_time, "last_dt": last_dt, "last_px": last_px, "stop": stop}


def read_manual_inputs() -> list[tuple[str, str]]:
    """
    让你一行一行输入：code t(yyyymmdd)
    输入空行结束。
    """
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

    # 先把基础信息算出来（t-1/t/t+1/L/stop）
    for code, t_in in pairs:
        try:
            if is_excluded_prefix(code):
                raise RuntimeError(f"code前缀被过滤（688/300/9）：{code}")
            ths = to_ths_code(code)

            # 拉一段日线，保证含 t 前后
            t_dt = pd.to_datetime(t_in, format="%Y%m%d", errors="coerce")
            startdate = ymd_to_dash((t_dt - pd.Timedelta(days=120)).strftime("%Y%m%d"))
            enddate   = ymd_to_dash((t_dt + pd.Timedelta(days=60)).strftime("%Y%m%d"))
            daily = fetch_daily(ths, startdate, enddate)
            if daily.empty:
                raise RuntimeError("daily empty")

            tminus1, t_trade, tplus1 = resolve_t_triplet(daily, t_in)
            dmap = daily.set_index("trade_date")
            L = float(dmap.loc[t_trade, "low"])
            stop = INIT_STOP_MULT * L

            name, industry = fetch_name_industry(ths)
            if ENABLE_EXCLUDE_ST and is_st_name(name):
                raise RuntimeError(f"ST股票被过滤：name={name}")

            rows.append({
                "code": code,
                "ths_code": ths,
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
        print("全部都被过滤/或准备失败，退出。")
        return

    # 实时循环
    while True:
        now = pd.Timestamp.now()
        if now >= close_dt + pd.Timedelta(minutes=1):
            print("[EOD] market closed, stop.")
            break

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

                action, info = intraday_decision(r["ths_code"], today_ymd, float(r["L"]), float(r["stop"]))
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

        # SELL/BUY 排前
        pri = {"SELL(stop)": 0, "BUY@L / HOLD": 1, "WAIT": 2, "NOT_TPLUS1": 3, "NO_DATA": 4, "ERR": 5}
        df_rt["_pri"] = df_rt["action"].map(pri).fillna(9)
        df_rt = df_rt.sort_values(["_pri","code"]).drop(columns=["_pri"]).reset_index(drop=True)

        out_path = safe_excel_path(OUTPUT_XLSX)
        with pd.ExcelWriter(out_path, engine="openpyxl") as w:
            df_rt.to_excel(w, index=False, sheet_name="realtime")
            pd.DataFrame(errors).to_excel(w, index=False, sheet_name="errors")
        print(f"[Tick] {now.strftime('%H:%M:%S')} saved={out_path}")

        time.sleep(POLL_SEC)

    # 最后保存一次
    out_path = safe_excel_path(OUTPUT_XLSX)
    with pd.ExcelWriter(out_path, engine="openpyxl") as w:
        df_rt.to_excel(w, index=False, sheet_name="realtime")
        pd.DataFrame(errors).to_excel(w, index=False, sheet_name="errors")
    print("Saved final:", out_path)


if __name__ == "__main__":
    main()
