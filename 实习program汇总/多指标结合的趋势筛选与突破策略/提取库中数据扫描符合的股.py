import re
from pathlib import Path
from decimal import Decimal, ROUND_HALF_UP

import numpy as np
import pandas as pd
from tqdm import tqdm

# ===================== 你要改的配置 =====================
INPUT_CSV  = Path(r"C:\Users\24106\Desktop\daily_bar.csv")          # <-- 改成你的CSV路径
OUTPUT_XLSX = Path(r"D:\candidates_scan_from_csv.xlsx")

ASOF_DATE = pd.Timestamp.now().strftime("%Y%m%d")      # 扫描截止日（<=该日）
LOOKBACK_DAYS = 90                                     # 只用近 N 天数据（按自然日粗截）
SCAN_LAST_N_DAYS = 10                                  # 回扫最近 N 个 t（交易日）
MAX_CANDIDATES = 800

# ===================== 入场条件（不改思路） =====================
BEAR_BODY_TH = -0.03            # t 实体跌：close/open - 1 <= -3%
VOL_MULT = 3.0                  # t 成交量 >= t-1 * 3

# 过热过滤（三者满足其一就不入）
USE_CALENDAR_LOOKBACK = True
LU_WINDOW_DAYS = 5
LU_MIN_CNT = 2.5
RUNUP_WINDOW_DAYS = 10
RUNUP_TH = 0.30
T_BODY_DROP_FILTER_TH = -0.09   # t 实体<=-9% 不入

# 输出关键价
INIT_STOP_MULT = 0.90           # 初始止损 = 0.9*L

# 硬过滤
ENABLE_EXCLUDE_PREFIX = True
EXCLUDE_PREFIXES = ("688", "9")  # 688/300/9 跳过
ENABLE_EXCLUDE_ST = True                # 需要CSV有 name 列才可靠

# ===================== 工具函数 =====================
def code6(x) -> str:
    s = re.sub(r"\D", "", str(x).strip())
    return s.zfill(6)

def is_excluded_prefix(code6_str: str) -> bool:
    return code6(code6_str).startswith(EXCLUDE_PREFIXES)

def is_st_name(name: str) -> bool:
    """
    名称里含 ST/*ST/ＳＴ 都判定为 ST
    """
    if not name or str(name).strip().lower() == "nan":
        return False
    s = str(name).strip().upper()
    s = s.replace(" ", "")
    s = s.replace("＊", "*")     # 全角星号
    s = s.replace("ＳＴ", "ST")  # 全角ST
    return ("ST" in s)           # 覆盖 ST / *ST

def round_2(x: float) -> float:
    # 用 Decimal 四舍五入到 0.01（更稳）
    return float(Decimal(str(x)).quantize(Decimal("0.01"), rounding=ROUND_HALF_UP))

def infer_limit_ratio(code6_str: str, is_st: bool) -> float:
    """
    常见涨跌幅限制（不考虑新股首日等特例）：
    - ST：5%
    - 300/688：20%
    - 北交所 8/4：30%
    - 其他：10%
    """
    c = code6(code6_str)
    if is_st:
        return 0.05
    if c.startswith(("300", "688")):
        return 0.20
    if c.startswith(("8", "4")):
        return 0.30
    return 0.10

def limit_up_price(prev_close: float, ratio: float) -> float:
    return round_2(prev_close * (1.0 + ratio))

def is_strict_limit_up(code6_str: str, is_st: bool, prev_close: float, close_tm1: float) -> bool:
    """
    严格涨停：t-1 收盘价 == 按 t-2 收盘与涨幅限制计算出来的涨停价
    """
    if not np.isfinite(prev_close) or prev_close <= 0:
        return False
    lu = limit_up_price(prev_close, infer_limit_ratio(code6_str, is_st))
    return abs(float(close_tm1) - lu) <= 1e-6

def _slice_by_calendar_days_include_end(daily: pd.DataFrame, end_trade_date: str, window_days: int) -> pd.DataFrame:
    if daily.empty:
        return daily.iloc[0:0]
    end_dt = pd.to_datetime(end_trade_date, format="%Y%m%d", errors="coerce")
    if pd.isna(end_dt):
        return daily.iloc[0:0]
    win_start = end_dt - pd.Timedelta(days=window_days - 1)
    dts = pd.to_datetime(daily["trade_date"], format="%Y%m%d", errors="coerce")
    return daily.loc[(dts >= win_start) & (dts <= end_dt)].copy()

def overheat_filter(daily: pd.DataFrame, idx_t: int, t: str, tminus1: str) -> tuple[bool, str]:
    row_t = daily.iloc[idx_t]
    ot = float(row_t["open"]); ct = float(row_t["close"])
    body_t = (ct / ot - 1.0) if ot > 0 else np.nan
    hit3 = (np.isfinite(body_t) and body_t <= T_BODY_DROP_FILTER_TH)

    w_lu = _slice_by_calendar_days_include_end(daily, tminus1, LU_WINDOW_DAYS) if USE_CALENDAR_LOOKBACK else daily.iloc[max(0, idx_t-6):idx_t].copy()
    w_ru = _slice_by_calendar_days_include_end(daily, tminus1, RUNUP_WINDOW_DAYS) if USE_CALENDAR_LOOKBACK else daily.iloc[max(0, idx_t-11):idx_t].copy()

    hit1 = False; lu_cnt = np.nan
    if not w_lu.empty:
        # 过热过滤仍用你原来的“收盘=最高”计数（不改思路）
        lu_cnt = int((w_lu["close"].astype(float) == w_lu["high"].astype(float)).sum())
        hit1 = (lu_cnt >= LU_MIN_CNT)

    hit2 = False; runup = np.nan
    if len(w_ru) >= 2:
        c0 = float(w_ru.iloc[0]["close"]); c1 = float(w_ru.iloc[-1]["close"])
        if c0 > 0:
            runup = c1 / c0 - 1.0
            hit2 = (np.isfinite(runup) and runup > RUNUP_TH)

    if hit1 or hit2 or hit3:
        parts = []
        if hit1: parts.append(f"limitup_cnt={lu_cnt}>=3")
        if hit2: parts.append(f"runup={runup:.2%}>30%")
        if hit3: parts.append(f"t_body={body_t:.2%}<=-9%")
        return True, " | ".join(parts)
    return False, f"ok(t_body={body_t:.2%})"

# ===================== 主逻辑 =====================
def main():
    if not INPUT_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {INPUT_CSV}")

    df = pd.read_csv(INPUT_CSV)

    # 期望至少有这些列
    need_cols = {"code","trade_date","open","high","low","close","volume"}
    miss = need_cols - set(df.columns)
    if miss:
        raise RuntimeError(f"CSV 缺列: {sorted(miss)}；至少需要 {sorted(need_cols)}")

    # 标准化
    df["code"] = df["code"].apply(code6)
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False).str.slice(0, 8)

    for c in ["open","high","low","close","volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna(subset=["code","trade_date","open","high","low","close"]).copy()

    # 只取 lookback 窗口（粗截：按自然日）
    end_dt = pd.to_datetime(ASOF_DATE, format="%Y%m%d", errors="coerce")
    if pd.isna(end_dt):
        raise RuntimeError(f"ASOF_DATE 不合法: {ASOF_DATE}")
    start_dt = end_dt - pd.Timedelta(days=LOOKBACK_DAYS)
    dts = pd.to_datetime(df["trade_date"], format="%Y%m%d", errors="coerce")
    df = df.loc[(dts >= start_dt) & (dts <= end_dt)].copy()

    if df.empty:
        raise RuntimeError("CSV 在指定窗口内没有数据。")

    has_name = "name" in df.columns
    if ENABLE_EXCLUDE_ST and (not has_name):
        print("[WARN] 你开了 ENABLE_EXCLUDE_ST=True，但 CSV 没有 name 列，无法可靠剔除ST；将自动跳过ST过滤。")

    rows = []
    errs = []

    # 关键：保证全局排序（即使CSV乱序也不怕）
    df = df.sort_values(["code","trade_date"]).reset_index(drop=True)

    for code, g in tqdm(df.groupby("code", sort=False), desc="Scan CSV"):
        try:
            if ENABLE_EXCLUDE_PREFIX and is_excluded_prefix(code):
                continue

            daily = g.sort_values("trade_date").reset_index(drop=True)
            if len(daily) < 8:
                continue

            # name/industry（如果有）
            name = ""
            industry = ""
            if "name" in daily.columns:
                tmp = daily["name"].dropna()
                name = str(tmp.iloc[-1]).strip() if len(tmp) else ""
            if "industry" in daily.columns:
                tmp = daily["industry"].dropna()
                industry = str(tmp.iloc[-1]).strip() if len(tmp) else ""

            # ST 硬过滤（如果能做）
            if ENABLE_EXCLUDE_ST and has_name and is_st_name(name):
                continue

            dates = daily["trade_date"].astype(str).tolist()
            tail = dates[-(SCAN_LAST_N_DAYS + 2):]  # 留 t-1、t+1
            dmap = daily.set_index("trade_date")

            for j in range(1, len(tail)-1):
                t = tail[j]
                tminus1 = tail[j-1]
                tplus1 = tail[j+1]

                if t not in dmap.index or tminus1 not in dmap.index:
                    continue

                row_t = dmap.loc[t]
                row_tm1 = dmap.loc[tminus1]

                # ===== 条件1：t-1 严格涨停（用涨停价）=====
                # 需要 t-2 收盘
                pos_tm1_arr = np.where(daily["trade_date"].astype(str).values == tminus1)[0]
                if len(pos_tm1_arr) == 0:
                    continue
                pos_tm1 = int(pos_tm1_arr[0])
                if pos_tm1 - 1 < 0:
                    continue
                prev_close = float(daily.iloc[pos_tm1 - 1]["close"])
                close_tm1 = float(row_tm1["close"])

                # 如果你未来要保留ST但按5%判定，把 is_st 改成 is_st_name(name)
                is_st = False
                if not is_strict_limit_up(code, is_st=is_st, prev_close=prev_close, close_tm1=close_tm1):
                    continue

                # ===== 条件2：t 大阴线 =====
                o = float(row_t["open"])
                c = float(row_t["close"])
                if o <= 0 or not np.isfinite(o) or not np.isfinite(c):
                    continue
                body = c / o - 1.0
                if not (body <= BEAR_BODY_TH):
                    continue

                # ===== 条件3：t 放量 =====
                v_t = float(row_t["volume"]) if pd.notna(row_t["volume"]) else np.nan
                v_tm1 = float(row_tm1["volume"]) if pd.notna(row_tm1["volume"]) else np.nan
                if not (np.isfinite(v_t) and np.isfinite(v_tm1) and v_tm1 > 0 and v_t >= VOL_MULT * v_tm1):
                    continue

                # ===== 过热过滤 =====
                idx_t_arr = np.where(daily["trade_date"].astype(str).values == t)[0]
                if len(idx_t_arr) == 0:
                    continue
                idx_t = int(idx_t_arr[0])

                hit, why = overheat_filter(daily, idx_t, t, tminus1)
                if hit:
                    continue

                L = float(row_t["low"])
                stop = INIT_STOP_MULT * L

                rows.append({
                    "asof_date": ASOF_DATE,
                    "code": code,
                    "name": name,
                    "industry": industry,
                    "tminus1": tminus1,
                    "t": t,
                    "tplus1": tplus1,
                    "L": L,
                    "stop": stop,
                    "body_t": body,
                    "vol_mult": (v_t / v_tm1),
                    "note": f"ok | body={body:.2%} vol_mult={v_t/v_tm1:.2f} | overheat={why}"
                })

        except Exception as e:
            errs.append({"code": code, "stage": "scan", "err": repr(e)})

    out = pd.DataFrame(rows)
    if not out.empty:
        out["_score"] = (-out["body_t"].astype(float)) * 1.0 + (out["vol_mult"].astype(float)) * 0.2
        out = out.sort_values(["t", "_score"], ascending=[False, False]).drop(columns=["_score"]).reset_index(drop=True)
        if MAX_CANDIDATES and len(out) > MAX_CANDIDATES:
            out = out.head(MAX_CANDIDATES).copy()

    OUTPUT_XLSX.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(OUTPUT_XLSX, engine="openpyxl") as w:
        out.to_excel(w, index=False, sheet_name="candidates")
        pd.DataFrame(errs).to_excel(w, index=False, sheet_name="errors")

    print("Saved:", OUTPUT_XLSX, "candidates=", len(out), "errors=", len(errs))

if __name__ == "__main__":
    main()
