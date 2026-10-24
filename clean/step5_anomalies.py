"""第 5 步：先标记异常，再按规则剔除硬异常。"""

import pandas as pd

from .common import LAT_RANGE, LNG_RANGE, LONG_RIDE_SECONDS, MAX_RIDE_SECONDS, MIN_RIDE_SECONDS

RULES = {
    "flag_nonpositive_duration": "骑行时长<=0",
    "flag_short_duration": f"骑行时长<{MIN_RIDE_SECONDS}秒",
    "flag_long_duration": f"骑行时长>{LONG_RIDE_SECONDS // 3600}小时",
    "flag_over_24h": "骑行时长>24小时",
    "flag_bad_start_coord": "起点坐标越界",
    "flag_bad_end_coord": "终点坐标越界",
    "flag_out_of_month": "不在2024年1月",
}

# 无法解释为真实骑行的硬异常：直接剔除
HARD_FLAGS = [
    "flag_nonpositive_duration",
    "flag_over_24h",
    "flag_bad_start_coord",
    "flag_bad_end_coord",
    "flag_out_of_month",
]


def _coord_out_of_range(lat: pd.Series, lng: pd.Series) -> pd.Series:
    return (
        (lat < LAT_RANGE[0])
        | (lat > LAT_RANGE[1])
        | (lng < LNG_RANGE[0])
        | (lng > LNG_RANGE[1])
    )


def mark(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """返回（全部记录 + 标记，剔除硬异常后的分析表）。"""
    df = df.copy()
    duration = df["duration_sec"]

    df["flag_nonpositive_duration"] = duration <= 0
    df["flag_short_duration"] = (duration > 0) & (duration < MIN_RIDE_SECONDS)
    df["flag_long_duration"] = duration > LONG_RIDE_SECONDS
    df["flag_over_24h"] = duration > MAX_RIDE_SECONDS
    df["flag_bad_start_coord"] = _coord_out_of_range(df["start_lat"], df["start_lng"])
    df["flag_bad_end_coord"] = _coord_out_of_range(df["end_lat"], df["end_lng"])
    df["flag_out_of_month"] = ~df["started_at"].between("2024-01-01", "2024-02-01")

    # 逐条规则标记，保留全部判据，便于回溯为什么被标记
    reason = pd.Series("", index=df.index, dtype="string")
    for flag, label in RULES.items():
        reason = reason + df[flag].map({True: f"{label}；", False: ""})
    df["anomaly_reason"] = reason.str.rstrip("；")
    df["is_anomaly"] = df["anomaly_reason"] != ""

    for flag, label in RULES.items():
        print(f"[标记异常] {label}: {int(df[flag].sum())} 行")

    hard_mask = df[HARD_FLAGS].any(axis=1)
    cleaned = df.loc[~hard_mask].reset_index(drop=True)

    print(
        f"[标记异常] 标记 {int(df['is_anomaly'].sum())} 行；"
        f"按硬规则剔除 {int(hard_mask.sum())} 行，保留 {len(cleaned)} 行"
    )
    return df, cleaned
