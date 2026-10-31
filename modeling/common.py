"""建模与调度仿真的公共配置、数据读取和通用工具。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
CLEAN_CSV = OUTPUT_DIR / "rides_clean.csv"
MODEL_DIR = OUTPUT_DIR / "models"

PERIOD_BINS = [-1, 5, 9, 16, 19, 23]
PERIOD_LABELS = ["凌晨", "早高峰", "平峰", "晚高峰", "夜间"]
RANDOM_STATE = 42

TARGET_COLUMN = "demand"

# 这些列是模型真正读取的特征。会员/车辆结构使用“前一日滞后值”和
# “仅由训练期估计的画像值”，当前小时的构成不会进入特征，避免标签泄漏。
FEATURE_COLUMNS = [
    "hour",
    "weekday",
    "is_weekend",
    "day_of_month",
    "hour_sin",
    "hour_cos",
    "weekday_sin",
    "weekday_cos",
    "is_am_peak",
    "is_pm_peak",
    "is_night",
    "station_lat",
    "station_lng",
    "station_demand_profile",
    "station_hour_demand_profile",
    "member_ratio_profile",
    "electric_ratio_profile",
    "member_ratio_lag_24h",
    "electric_ratio_lag_24h",
    "demand_lag_1h",
    "demand_lag_24h",
    "demand_lag_168h",
    "demand_rolling_24h",
    "city_demand_lag_1h",
    "city_demand_lag_24h",
]


def normalize_station_key(series: pd.Series) -> pd.Series:
    """把站点编号统一成无小数点后缀的字符串，兼容 CSV 的 float 推断。"""
    result = series.astype("string").str.strip()
    result = result.str.replace(r"\.0+$", "", regex=True)
    return result.mask(result.isin(["", "nan", "None", "<NA>"]))


def hour_to_period(hours: pd.Series | np.ndarray) -> np.ndarray:
    """沿用 A 模块的时段口径。"""
    values = np.asarray(hours, dtype=int)
    return np.select(
        [values <= 5, values <= 9, values <= 16, values <= 19],
        PERIOD_LABELS[:4],
        default=PERIOD_LABELS[4],
    )


def load_clean_data() -> pd.DataFrame:
    """读取 A 的清洗表，只保留有起点站点编号的可建模记录。"""
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {CLEAN_CSV}，请先运行 python -m clean.clean 或根目录 pipeline.py"
        )

    df = pd.read_csv(
        CLEAN_CSV,
        dtype={
            "ride_id": "string",
            "rideable_type": "string",
            "member_casual": "string",
            "start_station_id": "string",
            "end_station_id": "string",
            "start_station_name": "string",
        },
        parse_dates=["started_at", "ended_at"],
    )
    required = {
        "ride_id",
        "started_at",
        "start_station_id",
        "start_station_name",
        "start_lat",
        "start_lng",
        "member_casual",
        "rideable_type",
    }
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"清洗表缺少建模必需字段: {sorted(missing)}")

    total = len(df)
    df = df.loc[df["start_station_id"].notna() & df["started_at"].notna()].copy()
    df = df.loc[df["member_casual"].notna() & df["rideable_type"].notna()].copy()
    df["start_station_id"] = normalize_station_key(df["start_station_id"])
    df = df.loc[df["start_station_id"].notna()].copy()
    df["member_casual"] = df["member_casual"].str.strip().str.lower()
    df["rideable_type"] = df["rideable_type"].str.strip().str.lower()
    df["started_at"] = pd.to_datetime(df["started_at"], errors="coerce")
    df = df.loc[df["started_at"].notna()].copy()

    print(
        f"[建模数据] 清洗表 {total} 行，保留起点站点/时间/类型有效的 "
        f"{len(df)} 行（{len(df) / total:.1%}），站点 {df['start_station_id'].nunique()} 个"
    )
    return df.reset_index(drop=True)


def save_csv(df: pd.DataFrame, name: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    df.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """两点球面距离，单位 km。"""
    radius = 6371.0088
    phi1, phi2 = np.deg2rad(lat1), np.deg2rad(lat2)
    d_phi = np.deg2rad(lat2 - lat1)
    d_lambda = np.deg2rad(lon2 - lon1)
    a = np.sin(d_phi / 2) ** 2 + np.cos(phi1) * np.cos(phi2) * np.sin(d_lambda / 2) ** 2
    return float(2 * radius * np.arcsin(np.sqrt(a)))