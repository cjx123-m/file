"""清洗管线的公共配置与工具。"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
RAW_CSV = REPO_ROOT / "题5_共享单车骑行_202401.csv"
OUTPUT_DIR = REPO_ROOT / "output"

RIDE_ID = "ride_id"
TIMESTAMP_COLUMNS = ["started_at", "ended_at"]
COORDINATE_COLUMNS = ["start_lat", "start_lng", "end_lat", "end_lng"]
STATION_NAME_COLUMNS = ["start_station_name", "end_station_name"]
STATION_ID_COLUMNS = ["start_station_id", "end_station_id"]
CATEGORY_COLUMNS = ["rideable_type", "member_casual"]

EXPECTED_CATEGORIES = {
    "rideable_type": {"classic_bike", "electric_bike"},
    "member_casual": {"member", "casual"},
}

# 关键字段：缺失后无法参与任何统计，只能删除整行
CRITICAL_COLUMNS = [
    "ride_id",
    "started_at",
    "ended_at",
    "start_lat",
    "start_lng",
    "end_lat",
    "end_lng",
]

# 华盛顿特区服务区外接矩形（含 Arlington / Alexandria 周边）
LAT_RANGE = (38.4, 39.4)
LNG_RANGE = (-77.6, -76.6)

MIN_RIDE_SECONDS = 60
LONG_RIDE_SECONDS = 6 * 3600
MAX_RIDE_SECONDS = 24 * 3600

UNKNOWN_STATION = "未知站点"


def read_raw() -> pd.DataFrame:
    """按文本读入原始数据，保留空串，避免 pandas 提前推断类型。"""
    if not RAW_CSV.exists():
        raise FileNotFoundError(f"找不到原始数据: {RAW_CSV}")
    return pd.read_csv(RAW_CSV, dtype=str, keep_default_na=False)


def log_step(title: str, before: int, after: int, note: str = "") -> None:
    removed = before - after
    suffix = f"；{note}" if note else ""
    print(f"[{title}] {before} -> {after} 行，去除 {removed} 行{suffix}")
