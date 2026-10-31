"""空间、潮汐与供需分析的公共工具：加载清洗表、统一时段口径、构造站点映射。

输入：output/rides_clean.csv（A 在 05-标记异常 分支产出的分析表）
输出：无（供 spatial / tide / imbalance / visualize 复用）
运行：由上层模块导入，不单独运行
"""

from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import PERIOD_BINS, PERIOD_LABELS

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
CLEAN_CSV = OUTPUT_DIR / "rides_clean.csv"

UNKNOWN_STATION_NAME = "未知站点"
EARTH_RADIUS_KM = 6371.0088


def save_csv(table: pd.DataFrame, name: str) -> Path:
    """统一用 utf-8-sig 输出，Excel 直接打开不乱码（与清洗结果保持一致）。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    path = OUTPUT_DIR / name
    table.to_csv(path, index=False, encoding="utf-8-sig")
    return path


def load_clean() -> pd.DataFrame:
    """读入 A 的清洗表，补充小时/时段维度与起终点站点主键。

    只有两端都有站点编号的记录才能落到具体桩点，因此这里统一剔除
    “未知站点”记录（占 15% 左右，其坐标只到街区级、起终点常相同），
    保证热力图、OD 与供需统计都是桩点口径。剔除比例在报告里明示。
    """
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {CLEAN_CSV}，请先在 05-标记异常 分支运行 python -m clean.clean"
        )

    df = pd.read_csv(
        CLEAN_CSV,
        dtype={
            "ride_id": str,
            "rideable_type": str,
            "member_casual": str,
            "start_station_id": str,
            "end_station_id": str,
        },
        parse_dates=["started_at", "ended_at"],
    )

    total = len(df)
    mask = df["start_station_id"].notna() & df["end_station_id"].notna()
    df = df.loc[mask].reset_index(drop=True)

    df["hour"] = df["started_at"].dt.hour
    df["period"] = pd.Categorical(
        pd.cut(df["hour"], bins=PERIOD_BINS, labels=PERIOD_LABELS),
        categories=PERIOD_LABELS,
        ordered=True,
    )

    for side in ("start", "end"):
        df[f"{side}_key"] = df[f"{side}_station_id"]
        df[f"{side}_display"] = df[f"{side}_station_name"].where(
            df[f"{side}_station_name"] != UNKNOWN_STATION_NAME, df[f"{side}_key"]
        )

    print(
        f"[加载] 清洗表 {total} 行，剔除起终点缺站点编号的 {total - len(df)} 行"
        f"（保留 {len(df) / total:.1%}）；"
        f"起点桩点 {df['start_key'].nunique()} 个，终点桩点 {df['end_key'].nunique()} 个"
    )
    df.attrs["clean_rows"] = total
    return df


def station_coordinates(df: pd.DataFrame) -> pd.DataFrame:
    """把起终点两侧的记录合并成“站点 -> 名称 + 坐标”映射表。

    同一站点在不同记录中坐标会有米级抖动，取中位数作为站点坐标；
    一个编号对应多个站点名时按字典序取第一个，保证结果可复现。
    """
    points = pd.concat(
        [
            df[["start_key", "start_display", "start_lat", "start_lng"]].set_axis(
                ["key", "name", "lat", "lng"], axis=1
            ),
            df[["end_key", "end_display", "end_lat", "end_lng"]].set_axis(
                ["key", "name", "lat", "lng"], axis=1
            ),
        ],
        ignore_index=True,
    )

    coords = points.groupby("key", as_index=False).agg(
        lat=("lat", "median"), lng=("lng", "median")
    )

    named = points.loc[points["name"] != UNKNOWN_STATION_NAME, ["key", "name"]]
    named = named.drop_duplicates("key").sort_values(["key", "name"])
    coords = coords.merge(named, on="key", how="left")
    coords["name"] = coords["name"].fillna(coords["key"])
    return coords.sort_values("key", ignore_index=True)


def build_stations(df: pd.DataFrame) -> pd.DataFrame:
    """站点维表：名称、坐标、全月流出（借车）与流入（还车）量。"""
    stations = station_coordinates(df)
    outflow = df.groupby("start_key").size().rename("outflow")
    inflow = df.groupby("end_key").size().rename("inflow")

    stations = stations.merge(outflow, left_on="key", right_index=True, how="left")
    stations = stations.merge(inflow, left_on="key", right_index=True, how="left")
    for col in ("outflow", "inflow"):
        stations[col] = stations[col].fillna(0).astype(int)

    stations["total_flow"] = stations["outflow"] + stations["inflow"]
    stations["net_inflow"] = stations["inflow"] - stations["outflow"]
    return stations.rename(columns={"key": "station_key", "name": "station_name"})


def station_period_flow(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """站点 × 时段 的流出/流入，并补齐没有流量的组合（保证 5 个时段齐全）。"""
    keys = stations["station_key"].tolist()
    grid = pd.MultiIndex.from_product([keys, PERIOD_LABELS], names=["station_key", "period"])
    flow = grid.to_frame(index=False)
    flow["period"] = flow["period"].astype(str)

    outflow = (
        df.groupby(["start_key", "period"], observed=True)
        .size()
        .rename("outflow")
        .reset_index()
        .rename(columns={"start_key": "station_key"})
    )
    inflow = (
        df.groupby(["end_key", "period"], observed=True)
        .size()
        .rename("inflow")
        .reset_index()
        .rename(columns={"end_key": "station_key"})
    )
    for table in (outflow, inflow):
        table["period"] = table["period"].astype(str)

    flow = flow.merge(outflow, on=["station_key", "period"], how="left")
    flow = flow.merge(inflow, on=["station_key", "period"], how="left")
    for col in ("outflow", "inflow"):
        flow[col] = flow[col].fillna(0).astype(int)

    flow["net_inflow"] = flow["inflow"] - flow["outflow"]
    flow["abs_gap"] = flow["net_inflow"].abs()
    flow["total_flow"] = flow["inflow"] + flow["outflow"]
    flow["period"] = pd.Categorical(flow["period"], categories=PERIOD_LABELS, ordered=True)

    flow = flow.merge(
        stations[["station_key", "station_name", "lat", "lng"]],
        on="station_key",
        how="left",
    )
    return flow.sort_values(
        ["period", "abs_gap"], ascending=[True, False], ignore_index=True
    )


def haversine_km(
    lat1: pd.Series, lng1: pd.Series, lat2: pd.Series, lng2: pd.Series
) -> np.ndarray:
    """两点球面距离（公里），用于刻画 OD 的空间跨度。"""
    lat1, lng1, lat2, lng2 = (
        np.radians(np.asarray(v, dtype=float)) for v in (lat1, lng1, lat2, lng2)
    )
    dlat = lat2 - lat1
    dlng = lng2 - lng1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlng / 2) ** 2
    return 2 * EARTH_RADIUS_KM * np.arcsin(np.sqrt(a))
