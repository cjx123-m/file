"""空间分析：站点经纬度映射、热力网格、骑行 OD 起终点流向。

输入：output/rides_clean.csv（A 的清洗表）
输出：output/stations.csv、output/heatmap_grid.csv、output/od_flows.csv、output/spatial_report.txt
运行：python -m analysis.spatial
"""

import pandas as pd

from .spatial_common import (
    OUTPUT_DIR,
    build_stations,
    haversine_km,
    load_clean,
    save_csv,
)

GRID_SIZE = 0.004  # 热力网格边长（度），约 400 m，兼顾分辨率与图面可读性
OD_MIN_RIDES = 20  # OD 对最小骑行量阈值，过滤长尾噪声
TOP_N = 10


def build_heatmap_grid(df: pd.DataFrame) -> pd.DataFrame:
    """把起终点坐标落进规则网格，得到可直接绘图的计数表。"""
    cells = []
    for side, col in (("start", "start_trips"), ("end", "end_trips")):
        cell = pd.DataFrame(
            {
                "lat": (df[f"{side}_lat"] // GRID_SIZE) * GRID_SIZE,
                "lng": (df[f"{side}_lng"] // GRID_SIZE) * GRID_SIZE,
            }
        )
        cells.append(cell.groupby(["lat", "lng"]).size().rename(col))

    grid = pd.concat(cells, axis=1).fillna(0).reset_index()
    grid[["lat", "lng"]] = grid[["lat", "lng"]].round(4)
    # 转成网格中心点，绘图时直接当作像元坐标
    grid["lat"] = grid["lat"] + GRID_SIZE / 2
    grid["lng"] = grid["lng"] + GRID_SIZE / 2
    for col in ("start_trips", "end_trips"):
        grid[col] = grid[col].astype(int)
    grid["total_trips"] = grid["start_trips"] + grid["end_trips"]
    return grid.sort_values("total_trips", ascending=False, ignore_index=True)


def build_od(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """按起点站点 -> 终点站点聚合，得到 OD 流向表（含直线距离与骑行时长）。"""
    od = (
        df.groupby(["start_key", "end_key"], as_index=False)
        .agg(rides=("ride_id", "size"), avg_duration_sec=("duration_sec", "mean"))
        .rename(columns={"start_key": "start_station_key", "end_key": "end_station_key"})
    )

    coords = stations[["station_key", "station_name", "lat", "lng"]]
    od = od.merge(
        coords.rename(
            columns={
                "station_key": "start_station_key",
                "station_name": "start_station_name",
                "lat": "start_lat",
                "lng": "start_lng",
            }
        ),
        on="start_station_key",
        how="left",
    )
    od = od.merge(
        coords.rename(
            columns={
                "station_key": "end_station_key",
                "station_name": "end_station_name",
                "lat": "end_lat",
                "lng": "end_lng",
            }
        ),
        on="end_station_key",
        how="left",
    )

    od["distance_km"] = haversine_km(
        od["start_lat"], od["start_lng"], od["end_lat"], od["end_lng"]
    ).round(3)
    od["avg_duration_sec"] = od["avg_duration_sec"].round(1)
    od["same_station"] = od["start_station_key"] == od["end_station_key"]

    od = od.sort_values("rides", ascending=False, ignore_index=True)
    return od


def run() -> dict[str, pd.DataFrame]:
    df = load_clean()
    stations = build_stations(df)
    grid = build_heatmap_grid(df)
    od_all = build_od(df, stations)
    od = od_all.loc[od_all["rides"] >= OD_MIN_RIDES].reset_index(drop=True)

    save_csv(stations, "stations.csv")
    save_csv(grid, "heatmap_grid.csv")
    save_csv(od, "od_flows.csv")

    # 距离统计：同站借还（原地换车）距离恒为 0，单独剥离，不参与平均距离
    travel = od_all.loc[~od_all["same_station"]].sort_values("distance_km")
    weight = travel["rides"] / travel["rides"].sum()
    avg_distance = float((travel["distance_km"] * weight).sum())
    median_distance = float(travel.loc[weight.cumsum() >= 0.5, "distance_km"].iloc[0])

    top_start = stations.nlargest(TOP_N, "outflow")
    top_end = stations.nlargest(TOP_N, "inflow")
    top_od = od.head(TOP_N)

    lines = [
        "共享单车 2024-01 空间分析报告",
        f"分析记录数: {len(df)}（占清洗表 {len(df) / df.attrs['clean_rows']:.1%}，"
        "仅保留起终点都有站点编号、能落到具体桩点的记录）",
        f"桩点数: {len(stations)} 个",
        f"坐标范围: 纬度 {stations['lat'].min():.4f} ~ {stations['lat'].max():.4f}，"
        f"经度 {stations['lng'].min():.4f} ~ {stations['lng'].max():.4f}",
        f"热力网格: {GRID_SIZE:.3f}°（约 {GRID_SIZE * 111_000:.0f} m），"
        f"有骑行落点的网格 {len(grid)} 个",
        f"平均骑行直线距离: {avg_distance:.2f} km，中位数 {median_distance:.2f} km",
        f"同站借还占比: {od_all['same_station'].mul(od_all['rides']).sum() / od_all['rides'].sum():.1%}",
        f"OD 对总数: {len(od_all)} 个，骑行量 ≥ {OD_MIN_RIDES} 次的高频 OD 对 {len(od)} 个",
        "",
        f"Top {TOP_N} 起点站点（借车量）:",
    ]
    for rank, row in enumerate(top_start.itertuples(), start=1):
        lines.append(f"  {rank:2d}. {row.station_name}：{row.outflow} 次，净流入 {row.net_inflow:+d}")

    lines.append(f"\nTop {TOP_N} 终点站点（还车量）:")
    for rank, row in enumerate(top_end.itertuples(), start=1):
        lines.append(f"  {rank:2d}. {row.station_name}：{row.inflow} 次，净流入 {row.net_inflow:+d}")

    lines.append(f"\nTop {TOP_N} 高频 OD 走廊（图上用箭头可视化）:")
    for rank, row in enumerate(top_od.itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.start_station_name} -> {row.end_station_name}："
            f"{row.rides} 次，{row.distance_km:.2f} km，平均 {row.avg_duration_sec / 60:.1f} 分钟"
        )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "spatial_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"空间分析结果已输出到: {OUTPUT_DIR}")
    return {"stations": stations, "heatmap_grid": grid, "od_flows": od}


if __name__ == "__main__":
    run()
