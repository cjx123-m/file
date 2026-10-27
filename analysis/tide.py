"""潮汐规律识别：逐小时潮汐曲线、早高峰流出热点、晚高峰流入热点。

潮汐现象：早高峰从居住区骑往办公区，晚高峰反向回流。因此
居住型站点“早高峰净流出、晚高峰净流入”，办公型站点正好相反。
口径与 A 的聚合统计一致：按骑行开始时刻归属小时/时段，
“净流入 = 还入 - 借出”，正值表示车辆在该时段净淤积。

输入：output/rides_clean.csv（A 的清洗表）
输出：output/tide_hourly.csv、output/station_period_flow.csv、
      output/tide_hotspots.csv、output/tide_report.txt
运行：python -m analysis.tide
"""

import numpy as np
import pandas as pd

from .spatial_common import (
    OUTPUT_DIR,
    build_stations,
    load_clean,
    save_csv,
    station_period_flow,
)

MORNING = "早高峰"
EVENING = "晚高峰"
TIDE_MIN_GAP = 10  # 判定潮汐方向所需的最小净流量（次/月），过滤噪声
TOP_N = 15


def hourly_tide(df: pd.DataFrame, stations: pd.DataFrame) -> pd.DataFrame:
    """逐小时曲线：借车量、净流出/净流入站点数、跨站迁移量（潮汐强度）。"""
    hours = pd.Index(range(24), name="hour")
    keys = stations["station_key"]

    outflow = (
        df.groupby(["start_key", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=keys, columns=hours, fill_value=0)
    )
    inflow = (
        df.groupby(["end_key", "hour"])
        .size()
        .unstack(fill_value=0)
        .reindex(index=keys, columns=hours, fill_value=0)
    )
    net = inflow - outflow

    member = (
        df.loc[df["member_casual"].eq("member")]
        .groupby("hour")
        .size()
        .reindex(hours, fill_value=0)
    )

    hourly = pd.DataFrame(
        {
            "trips": outflow.sum(),
            "member_trips": member,
            "net_out_stations": (net < 0).sum(),
            "net_in_stations": (net > 0).sum(),
            "balanced_stations": (net == 0).sum(),
            # Σ|净流量| 的一半即“需要调度的车次数”，直接度量潮汐强度
            "tide_volume": (net.abs().sum() / 2).astype(int),
        }
    ).reset_index()
    hourly["member_share"] = (hourly["member_trips"] / hourly["trips"]).round(4)
    hourly["tide_intensity"] = (hourly["tide_volume"] / hourly["trips"]).round(3)
    return hourly


def _tide_type(hot: pd.DataFrame) -> pd.Series:
    """按早晚高峰净流入方向给站点分型。"""
    morning = hot["am_net_inflow"]
    evening = hot["pm_net_inflow"]
    residential = (morning <= -TIDE_MIN_GAP) & (evening >= TIDE_MIN_GAP)
    office = (morning >= TIDE_MIN_GAP) & (evening <= -TIDE_MIN_GAP)
    labels = np.select([residential, office], ["居住型", "办公型"], default="潮汐不明显")
    return pd.Series(
        pd.Categorical(labels, categories=["居住型", "办公型", "潮汐不明显"]),
        index=hot.index,
    )


def build_hotspots(flow: pd.DataFrame) -> pd.DataFrame:
    """站点级早晚高峰画像：早高峰流出、晚高峰流入与潮汐类型。"""
    am = flow.loc[flow["period"] == MORNING].set_index("station_key")
    pm = flow.loc[flow["period"] == EVENING].set_index("station_key")

    hotspots = pd.DataFrame(
        {
            "station_name": am["station_name"],
            "lat": am["lat"],
            "lng": am["lng"],
            "am_outflow": am["outflow"],
            "am_inflow": am["inflow"],
            "am_net_inflow": am["net_inflow"],
            "pm_outflow": pm["outflow"],
            "pm_inflow": pm["inflow"],
            "pm_net_inflow": pm["net_inflow"],
        }
    )
    hotspots["am_out_rank"] = hotspots["am_outflow"].rank(ascending=False, method="min").astype(int)
    hotspots["pm_in_rank"] = hotspots["pm_inflow"].rank(ascending=False, method="min").astype(int)
    hotspots["tide_strength"] = hotspots["am_net_inflow"].abs() + hotspots["pm_net_inflow"].abs()
    hotspots["tide_type"] = _tide_type(hotspots)
    hotspots = hotspots.reset_index()
    return hotspots.sort_values("tide_strength", ascending=False, ignore_index=True)


def _period_summary(flow: pd.DataFrame, period: str) -> dict:
    sub = flow.loc[flow["period"] == period]
    return {
        "trips": int(sub["outflow"].sum()),
        "net_out": int((sub["net_inflow"] < 0).sum()),
        "net_in": int((sub["net_inflow"] > 0).sum()),
        "volume": int(sub["abs_gap"].sum() / 2),
    }


def run() -> dict[str, pd.DataFrame]:
    df = load_clean()
    stations = build_stations(df)
    flow = station_period_flow(df, stations)
    hourly = hourly_tide(df, stations)
    hotspots = build_hotspots(flow)

    save_csv(hourly, "tide_hourly.csv")
    save_csv(flow, "station_period_flow.csv")
    save_csv(
        hotspots.assign(
            tide_type=hotspots["tide_type"].astype(str)
        ),
        "tide_hotspots.csv",
    )

    am = _period_summary(flow, MORNING)
    pm = _period_summary(flow, EVENING)
    peak_volume_hour = hourly.loc[hourly["tide_volume"].idxmax()]
    peak_trip_hour = hourly.loc[hourly["trips"].idxmax()]
    types = hotspots["tide_type"].value_counts()

    lines = [
        "共享单车 2024-01 潮汐规律报告",
        f"分析记录数: {len(df)}（桩点口径，占清洗表 {len(df) / df.attrs['clean_rows']:.1%}）",
        f"早高峰（6-9 点）: 借出 {am['trips']} 次；净流出站点 {am['net_out']} 个、"
        f"净流入站点 {am['net_in']} 个；跨站迁移 {am['volume']} 次"
        f"（占该时段 {am['volume'] / am['trips']:.1%}）",
        f"晚高峰（17-19 点）: 借出 {pm['trips']} 次；净流出站点 {pm['net_out']} 个、"
        f"净流入站点 {pm['net_in']} 个；跨站迁移 {pm['volume']} 次"
        f"（占该时段 {pm['volume'] / pm['trips']:.1%}）",
        f"潮汐最强小时: {int(peak_volume_hour['hour'])} 点，迁移 {int(peak_volume_hour['tide_volume'])} 次"
        f"（每单净迁移 {peak_volume_hour['tide_intensity']:.2f} 辆）",
        f"骑行量峰值小时: {int(peak_trip_hour['hour'])} 点，{int(peak_trip_hour['trips'])} 次",
        "",
        f"早高峰流出 Top {TOP_N} 站点（车辆被骑走最多，早高峰需要补车）:",
    ]
    top_am = hotspots.nlargest(TOP_N, "am_outflow")
    for rank, row in enumerate(top_am.itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.station_name}：流出 {row.am_outflow} 次，"
            f"净流入 {row.am_net_inflow:+d}"
        )

    lines.append(f"\n晚高峰流入 Top {TOP_N} 站点（车辆回流最多，晚高峰容易堆车）:")
    top_pm = hotspots.nlargest(TOP_N, "pm_inflow")
    for rank, row in enumerate(top_pm.itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.station_name}：流入 {row.pm_inflow} 次，"
            f"净流入 {row.pm_net_inflow:+d}"
        )

    lines.append("\n站点潮汐分型（阈值：早晚高峰净流量绝对值 ≥ %d 次/月）:" % TIDE_MIN_GAP)
    for tide_type in ["居住型", "办公型", "潮汐不明显"]:
        count = int(types.get(tide_type, 0))
        lines.append(f"  {tide_type}: {count} 个站点（占 {count / len(hotspots):.1%}）")

    lines.append(f"\n典型潮汐站点 Top {TOP_N}（早晚高峰净迁移量最大）:")
    for rank, row in enumerate(hotspots.head(TOP_N).itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.station_name}：{row.tide_type}，"
            f"早高峰净流入 {row.am_net_inflow:+d}，晚高峰净流入 {row.pm_net_inflow:+d}"
        )

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "tide_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"潮汐分析结果已输出到: {OUTPUT_DIR}")
    return {"tide_hourly": hourly, "station_period_flow": flow, "tide_hotspots": hotspots}


if __name__ == "__main__":
    run()
