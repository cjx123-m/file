"""时空可视化：站点热力图、OD 流向图、潮汐曲线、站点失衡柱状图与点位×时段热力图。

只读取前面三个模块产出的聚合表，不再碰 55 MB 的明细数据，保证出图快且可复现。

输入：output/heatmap_grid.csv、output/stations.csv、output/od_flows.csv、
      output/tide_hourly.csv、output/station_imbalance.csv、
      output/station_period_imbalance.csv
输出：output/fig_heatmap.png、output/fig_od_flow.png、output/fig_tide_curve.png、
      output/fig_imbalance_bar.png、output/fig_imbalance_heatmap.png
运行：python -m analysis.visualize
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # 无界面环境下出图
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D

from .aggregate import PERIOD_LABELS
from .spatial_common import OUTPUT_DIR

GRID_SIZE = 0.004
DPI = 150

plt.rcParams["font.sans-serif"] = [
    "Microsoft YaHei",
    "SimHei",
    "Noto Sans SC",
    "DejaVu Sans",
]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["axes.grid"] = True
plt.rcParams["grid.alpha"] = 0.25


def _read(name: str, **kwargs) -> pd.DataFrame:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(f"找不到 {path}，请先运行对应的分析模块（spatial/tide/imbalance）")
    return pd.read_csv(path, encoding="utf-8-sig", **kwargs)


def _short(name: str, limit: int = 30) -> str:
    """图上的站点名做截断，避免长英文站名挤压坐标轴。"""
    return name if len(name) <= limit else name[: limit - 1] + "…"


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[出图] {path.name}")
    return path


def plot_heatmap(grid: pd.DataFrame) -> Path:
    """起点/终点落点热力图：对数色标，突出核心区的密度差异。"""
    lats = np.sort(grid["lat"].unique())
    lngs = np.sort(grid["lng"].unique())
    x_edges = np.append(lngs - GRID_SIZE / 2, lngs[-1] + GRID_SIZE / 2)
    y_edges = np.append(lats - GRID_SIZE / 2, lats[-1] + GRID_SIZE / 2)
    aspect = 1 / np.cos(np.deg2rad(float(np.mean(lats))))

    panels = [("start_trips", "起点（借车）"), ("end_trips", "终点（还车）")]
    values = grid[["start_trips", "end_trips"]].to_numpy()
    # 用 99.5 分位截断，避免个别超级热点把整张图压暗
    vmax = float(np.percentile(values[values > 0], 99.5))
    fig, axes = plt.subplots(1, 2, figsize=(13, 6.2), sharey=True)
    mesh = None
    for ax, (col, title) in zip(axes, panels):
        matrix = (
            grid.pivot_table(index="lat", columns="lng", values=col, aggfunc="sum")
            .reindex(index=lats, columns=lngs)
            .to_numpy()
        )
        mesh = ax.pcolormesh(
            x_edges,
            y_edges,
            np.ma.masked_less(matrix, 1),
            cmap="inferno",
            norm=LogNorm(vmin=1, vmax=vmax),
        )
        ax.set_title(f"{title} 落点热力（{GRID_SIZE:.3f}° 网格，约 {GRID_SIZE * 111_000:.0f} m）")
        ax.set_xlabel("经度")
        ax.set_aspect(aspect)
        ax.grid(False)

    axes[0].set_ylabel("纬度")
    cbar = fig.colorbar(mesh, ax=axes, pad=0.02, fraction=0.035)
    cbar.set_label("网格内骑行次数（对数色标）")
    fig.suptitle("2024-01 共享单车起终点空间热力图", fontsize=13)
    return _save(fig, "fig_heatmap.png")


def plot_od_flow(od: pd.DataFrame, stations: pd.DataFrame, top_n: int = 40) -> Path:
    """OD 起终点流向图：左图看全城分布，右图放大中心城区看走廊结构与方向。

    线宽与颜色都随骑行量变化，粗箭头标出最强的若干条走廊的终点方向。
    OD 走廊长度多在 1 km 量级，因此单独做放大子图，否则在全城尺度上看不清。
    """
    flows = od.loc[~od["same_station"]].nlargest(top_n, "rides").reset_index(drop=True)
    lo, hi = float(flows["rides"].min()), float(flows["rides"].max())
    widths = 0.4 + 1.6 * (flows["rides"] - lo) / max(hi - lo, 1)
    aspect = 1 / np.cos(np.deg2rad(float(stations["lat"].mean())))

    fig, (ax_all, ax_zoom) = plt.subplots(1, 2, figsize=(14, 7.4))
    segments = np.stack(
        [flows[["start_lng", "start_lat"]].to_numpy(), flows[["end_lng", "end_lat"]].to_numpy()],
        axis=1,
    )

    # 左：全城概览（线细、淡，只表达分布）
    ax_all.scatter(
        stations["lng"],
        stations["lat"],
        s=np.sqrt(stations["total_flow"]) * 0.7,
        c="#cbd5dd",
        edgecolors="none",
        zorder=1,
    )
    overview = LineCollection(
        segments,
        cmap="plasma",
        norm=plt.Normalize(lo, hi),
        array=flows["rides"].to_numpy(),
        linewidths=widths,
        alpha=0.7,
        zorder=2,
    )
    ax_all.add_collection(overview)
    ax_all.set_title("全城视角（灰点 = 全部桩点）", fontsize=10)
    ax_all.set_xlabel("经度")
    ax_all.set_ylabel("纬度")
    ax_all.set_aspect(aspect)
    ax_all.grid(False)

    # 右：中心城区放大（标注站名与方向箭头）
    zoom = flows.nlargest(15, "rides")
    ax_zoom.scatter(
        stations["lng"],
        stations["lat"],
        s=np.sqrt(stations["total_flow"]) * 1.2,
        c="#e3e9ee",
        edgecolors="none",
        zorder=1,
    )
    zoom_segments = np.stack(
        [zoom[["start_lng", "start_lat"]].to_numpy(), zoom[["end_lng", "end_lat"]].to_numpy()],
        axis=1,
    )
    detail = LineCollection(
        zoom_segments,
        cmap="plasma",
        norm=plt.Normalize(lo, hi),
        array=zoom["rides"].to_numpy(),
        linewidths=1.2 + 3.0 * (zoom["rides"] - lo) / max(hi - lo, 1),
        alpha=0.9,
        zorder=2,
    )
    ax_zoom.add_collection(detail)

    for row in zoom.head(8).itertuples():
        ax_zoom.annotate(
            "",
            xy=(row.end_lng, row.end_lat),
            xytext=(row.start_lng, row.start_lat),
            arrowprops=dict(
                arrowstyle="-|>", color="#08306b", lw=1.3, shrinkA=5, shrinkB=7, alpha=0.95
            ),
            zorder=3,
        )

    # 只标注最强 5 条走廊的端点，交替上下、靠右侧的站名朝左排，避免压到色标
    ends = pd.concat(
        [
            zoom.head(5)[["start_station_key", "start_station_name", "start_lat", "start_lng"]]
            .set_axis(["key", "name", "lat", "lng"], axis=1),
            zoom.head(5)[["end_station_key", "end_station_name", "end_lat", "end_lng"]]
            .set_axis(["key", "name", "lat", "lng"], axis=1),
        ]
    ).drop_duplicates("key")
    lng_vals = zoom[["start_lng", "end_lng"]].to_numpy()
    lat_vals = zoom[["start_lat", "end_lat"]].to_numpy()
    mid_lng = float(lng_vals.mean())
    offsets = [(12, 18), (-12, -22), (12, 34), (-12, -38), (12, 50), (-12, -54)]
    for i, row in enumerate(ends.itertuples()):
        dx, dy = offsets[i % len(offsets)]
        ax_zoom.annotate(
            _short(row.name, 30),
            (row.lng, row.lat),
            xytext=(dx, dy),
            textcoords="offset points",
            ha="right" if dx < 0 else "left",
            fontsize=7.5,
            color="#222222",
            bbox=dict(boxstyle="round,pad=0.18", fc="white", ec="none", alpha=0.85),
            arrowprops=dict(arrowstyle="-", color="#8c8c8c", lw=0.6, shrinkA=0, shrinkB=2),
            zorder=4,
        )

    pad_lng = (np.ptp(lng_vals) or 0.02) * 0.25
    pad_lat = (np.ptp(lat_vals) or 0.02) * 0.45
    ax_zoom.set_xlim(lng_vals.min() - pad_lng, lng_vals.max() + pad_lng)
    ax_zoom.set_ylim(lat_vals.min() - pad_lat, lat_vals.max() + pad_lat)
    ax_zoom.set_title(f"中心城区放大（Top {len(zoom)} 走廊，箭头指向终点）", fontsize=10)
    ax_zoom.set_xlabel("经度")
    ax_zoom.set_aspect(aspect)
    ax_zoom.grid(False)

    cbar = fig.colorbar(
        detail, ax=[ax_all, ax_zoom], location="bottom", pad=0.08, fraction=0.04, aspect=45
    )
    cbar.set_label(f"OD 骑行量（次，Top {top_n} 走廊）")
    fig.suptitle(
        f"2024-01 骑行 OD 起终点流向图（线宽/颜色 ∝ 骑行量；{lo:.0f}–{hi:.0f} 次）",
        fontsize=12,
    )
    return _save(fig, "fig_od_flow.png")


def plot_tide_curve(hourly: pd.DataFrame) -> Path:
    """时序潮汐曲线：借出量 + 净流入/净流出站点数 + 跨站迁移量。"""
    fig, (ax_trip, ax_tide) = plt.subplots(
        2, 1, figsize=(11, 8), sharex=True, height_ratios=[3, 2]
    )

    ax_trip.bar(hourly["hour"], hourly["trips"], color="#9ecae1", label="每小时借出量")
    ax_trip.set_ylabel("借出量（次）")

    ax_station = ax_trip.twinx()
    ax_station.plot(
        hourly["hour"],
        hourly["net_out_stations"],
        color="#d62728",
        lw=1.8,
        marker="o",
        ms=3,
        label="净流出站点数（借多于还）",
    )
    ax_station.plot(
        hourly["hour"],
        hourly["net_in_stations"],
        color="#1f77b4",
        lw=1.8,
        marker="s",
        ms=3,
        label="净流入站点数（还多于借）",
    )
    ax_station.set_ylabel("站点数")
    ax_station.grid(False)

    for lo_hour, hi_hour, label, color in [
        (6, 9, "早高峰", "#fdd0a2"),
        (17, 19, "晚高峰", "#c6dbef"),
    ]:
        for ax in (ax_trip, ax_tide):
            ax.axvspan(lo_hour - 0.5, hi_hour + 0.5, color=color, alpha=0.5, zorder=0)
        ax_trip.annotate(
            label,
            xy=((lo_hour + hi_hour) / 2, 0.94),
            xycoords=("data", "axes fraction"),
            ha="center",
            fontsize=9,
            color="#7f2704",
        )

    handles, labels = ax_trip.get_legend_handles_labels()
    extra_handles, extra_labels = ax_station.get_legend_handles_labels()
    ax_trip.legend(
        handles + extra_handles, labels + extra_labels, loc="upper left", fontsize=8
    )

    bar_colors = [
        "#e6550d" if 6 <= h <= 9 else "#3182bd" if 17 <= h <= 19 else "#bdbdbd"
        for h in hourly["hour"]
    ]
    ax_tide.bar(hourly["hour"], hourly["tide_volume"], color=bar_colors, label="跨站迁移量")
    ax_tide.set_ylabel("跨站迁移量（次）")
    ax_tide.set_xlabel("小时（0-23）")
    ax_tide.set_xticks(range(24))
    ax_tide.annotate(
        f"峰值 {int(hourly['tide_volume'].max())} 次\n{int(hourly.loc[hourly['tide_volume'].idxmax(), 'hour'])} 点",
        xy=(int(hourly.loc[hourly["tide_volume"].idxmax(), "hour"]), hourly["tide_volume"].max()),
        xytext=(0, 8),
        textcoords="offset points",
        ha="center",
        fontsize=8,
        color="#a63603",
    )
    ax_tide.legend(loc="upper left", fontsize=8)

    fig.suptitle(
        "2024-01 潮汐曲线：早高峰净流出站点多、晚高峰净流入站点多，跨站迁移在 8 点与 17 点双峰",
        fontsize=12,
    )
    return _save(fig, "fig_tide_curve.png")


def plot_imbalance_bar(table: pd.DataFrame, top_n: int = 12) -> Path:
    """站点失衡柱状图：全月缺车最多的站点向正方向、淤积最多的向负方向。"""
    short = table.nlargest(top_n, "gap")
    surplus = table.nsmallest(top_n, "gap")
    bars = pd.concat([short, surplus]).sort_values("gap").reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(10.5, 8))
    colors = ["#d62728" if gap > 0 else "#1f77b4" for gap in bars["gap"]]
    ax.barh(np.arange(len(bars)), bars["gap"], color=colors, alpha=0.88)
    ax.set_yticks(np.arange(len(bars)))
    ax.set_yticklabels(
        [_short(f"{name}（{key}）", 34) for name, key in zip(bars["station_name"], bars["station_key"])],
        fontsize=8,
    )
    ax.axvline(0, color="#333333", lw=0.8)
    ax.set_xlabel("供需差 gap = 借出 - 还入（辆/月）；正值 = 缺车，负值 = 淤积")

    span = float(bars["gap"].abs().max())
    for i, (gap, level) in enumerate(zip(bars["gap"], bars["level"])):
        ax.text(
            gap + (span * 0.01 if gap > 0 else -span * 0.01),
            i,
            f"{int(gap):+d}（{level}）",
            va="center",
            ha="left" if gap > 0 else "right",
            fontsize=7.5,
            color="#7f2704" if gap > 0 else "#08306b",
        )

    ax.set_xlim(-span * 1.35, span * 1.35)
    ax.set_title(
        f"全月站点供需差 Top {top_n}（红 = 缺车需补车，蓝 = 淤积需回收）", fontsize=12
    )
    return _save(fig, "fig_imbalance_bar.png")


def plot_imbalance_heatmap(period_table: pd.DataFrame, top_n: int = 15) -> Path:
    """点位 × 时段供需差热力图：直接定位“哪个站点在哪个时段最失衡”。"""
    eligible = period_table.loc[period_table["total_flow"] >= 30]
    worst_keys = eligible.groupby("station_key")["abs_gap"].max().nlargest(top_n).index
    sub = eligible.loc[eligible["station_key"].isin(worst_keys)]

    matrix = (
        sub.pivot_table(index="station_key", columns="period", values="gap", aggfunc="sum")
        .reindex(columns=PERIOD_LABELS)
        .fillna(0)
    )
    order = matrix.abs().max(axis=1).sort_values(ascending=False).index
    matrix = matrix.loc[order]

    names = sub.drop_duplicates("station_key").set_index("station_key")["station_name"]
    labels = [_short(names.get(key, key), 30) for key in matrix.index]

    limit = float(np.abs(matrix.to_numpy()).max())
    fig, ax = plt.subplots(figsize=(9.5, 7.5))
    image = ax.imshow(
        matrix.to_numpy(),
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vcenter=0, vmin=-limit, vmax=limit),
        aspect="auto",
    )
    ax.set_xticks(np.arange(len(PERIOD_LABELS)))
    ax.set_xticklabels(PERIOD_LABELS)
    ax.set_yticks(np.arange(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.grid(False)

    threshold = limit * 0.35
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            value = int(matrix.iloc[i, j])
            ax.text(
                j,
                i,
                f"{value:+d}",
                ha="center",
                va="center",
                fontsize=7.5,
                color="white" if abs(value) > threshold else "#333333",
            )

    cbar = fig.colorbar(image, ax=ax, pad=0.02, fraction=0.04)
    cbar.set_label("供需差 gap（借出 - 还入）；红 = 缺车，蓝 = 淤积")
    ax.set_title(f"供需矛盾最突出的 {top_n} 个点位 × 时段（辆，流量 ≥ 30 次）", fontsize=12)
    return _save(fig, "fig_imbalance_heatmap.png")


def run() -> dict[str, Path]:
    grid = _read("heatmap_grid.csv")
    stations = _read("stations.csv", dtype={"station_key": str})
    od = _read("od_flows.csv", dtype={"start_station_key": str, "end_station_key": str})
    hourly = _read("tide_hourly.csv")
    station_table = _read("station_imbalance.csv", dtype={"station_key": str})
    period_table = _read(
        "station_period_imbalance.csv", dtype={"station_key": str, "period": str}
    )

    figures = {
        "heatmap": plot_heatmap(grid),
        "od_flow": plot_od_flow(od, stations),
        "tide_curve": plot_tide_curve(hourly),
        "imbalance_bar": plot_imbalance_bar(station_table),
        "imbalance_heatmap": plot_imbalance_heatmap(period_table),
    }

    print(f"\n共输出 {len(figures)} 张图，目录: {OUTPUT_DIR}")
    return figures


if __name__ == "__main__":
    run()
