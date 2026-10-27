"""供需失衡量化：计算各站点供需差，识别矛盾突出的点位与时段。

口径：
- 需求 demand = 借出量 outflow（用户把车骑走）
- 供给 supply = 还入量 inflow（用户把车骑回来）
- 供需差 gap = 借出 - 还入；gap > 0 表示该点位车不够用（缺车），
  gap < 0 表示车堆积（淤积），gap 为 0 表示当期内自平衡
- 失衡率 = |gap| / (借出 + 还入)，用于横向比较不同量级站点

输入：output/rides_clean.csv（A 的清洗表）
输出：output/station_imbalance.csv、output/station_period_imbalance.csv、
      output/imbalance_hotspots.csv、output/imbalance_report.txt
运行：python -m analysis.imbalance
"""

import numpy as np
import pandas as pd

from .aggregate import PERIOD_LABELS
from .spatial_common import (
    OUTPUT_DIR,
    build_stations,
    load_clean,
    save_csv,
    station_period_flow,
)

# 分级阈值（次/月）。站点月均借还量在百级，15/40/80 能区分出明显矛盾点位
MILD_GAP = 15
CLEAR_GAP = 40
SEVERE_GAP = 80
MIN_FLOW = 30  # 参与矛盾排行的最小流量，避免 3 单里 2 单的小样本放大
TOP_N = 15


def grade(gap: pd.Series) -> pd.DataFrame:
    """按供需差绝对值分级，并按符号标出方向。"""
    abs_gap = gap.abs()
    level = np.select(
        [abs_gap >= SEVERE_GAP, abs_gap >= CLEAR_GAP, abs_gap >= MILD_GAP],
        ["严重", "明显", "轻度"],
        default="基本平衡",
    )
    direction = np.select([gap >= MILD_GAP, gap <= -MILD_GAP], ["缺车", "淤积"], default="平衡")
    return pd.DataFrame({"direction": direction, "level": level}, index=gap.index)


def _with_metrics(table: pd.DataFrame) -> pd.DataFrame:
    table = table.copy()
    table["gap"] = table["outflow"] - table["inflow"]
    table["abs_gap"] = table["gap"].abs()
    table["imbalance_rate"] = (table["abs_gap"] / table["total_flow"].clip(lower=1)).round(3)
    table = pd.concat([table, grade(table["gap"])], axis=1)
    table["rank"] = table["abs_gap"].rank(ascending=False, method="min").astype(int)
    return table.sort_values(["abs_gap", "total_flow"], ascending=False, ignore_index=True)


def station_imbalance(stations: pd.DataFrame) -> pd.DataFrame:
    """全月站点供需差。"""
    table = stations[
        ["station_key", "station_name", "lat", "lng", "outflow", "inflow", "total_flow"]
    ]
    return _with_metrics(table)


def period_imbalance(flow: pd.DataFrame) -> pd.DataFrame:
    """站点 × 时段供需差，直接定位“哪个点位在哪个时段失衡”。"""
    table = flow[
        ["station_key", "station_name", "lat", "lng", "period", "outflow", "inflow", "total_flow"]
    ]
    return _with_metrics(table).sort_values(
        ["period", "abs_gap"], ascending=[True, False], ignore_index=True
    )


def _direction_counts(table: pd.DataFrame, direction: str) -> dict:
    mask = table["direction"] == direction
    levels = table.loc[mask, "level"].value_counts()
    return {
        "total": int(mask.sum()),
        "severe": int(levels.get("严重", 0)),
        "clear": int(levels.get("明显", 0)),
        "mild": int(levels.get("轻度", 0)),
    }


def run() -> dict[str, pd.DataFrame]:
    df = load_clean()
    stations = build_stations(df)
    flow = station_period_flow(df, stations)
    station_table = station_imbalance(stations)
    period_table = period_imbalance(flow)

    hotspots = (
        period_table.loc[period_table["total_flow"] >= MIN_FLOW]
        .nlargest(TOP_N, "abs_gap")
        .reset_index(drop=True)
    )
    hotspots.insert(0, "hotspot_rank", range(1, len(hotspots) + 1))

    save_csv(station_table, "station_imbalance.csv")
    save_csv(period_table, "station_period_imbalance.csv")
    save_csv(hotspots, "imbalance_hotspots.csv")

    short = _direction_counts(station_table, "缺车")
    long_ = _direction_counts(station_table, "淤积")
    balanced = int((station_table["direction"] == "平衡").sum())
    top_short = station_table.loc[station_table["direction"] == "缺车"].head(10)
    top_long = station_table.loc[station_table["direction"] == "淤积"].head(5)

    lines = [
        "共享单车 2024-01 供需失衡报告",
        "供需差口径: gap = 借出(outflow) - 还入(inflow)；gap > 0 表示缺车，gap < 0 表示淤积",
        f"分析记录数: {len(df)}（桩点口径，占清洗表 {len(df) / df.attrs['clean_rows']:.1%}），"
        f"桩点 {len(stations)} 个",
        f"分级阈值: 轻度 |gap| ≥ {MILD_GAP}、明显 ≥ {CLEAR_GAP}、严重 ≥ {SEVERE_GAP} 次/月；"
        f"矛盾排行要求当期流量 ≥ {MIN_FLOW} 次",
        "",
        "全月站点失衡结构:",
        f"  缺车站点 {short['total']} 个（严重 {short['severe']}、明显 {short['clear']}、"
        f"轻度 {short['mild']}），占 {short['total'] / len(station_table):.1%}",
        f"  淤积站点 {long_['total']} 个（严重 {long_['severe']}、明显 {long_['clear']}、"
        f"轻度 {long_['mild']}），占 {long_['total'] / len(station_table):.1%}",
        f"  基本平衡 {balanced} 个，占 {balanced / len(station_table):.1%}",
        "  注意：全月净差会把早高峰的流出与晚高峰的流入相互抵消，"
        "潮汐型站点的矛盾要落到“点位 × 时段”才看得出来（见下表）",
        "",
        "缺车最严重的点位 Top 10（早高峰前需要补车）:",
    ]
    for rank, row in enumerate(top_short.itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.station_name}（{row.station_key}）：缺车 {row.gap} 辆，"
            f"借出 {row.outflow} / 还入 {row.inflow}，失衡率 {row.imbalance_rate:.1%}，{row.level}"
        )

    lines.append("\n淤积最严重的点位 Top 5（晚高峰后需要回收）:")
    for rank, row in enumerate(top_long.itertuples(), start=1):
        lines.append(
            f"  {rank:2d}. {row.station_name}（{row.station_key}）：淤积 {abs(row.gap)} 辆，"
            f"借出 {row.outflow} / 还入 {row.inflow}，失衡率 {row.imbalance_rate:.1%}，{row.level}"
        )

    lines.append(f"\n供需矛盾最突出的点位 × 时段 Top {TOP_N}:")
    for row in hotspots.itertuples():
        lines.append(
            f"  {row.hotspot_rank:2d}. {row.period}｜{row.station_name}："
            f"{row.direction} {row.abs_gap} 辆，借出 {row.outflow} / 还入 {row.inflow}，"
            f"失衡率 {row.imbalance_rate:.1%}，{row.level}"
        )

    lines.append("\n各时段失衡结构:")
    for period in PERIOD_LABELS:
        sub = period_table.loc[period_table["period"] == period]
        if sub.empty:
            continue
        lines.append(
            f"  {period}: 缺车 {int((sub['direction'] == '缺车').sum())} 个、"
            f"淤积 {int((sub['direction'] == '淤积').sum())} 个，"
            f"最大缺口 {int(sub['abs_gap'].max())} 辆，"
            f"严重矛盾组合 {int((sub['level'] == '严重').sum())} 个"
        )

    peaking = hotspots.iloc[0]
    lines += [
        "",
        "结论提示:",
        f"  1. 全月 {short['total']} 个点位借多还少、{long_['total']} 个点位还多借少，"
        f"说明车辆在空间上持续单向迁移，需要按潮汐方向调度；",
        f"  2. 最突出的矛盾是 {peaking['period']} 的 {peaking['station_name']}"
        f"（{peaking['direction']} {int(peaking['abs_gap'])} 辆），建议在该时段前完成车辆腾挪；",
        f"  3. 缺车点位（如 {top_short['station_name'].iloc[0]}）与淤积点位"
        f"（如 {top_long['station_name'].iloc[0]}）在空间上互补，"
        "适合按“缺车点 -> 淤积点”配对做调度闭环。",
    ]

    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "imbalance_report.txt").write_text(report, encoding="utf-8")
    print(report)
    print(f"供需失衡结果已输出到: {OUTPUT_DIR}")
    return {
        "station_imbalance": station_table,
        "station_period_imbalance": period_table,
        "imbalance_hotspots": hotspots,
    }


if __name__ == "__main__":
    run()
