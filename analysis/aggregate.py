"""基础聚合统计：按时段、星期和日期统计骑行量。

输入：output/rides_clean.csv（清洗后的分析表）
输出：output/stats_*.csv、output/aggregation_report.txt
运行：python -m analysis.aggregate
"""

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = REPO_ROOT / "output"
CLEAN_CSV = OUTPUT_DIR / "rides_clean.csv"

WEEKDAYS = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]

# 时段划分：凌晨 0-5，早高峰 6-9，平峰 10-16，晚高峰 17-19，夜间 20-23
PERIOD_BINS = [-1, 5, 9, 16, 19, 23]
PERIOD_LABELS = ["凌晨", "早高峰", "平峰", "晚高峰", "夜间"]


def load_clean() -> pd.DataFrame:
    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"找不到 {CLEAN_CSV}，请先在 05-标记异常 分支运行 python -m clean.clean"
        )

    df = pd.read_csv(
        CLEAN_CSV,
        dtype={"ride_id": str, "rideable_type": str, "member_casual": str},
        parse_dates=["started_at", "ended_at"],
    )

    df["date"] = df["started_at"].dt.normalize()
    df["hour"] = df["started_at"].dt.hour
    df["weekday_idx"] = df["started_at"].dt.day_of_week
    df["weekday"] = pd.Categorical(
        df["weekday_idx"].map(dict(enumerate(WEEKDAYS))),
        categories=WEEKDAYS,
        ordered=True,
    )
    df["period"] = pd.Categorical(
        pd.cut(df["hour"], bins=PERIOD_BINS, labels=PERIOD_LABELS),
        categories=PERIOD_LABELS,
        ordered=True,
    )

    # 计数用的 0/1 标记，便于一次性聚合出分类型骑行量
    df["member_trips"] = df["member_casual"].eq("member").astype(int)
    df["casual_trips"] = df["member_casual"].eq("casual").astype(int)
    df["electric_trips"] = df["rideable_type"].eq("electric_bike").astype(int)
    df["classic_trips"] = df["rideable_type"].eq("classic_bike").astype(int)
    return df


def _aggregate(df: pd.DataFrame, key: str) -> pd.DataFrame:
    grouped = df.groupby(key, observed=True)
    out = pd.DataFrame(
        {
            "trips": grouped.size(),
            "member_trips": grouped["member_trips"].sum(),
            "casual_trips": grouped["casual_trips"].sum(),
            "electric_trips": grouped["electric_trips"].sum(),
            "classic_trips": grouped["classic_trips"].sum(),
            "avg_duration_sec": grouped["duration_sec"].mean().round(1),
        }
    ).reset_index()
    out["share"] = (out["trips"] / len(df)).round(4)
    return out


def run() -> dict[str, pd.DataFrame]:
    df = load_clean()
    OUTPUT_DIR.mkdir(exist_ok=True)

    by_period = _aggregate(df, "period")
    by_weekday = _aggregate(df, "weekday")
    by_hour = _aggregate(df, "hour")

    # 按日期汇总成时序，补齐没有骑行的日期，保证时间轴完整
    daily = _aggregate(df, "date")
    full_dates = pd.date_range(df["date"].min(), df["date"].max(), freq="D", name="date")
    daily = daily.set_index("date").reindex(full_dates)
    for col in ["trips", "member_trips", "casual_trips", "electric_trips", "classic_trips"]:
        daily[col] = daily[col].fillna(0).astype(int)
    daily["share"] = (daily["trips"] / len(df)).round(4)
    daily = daily.reset_index()

    outputs = {
        "stats_by_period.csv": by_period,
        "stats_by_weekday.csv": by_weekday,
        "stats_by_hour.csv": by_hour,
        "stats_daily.csv": daily,
    }
    for name, table in outputs.items():
        table.to_csv(OUTPUT_DIR / name, index=False, encoding="utf-8-sig")

    peak_day = daily.loc[daily["trips"].idxmax()]
    peak_hour = by_hour.loc[by_hour["trips"].idxmax()]
    peak_weekday = by_weekday.loc[by_weekday["trips"].idxmax()]
    peak_period = by_period.loc[by_period["trips"].idxmax()]

    lines = [
        "共享单车 2024-01 聚合统计报告",
        f"统计记录数: {len(df)}",
        f"统计区间: {df['date'].min():%Y-%m-%d} ~ {df['date'].max():%Y-%m-%d}",
        f"日均骑行量: {daily['trips'].mean():.1f}",
        f"单日最高: {peak_day['date']:%Y-%m-%d}，{int(peak_day['trips'])} 次",
        f"最忙时段: {peak_period['period']}，{int(peak_period['trips'])} 次"
        f"（占 {peak_period['share']:.1%}）",
        f"最忙小时: {int(peak_hour['hour'])} 点，{int(peak_hour['trips'])} 次",
        f"最忙星期: {peak_weekday['weekday']}，{int(peak_weekday['trips'])} 次"
        f"（占 {peak_weekday['share']:.1%}）",
    ]
    (OUTPUT_DIR / "aggregation_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("\n".join(lines))
    print(f"\n统计结果已输出到: {OUTPUT_DIR}")
    return outputs


if __name__ == "__main__":
    run()
