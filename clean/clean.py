"""清洗管线入口。

运行：python -m clean.clean
输出：output/rides_marked.csv、output/rides_clean.csv、output/cleaning_report.txt
"""

from pandas import DataFrame

from . import (
    step1_dedupe,
    step2_normalize,
    step3_convert_types,
    step4_missing,
    step5_anomalies,
)
from .common import OUTPUT_DIR, read_raw


def run() -> DataFrame:
    df = read_raw()
    raw_rows = len(df)
    print(f"原始数据: {raw_rows} 行 x {df.shape[1]} 列\n")

    df = step1_dedupe.clean(df)
    after_dedupe = len(df)
    df = step2_normalize.clean(df)
    df = step3_convert_types.clean(df)
    df = step4_missing.clean(df)
    after_missing = len(df)
    marked, cleaned = step5_anomalies.mark(df)

    OUTPUT_DIR.mkdir(exist_ok=True)
    marked.to_csv(OUTPUT_DIR / "rides_marked.csv", index=False, encoding="utf-8-sig")
    cleaned.to_csv(OUTPUT_DIR / "rides_clean.csv", index=False, encoding="utf-8-sig")

    print("\n按会员类型统计骑行时长（秒，清洗后）:")
    summary = (
        cleaned.groupby("member_casual")["duration_sec"]
        .agg(["count", "mean", "median"])
        .round(1)
    )
    print(summary.to_string())

    report = OUTPUT_DIR / "cleaning_report.txt"
    with report.open("w", encoding="utf-8") as fh:
        fh.write("共享单车 2024-01 清洗报告\n")
        fh.write(f"原始行数: {raw_rows}\n")
        fh.write(f"去重后行数: {after_dedupe}\n")
        fh.write(f"补缺删缺后行数: {after_missing}\n")
        fh.write(f"异常标记行数: {int(marked['is_anomaly'].sum())}\n")
        fh.write(f"剔除硬异常后行数: {len(cleaned)}\n\n")
        fh.write(summary.to_string())
        fh.write("\n")

    print(f"\n已输出到: {OUTPUT_DIR}")
    return cleaned


if __name__ == "__main__":
    run()
