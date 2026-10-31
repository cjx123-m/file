"""完整项目流水线：A 清洗 -> A/B 时空分析 -> C 需求预测 -> D 调度测算。

示例：
    python pipeline.py
    python pipeline.py --skip-cleaning --skip-analysis
    python pipeline.py --quick
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

from analysis import aggregate, run_all as analysis_run_all
from clean import clean
from modeling import run_all as modeling_run_all
from modeling.common import CLEAN_CSV, OUTPUT_DIR

REQUIRED_B_OUTPUTS = (
    "station_period_flow.csv",
    "station_imbalance.csv",
    "tide_hourly.csv",
    "stations.csv",
)


def _require(name: str) -> None:
    path = OUTPUT_DIR / name
    if not path.exists():
        raise FileNotFoundError(
            f"缺少上游产物 {path}；请去掉 --skip-cleaning/--skip-analysis 后重跑"
        )


def run(skip_cleaning: bool = False, skip_analysis: bool = False, quick: bool = False) -> None:
    started = time.perf_counter()
    steps: list[str] = []

    if skip_cleaning:
        _require(CLEAN_CSV.name)
        steps.append("A 清洗: 复用现有 output/rides_clean.csv")
        print("[流水线] 复用 A 的清洗结果")
    else:
        print("\n" + "=" * 72)
        print("A 数据清洗")
        print("=" * 72)
        clean.run()
        steps.append("A 清洗: 完成")

    print("\n" + "=" * 72)
    print("A 基础聚合")
    print("=" * 72)
    aggregate.run()
    steps.append("A 基础聚合: 完成")

    if skip_analysis:
        for name in REQUIRED_B_OUTPUTS:
            _require(name)
        steps.append("B 空间/潮汐/失衡: 复用现有 output 产物")
        print("[流水线] 复用 B 的空间、潮汐与供需失衡结果")
    else:
        analysis_run_all.run()
        steps.append("B 空间/潮汐/失衡/可视化: 完成")

    modeling_result = modeling_run_all.run(
        quick=quick,
        test_days=7,
    )
    steps.append(f"C 需求预测: 最佳模型 {modeling_result['training']['best_name']}")
    steps.append("D 调度测算: 完成")

    elapsed = time.perf_counter() - started
    summary = modeling_result["dispatch"]["summary"]
    lines = [
        "完整项目流水线报告",
        "=" * 48,
        *steps,
        f"总用时: {elapsed:.1f} 秒",
        "",
        "调度情景汇总:",
        summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
    ]
    (OUTPUT_DIR / "pipeline_report.txt").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print("\n" + "\n".join(lines))
    print("\n[流水线完成] 结果目录: output/")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip-cleaning", action="store_true", help="复用现有清洗表")
    parser.add_argument("--skip-analysis", action="store_true", help="复用现有 B 模块产物")
    parser.add_argument("--quick", action="store_true", help="减少树模型迭代，用于快速联调")
    args = parser.parse_args()
    run(
        skip_cleaning=args.skip_cleaning,
        skip_analysis=args.skip_analysis,
        quick=args.quick,
    )


if __name__ == "__main__":
    main()