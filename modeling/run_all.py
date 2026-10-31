"""建模与调度子流水线入口：特征 -> 训练评估 -> 预测图 -> 调度测算。"""

from __future__ import annotations

from typing import Any

from . import dispatch, features, train, visualize


def run(quick: bool = False, test_days: int = 7) -> dict[str, Any]:
    print("\n" + "=" * 72)
    print("C/D 需求预测、影响因子建模与调度投放测算")
    print("=" * 72)
    panel, metadata = features.build_and_save(test_days=test_days)
    training = train.run(panel, metadata, quick=quick)
    figures = visualize.run(
        training["metrics"], training["predictions"], training["importance"]
    )
    dispatch_result = dispatch.run()
    return {
        "features": panel,
        "training": training,
        "figures": figures,
        "dispatch": dispatch_result,
    }


if __name__ == "__main__":
    run()