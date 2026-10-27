"""一键运行空间分析流水线：契约校验 -> 空间分析 -> 潮汐 -> 供需失衡 -> 可视化。

任务 1-4 的执行顺序就是本模块的调用顺序，单跑某一步时用对应的
python -m analysis.spatial / tide / imbalance / visualize。

运行：python -m analysis.run_all
输出：output/ 下的全部表、报告与图片
"""

from pathlib import Path

from . import check_contract, imbalance, spatial, tide, visualize

STEPS = (
    ("0/4 清洗表字段契约校验", check_contract.run),
    ("1/4 空间分析（经纬度映射 / 热力网格 / OD 流向）", spatial.run),
    ("2/4 潮汐规律识别（早高峰流出 / 晚高峰流入）", tide.run),
    ("3/4 供需失衡量化（站点 × 时段）", imbalance.run),
    ("4/4 时空可视化（热力图 / 潮汐曲线 / 失衡柱状图）", visualize.run),
)


def run() -> dict[str, Path]:
    outputs: dict[str, Path] = {}
    for title, step in STEPS:
        print("\n" + "=" * 68)
        print(title)
        print("=" * 68)
        result = step()
        if isinstance(result, dict):
            outputs.update(result)
    print("\n全部步骤完成，结果目录: output/")
    return outputs


if __name__ == "__main__":
    run()
