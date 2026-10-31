# 共享单车需求预测与调度投放项目

本项目读取组员在 `06-基础聚合统计` / `07-空间分析与潮汐` 分支中已经完成的数据清洗与时空分析结果，
在独立分支 `codex/influence-factor-model` 上完成：

1. **影响因子与特征工程**：会员类型、车辆类型、小时/星期/高峰时段、站点空间与历史需求特征；
2. **需求回归模型**：均值基线、Ridge、RandomForest、HistGradientBoosting；
3. **模型评估与对比**：MAE、RMSE、R²、WAPE，输出预测效果图；
4. **调度投放测算**：结合 B 的站点×时段流入比例，模拟安全库存阈值 + 最近邻调车；
5. **完整流水线**：清洗 -> 聚合/空间/潮汐/失衡 -> 需求模型 -> 调度测算。

## 快速运行

安装依赖：

```bash
python -m pip install -r requirements.txt
```

完整流水线（默认重跑 A 清洗、B 分析和 C/D 建模）：

```bash
python pipeline.py
```

复用已有 A/B 产物，只跑建模与调度（开发联调最快）：

```bash
python pipeline.py --skip-cleaning --skip-analysis
```

快速冒烟测试：

```bash
python pipeline.py --skip-cleaning --skip-analysis --quick
```

也可以分模块运行：

```bash
python -m clean.clean
python -m analysis.aggregate
python -m analysis.run_all
python -m modeling.run_all
```

## 建模口径

- 需求目标：`站点 × 小时` 的借出骑行次数；无借出的站点小时补 0。
- 时间切分：前 24 天训练，最后 7 天测试，不随机打乱。
- 会员/车辆类型：使用训练期画像 `member_ratio_profile`、`electric_ratio_profile`，以及前一日同小时占比。
- 时段特征：小时、星期、周末、早晚高峰、夜间及周期正余弦编码。
- 历史需求：前一小时、前一日同小时、上周同小时、24 小时滚动均值。
- 防泄漏：站点/会员/车型画像只由训练期计算；当前小时的需求构成不进入特征。
- 评估为 one-step-ahead 回测：前一小时需求等滞后值在预测时已可观测。

## 主要输出

| 文件 | 内容 |
| --- | --- |
| `output/model_metrics.csv` | 各模型 MAE/RMSE/R²/WAPE |
| `output/model_report.txt` | 建模口径、模型对比与特征重要性 |
| `output/test_predictions.csv` | 测试期实际值/预测值 |
| `output/feature_importance.csv` | 影响因子重要性 |
| `output/fig_model_comparison.png` | 模型误差对比 |
| `output/fig_prediction_daily.png` | 逐日预测效果 |
| `output/fig_prediction_scatter.png` | 实际 vs 预测散点 |
| `output/fig_prediction_hourly.png` | 小时需求曲线 |
| `output/fig_feature_importance.png` | 影响因子排序 |
| `output/dispatch_orders.csv` | 模拟调度单 |
| `output/dispatch_summary.csv` | 不调度 vs 阈值调度 KPI |
| `output/dispatch_report.txt` | 调度参数、结果与限制 |
| `output/fig_dispatch_comparison.png` | 调度策略效果对比 |
| `output/pipeline_report.txt` | 完整流水线运行报告 |

## 调度测算说明

调度仿真消费：

- A 的清洗结果与需求预测；
- B 的 `output/station_period_flow.csv`（站点×时段流入/流出）；
- B 的 `output/stations.csv`（站点坐标）。

策略按小时检查预计库存，低于安全库存时从最近的高库存站点调车。默认参数（初始 3 辆、容量 20、安全库存 3、单车次最多 12 辆、每小时最多 80 辆）仅用于演示，不代表真实车队运营参数；生产化应替换为真实容量、库存、还车率、人员排班与路网时延。

## 分支说明

本次作业工作分支：`codex/influence-factor-model`，基于组员的 `origin/06-基础聚合统计`（其中已合并 `07-空间分析与潮汐`）。