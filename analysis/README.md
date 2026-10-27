# 分析模块

分析代码按分支推进，每个模块只读上游 `output/` 的产物，互不改写对方数据。

## 06 基础聚合统计

基于 `05-标记异常` 分支产出的 `output/rides_clean.csv`，对骑行量做基础聚合统计。

| 输出文件 | 内容 |
| --- | --- |
| `stats_by_period.csv` | 按时段（凌晨/早高峰/平峰/晚高峰/夜间）统计骑行量 |
| `stats_by_weekday.csv` | 按星期（周一至周日）统计骑行量 |
| `stats_by_hour.csv` | 按 0–23 点统计骑行量 |
| `stats_daily.csv` | 按日期统计骑行量时序 |
| `aggregation_report.txt` | 关键指标汇总 |

每张表都包含总骑行量、会员/非会员拆分、电动车/经典车拆分和平均时长。

```bash
python -m analysis.aggregate
```

## 07 空间分析 · 潮汐规律 · 供需失衡

在同一份清洗表上完成空间分析、潮汐识别、供需失衡量化与时空可视化，
`python -m analysis.run_all` 一次跑完全部步骤。

### 数据口径（与 A 的清洗结果对接）

| 项 | 口径 |
| --- | --- |
| 时段 | 沿用 `analysis.aggregate` 的划分：凌晨 0-5、早高峰 6-9、平峰 10-16、晚高峰 17-19、夜间 20-23，按骑行开始时刻归属 |
| 站点主键 | `start_station_id` / `end_station_id`；站点坐标取该站点全部记录的经纬度中位数 |
| 记录筛选 | 仅保留起终点都有站点编号的记录（183,863 行，占清洗表 77.1%）。其余 54,608 行在清洗时站点为空、被填为“未知站点”，坐标只落到街区级且起终点常相同，会污染热力图与 OD 距离，故在站点级统计中剔除，比例在报告中明示 |
| 借出 / 还入 | 借出 outflow = 以该站为起点的骑行数；还入 inflow = 以该站为终点的骑行数 |
| 供需差 | gap = 借出 - 还入；gap > 0 表示缺车，gap < 0 表示淤积 |
| 潮汐方向 | 早高峰净流入 ≤ -10 且晚高峰净流入 ≥ +10 记为居住型，方向相反记为办公型，其余为潮汐不明显 |

### 模块与输出

| 模块 | 命令 | 输出 |
| --- | --- | --- |
| 字段契约校验（联调） | `python -m analysis.check_contract` | `contract_report.txt` |
| 1 空间分析 | `python -m analysis.spatial` | `stations.csv`、`heatmap_grid.csv`、`od_flows.csv`、`spatial_report.txt` |
| 2 潮汐规律 | `python -m analysis.tide` | `tide_hourly.csv`、`station_period_flow.csv`、`tide_hotspots.csv`、`tide_report.txt` |
| 3 供需失衡 | `python -m analysis.imbalance` | `station_imbalance.csv`、`station_period_imbalance.csv`、`imbalance_hotspots.csv`、`imbalance_report.txt` |
| 4 时空可视化 | `python -m analysis.visualize` | `fig_heatmap.png`、`fig_od_flow.png`、`fig_tide_curve.png`、`fig_imbalance_bar.png`、`fig_imbalance_heatmap.png` |
| 一键全流程 | `python -m analysis.run_all` | 以上全部 |

所有表都用 `utf-8-sig` 输出，Excel 直接打开不乱码。

### 图表说明

| 图 | 内容 |
| --- | --- |
| `fig_heatmap.png` | 起点/终点落点热力图（0.004°，约 444 m 网格，对数色标） |
| `fig_od_flow.png` | OD 流向图：左全城、右中心城区放大；线宽/颜色 ∝ 骑行量，箭头指向终点 |
| `fig_tide_curve.png` | 潮汐曲线：借出量与净流出/净流入站点数（上）、跨站迁移量（下），阴影为早晚高峰 |
| `fig_imbalance_bar.png` | 站点失衡柱状图：红 = 缺车、蓝 = 淤积，标注缺口与分级 |
| `fig_imbalance_heatmap.png` | 矛盾点位 × 时段供需差热力图，直接定位点位与时段 |

### 关键结论（2024-01）

- 空间：760 个桩点，骑行平均直线距离 1.82 km（中位数 1.49 km），同站借还仅占 3.8%；
  高频走廊集中在 Union Station / NoMa / Eastern Market 一带的 1 km 短途接驳。
- 潮汐：8 点净流出站点 337 个 vs 净流入 243 个，17 点方向反转（净流入 330 个）；
  跨站迁移峰值 7,338 次/小时，相当于该小时每单净迁移 0.50 辆。
- 类型：居住型 160 个、办公型 125 个、潮汐不明显 475 个站点。
- 失衡：全月 112 个点位缺车、104 个点位淤积；最突出矛盾为早高峰 `17th & G St NW`
  淤积 376 辆、晚高峰 `Columbus Circle / Union Station` 缺车 343 辆。

### 运行环境

`requirements.txt` 已包含 pandas、numpy、matplotlib。Windows 下 matplotlib 用
Microsoft YaHei / SimHei 渲染中文，若本机没有这两个字体，请在
`analysis/visualize.py` 的 `font.sans-serif` 列表里补上本机中文字体。
