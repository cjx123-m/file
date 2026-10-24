# 基础聚合统计

基于 `05-标记异常` 分支产出的 `output/rides_clean.csv`，对骑行量做基础聚合统计。

## 统计内容

| 输出文件 | 内容 |
| --- | --- |
| `stats_by_period.csv` | 按时段（凌晨/早高峰/平峰/晚高峰/夜间）统计骑行量 |
| `stats_by_weekday.csv` | 按星期（周一至周日）统计骑行量 |
| `stats_by_hour.csv` | 按 0–23 点统计骑行量 |
| `stats_daily.csv` | 按日期统计骑行量时序 |
| `aggregation_report.txt` | 关键指标汇总 |

每张表都包含总骑行量、会员/非会员拆分、电动车/经典车拆分和平均时长。

## 运行

```bash
python -m analysis.aggregate
```
