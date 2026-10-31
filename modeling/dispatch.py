"""调度投放策略仿真：预测需求 + B 的站点时段流入比例 + 最近车辆调度。

这是可解释的测算演示，不代表真实运营参数：
- 每个小时开始，根据预测需求与历史还入比例估算小时末库存；
- 低于安全库存的站点向最近的高库存站点调车；
- 单次最多装 truck_capacity 辆，每小时最多移动 max_moves_per_hour 辆；
- 同一测试期分别运行“不调度”和“阈值 + 最近邻调度”两种情景。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (
    OUTPUT_DIR,
    haversine_km,
    hour_to_period,
    normalize_station_key,
    save_csv,
)

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


@dataclass(frozen=True)
class DispatchConfig:
    """演示参数，可在此调整以做敏感性分析。"""

    initial_inventory: float = 3.0
    capacity: float = 20.0
    safety_stock: float = 3.0
    donor_target_ratio: float = 0.75
    truck_capacity: float = 12.0
    max_moves_per_hour: float = 80.0
    max_distance_km: float = 8.0
    return_factor_floor: float = 0.25
    return_factor_cap: float = 2.5


def _read_inputs() -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    prediction_path = OUTPUT_DIR / "test_predictions.csv"
    flow_path = OUTPUT_DIR / "station_period_flow.csv"
    stations_path = OUTPUT_DIR / "stations.csv"
    for path in (prediction_path, flow_path, stations_path):
        if not path.exists():
            raise FileNotFoundError(f"找不到 {path}，请先完成 A/B 分析与需求模型")

    predictions = pd.read_csv(prediction_path, dtype={"station_key": "string"})
    predictions["station_key"] = normalize_station_key(predictions["station_key"])
    predictions["timestamp"] = pd.to_datetime(predictions["timestamp"], errors="coerce")
    predictions = predictions.loc[predictions["station_key"].notna()].copy()
    predictions["hour"] = predictions["timestamp"].dt.hour
    predictions["period"] = hour_to_period(predictions["hour"])

    flow = pd.read_csv(flow_path, dtype={"station_key": "string", "period": "string"})
    flow["station_key"] = normalize_station_key(flow["station_key"])
    flow["return_factor"] = (
        flow["inflow"] / flow["outflow"].clip(lower=1)
    ).clip(lower=0.05, upper=5.0)
    flow = flow[["station_key", "period", "outflow", "inflow", "return_factor"]]

    stations = pd.read_csv(
        stations_path, dtype={"station_key": "string"}, encoding="utf-8-sig"
    )
    stations["station_key"] = normalize_station_key(stations["station_key"])
    coords = {
        str(row.station_key): (float(row.lat), float(row.lng))
        for row in stations.itertuples(index=False)
        if pd.notna(row.lat) and pd.notna(row.lng)
    }

    frame = predictions.merge(flow, on=["station_key", "period"], how="left")
    frame["return_factor"] = (
        frame["return_factor"]
        .fillna(1.0)
        .clip(
            lower=DispatchConfig.return_factor_floor,
            upper=DispatchConfig.return_factor_cap,
        )
    )
    frame["prediction"] = frame["prediction"].clip(lower=0)
    frame["demand"] = frame["demand"].clip(lower=0)
    frame["predicted_return"] = frame["prediction"] * frame["return_factor"]
    frame["actual_return_proxy"] = frame["demand"] * frame["return_factor"]
    return frame, flow, coords


def _nearest_donor(
    receiver_key: str,
    donor_keys: list[str],
    coords: dict[str, tuple[float, float]],
) -> tuple[str | None, float]:
    if not donor_keys or receiver_key not in coords:
        return None, float("inf")
    lat, lon = coords[receiver_key]
    best_key = None
    best_distance = float("inf")
    for donor in donor_keys:
        if donor == receiver_key or donor not in coords:
            continue
        distance = haversine_km(lat, lon, coords[donor][0], coords[donor][1])
        if distance < best_distance:
            best_key, best_distance = donor, distance
    return best_key, best_distance


def _plan_dispatch(
    group: pd.DataFrame,
    inventory: dict[str, float],
    coords: dict[str, tuple[float, float]],
    config: DispatchConfig,
    timestamp: pd.Timestamp,
) -> list[dict[str, Any]]:
    """阈值触发 + 最近邻匹配；先处理缺口最大的站点。"""
    projected = {}
    receiver_need = {}
    donor_surplus = {}
    info = {}
    for row in group.itertuples(index=False):
        key = str(row.station_key)
        stock = inventory.get(key, config.initial_inventory)
        projected_end = stock - float(row.prediction) + float(row.predicted_return)
        projected[key] = projected_end
        need = max(0.0, config.safety_stock - projected_end)
        surplus = max(0.0, projected_end - config.capacity * config.donor_target_ratio)
        if need > 0.25:
            receiver_need[key] = need
        if surplus > 0.25:
            donor_surplus[key] = surplus
        info[key] = row

    orders: list[dict[str, Any]] = []
    moved_total = 0.0
    receivers = sorted(receiver_need, key=lambda key: receiver_need[key], reverse=True)
    for receiver in receivers:
        need = receiver_need[receiver]
        while need > 0.25 and moved_total < config.max_moves_per_hour:
            candidates = [key for key, amount in donor_surplus.items() if amount > 0.25]
            donor, distance = _nearest_donor(receiver, candidates, coords)
            if donor is None or distance > config.max_distance_km:
                break
            bikes = min(
                need,
                donor_surplus[donor],
                config.truck_capacity,
                config.max_moves_per_hour - moved_total,
            )
            if bikes <= 0.25:
                break
            donor_surplus[donor] -= bikes
            need -= bikes
            moved_total += bikes
            orders.append(
                {
                    "timestamp": timestamp,
                    "from_station_key": donor,
                    "to_station_key": receiver,
                    "bikes": bikes,
                    "distance_km": distance,
                    "receiver_projected_end_before": projected[receiver],
                    "receiver_need_before": receiver_need[receiver],
                }
            )
    return orders


def _simulate(
    frame: pd.DataFrame,
    coords: dict[str, tuple[float, float]],
    config: DispatchConfig,
    dispatch_enabled: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    keys = frame["station_key"].dropna().unique().tolist()
    inventory = {str(key): float(config.initial_inventory) for key in keys}
    records: list[dict[str, Any]] = []
    all_orders: list[dict[str, Any]] = []

    for timestamp, group in frame.groupby("timestamp", sort=True):
        group = group.sort_values("station_key")
        if dispatch_enabled:
            orders = _plan_dispatch(group, inventory, coords, config, timestamp)
            for order in orders:
                donor = order["from_station_key"]
                receiver = order["to_station_key"]
                bikes = order["bikes"]
                inventory[donor] = inventory.get(donor, config.initial_inventory) - bikes
                inventory[receiver] = inventory.get(receiver, config.initial_inventory) + bikes
            all_orders.extend(orders)

        for row in group.itertuples(index=False):
            key = str(row.station_key)
            stock_before = float(inventory.get(key, config.initial_inventory))
            actual_demand = float(row.demand)
            served = min(actual_demand, max(stock_before, 0.0))
            unmet = actual_demand - served
            stock_after_demand = max(stock_before - served, 0.0)

            returned = float(row.actual_return_proxy)
            accepted_returns = min(returned, max(config.capacity - stock_after_demand, 0.0))
            lost_returns = returned - accepted_returns
            stock_end = stock_after_demand + accepted_returns
            inventory[key] = stock_end
            records.append(
                {
                    "timestamp": timestamp,
                    "date": pd.Timestamp(timestamp).normalize(),
                    "station_key": key,
                    "predicted_demand": float(row.prediction),
                    "actual_demand": actual_demand,
                    "served_demand": served,
                    "unmet_demand": unmet,
                    "returned_demand": returned,
                    "lost_returns": lost_returns,
                    "stock_start": stock_before,
                    "stock_end": stock_end,
                }
            )
    return pd.DataFrame(records), pd.DataFrame(all_orders)


def _summary(metrics: pd.DataFrame, orders: pd.DataFrame, scenario: str) -> dict[str, Any]:
    moved = 0.0 if orders.empty else float(orders["bikes"].sum())
    distance = 0.0 if orders.empty else float((orders["bikes"] * orders["distance_km"]).sum())
    actual = float(metrics["actual_demand"].sum())
    served = float(metrics["served_demand"].sum())
    return {
        "scenario": scenario,
        "actual_demand": actual,
        "served_demand": served,
        "unmet_demand": float(metrics["unmet_demand"].sum()),
        "lost_returns": float(metrics["lost_returns"].sum()),
        "service_rate": served / actual if actual else float("nan"),
        "dispatch_bikes": moved,
        "dispatch_bike_km": distance,
        "dispatch_orders": int(len(orders)),
    }


def plot_dispatch_result(summary: pd.DataFrame, daily: pd.DataFrame) -> Path:
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    labels = summary["scenario"].tolist()
    x = np.arange(len(labels))
    width = 0.36

    axes[0].bar(x - width / 2, summary["unmet_demand"], width, label="未满足需求", color="#d62728")
    axes[0].bar(x + width / 2, summary["lost_returns"], width, label="容量溢出还车", color="#f1a340")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels)
    axes[0].set_ylabel("需求次数")
    axes[0].set_title("测试期服务缺口对比")
    axes[0].legend()

    for scenario, group in daily.groupby("scenario"):
        axes[1].plot(
            group["date"],
            group["service_rate"] * 100,
            marker="o",
            lw=1.8,
            label=scenario,
        )
    axes[1].set_ylim(0, 102)
    axes[1].set_ylabel("需求满足率（%）")
    axes[1].set_xlabel("日期")
    axes[1].set_title("逐日需求满足率")
    axes[1].legend()
    fig.autofmt_xdate()
    fig.suptitle("基于预测需求 + B 站点时段流入比例的调度投放测算", fontsize=12)
    path = OUTPUT_DIR / "fig_dispatch_comparison.png"
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[调度图] {path.name}")
    return path


def run(config: DispatchConfig | None = None) -> dict[str, Any]:
    config = config or DispatchConfig()
    OUTPUT_DIR.mkdir(exist_ok=True)
    frame, flow, coords = _read_inputs()

    # A/B 结果的调度压力表：测试期预测净缺口，按站点汇总。
    pressure = (
        frame.groupby(["station_key", "period"], as_index=False)
        .agg(
            predicted_demand=("prediction", "sum"),
            predicted_return=("predicted_return", "sum"),
        )
    )
    pressure["predicted_net_gap"] = pressure["predicted_demand"] - pressure["predicted_return"]
    pressure["abs_gap"] = pressure["predicted_net_gap"].abs()
    pressure["direction"] = np.select(
        [pressure["predicted_net_gap"] > 0, pressure["predicted_net_gap"] < 0],
        ["缺车", "淤积"],
        default="平衡",
    )
    save_csv(
        pressure.sort_values(["abs_gap", "predicted_demand"], ascending=False),
        "dispatch_pressure.csv",
    )

    base_metrics, _ = _simulate(frame, coords, config, dispatch_enabled=False)
    dispatch_metrics, orders = _simulate(frame, coords, config, dispatch_enabled=True)

    base_metrics["scenario"] = "不调度"
    dispatch_metrics["scenario"] = "阈值调度"
    all_metrics = pd.concat([base_metrics, dispatch_metrics], ignore_index=True)
    summary = pd.DataFrame(
        [
            _summary(base_metrics, pd.DataFrame(), "不调度"),
            _summary(dispatch_metrics, orders, "阈值调度"),
        ]
    )
    base_rate = float(summary.loc[summary["scenario"].eq("不调度"), "service_rate"].iloc[0])
    dispatch_rate = float(
        summary.loc[summary["scenario"].eq("阈值调度"), "service_rate"].iloc[0]
    )
    summary["service_rate_change_pp"] = (summary["service_rate"] - base_rate) * 100

    if orders.empty:
        orders = pd.DataFrame(
            columns=[
                "timestamp",
                "from_station_key",
                "to_station_key",
                "bikes",
                "distance_km",
                "receiver_projected_end_before",
                "receiver_need_before",
            ]
        )
    save_csv(summary, "dispatch_summary.csv")
    save_csv(orders, "dispatch_orders.csv")

    daily = (
        all_metrics.groupby(["date", "scenario"], as_index=False)
        .agg(
            actual_demand=("actual_demand", "sum"),
            served_demand=("served_demand", "sum"),
            unmet_demand=("unmet_demand", "sum"),
            lost_returns=("lost_returns", "sum"),
        )
    )
    daily["service_rate"] = daily["served_demand"] / daily["actual_demand"].clip(lower=1)
    save_csv(daily, "dispatch_daily_metrics.csv")
    figure = plot_dispatch_result(summary, daily)

    config_dict = asdict(config)
    report = [
        "共享单车调度投放策略测算报告",
        "=" * 48,
        "输入: A 的清洗结果 -> 需求模型预测；B 的 station_period_flow.csv -> 还入比例。",
        "策略: 小时末库存低于安全库存时，按缺口从最近的高库存站点调车。",
        "口径: 调度决策使用预测需求；服务评价使用测试期实际需求。",
        "",
        "演示参数（非真实运营参数）:",
        *[f"  {key} = {value}" for key, value in config_dict.items()],
        "",
        "测试期汇总:",
        summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"),
        "",
        f"满足率变化: {(dispatch_rate - base_rate) * 100:+.3f} 个百分点",
        f"调度单数量: {len(orders)}；累计移动 {summary.loc[1, 'dispatch_bikes']:.0f} 辆；"
        f"载车公里 {summary.loc[1, 'dispatch_bike_km']:.1f} km",
        "",
        "说明: 这是策略测算演示，不是整数车辆/路径/人员排班的精确优化。",
        "生产化时应替换为真实站点容量、车辆初始库存、还车率、车队规模与路网时延。",
    ]
    (OUTPUT_DIR / "dispatch_report.txt").write_text(
        "\n".join(report) + "\n", encoding="utf-8"
    )

    print(
        f"[调度完成] 不调度满足率 {base_rate:.2%} -> 阈值调度 {dispatch_rate:.2%}；"
        f"移动 {summary.loc[1, 'dispatch_bikes']:.0f} 辆"
    )
    return {
        "summary": summary,
        "orders": orders,
        "daily": daily,
        "pressure": pressure,
        "figure": figure,
        "config": config_dict,
    }