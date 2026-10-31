"""需求预测效果图：模型对比、时间序列、散点与特征重要性。"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import OUTPUT_DIR

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


def _save(fig: plt.Figure, name: str) -> Path:
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    print(f"[预测图] {path.name}")
    return path


def plot_model_comparison(metrics: pd.DataFrame) -> Path:
    ordered = metrics.sort_values("mae", ascending=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    colors = ["#d62728" if bool(flag) else "#4c78a8" for flag in ordered["is_best"]]
    axes[0].barh(ordered["model"], ordered["mae"], color=colors, alpha=0.9)
    axes[0].invert_yaxis()
    axes[0].set_title("MAE（越小越好）")
    axes[0].set_xlabel("次 / 站点小时")
    axes[1].barh(ordered["model"], ordered["rmse"], color=colors, alpha=0.9)
    axes[1].invert_yaxis()
    axes[1].set_title("RMSE（越小越好）")
    axes[1].set_xlabel("次 / 站点小时")
    for ax, metric in zip(axes, ["mae", "rmse"]):
        for index, value in enumerate(ordered[metric]):
            ax.text(value, index, f" {value:.3f}", va="center", fontsize=8)
        ax.margins(x=0.16)
    fig.suptitle("需求回归模型对比（红色为最佳模型）", fontsize=13)
    return _save(fig, "fig_model_comparison.png")


def plot_daily_prediction(predictions: pd.DataFrame) -> Path:
    frame = predictions.copy()
    frame["date"] = pd.to_datetime(frame["timestamp"]).dt.normalize()
    daily = (
        frame.groupby("date", as_index=False)
        .agg(actual=("demand", "sum"), prediction=("prediction", "sum"))
        .sort_values("date")
    )
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.plot(daily["date"], daily["actual"], marker="o", lw=2, label="实际需求")
    ax.plot(daily["date"], daily["prediction"], marker="s", lw=2, label="预测需求")
    ax.set_title("测试期逐日需求量：实际 vs 预测")
    ax.set_ylabel("借出骑行量（次/日）")
    ax.set_xlabel("日期")
    ax.legend()
    fig.autofmt_xdate()
    return _save(fig, "fig_prediction_daily.png")


def plot_prediction_scatter(predictions: pd.DataFrame) -> Path:
    sample = predictions.sample(n=min(20_000, len(predictions)), random_state=42)
    actual = sample["demand"].to_numpy(dtype=float)
    predicted = sample["prediction"].to_numpy(dtype=float)
    peak = max(float(np.quantile(actual, 0.995)), float(np.quantile(predicted, 0.995)), 1.0)

    fig, ax = plt.subplots(figsize=(7.2, 6))
    image = ax.hexbin(
        actual,
        predicted,
        gridsize=45,
        extent=(0, peak, 0, peak),
        mincnt=1,
        cmap="viridis",
        bins="log",
    )
    ax.plot([0, peak], [0, peak], color="#d62728", lw=1.2, ls="--", label="完美预测")
    ax.set_xlim(0, peak)
    ax.set_ylim(0, peak)
    ax.set_xlabel("实际站点小时需求")
    ax.set_ylabel("预测站点小时需求")
    ax.set_title("测试集预测散点（样本 20,000，颜色为对数计数）")
    ax.legend(loc="upper left")
    fig.colorbar(image, ax=ax, label="样本数（对数）")
    return _save(fig, "fig_prediction_scatter.png")


def plot_hourly_profile(predictions: pd.DataFrame) -> Path:
    profile = (
        predictions.groupby("hour", as_index=False)
        .agg(actual=("demand", "mean"), prediction=("prediction", "mean"))
        .sort_values("hour")
    )
    fig, ax = plt.subplots(figsize=(9, 4.8))
    ax.plot(profile["hour"], profile["actual"], marker="o", lw=2, label="实际")
    ax.plot(profile["hour"], profile["prediction"], marker="s", lw=2, label="预测")
    ax.set_xticks(range(24))
    ax.set_xlabel("小时")
    ax.set_ylabel("平均站点需求（次/站点小时）")
    ax.set_title("测试期小时需求曲线：模型已学习早晚高峰特征")
    ax.legend()
    return _save(fig, "fig_prediction_hourly.png")


def plot_feature_importance(importance: pd.DataFrame, top_n: int = 15) -> Path:
    table = importance.head(top_n).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9.5, 6.2))
    ax.barh(table["feature"], table["importance"], color="#4c78a8", alpha=0.9)
    ax.set_xlabel("重要性")
    ax.set_title(f"最佳模型影响因子 Top {min(top_n, len(table))}")
    ax.margins(x=0.15)
    for index, value in enumerate(table["importance"]):
        ax.text(value, index, f" {value:.3g}", va="center", fontsize=7.5)
    return _save(fig, "fig_feature_importance.png")


def run(
    metrics: pd.DataFrame,
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
) -> dict[str, Path]:
    OUTPUT_DIR.mkdir(exist_ok=True)
    return {
        "model_comparison": plot_model_comparison(metrics),
        "prediction_daily": plot_daily_prediction(predictions),
        "prediction_scatter": plot_prediction_scatter(predictions),
        "prediction_hourly": plot_hourly_profile(predictions),
        "feature_importance": plot_feature_importance(importance),
    }