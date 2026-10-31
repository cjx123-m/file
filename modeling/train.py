"""需求回归模型训练、时间外推评估与特征重要性输出。"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .common import (
    FEATURE_COLUMNS,
    MODEL_DIR,
    OUTPUT_DIR,
    RANDOM_STATE,
    TARGET_COLUMN,
    save_csv,
)


def _wrap_log_target(estimator: Any) -> TransformedTargetRegressor:
    """需求右偏且含大量 0，用 log1p 目标让线性模型更稳定。"""
    return TransformedTargetRegressor(
        regressor=estimator,
        func=np.log1p,
        inverse_func=np.expm1,
        check_inverse=False,
    )


def _model_zoo(quick: bool) -> dict[str, TransformedTargetRegressor]:
    tree_estimators = 30 if quick else 100
    hist_iterations = 50 if quick else 240
    n_jobs = -1
    return {
        "BaselineMean": _wrap_log_target(DummyRegressor(strategy="mean")),
        "Ridge": _wrap_log_target(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                    ("ridge", Ridge(alpha=2.0)),
                ]
            )
        ),
        "RandomForest": _wrap_log_target(
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "rf",
                        RandomForestRegressor(
                            n_estimators=tree_estimators,
                            max_depth=14,
                            min_samples_leaf=8,
                            max_features=0.8,
                            n_jobs=n_jobs,
                            random_state=RANDOM_STATE,
                        ),
                    ),
                ]
            )
        ),
        "HistGradientBoosting": _wrap_log_target(
            HistGradientBoostingRegressor(
                max_iter=hist_iterations,
                learning_rate=0.07,
                max_leaf_nodes=31,
                min_samples_leaf=20,
                l2_regularization=0.2,
                early_stopping=False,
                random_state=RANDOM_STATE,
            )
        ),
    }


def _metrics(actual: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    prediction = np.clip(prediction, 0, None)
    residual = actual - prediction
    positive = actual > 0
    denominator = float(np.abs(actual).sum())
    return {
        "mae": float(mean_absolute_error(actual, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(actual, prediction))),
        "r2": float(r2_score(actual, prediction)),
        "wape": float(np.abs(residual).sum() / denominator) if denominator else float("nan"),
        "mae_positive": (
            float(mean_absolute_error(actual[positive], prediction[positive]))
            if positive.any()
            else float("nan")
        ),
        "mean_actual": float(actual.mean()),
        "mean_prediction": float(prediction.mean()),
    }


def _segment_metrics(
    frame: pd.DataFrame, actual: np.ndarray, prediction: np.ndarray, model_name: str
) -> list[dict[str, Any]]:
    segments = {
        "全部": np.ones(len(frame), dtype=bool),
        "早晚高峰": frame["is_am_peak"].eq(1).to_numpy()
        | frame["is_pm_peak"].eq(1).to_numpy(),
        "非高峰": ~(
            frame["is_am_peak"].eq(1).to_numpy() | frame["is_pm_peak"].eq(1).to_numpy()
        ),
        "周末": frame["is_weekend"].eq(1).to_numpy(),
        "工作日": frame["is_weekend"].eq(0).to_numpy(),
    }
    rows: list[dict[str, Any]] = []
    for segment, mask in segments.items():
        if not mask.any():
            continue
        row = {"model": model_name, "segment": segment, "rows": int(mask.sum())}
        row.update(_metrics(actual[mask], prediction[mask]))
        rows.append(row)
    return rows


def _extract_importance(
    model: TransformedTargetRegressor,
    X_sample: pd.DataFrame,
    y_sample: pd.Series,
) -> pd.DataFrame:
    estimator = model.regressor_
    if isinstance(estimator, Pipeline):
        final_name = estimator.steps[-1][0]
        final = estimator.steps[-1][1]
        if final_name == "ridge" and hasattr(final, "coef_"):
            values = np.abs(np.asarray(final.coef_)).ravel()
            method = "标准化后的 Ridge 系数绝对值"
        elif hasattr(final, "feature_importances_"):
            values = np.asarray(final.feature_importances_).ravel()
            method = "树模型不纯度重要性"
        else:
            values = None
    elif hasattr(estimator, "feature_importances_"):
        values = np.asarray(estimator.feature_importances_).ravel()
        method = "树模型不纯度重要性"
    else:
        values = None

    if values is None:
        result = permutation_importance(
            model,
            X_sample,
            y_sample,
            scoring="neg_mean_absolute_error",
            n_repeats=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
        values = np.maximum(result.importances_mean, 0)
        method = "测试集置换重要性（MAE）"

    table = pd.DataFrame(
        {
            "feature": FEATURE_COLUMNS,
            "importance": values,
            "importance_method": method,
        }
    )
    total = table["importance"].sum()
    table["share"] = table["importance"] / total if total > 0 else 0.0
    return table.sort_values("importance", ascending=False, ignore_index=True)


def run(panel: pd.DataFrame, metadata: dict[str, Any], quick: bool = False) -> dict[str, Any]:
    """训练多个回归器，按时间后 7 天测试并输出全部模型产物。"""
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

    train = panel.loc[panel["is_test"].eq(0)].copy()
    test = panel.loc[panel["is_test"].eq(1)].copy()
    X_train = train[FEATURE_COLUMNS]
    y_train = train[TARGET_COLUMN].astype(float)
    X_test = test[FEATURE_COLUMNS]
    y_test = test[TARGET_COLUMN].astype(float)

    models = _model_zoo(quick)
    fitted: dict[str, TransformedTargetRegressor] = {}
    predictions: dict[str, np.ndarray] = {}
    metric_rows: list[dict[str, Any]] = []
    segment_rows: list[dict[str, Any]] = []

    print(
        f"\n[模型训练] 时间外推切分：训练 {len(train)} 行，测试 {len(test)} 行；"
        f"目标为站点小时借出需求"
    )
    for name, model in models.items():
        started = time.perf_counter()
        model.fit(X_train, y_train)
        elapsed = time.perf_counter() - started
        prediction = np.clip(model.predict(X_test), 0, None)
        fitted[name] = model
        predictions[name] = prediction
        row = {"model": name, "train_seconds": elapsed, "rows": int(len(test))}
        row.update(_metrics(y_test.to_numpy(), prediction))
        metric_rows.append(row)
        segment_rows.extend(
            _segment_metrics(test, y_test.to_numpy(), prediction, name)
        )
        print(
            f"  {name:<22} MAE={row['mae']:.4f}  RMSE={row['rmse']:.4f}  "
            f"R2={row['r2']:.4f}  用时={elapsed:.1f}s"
        )

    metrics = pd.DataFrame(metric_rows).sort_values("mae", ignore_index=True)
    metrics["is_best"] = False
    metrics.loc[metrics.index[0], "is_best"] = True
    best_name = str(metrics.loc[0, "model"])
    best_model = fitted[best_name]
    best_prediction = predictions[best_name]
    save_csv(metrics, "model_metrics.csv")
    save_csv(pd.DataFrame(segment_rows), "model_metrics_by_segment.csv")

    prediction_table = test[
        [
            "timestamp",
            "station_key",
            "station_name",
            "period",
            "hour",
            "weekday",
            "is_weekend",
            "is_am_peak",
            "is_pm_peak",
            "demand",
        ]
    ].copy()
    prediction_table["prediction"] = best_prediction
    prediction_table["error"] = prediction_table["prediction"] - prediction_table["demand"]
    prediction_table["abs_error"] = prediction_table["error"].abs()
    save_csv(prediction_table, "test_predictions.csv")

    importance_sample = test.sample(n=min(6000, len(test)), random_state=RANDOM_STATE)
    importance = _extract_importance(
        best_model,
        importance_sample[FEATURE_COLUMNS],
        importance_sample[TARGET_COLUMN].astype(float),
    )
    importance["model"] = best_name
    importance.to_csv(
        OUTPUT_DIR / "feature_importance.csv", index=False, encoding="utf-8-sig"
    )

    model_path = MODEL_DIR / "best_model.joblib"
    joblib.dump(
        {
            "model_name": best_name,
            "features": FEATURE_COLUMNS,
            "model": best_model,
            "metadata": metadata,
        },
        model_path,
    )

    metadata = dict(metadata)
    metadata.update(
        {
            "best_model": best_name,
            "best_metrics": metrics.iloc[0].drop(labels=["is_best"]).to_dict(),
            "model_path": str(model_path.relative_to(OUTPUT_DIR.parent)),
            "training": "按时间外推：前段训练，最后 7 天测试",
            "evaluation_scope": "站点小时需求量；滞后特征为一小时前可观测值",
        }
    )
    with (OUTPUT_DIR / "model_metadata.json").open("w", encoding="utf-8") as fh:
        json.dump(metadata, fh, ensure_ascii=False, indent=2)

    report_lines = [
        "共享单车影响因子与需求预测报告",
        "=" * 48,
        "目标: 站点 × 小时借出骑行量 demand",
        f"样本: 面板 {len(panel)} 行，训练 {len(train)} 行，测试 {len(test)} 行",
        f"测试期: {metadata['test_start']} ~ {metadata['test_end']}",
        f"特征: {len(FEATURE_COLUMNS)} 个（会员类型、车辆类型、时段、空间与历史需求）",
        "",
        "模型对比（按 MAE 升序）:",
        metrics.to_string(index=False, float_format=lambda value: f"{value:.4f}"),
        "",
        f"最佳模型: {best_name}",
        "评价说明: MAE/RMSE 单位均为“次/站点小时”；WAPE 为绝对误差总量占实际需求总量比例。",
        "泄漏控制: 站点/会员/车型画像仅由训练期估计；当前小时标签不进入特征。",
        "预测口径: 一小时前需求、前一日同小时等滞后特征可用，属于 one-step-ahead 回测。",
        "",
        "特征重要性 Top 10:",
        importance.head(10)[["feature", "importance", "share"]].to_string(
            index=False, float_format=lambda value: f"{value:.4f}"
        ),
    ]
    (OUTPUT_DIR / "model_report.txt").write_text(
        "\n".join(report_lines) + "\n", encoding="utf-8"
    )

    print(f"[模型完成] 最佳模型 {best_name}，模型文件: {model_path}")
    return {
        "metrics": metrics,
        "predictions": prediction_table,
        "importance": importance,
        "best_model": best_model,
        "best_name": best_name,
        "metadata": metadata,
    }