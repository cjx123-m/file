"""特征工程：构造“站点 × 小时”需求面板及滞后/结构特征。

需求定义：某站点在某小时内被借出的骑行次数。缺失的站点小时补 0，
形成完整面板，才能正确训练回归模型并用于调度仿真。
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from .common import (
    FEATURE_COLUMNS,
    OUTPUT_DIR,
    TARGET_COLUMN,
    hour_to_period,
    load_clean_data,
    save_csv,
)


def aggregate_station_hour(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """把骑行明细聚合为站点小时需求，并返回站点维表。"""
    rides = df.copy()
    rides["timestamp"] = rides["started_at"].dt.floor("h")
    rides["is_member"] = rides["member_casual"].eq("member").astype(int)
    rides["is_electric"] = rides["rideable_type"].eq("electric_bike").astype(int)

    station_meta = (
        rides.groupby("start_station_id", as_index=False)
        .agg(
            station_name=("start_station_name", "first"),
            station_lat=("start_lat", "median"),
            station_lng=("start_lng", "median"),
        )
        .rename(columns={"start_station_id": "station_key"})
    )

    activity = (
        rides.groupby(["start_station_id", "timestamp"], as_index=False)
        .agg(
            demand=("ride_id", "size"),
            member_demand=("is_member", "sum"),
            electric_demand=("is_electric", "sum"),
            avg_duration_sec=("duration_sec", "mean"),
        )
        .rename(columns={"start_station_id": "station_key"})
    )

    timestamps = pd.date_range(
        rides["timestamp"].min().floor("h"),
        rides["timestamp"].max().floor("h"),
        freq="h",
        name="timestamp",
    )
    panel = station_meta.merge(pd.DataFrame({"timestamp": timestamps}), how="cross")
    panel = panel.merge(activity, on=["station_key", "timestamp"], how="left")

    count_columns = ["demand", "member_demand", "electric_demand"]
    for col in count_columns:
        panel[col] = panel[col].fillna(0).astype(int)
    panel["casual_demand"] = panel["demand"] - panel["member_demand"]
    panel["classic_demand"] = panel["demand"] - panel["electric_demand"]
    panel["avg_duration_sec"] = panel["avg_duration_sec"].fillna(0.0)
    return panel, station_meta


def add_calendar_features(panel: pd.DataFrame) -> pd.DataFrame:
    out = panel.copy()
    ts = pd.to_datetime(out["timestamp"])
    out["date"] = ts.dt.normalize()
    out["hour"] = ts.dt.hour
    out["weekday"] = ts.dt.day_of_week
    out["is_weekend"] = out["weekday"].ge(5).astype(int)
    out["day_of_month"] = ts.dt.day
    out["period"] = hour_to_period(out["hour"])
    out["hour_sin"] = np.sin(2 * np.pi * out["hour"] / 24)
    out["hour_cos"] = np.cos(2 * np.pi * out["hour"] / 24)
    out["weekday_sin"] = np.sin(2 * np.pi * out["weekday"] / 7)
    out["weekday_cos"] = np.cos(2 * np.pi * out["weekday"] / 7)
    out["is_am_peak"] = out["hour"].between(6, 9).astype(int)
    out["is_pm_peak"] = out["hour"].between(17, 19).astype(int)
    out["is_night"] = out["hour"].ge(20).astype(int)
    return out


def add_lag_features(panel: pd.DataFrame) -> pd.DataFrame:
    """加入一小时前、前一日同小时、上周同小时及 24h 滚动需求。"""
    out = panel.sort_values(["station_key", "timestamp"]).copy()
    out["member_ratio"] = np.where(
        out["demand"].gt(0), out["member_demand"] / out["demand"].clip(lower=1), np.nan
    )
    out["electric_ratio"] = np.where(
        out["demand"].gt(0), out["electric_demand"] / out["demand"].clip(lower=1), np.nan
    )

    grouped = out.groupby("station_key", sort=False)["demand"]
    out["demand_lag_1h"] = grouped.shift(1)
    out["demand_lag_24h"] = grouped.shift(24)
    out["demand_lag_168h"] = grouped.shift(168)
    out["demand_rolling_24h"] = out.groupby("station_key", sort=False)["demand"].transform(
        lambda values: values.shift(1).rolling(24, min_periods=12).mean()
    )
    out["member_ratio_lag_24h"] = out.groupby("station_key", sort=False)[
        "member_ratio"
    ].shift(24)
    out["electric_ratio_lag_24h"] = out.groupby("station_key", sort=False)[
        "electric_ratio"
    ].shift(24)

    city_demand = out.groupby("timestamp", sort=True)["demand"].sum()
    out["city_demand_lag_1h"] = out["timestamp"].map(city_demand.shift(1))
    out["city_demand_lag_24h"] = out["timestamp"].map(city_demand.shift(24))
    return out


def add_training_profiles(panel: pd.DataFrame, train_mask: pd.Series) -> pd.DataFrame:
    """仅用训练期估计站点与站点小时画像，避免测试期信息泄漏。"""
    out = panel.copy()
    train = out.loc[train_mask].copy()

    station_profile = (
        train.groupby("station_key", as_index=False)["demand"]
        .mean()
        .rename(columns={"demand": "station_demand_profile"})
    )
    out = out.merge(station_profile, on="station_key", how="left")

    profile = (
        train.groupby(["station_key", "hour"], as_index=False)
        .agg(
            demand_sum=("demand", "sum"),
            member_sum=("member_demand", "sum"),
            electric_sum=("electric_demand", "sum"),
        )
    )
    profile["station_hour_demand_profile"] = profile["demand_sum"] / max(1, 24 * 7)
    profile["member_ratio_profile"] = np.where(
        profile["demand_sum"].gt(0),
        profile["member_sum"] / profile["demand_sum"].clip(lower=1),
        np.nan,
    )
    profile["electric_ratio_profile"] = np.where(
        profile["demand_sum"].gt(0),
        profile["electric_sum"] / profile["demand_sum"].clip(lower=1),
        np.nan,
    )
    # profile 为训练期每个 station × hour 的总需求；换算成“每个同小时样本的均值”
    train_hour_counts = train.groupby(["station_key", "hour"]).size().rename("n_samples")
    profile = profile.merge(train_hour_counts, on=["station_key", "hour"], how="left")
    profile["station_hour_demand_profile"] = (
        profile["demand_sum"] / profile["n_samples"].clip(lower=1)
    )
    profile_cols = [
        "station_key",
        "hour",
        "station_hour_demand_profile",
        "member_ratio_profile",
        "electric_ratio_profile",
    ]
    out = out.merge(profile[profile_cols], on=["station_key", "hour"], how="left")

    overall_member_ratio = float(
        train["member_demand"].sum() / max(1, train["demand"].sum())
    )
    overall_electric_ratio = float(
        train["electric_demand"].sum() / max(1, train["demand"].sum())
    )
    out["member_ratio_profile"] = out["member_ratio_profile"].fillna(overall_member_ratio)
    out["electric_ratio_profile"] = out["electric_ratio_profile"].fillna(overall_electric_ratio)
    out["station_demand_profile"] = out["station_demand_profile"].fillna(
        float(train["demand"].mean())
    )
    out["station_hour_demand_profile"] = out["station_hour_demand_profile"].fillna(
        float(train["demand"].mean())
    )
    return out


def _feature_dictionary() -> pd.DataFrame:
    rows = [
        ("hour / weekday / day_of_month", "时段", "小时、星期与月内日期"),
        ("hour_sin / hour_cos / weekday_sin / weekday_cos", "时段", "周期编码，保留 0 点与 23 点、周日与周一邻近关系"),
        ("is_weekend / is_am_peak / is_pm_peak / is_night", "时段", "周末、早晚高峰、夜间标记"),
        ("station_lat / station_lng", "空间", "站点位置，区分不同需求区域"),
        ("station_demand_profile / station_hour_demand_profile", "需求历史", "仅用训练期估计的站点、站点小时基准需求"),
        ("member_ratio_profile / member_ratio_lag_24h", "会员类型", "训练期会员占比画像与前一日同小时会员占比"),
        ("electric_ratio_profile / electric_ratio_lag_24h", "车辆类型", "训练期电动车占比画像与前一日同小时电动车占比"),
        ("demand_lag_1h / 24h / 168h", "需求历史", "一小时前、前一日同小时、上周同小时需求"),
        ("demand_rolling_24h", "需求历史", "过去 24 小时平均需求"),
        ("city_demand_lag_1h / city_demand_lag_24h", "城市需求", "全城前一小时/前一日同小时总需求"),
    ]
    return pd.DataFrame(rows, columns=["feature", "category", "description"])


def build_and_save(test_days: int = 7) -> tuple[pd.DataFrame, dict[str, Any]]:
    """执行完整特征工程并输出完整面板、样本和字段说明。"""
    df = load_clean_data()
    panel, _ = aggregate_station_hour(df)
    panel = add_calendar_features(panel)
    panel = add_lag_features(panel)

    max_ts = panel["timestamp"].max().normalize()
    test_start = max_ts - pd.Timedelta(days=test_days - 1)
    train_mask = panel["timestamp"].lt(test_start)
    test_mask = ~train_mask
    panel = add_training_profiles(panel, train_mask)
    panel["is_test"] = test_mask.astype(int)
    panel = panel.sort_values(["timestamp", "station_key"], ignore_index=True)

    missing_features = [col for col in FEATURE_COLUMNS if col not in panel.columns]
    if missing_features:
        raise ValueError(f"特征构造不完整，缺少: {missing_features}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    feature_path = OUTPUT_DIR / "station_hour_features.csv.gz"
    panel.to_csv(feature_path, index=False, encoding="utf-8-sig", compression="gzip")
    sample = panel.sample(n=min(5000, len(panel)), random_state=42).sort_values(
        ["timestamp", "station_key"]
    )
    save_csv(sample, "station_hour_features_sample5000.csv")
    save_csv(_feature_dictionary(), "feature_dictionary.csv")

    metadata = {
        "clean_rows": int(len(df)),
        "station_count": int(panel["station_key"].nunique()),
        "timestamp_count": int(panel["timestamp"].nunique()),
        "panel_rows": int(len(panel)),
        "train_rows": int(train_mask.sum()),
        "test_rows": int(test_mask.sum()),
        "test_start": str(test_start),
        "test_end": str(max_ts),
        "feature_count": len(FEATURE_COLUMNS),
        "features": FEATURE_COLUMNS,
        "panel_path": str(feature_path.relative_to(OUTPUT_DIR.parent)),
    }
    print(
        f"[特征工程] 面板 {len(panel)} 行 × {len(FEATURE_COLUMNS)} 个模型特征；"
        f"训练 {int(train_mask.sum())}、测试 {int(test_mask.sum())}；"
        f"测试期 {test_start:%Y-%m-%d} ~ {max_ts:%Y-%m-%d}"
    )
    return panel, metadata