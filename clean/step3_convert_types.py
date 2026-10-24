"""第 3 步：转数值（时间戳、经纬度），并派生骑行时长。"""

import pandas as pd

from .common import COORDINATE_COLUMNS, TIMESTAMP_COLUMNS


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 时间戳统一转成 datetime；无法解析的值收敛为 NaT，交给补缺步骤处理
    for col in TIMESTAMP_COLUMNS:
        parsed = pd.to_datetime(df[col], errors="coerce")
        print(
            f"[转数值] {col} 解析失败 {int(parsed.isna().sum())} 行，"
            f"范围 {parsed.min()} ~ {parsed.max()}"
        )
        df[col] = parsed

    # 先转时间，再派生时长，后续异常判据才能直接做数值比较
    df["duration_sec"] = (df["ended_at"] - df["started_at"]).dt.total_seconds()

    for col in COORDINATE_COLUMNS:
        parsed = pd.to_numeric(df[col], errors="coerce")
        print(
            f"[转数值] {col} 无法转成数值 {int(parsed.isna().sum())} 行，"
            f"范围 {parsed.min():.5f} ~ {parsed.max():.5f}"
        )
        df[col] = parsed

    print(
        f"[转数值] duration_sec 范围 "
        f"{df['duration_sec'].min():.0f} ~ {df['duration_sec'].max():.0f} 秒"
    )
    return df
