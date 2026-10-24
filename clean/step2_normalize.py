"""第 2 步：归一类别写法（去空白、统一大小写、同义词映射）。"""

import pandas as pd

from .common import (
    CATEGORY_COLUMNS,
    EXPECTED_CATEGORIES,
    STATION_ID_COLUMNS,
    STATION_NAME_COLUMNS,
)

# 业务同义词 -> 规范写法；规范取值见 common.EXPECTED_CATEGORIES
SYNONYMS = {
    "rideable_type": {
        "classic": "classic_bike",
        "classic bike": "classic_bike",
        "electric": "electric_bike",
        "ebike": "electric_bike",
        "e-bike": "electric_bike",
        "electric bike": "electric_bike",
    },
    "member_casual": {
        "subscriber": "member",
        "registered": "member",
        "annual": "member",
        "customer": "casual",
        "non-member": "casual",
        "nonmember": "casual",
    },
}


def _clean_text(series: pd.Series) -> pd.Series:
    """NBSP 换成普通空格，去掉首尾空白，把连续空白压成一个空格。"""
    return (
        series.astype("string")
        .str.replace("\xa0", " ", regex=False)
        .str.replace(r"\s+", " ", regex=True)
        .str.strip()
    )


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    changed_station_names = 0
    for col in STATION_NAME_COLUMNS:
        before = df[col].astype("string")
        df[col] = _clean_text(df[col])
        changed_station_names += int((before != df[col]).sum())

    # 站点编号也要去空白，保证同一站点编号只对应一种写法
    for col in STATION_ID_COLUMNS:
        df[col] = _clean_text(df[col])

    for col in CATEGORY_COLUMNS:
        before = df[col].astype("string")
        df[col] = _clean_text(df[col]).str.lower()
        df[col] = df[col].replace(SYNONYMS.get(col, {}))

        unknown = sorted(set(df[col].dropna().unique()) - EXPECTED_CATEGORIES[col] - {""})
        if unknown:
            print(f"[归一类别] {col} 仍有未识别写法: {unknown}")
        changed = int((before != df[col]).sum())
        values = sorted(set(df[col].dropna().unique()) - {""})
        print(f"[归一类别] {col} 改写 {changed} 行，取值 {values}")

    print(f"[归一类别] 站点名称去空白/规范化 {changed_station_names} 行")
    return df
