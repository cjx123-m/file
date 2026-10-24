"""第 4 步：补缺删缺（先诊断，再按字段性质处置）。"""

import pandas as pd

from .common import CRITICAL_COLUMNS, STATION_ID_COLUMNS, STATION_NAME_COLUMNS, UNKNOWN_STATION


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    before = len(df)

    # 站点字段在前一步还是文本，空串要先统一成 NA 才能被 isna 统计
    for col in STATION_NAME_COLUMNS + STATION_ID_COLUMNS:
        df[col] = df[col].replace("", pd.NA)

    print("[补缺删缺] 缺失诊断（行数）:")
    missing = df.isna().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(missing.to_string())
    else:
        print("  无缺失")

    # 关键字段缺失（时间戳、坐标）无法参与时长/距离统计，删除整行
    df = df.dropna(subset=CRITICAL_COLUMNS).reset_index(drop=True)

    # 站点名称缺失往往意味着无桩停放/站点记录不全，不代表这趟骑行无效：
    # 保留记录并显式标注，避免按比例删除造成样本偏差
    for col in STATION_NAME_COLUMNS:
        df[col] = df[col].fillna(UNKNOWN_STATION)

    print(
        f"[补缺删缺] {before} -> {len(df)} 行，删除 {before - len(df)} 行；"
        f"站点名称缺失填为“{UNKNOWN_STATION}”"
    )
    return df
