"""第 1 步：按业务主键去重。"""

import pandas as pd

from .common import RIDE_ID, log_step


def clean(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.copy()

    # 判重依据是业务主键 ride_id：同一笔订单只保留第一次出现的记录
    repeated_ids = int(df.duplicated(subset=[RIDE_ID], keep=False).sum())
    exact_rows = int(df.duplicated(keep=False).sum())
    print(f"[去重] 重复 ride_id 涉及 {repeated_ids} 行；整行完全重复 {exact_rows} 行")

    df = df.drop_duplicates(subset=[RIDE_ID], keep="first").reset_index(drop=True)
    log_step("去重", before, len(df))
    return df
