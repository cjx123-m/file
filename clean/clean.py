"""清洗管线入口。

运行：python -m clean.clean
"""

from pandas import DataFrame

from . import step1_dedupe, step2_normalize, step3_convert_types, step4_missing
from .common import read_raw


def run() -> DataFrame:
    df = read_raw()
    print(f"原始数据: {len(df)} 行 x {df.shape[1]} 列\n")

    df = step1_dedupe.clean(df)
    df = step2_normalize.clean(df)
    df = step3_convert_types.clean(df)
    df = step4_missing.clean(df)

    print(f"\n清洗后: {len(df)} 行")
    return df


if __name__ == "__main__":
    run()
