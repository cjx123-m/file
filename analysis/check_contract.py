"""对接联调：校验 A 的清洗表是否满足空间分析所需的最小字段契约。

上游是 05-标记异常 分支产出的 output/rides_clean.csv，本模块只读不写数据，
逐项检查字段齐全性、类型可转换性、取值范围与关键字段覆盖率，
把“能不能直接接上 A 的清洗结果”变成一个可复现的检查项。

运行：python -m analysis.check_contract
输出：output/contract_report.txt
"""

import pandas as pd

from clean.common import LAT_RANGE, LNG_RANGE

from .spatial_common import CLEAN_CSV, OUTPUT_DIR

REQUIRED_DTYPES = {
    "ride_id": "str",
    "started_at": "datetime",
    "ended_at": "datetime",
    "start_station_id": "str",
    "start_station_name": "str",
    "end_station_id": "str",
    "end_station_name": "str",
    "start_lat": "float",
    "start_lng": "float",
    "end_lat": "float",
    "end_lng": "float",
    "rideable_type": "str",
    "member_casual": "str",
    "duration_sec": "float",
}
EXPECTED_CATEGORIES = {
    "rideable_type": {"classic_bike", "electric_bike"},
    "member_casual": {"member", "casual"},
}
MIN_STATION_COVERAGE = 0.70  # 起终点站点编号覆盖率下限（低于此值站点级结论不可信）
MAX_RIDE_HOURS = 24


def _check_columns(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results = []
    missing = [col for col in REQUIRED_DTYPES if col not in df.columns]
    results.append(
        ("必需字段齐全", not missing, "缺少: " + "、".join(missing) if missing else f"共 {len(REQUIRED_DTYPES)} 列")
    )
    for flag in ("is_anomaly", "anomaly_reason"):
        results.append(
            (f"异常标记列 {flag}", flag in df.columns, "存在" if flag in df.columns else "缺失，可能不是 05 分支产物")
        )
    return results


def _check_values(df: pd.DataFrame) -> list[tuple[str, bool, str]]:
    results = []

    for col in ("started_at", "ended_at"):
        parsed = pd.to_datetime(df[col], errors="coerce")
        results.append(
            (
                f"{col} 可解析为时间",
                parsed.notna().all(),
                f"解析失败 {int(parsed.isna().sum())} 行，范围 {parsed.min()} ~ {parsed.max()}",
            )
        )

    coords = ["start_lat", "start_lng", "end_lat", "end_lng"]
    numeric = df[coords].apply(pd.to_numeric, errors="coerce")
    ok_numeric = numeric.notna().all().all()
    results.append(("经纬度可转数值", ok_numeric, f"缺失 {int(numeric.isna().any(axis=1).sum())} 行"))

    def _in_range(frame: pd.DataFrame, low: float, high: float) -> bool:
        return bool(((frame >= low) & (frame <= high)).to_numpy().all())

    lat_ok = _in_range(numeric[["start_lat", "end_lat"]], LAT_RANGE[0], LAT_RANGE[1])
    lng_ok = _in_range(numeric[["start_lng", "end_lng"]], LNG_RANGE[0], LNG_RANGE[1])
    results.append(
        (
            "经纬度落在服务区范围",
            lat_ok and lng_ok,
            f"纬度 {LAT_RANGE}、经度 {LNG_RANGE}",
        )
    )

    coverage = float(
        (df["start_station_id"].notna() & df["end_station_id"].notna()).mean()
    )
    results.append(
        (
            "起终点站点编号覆盖率",
            coverage >= MIN_STATION_COVERAGE,
            f"{coverage:.1%}（下限 {MIN_STATION_COVERAGE:.0%}）",
        )
    )

    duration = pd.to_numeric(df["duration_sec"], errors="coerce")
    results.append(
        (
            "骑行时长非负且 ≤ 24 小时",
            bool((duration.dropna() >= 0).all() and (duration.max() <= MAX_RIDE_HOURS * 3600)),
            f"范围 {duration.min():.0f} ~ {duration.max():.0f} 秒",
        )
    )

    for col, allowed in EXPECTED_CATEGORIES.items():
        values = set(df[col].dropna().unique())
        results.append(
            (f"{col} 取值规范化", values <= allowed, "取值: " + "、".join(sorted(values)))
        )

    duplicates = int(df["ride_id"].duplicated().sum())
    results.append(("ride_id 唯一", duplicates == 0, f"重复 {duplicates} 行"))
    return results


def run(strict: bool = True) -> bool:
    lines = ["空间分析 · 清洗表字段契约校验"]

    if not CLEAN_CSV.exists():
        message = f"找不到 {CLEAN_CSV}，请先在 05-标记异常 分支运行 python -m clean.clean"
        lines.append(f"[失败] 文件存在: {message}")
        (OUTPUT_DIR / "contract_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
        print("\n".join(lines))
        if strict:
            raise FileNotFoundError(message)
        return False

    df = pd.read_csv(
        CLEAN_CSV,
        dtype={"ride_id": str, "rideable_type": str, "member_casual": str,
               "start_station_id": str, "end_station_id": str},
    )
    lines.append(f"[通过] 文件存在: {CLEAN_CSV.name}，{len(df)} 行 × {df.shape[1]} 列")

    results = _check_columns(df) + _check_values(df)
    for name, ok, detail in results:
        lines.append(f"[{'通过' if ok else '失败'}] {name}: {detail}")

    failed = [name for name, ok, _ in results if not ok]
    lines.append("")
    if failed:
        lines.append(f"结论: {len(failed)} 项未通过（{'、'.join(failed)}），需与 A 确认清洗表后重跑")
    else:
        lines.append("结论: 契约校验全部通过，可直接进入 spatial / tide / imbalance / visualize")

    (OUTPUT_DIR / "contract_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))

    if strict and failed:
        raise ValueError("清洗表契约校验未通过，详见 output/contract_report.txt")
    return not failed


if __name__ == "__main__":
    run()
