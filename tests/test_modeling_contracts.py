"""不依赖真实数据文件的建模契约测试。"""

from __future__ import annotations

import unittest

import pandas as pd

from modeling.common import FEATURE_COLUMNS, haversine_km, hour_to_period
from modeling.features import (
    add_calendar_features,
    add_lag_features,
    add_training_profiles,
    aggregate_station_hour,
)


class ModelingContractTest(unittest.TestCase):
    def setUp(self) -> None:
        timestamps = pd.date_range("2024-01-01", periods=50, freq="h")
        rows = []
        for index, timestamp in enumerate(timestamps):
            for station in ("A", "B"):
                if (index + ord(station)) % 3:
                    rows.append(
                        {
                            "ride_id": f"{station}-{index}",
                            "started_at": timestamp + pd.Timedelta(minutes=5),
                            "start_station_id": station,
                            "start_station_name": f"Station {station}",
                            "start_lat": 38.9,
                            "start_lng": -77.0,
                            "member_casual": "member" if index % 2 else "casual",
                            "rideable_type": "electric_bike" if index % 3 else "classic_bike",
                            "duration_sec": 600.0,
                        }
                    )
        self.df = pd.DataFrame(rows)

    def test_panel_calendar_lag_and_features(self) -> None:
        panel, meta = aggregate_station_hour(self.df)
        self.assertEqual(len(meta), 2)
        self.assertEqual(len(panel), 2 * 50)
        panel = add_calendar_features(panel)
        panel = add_lag_features(panel)
        mask = panel["timestamp"].lt(panel["timestamp"].max() - pd.Timedelta(hours=10))
        panel = add_training_profiles(panel, mask)
        self.assertTrue(set(FEATURE_COLUMNS).issubset(panel.columns))
        self.assertTrue(panel["demand"].ge(0).all())
        self.assertIn("hour_sin", panel.columns)

    def test_time_period_and_distance(self) -> None:
        periods = hour_to_period(pd.Series([0, 8, 12, 18, 22]))
        self.assertEqual(periods.tolist(), ["凌晨", "早高峰", "平峰", "晚高峰", "夜间"])
        self.assertAlmostEqual(haversine_km(38.9, -77.0, 38.9, -77.0), 0.0, places=6)
        self.assertGreater(haversine_km(38.9, -77.0, 38.91, -77.0), 1.0)


if __name__ == "__main__":
    unittest.main()