from datetime import datetime

import polars as pl

from bikeshare.aggregate import (
    add_calendar,
    add_weather,
    drop_nonexistent_hours,
    hourly_events,
    zero_fill,
)


def _trips() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "start_station_id": [1, 1, 1],
            "start_time": [
                datetime(2024, 1, 1, 8, 5),
                datetime(2024, 1, 1, 8, 50),
                datetime(2024, 1, 1, 11, 0),
            ],
            "end_station_id": [2, 2, None],
            "end_time": [datetime(2024, 1, 1, 8, 20), datetime(2024, 1, 1, 9, 10), None],
        }
    )


def test_hourly_events_counts_departures_and_arrivals() -> None:
    ev = hourly_events(_trips()).sort("station_id", "hour")
    row = lambda s, h: ev.filter((pl.col("station_id") == s) & (pl.col("hour") == h)).row(0)  # noqa: E731
    assert row(1, datetime(2024, 1, 1, 8))[2:] == (2, 0)
    assert row(2, datetime(2024, 1, 1, 8))[2:] == (0, 1)
    assert row(2, datetime(2024, 1, 1, 9))[2:] == (0, 1)
    assert ev["arrivals"].sum() == 2  # null end station is not an arrival


def test_zero_fill_stays_within_each_stations_active_span() -> None:
    filled = zero_fill(hourly_events(_trips()))
    s1 = filled.filter(pl.col("station_id") == 1)
    assert s1["hour"].min() == datetime(2024, 1, 1, 8)
    assert s1["hour"].max() == datetime(2024, 1, 1, 11)
    assert s1.height == 4
    assert s1.filter(pl.col("hour") == datetime(2024, 1, 1, 9))["departures"][0] == 0
    s2 = filled.filter(pl.col("station_id") == 2)
    assert s2.height == 2  # 08:00 and 09:00 only


def test_add_weather_keeps_row_count_and_hour_alignment() -> None:
    panel = zero_fill(hourly_events(_trips()))
    wx = pl.DataFrame({"hour": [datetime(2024, 1, 1, 8)], "temp_c": [-3.0]})
    out = add_weather(panel, wx)
    assert out.height == panel.height
    assert out.filter(pl.col("temp_c").is_not_null())["hour"].unique().to_list() == [
        datetime(2024, 1, 1, 8)
    ]


def test_add_calendar_uses_local_date_for_holidays() -> None:
    df = pl.DataFrame(
        {"hour": [datetime(2025, 7, 1, 23), datetime(2025, 7, 2, 0), datetime(2025, 3, 10, 9)]}
    )
    out = add_calendar(df)
    assert out["is_holiday"].to_list() == [True, False, False]  # Canada Day, then not
    assert out["hour_of_day"].to_list() == [23, 0, 9]
    assert out["day_of_week"].to_list() == [2, 3, 1]  # Tue, Wed, Mon
    assert out["month"].to_list() == [7, 7, 3]


def test_drop_nonexistent_hours_removes_spring_forward_gap_only() -> None:
    df = pl.DataFrame(
        {
            "hour": [
                datetime(2025, 3, 9, 1),
                datetime(2025, 3, 9, 2),  # does not exist in America/Toronto
                datetime(2025, 3, 9, 3),
                datetime(2025, 11, 2, 1),  # ambiguous (fall-back) but exists: kept
            ]
        }
    )
    out = drop_nonexistent_hours(df, "America/Toronto")
    assert out["hour"].to_list() == [
        datetime(2025, 3, 9, 1),
        datetime(2025, 3, 9, 3),
        datetime(2025, 11, 2, 1),
    ]
