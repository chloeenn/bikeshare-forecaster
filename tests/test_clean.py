from datetime import datetime
from pathlib import Path

import polars as pl
import pytest

from bikeshare.clean import cast_trips, clean_trips, normalize_header, read_trips_csv

HEADER_2023 = (
    "﻿Trip Id,Trip  Duration,Start Station Id,Start Time,Start Station Name,"
    "End Station Id,End Time,End Station Name,Bike Id,User Type\n"
)
HEADER_2025 = (
    "Trip_Id,Trip_Duration,Start_Station_Id,Start_Time,Start_Station_Name,"
    "End_Station_Id,End_Time,End_Station_Name,Bike_Id,User_Type,Bike_Model\n"
)

ROW_2023 = "1,840,7022,01/31/2023 13:05,A,7703,01/31/2023 13:19,NULL,9,Casual Member\n"
ROW_2025 = "2,193,7192,2025-01-01 00:02:15,A,7155.0,2025-01-01 00:05:28,B,9,Member,ICONIC\n"


def test_normalize_header_handles_bom_spaces_and_underscores() -> None:
    assert normalize_header("﻿Trip  Duration") == "trip_duration"
    assert normalize_header("Trip_Duration") == "trip_duration"


def test_both_schemas_map_to_same_columns(tmp_path: Path) -> None:
    old = tmp_path / "old.csv"
    old.write_text(
        HEADER_2023 + "1,840,7022,01/31/2023 13:05,A,7703,01/31/2023 13:19,NULL,9,Casual Member\n"
    )
    new = tmp_path / "new.csv"
    new.write_text(
        HEADER_2025
        + "2,193,7192,2025-01-01 00:02:15,A,7155.0,2025-01-01 00:05:28,B,9,Member,ICONIC\n"
    )
    a, b = cast_trips(read_trips_csv(old)), cast_trips(read_trips_csv(new))
    assert a.columns == b.columns
    assert a["start_time"][0] == datetime(2023, 1, 31, 13, 5)  # MM/DD/YYYY, not DD/MM
    assert b["end_station_id"][0] == 7155  # '7155.0' must not become null
    assert a["bike_model"][0] is None


def test_missing_required_column_raises(tmp_path: Path) -> None:
    bad = tmp_path / "bad.csv"
    bad.write_text("Trip_Id,Foo\n1,2\n")
    with pytest.raises(ValueError, match="missing"):
        read_trips_csv(bad)


def _trips(rows: list[tuple]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema=[
            "trip_id",
            "duration_s",
            "start_station_id",
            "start_time",
            "end_station_id",
            "end_time",
        ],
        orient="row",
    ).with_columns(pl.col("start_time", "end_time").str.to_datetime())


def test_clean_trips_filters_dedupes_and_counts_reasons() -> None:
    t = "2024-01-01 00:30:00"
    df = _trips(
        [
            (1, 300, 10, "2024-01-01 00:00:00", 11, t),
            (1, 300, 10, "2024-01-01 00:00:00", 11, t),  # duplicate id
            (2, 20, 10, "2024-01-01 01:00:00", 10, t),  # false start
            (3, 20, 10, "2024-01-01 02:00:00", 11, t),  # short but real trip: kept
            (4, 999999, 10, "2024-01-01 03:00:00", 11, t),  # too long
            (5, 100, None, "2024-01-01 04:00:00", 11, t),  # no start station
            (6, 100, 10, "2024-01-01 05:00:00", None, t),  # no end station: still a departure
            (7, 0, 10, "2024-01-01 06:00:00", None, None),  # unfinished rental
            (8, 20, 10, "2024-01-01 07:00:00", None, t),  # short, end unknown: kept
            (None, 100, 10, "2024-01-01 08:00:00", 11, t),  # no trip id
        ]
    )
    out, drops = clean_trips(df, min_duration_s=60, max_duration_s=86400)
    assert out["trip_id"].to_list() == [1, 3, 6, 8]
    assert drops == {
        "missing_key": 1,
        "missing_station": 1,
        "unfinished_rental": 1,
        "invalid_duration": 1,
        "false_start": 1,
        "duplicate": 1,
    }
    assert df.height == out.height + sum(drops.values())
