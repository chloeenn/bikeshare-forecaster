from datetime import datetime

import polars as pl

from bikeshare.weather import lst_to_local


def _lst(*stamps: str) -> pl.DataFrame:
    return pl.DataFrame({"ts_lst": [datetime.fromisoformat(s) for s in stamps], "temp_c": 1.0})


def test_winter_local_equals_lst() -> None:
    out = lst_to_local(_lst("2024-01-15 08:00"), -5, "America/Toronto")
    assert out["hour"][0] == datetime(2024, 1, 15, 8)


def test_summer_local_is_lst_plus_one_hour() -> None:
    out = lst_to_local(_lst("2024-07-15 08:00"), -5, "America/Toronto")
    assert out["hour"][0] == datetime(2024, 7, 15, 9)


def test_fall_back_hour_is_deduplicated() -> None:
    # 2024-11-03 01:00 EDT and 01:00 EST both read 01:00 on the wall clock.
    out = lst_to_local(
        _lst("2024-11-02 21:00", "2024-11-02 22:00", "2024-11-02 23:00"), -5, "America/Toronto"
    )
    assert out["hour"].is_unique().all()
