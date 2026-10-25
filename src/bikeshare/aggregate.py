import logging
from pathlib import Path
from typing import Any

import holidays
import polars as pl

log = logging.getLogger(__name__)


def hourly_events(trips: pl.DataFrame) -> pl.DataFrame:
    """Count departures and arrivals per (station_id, hour) from trip rows."""
    deps = trips.group_by(
        pl.col("start_station_id").alias("station_id"),
        pl.col("start_time").dt.truncate("1h").alias("hour"),
    ).agg(pl.len().cast(pl.Int32).alias("departures"))
    arrs = (
        trips.filter(pl.col("end_station_id").is_not_null() & pl.col("end_time").is_not_null())
        .group_by(
            pl.col("end_station_id").alias("station_id"),
            pl.col("end_time").dt.truncate("1h").alias("hour"),
        )
        .agg(pl.len().cast(pl.Int32).alias("arrivals"))
    )
    return deps.join(arrs, on=["station_id", "hour"], how="full", coalesce=True).fill_null(0)


def zero_fill(events: pl.DataFrame) -> pl.DataFrame:
    """Expand to a complete hourly grid between each station's first and last activity.

    Hours outside that span are not zero-filled: the station did not exist yet or was retired,
    so a 0 there would be a fabricated label.
    """
    spans = events.group_by("station_id").agg(
        pl.col("hour").min().alias("first_hour"), pl.col("hour").max().alias("last_hour")
    )
    grid = spans.select(
        "station_id",
        pl.datetime_ranges("first_hour", "last_hour", interval="1h").alias("hour"),
    ).explode("hour")
    return (
        grid.join(events, on=["station_id", "hour"], how="left")
        .with_columns(pl.col("departures", "arrivals").fill_null(0))
        .sort("station_id", "hour")
    )


def drop_nonexistent_hours(panel: pl.DataFrame, tz: str) -> pl.DataFrame:
    """Remove wall-clock hours skipped by the spring-forward DST jump (e.g. 02:00 on 2025-03-09).

    Naive hourly ranges include them, but no trip or weather reading can exist there, so a
    0 label would be fabricated.
    """
    exists = (
        pl.col("hour")
        .dt.replace_time_zone(tz, non_existent="null", ambiguous="earliest")
        .is_not_null()
    )
    return panel.filter(exists)


def add_weather(panel: pl.DataFrame, weather: pl.DataFrame) -> pl.DataFrame:
    """Left-join same-hour observed weather. Features built from it later must be lagged."""
    return panel.join(weather, on="hour", how="left")


def add_calendar(panel: pl.DataFrame) -> pl.DataFrame:
    """Add hour_of_day, day_of_week (1=Mon..7=Sun), month and an Ontario holiday flag.

    ``hour`` is naive local wall-clock time, so the holiday check uses the local date.
    """
    years = range(panel["hour"].min().year, panel["hour"].max().year + 1)
    days = list(holidays.Canada(subdiv="ON", years=years))
    return panel.with_columns(
        pl.col("hour").dt.hour().cast(pl.Int8).alias("hour_of_day"),
        pl.col("hour").dt.weekday().cast(pl.Int8).alias("day_of_week"),
        pl.col("hour").dt.month().cast(pl.Int8).alias("month"),
        pl.col("hour").dt.date().is_in(days).alias("is_holiday"),
    )


def build_panel(cfg: dict[str, Any]) -> Path:
    """Aggregate all interim trip years to a station-hour panel joined with weather."""
    interim = cfg["paths"]["interim"]
    files = [interim / f"trips_{y}.parquet" for y in cfg["trips"]["years"]]
    trips = pl.concat([pl.read_parquet(f) for f in files if f.exists()])
    # A trip spanning New Year is in one year's file only, so duplicates cannot occur across files.
    panel = add_weather(
        add_calendar(
            drop_nonexistent_hours(
                zero_fill(hourly_events(trips)), cfg["weather"]["local_timezone"]
            )
        ),
        pl.read_parquet(interim / "weather_hourly.parquet"),
    )
    log.info(
        "panel: %d rows, %d stations, %s -> %s",
        panel.height,
        panel["station_id"].n_unique(),
        panel["hour"].min(),
        panel["hour"].max(),
    )
    dest = cfg["paths"]["processed"] / "station_hourly.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    panel.write_parquet(dest)
    return dest
