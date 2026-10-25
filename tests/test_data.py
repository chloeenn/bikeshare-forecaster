"""Checks on the real pipeline output. Run `make process` first; a missing output is a failure."""

import json

import polars as pl
import pytest

from bikeshare.config import load_config

pytestmark = pytest.mark.data

CFG = load_config()
PANEL = CFG["paths"]["processed"] / "station_hourly.parquet"
SUMMARY = CFG["paths"]["interim"] / "cleaning_summary.json"
YEARS = CFG["trips"]["years"]
DROP_REASONS = [
    "missing_key",
    "missing_station",
    "unfinished_rental",
    "invalid_duration",
    "false_start",
    "duplicate",
]
EXPECTED_SCHEMA = {
    "station_id": pl.Int64,
    "hour": pl.Datetime("us"),
    "departures": pl.Int32,
    "arrivals": pl.Int32,
    "hour_of_day": pl.Int8,
    "day_of_week": pl.Int8,
    "month": pl.Int8,
    "is_holiday": pl.Boolean,
}


@pytest.fixture(scope="module")
def panel() -> pl.DataFrame:
    if not PANEL.exists():
        pytest.fail(f"{PANEL} is missing: run `make process` before `make test`")
    return pl.read_parquet(PANEL)


@pytest.fixture(scope="module")
def summary() -> dict[str, dict[str, int]]:
    if not SUMMARY.exists():
        pytest.fail(f"{SUMMARY} is missing: run `make process` before `make test`")
    return json.loads(SUMMARY.read_text())


@pytest.fixture(scope="module")
def cleaned() -> pl.DataFrame:
    files = [CFG["paths"]["interim"] / f"trips_{y}.parquet" for y in YEARS]
    missing = [f for f in files if not f.exists()]
    if missing:
        pytest.fail(f"missing cleaned trips {missing}: run `make process`")
    return pl.concat([pl.read_parquet(f) for f in files])


def test_schema(panel: pl.DataFrame) -> None:
    for col, dtype in EXPECTED_SCHEMA.items():
        assert panel.schema.get(col) == dtype, f"{col}: {panel.schema.get(col)} != {dtype}"
    assert {"temp_c", "precip_mm", "wind_kmh"} <= set(panel.columns)


def test_no_duplicate_station_hours(panel: pl.DataFrame) -> None:
    assert not panel.select(pl.struct("station_id", "hour").is_duplicated().any()).item()


def test_no_nulls_in_key_columns(panel: pl.DataFrame) -> None:
    for col in EXPECTED_SCHEMA:
        assert panel[col].null_count() == 0, col


def test_counts_are_non_negative(panel: pl.DataFrame) -> None:
    assert panel["departures"].min() >= 0
    assert panel["arrivals"].min() >= 0


def test_raw_rows_reconcile_with_csv_line_counts(summary: dict[str, dict[str, int]]) -> None:
    for year in YEARS:
        files = sorted((CFG["paths"]["raw"] / "trips" / str(year)).rglob("*.csv"))
        csv_rows = sum(
            pl.scan_csv(f, infer_schema=False).select(pl.len()).collect().item() for f in files
        )
        assert summary[str(year)]["raw_rows"] == csv_rows, year


def test_drops_account_for_every_raw_row(summary: dict[str, dict[str, int]]) -> None:
    for year, row in summary.items():
        assert row["raw_rows"] == row["kept"] + sum(row[r] for r in DROP_REASONS), year


def test_panel_departures_equal_cleaned_trips(
    panel: pl.DataFrame, summary: dict[str, dict[str, int]], cleaned: pl.DataFrame
) -> None:
    assert cleaned.height == sum(row["kept"] for row in summary.values())
    assert panel["departures"].sum() == cleaned.height


def test_panel_arrivals_equal_cleaned_trips_with_end_station(
    panel: pl.DataFrame, cleaned: pl.DataFrame
) -> None:
    assert panel["arrivals"].sum() == cleaned["end_station_id"].is_not_null().sum()


def test_panel_covers_configured_years(panel: pl.DataFrame) -> None:
    assert panel["hour"].min().year == min(YEARS)
    assert panel["hour"].max().year == max(YEARS)


def test_no_hours_inside_spring_forward_gap(panel: pl.DataFrame) -> None:
    tz = CFG["weather"]["local_timezone"]
    nonexistent = panel.filter(
        pl.col("hour").dt.replace_time_zone(tz, non_existent="null", ambiguous="earliest").is_null()
    )
    assert nonexistent.height == 0
