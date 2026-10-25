import random
from datetime import datetime, timedelta

import polars as pl
import pytest

from bikeshare.aggregate import drop_nonexistent_hours
from bikeshare.features import (
    ABLATION,
    FEATURES,
    SAME_HOW,
    WEATHER,
    build_features,
)

TZ = "America/Toronto"
START = datetime(2025, 2, 20)
N_HOURS = 24 * 40  # spans the 2025-03-09 spring-forward gap


def _panel(seed: int = 0, n_hours: int = N_HOURS) -> pl.DataFrame:
    rng = random.Random(seed)
    hours = pl.datetime_range(START, START + timedelta(hours=n_hours), "1h", eager=True)
    rows = []
    for sid in (7000, 7001):
        for h in hours:
            rows.append(
                {
                    "station_id": sid,
                    "hour": h,
                    "departures": rng.randint(0, 6),
                    **{c: rng.uniform(-5, 25) for c in WEATHER},
                }
            )
    panel = pl.DataFrame(rows).cast({"station_id": pl.Int64})
    return drop_nonexistent_hours(panel, TZ)


def _stations() -> pl.DataFrame:
    return pl.DataFrame({"station_id": [7000, 7001], "station_code": [0, 1]}).cast(
        {"station_code": pl.Int32}
    )


def _at(panel: pl.DataFrame, sid: int, ts: datetime, col: str) -> float | None:
    r = panel.filter((pl.col("station_id") == sid) & (pl.col("hour") == ts))
    return r[col][0] if r.height else None


@pytest.mark.parametrize("h", [1, 3])
def test_feature_values_match_manual_lookup(h: int) -> None:
    panel = _panel()
    feats = build_features(panel, _stations(), h)
    t = datetime(2025, 3, 20, 14)
    row = feats.filter((pl.col("station_id") == 7000) & (pl.col("hour") == t)).row(0, named=True)
    dep = lambda ts: _at(panel, 7000, ts, "departures")  # noqa: E731
    hr = timedelta(hours=1)
    assert row["target_hour"] == t + h * hr
    assert row["y"] == dep(t + h * hr)
    assert row["lag_0"] == dep(t)
    assert row["lag_1"] == dep(t - hr)
    assert row["lag_23"] == dep(t - 23 * hr)
    assert row["same_how_1w"] == dep(t + h * hr - timedelta(days=7))
    assert row["same_how_4w"] == dep(t + h * hr - timedelta(days=28))
    last24 = [dep(t - i * hr) for i in range(24)]
    assert row["roll_24h"] == pytest.approx(sum(last24) / 24, rel=1e-5)
    assert row["temp_c"] == pytest.approx(_at(panel, 7000, t, "temp_c"), rel=1e-5)
    assert row["temp_c_d3"] == pytest.approx(
        _at(panel, 7000, t, "temp_c") - _at(panel, 7000, t - 3 * hr, "temp_c"), rel=1e-4
    )
    assert row["temp_c_tgt"] == pytest.approx(_at(panel, 7000, t + h * hr, "temp_c"), rel=1e-5)
    assert row["hour_of_day"] == (t + h * hr).hour  # calendar describes the TARGET hour


def test_lags_are_wall_clock_across_dst_gap() -> None:
    # Origin one week after the gap: "168h back" must land on the same hour-of-week.
    panel = _panel()
    feats = build_features(panel, _stations(), 1)
    t = datetime(2025, 3, 16, 12)
    row = feats.filter((pl.col("station_id") == 7000) & (pl.col("hour") == t)).row(0, named=True)
    assert row["same_how_1w"] == _at(panel, 7000, datetime(2025, 3, 9, 13), "departures")


def test_rows_never_sit_on_nonexistent_hours() -> None:
    feats = build_features(_panel(), _stations(), 3)
    gap = datetime(2025, 3, 9, 2)
    assert feats.filter((pl.col("hour") == gap) | (pl.col("target_hour") == gap)).height == 0


def test_station_age_counts_hours_since_first_active_hour_and_is_capped() -> None:
    feats = build_features(_panel(n_hours=1500), _stations(), 1).filter(
        pl.col("station_id") == 7000
    )
    ordered = feats.sort("hour")
    first = ordered.row(0, named=True)
    assert first["hour"] == START and first["station_age_h"] == 0
    assert ordered["station_age_h"][100] == 100
    assert ordered["station_age_h"].max() == 1344


def test_features_at_t_do_not_depend_on_data_after_t() -> None:
    """Leakage test: rewrite all data after t0; features at origins <= t0 must not change."""
    h, t0 = 3, datetime(2025, 3, 20, 12)
    panel = _panel()
    changed = panel.with_columns(
        pl.when(pl.col("hour") > t0)
        .then(pl.col("departures") + 100)
        .otherwise(pl.col("departures"))
        .alias("departures"),
        *[
            pl.when(pl.col("hour") > t0).then(pl.col(c) + 50.0).otherwise(pl.col(c)).alias(c)
            for c in WEATHER
        ],
    )
    a = (
        build_features(panel, _stations(), h)
        .filter(pl.col("hour") <= t0)
        .sort("station_id", "hour")
    )
    b = (
        build_features(changed, _stations(), h)
        .filter(pl.col("hour") <= t0)
        .sort("station_id", "hour")
    )
    assert a.height == b.height > 0
    cols = list(dict.fromkeys([*FEATURES, *SAME_HOW, "hour", "target_hour", "station_id"]))
    assert a.select(cols).equals(b.select(cols))
    # Sanity: the perturbation is visible where it should be (targets and the t+h ablation columns).
    assert not a.select("y").equals(b.select("y"))
    assert not a.select(ABLATION).equals(b.select(ABLATION))
