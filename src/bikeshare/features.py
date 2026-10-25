"""Origin-time features for forecasting departures at station s in hour t+h.

Every column in FEATURES uses only data with timestamp <= t (calendar columns describe the target
hour, which is known in advance). ABLATION columns use weather observed at t+h and are NOT a real
forecast; they exist only to measure an upper bound.
"""

import logging
from typing import Any

import polars as pl

from bikeshare.aggregate import add_calendar

log = logging.getLogger(__name__)

WEATHER = [
    "temp_c",
    "dew_point_c",
    "humidity_pct",
    "precip_mm",
    "wind_kmh",
    "visibility_km",
    "pressure_kpa",
]
AGE_CAP_H = 1344
DEMAND = [
    "lag_0",
    "lag_1",
    "lag_2",
    "lag_23",
    "same_how_1w",
    "same_how_2w",
    "roll_24h",
    "roll_168h",
    "mean_28d",
]
CALENDAR = ["hour_of_day", "day_of_week", "month", "is_holiday"]
WEATHER_FEATURES = WEATHER + [f"{c}_d3" for c in WEATHER]
FEATURES = ["station_code", "station_age_h", *DEMAND, *CALENDAR, *WEATHER_FEATURES]
ABLATION = [f"{c}_tgt" for c in WEATHER]
# Baselines read these: demand at t+h-168k for k=1..4 (the first two are also features).
SAME_HOW = [f"same_how_{k}w" for k in (1, 2, 3, 4)]
META = ["station_id", "hour", "target_hour", "y", *SAME_HOW, "precip_mm_tgt"]
CATEGORICAL = ["station_code"]


def select_stations(panel: pl.DataFrame, cfg: dict[str, Any]) -> pl.DataFrame:
    """Station table: filter and volume tiers use data before ``filter_end`` only.

    Kept stations have >= min_departures before filter_end. Stations with no rows before
    filter_end are flagged ``is_new_2026`` and reported separately.
    """
    m = cfg["modeling"]
    before = panel.filter(pl.col("hour") < pl.lit(m["filter_end"]).str.to_datetime())
    stats = before.group_by("station_id").agg(
        pl.col("departures").sum().alias("departures_hist"),
        pl.len().alias("active_hours_hist"),
    )
    kept = stats.filter(pl.col("departures_hist") >= m["min_departures"]).with_columns(
        (pl.col("departures_hist") / pl.col("active_hours_hist")).alias("rate")
    )
    kept = kept.sort("station_id").with_columns(
        pl.int_range(pl.len()).cast(pl.Int32).alias("station_code"),
        pl.when(pl.col("rate") >= pl.col("rate").quantile(2 / 3))
        .then(pl.lit("top"))
        .when(pl.col("rate") >= pl.col("rate").quantile(1 / 3))
        .then(pl.lit("mid"))
        .otherwise(pl.lit("low"))
        .alias("tier"),
        pl.lit(False).alias("is_new_2026"),
    )
    seen = set(stats["station_id"])
    new = (
        panel.select("station_id")
        .unique()
        .filter(~pl.col("station_id").is_in(list(seen)))
        .with_columns(
            pl.lit(None, dtype=pl.Int32).alias("station_code"),
            pl.lit(None, dtype=pl.String).alias("tier"),
            pl.lit(True).alias("is_new_2026"),
        )
    )
    cols = ["station_id", "station_code", "tier", "is_new_2026", "departures_hist", "rate"]
    return pl.concat(
        [kept.select(cols), new.select([c for c in cols if c in new.columns])], how="diagonal"
    ).sort("station_id")


def _features_one_chunk(g: pl.DataFrame, h: int, codes: pl.DataFrame) -> pl.DataFrame:
    """Features for a chunk of stations. ``g`` has columns station_id, hour, departures, weather."""
    s = "station_id"
    g = g.sort(s, "hour").upsample(time_column="hour", every="1h", group_by=s, maintain_order=True)
    y = pl.col("departures").cast(pl.Float32)
    shifted = lambda e, k: e.shift(k).over(s)  # noqa: E731
    exprs = [
        y.alias("lag_0"),
        shifted(y, 1).alias("lag_1"),
        shifted(y, 2).alias("lag_2"),
        shifted(y, 23).alias("lag_23"),
        y.rolling_mean(24, min_samples=1).over(s).alias("roll_24h"),
        y.rolling_mean(168, min_samples=1).over(s).alias("roll_168h"),
        y.rolling_mean(672, min_samples=1).over(s).alias("mean_28d"),
        shifted(y, -h).alias("y"),
        pl.int_range(pl.len()).over(s).clip(upper_bound=AGE_CAP_H).alias("station_age_h"),
        (pl.col("hour") + pl.duration(hours=h)).alias("target_hour"),
    ]
    exprs += [shifted(y, 168 * k - h).alias(f"same_how_{k}w") for k in (1, 2, 3, 4)]
    for c in WEATHER:
        w = pl.col(c).cast(pl.Float32)
        exprs += [(w - shifted(w, 3)).alias(f"{c}_d3"), shifted(w, -h).alias(f"{c}_tgt")]
    g = g.with_columns(exprs)
    cal = add_calendar(g.select(pl.col("target_hour").alias("hour"))).drop("hour")
    g = g.hstack(cal.with_columns(pl.col("is_holiday").cast(pl.Int8)))
    g = g.join(codes, on=s, how="left")
    # Keep origins that exist (not re-inserted DST gap rows) and have a target.
    g = g.filter(pl.col("lag_0").is_not_null() & pl.col("y").is_not_null())
    return g.select(
        pl.col("station_id").cast(pl.Int32),
        "hour",
        "target_hour",
        *[pl.col(c).cast(pl.Float32) for c in ["y", *SAME_HOW, *DEMAND[:4], *DEMAND[6:]]],
        pl.col("station_code"),
        pl.col("station_age_h").cast(pl.Int16),
        pl.col("hour_of_day"),
        pl.col("day_of_week"),
        pl.col("month"),
        pl.col("is_holiday"),
        *[pl.col(c).cast(pl.Float32) for c in WEATHER],
        *[pl.col(f"{c}_d3") for c in WEATHER],
        *[pl.col(c) for c in ABLATION],
    )


def build_features(
    panel: pl.DataFrame, stations: pl.DataFrame, h: int, chunk: int = 100
) -> pl.DataFrame:
    """Feature table for horizon ``h`` over the stations in ``stations``.

    ``panel`` must already be limited to the modelling window. Chunking by station is exact
    because every feature is computed within a station.
    """
    codes = stations.select("station_id", "station_code")
    ids = stations["station_id"].to_list()
    cols = ["station_id", "hour", "departures", *WEATHER]
    parts = []
    for i in range(0, len(ids), chunk):
        part = panel.filter(pl.col("station_id").is_in(ids[i : i + chunk])).select(cols)
        parts.append(_features_one_chunk(part, h, codes))
    return pl.concat(parts)


def load_panel(cfg: dict[str, Any]) -> pl.DataFrame:
    """Load the hourly station panel, dropping the 2026-10-01 tail rows."""
    path = cfg["paths"]["processed"] / "station_hourly.parquet"
    last = pl.lit(cfg["modeling"]["last_hour"]).str.to_datetime()
    return pl.read_parquet(path).filter(pl.col("hour") <= last)


def build_all(cfg: dict[str, Any]) -> None:
    """Write stations.parquet and features_h{h}.parquet to data/processed."""
    panel = load_panel(cfg)
    stations = select_stations(panel, cfg)
    out = cfg["paths"]["processed"]
    stations.write_parquet(out / "stations.parquet")
    log.info(
        "stations: %d kept, %d new in 2026",
        stations.filter(~pl.col("is_new_2026")).height,
        stations.filter(pl.col("is_new_2026")).height,
    )
    for h in cfg["modeling"]["horizons"]:
        feats = build_features(panel, stations, h)
        feats.write_parquet(out / f"features_h{h}.parquet")
        log.info("h=%d: %d rows, %.1f GB in memory", h, feats.height, feats.estimated_size("gb"))
