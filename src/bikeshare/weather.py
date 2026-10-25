import logging
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

# Normalized ECCC header -> canonical name. Units: degC, %, mm, km/h, km, kPa.
COLUMNS = {
    "Date/Time (LST)": "ts_lst",
    "Temp (°C)": "temp_c",
    "Dew Point Temp (°C)": "dew_point_c",
    "Rel Hum (%)": "humidity_pct",
    "Precip. Amount (mm)": "precip_mm",
    "Wind Spd (km/h)": "wind_kmh",
    "Visibility (km)": "visibility_km",
    "Stn Press (kPa)": "pressure_kpa",
    "Weather": "condition",
}


def read_weather_csv(path: Path) -> pl.DataFrame:
    """Read one ECCC monthly CSV (BOM-prefixed) into canonical columns."""
    df = pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")
    df = df.rename({c: c.replace("﻿", "") for c in df.columns})
    missing = set(COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}")
    df = df.select(list(COLUMNS)).rename(COLUMNS)
    numeric = [c for c in df.columns if c not in ("ts_lst", "condition")]
    return df.with_columns(
        pl.col("ts_lst").str.strptime(pl.Datetime("us"), "%Y-%m-%d %H:%M"),
        *[pl.col(c).cast(pl.Float64, strict=False) for c in numeric],
        pl.col("condition").replace("", None),
    )


def lst_to_local(df: pl.DataFrame, offset_hours: int, tz: str) -> pl.DataFrame:
    """Convert fixed-offset standard time to naive local wall-clock time (DST-aware).

    The autumn fall-back hour appears twice in local wall-clock time; the first row is kept
    so ``hour`` stays unique, matching how trips are binned.
    """
    return (
        df.with_columns(
            (pl.col("ts_lst") - pl.duration(hours=offset_hours))
            .dt.replace_time_zone("UTC")
            .dt.convert_time_zone(tz)
            .dt.replace_time_zone(None)
            .alias("hour")
        )
        .drop("ts_lst")
        .unique(subset="hour", keep="first", maintain_order=True)
        .sort("hour")
    )


def build_weather(cfg: dict[str, Any]) -> Path:
    """Combine raw monthly CSVs into ``data/interim/weather_hourly.parquet``."""
    files = sorted((cfg["paths"]["raw"] / "weather").glob("*.csv"))
    if not files:
        raise FileNotFoundError("no weather CSVs; run the download step first")
    w = cfg["weather"]
    df = lst_to_local(
        pl.concat([read_weather_csv(f) for f in files]),
        w["source_utc_offset_hours"],
        w["local_timezone"],
    )
    temp_null_pct = 100 * df["temp_c"].null_count() / df.height
    log.info("weather: %d hourly rows, temp null %.1f%%", df.height, temp_null_pct)
    dest = cfg["paths"]["interim"] / "weather_hourly.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.write_parquet(dest)
    return dest
