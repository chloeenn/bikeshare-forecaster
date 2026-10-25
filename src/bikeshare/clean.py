import json
import logging
import re
from pathlib import Path
from typing import Any

import polars as pl

log = logging.getLogger(__name__)

# Normalized header -> canonical name. Headers differ by year ("Trip  Duration" vs "Trip_Duration").
REQUIRED = {
    "trip_id": "trip_id",
    "trip_duration": "duration_s",
    "start_station_id": "start_station_id",
    "start_time": "start_time",
    "end_station_id": "end_station_id",
    "end_time": "end_time",
}
OPTIONAL = {"user_type": "user_type", "bike_model": "bike_model"}
TIME_FORMATS = ["%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M", "%m/%d/%Y %H:%M", "%m/%d/%Y %H:%M:%S"]


def normalize_header(name: str) -> str:
    """'\\ufeffTrip  Duration' and 'Trip_Duration' both become 'trip_duration'."""
    return re.sub(r"[^a-z0-9]+", "_", name.replace("﻿", "").lower()).strip("_")


def parse_time(col: str) -> pl.Expr:
    """Parse a string column trying each known format; unparseable values become null."""
    return pl.coalesce(
        [pl.col(col).str.strptime(pl.Datetime("us"), f, strict=False) for f in TIME_FORMATS]
    )


def to_int(col: str) -> pl.Expr:
    """Cast numeric strings to Int64, including float-formatted ids such as '7130.0'."""
    return pl.col(col).cast(pl.Float64, strict=False).cast(pl.Int64, strict=False)


def read_trips_csv(path: Path) -> pl.DataFrame:
    """Read one monthly CSV as strings and map it onto the canonical column names."""
    df = pl.read_csv(path, infer_schema=False, encoding="utf8-lossy")
    df = df.rename({c: normalize_header(c) for c in df.columns})
    missing = set(REQUIRED) - set(df.columns)
    if missing:
        raise ValueError(f"{path.name}: missing {sorted(missing)}; columns are {df.columns}")
    mapping = REQUIRED | {k: v for k, v in OPTIONAL.items() if k in df.columns}
    df = df.select(list(mapping)).rename(mapping)
    for col in OPTIONAL.values():
        if col not in df.columns:
            df = df.with_columns(pl.lit(None, dtype=pl.String).alias(col))
    return df


def cast_trips(df: pl.DataFrame) -> pl.DataFrame:
    """Cast strings to typed columns. 'NULL'/empty station ids and bad numbers become null."""
    return df.select(
        to_int("trip_id"),
        to_int("duration_s"),
        to_int("start_station_id"),
        parse_time("start_time").alias("start_time"),
        to_int("end_station_id"),
        parse_time("end_time").alias("end_time"),
        pl.col("user_type").str.strip_chars(),
        pl.col("bike_model").str.strip_chars(),
    )


def clean_trips(
    df: pl.DataFrame, min_duration_s: int, max_duration_s: int
) -> tuple[pl.DataFrame, dict[str, int]]:
    """Drop unusable rows, returning the kept rows and a count per drop reason.

    Reasons are applied in order and are mutually exclusive: a row is counted under the first
    rule it fails. A missing end station is fine (still a valid departure); a missing end time
    means the rental never finished and is dropped.
    """
    same_station = (pl.col("start_station_id") == pl.col("end_station_id")).fill_null(False)
    rules = [
        ("missing_key", pl.col("trip_id").is_not_null() & pl.col("start_time").is_not_null()),
        ("missing_station", pl.col("start_station_id").is_not_null()),
        ("unfinished_rental", pl.col("end_time").is_not_null()),
        ("invalid_duration", pl.col("duration_s").is_between(0, max_duration_s)),
        ("false_start", ~((pl.col("duration_s") < min_duration_s) & same_station)),
    ]
    drops: dict[str, int] = {}
    for reason, keep in rules:
        before = df.height
        df = df.filter(keep)
        drops[reason] = before - df.height
    before = df.height
    df = df.unique(subset="trip_id", keep="first", maintain_order=True)
    drops["duplicate"] = before - df.height
    return df, drops


def clean_year(year: int, cfg: dict[str, Any]) -> dict[str, int]:
    """Clean one year's CSVs into ``data/interim/trips_<year>.parquet``; return row counts."""
    src = cfg["paths"]["raw"] / "trips" / str(year)
    files = sorted(src.rglob("*.csv"))
    if not files:
        raise FileNotFoundError(f"no CSVs under {src}")
    c = cfg["cleaning"]
    raw = pl.concat([cast_trips(read_trips_csv(f)) for f in files])
    out, drops = clean_trips(raw, c["min_duration_s"], c["max_duration_s"])
    dest = cfg["paths"]["interim"] / f"trips_{year}.parquet"
    dest.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(dest)
    return {"raw_rows": raw.height, **drops, "kept": out.height}


def write_summary(counts: dict[int, dict[str, int]], path: Path) -> str:
    """Save per-year cleaning counts as JSON and return a printable table."""
    path.write_text(json.dumps(counts, indent=2))
    cols = list(next(iter(counts.values())))
    lines = ["year " + " ".join(f"{c:>18}" for c in cols)]
    for year, row in counts.items():
        lines.append(f"{year} " + " ".join(f"{row[c]:>18,}" for c in cols))
    return "\n".join(lines)
