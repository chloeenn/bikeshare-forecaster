"""Metrics, the model-vs-baseline comparison table and slice analysis."""

from typing import Any

import numpy as np
import polars as pl

METRICS = ("rmse", "deviance", "mae")  # headline first; MAE rewards the median on sparse counts
BASELINES = ("bl_1w", "bl_4w")
HOUR_BUCKETS = [
    (0, 5, "night 00-05"),
    (6, 9, "morning peak 06-09"),
    (10, 15, "midday 10-15"),
    (16, 19, "evening peak 16-19"),
    (20, 23, "late 20-23"),
]


def metric_sums(y: np.ndarray, p: np.ndarray, eps: float) -> dict[str, float]:
    """Sufficient statistics so folds can be pooled exactly. Deviance clips predictions at eps."""
    y = y.astype(np.float64)
    p = p.astype(np.float64)
    mu = np.maximum(p, eps)
    term = np.where(y > 0, y * np.log(np.where(y > 0, y, 1.0) / mu), 0.0)
    return {
        "n": float(len(y)),
        "se": float(np.sum((y - p) ** 2)),
        "ae": float(np.sum(np.abs(y - p))),
        "dev": float(np.sum(2.0 * (term - (y - mu)))),
    }


def combine(parts: list[dict[str, float]]) -> dict[str, float]:
    return {k: sum(p[k] for p in parts) for k in ("n", "se", "ae", "dev")}


def finalize(s: dict[str, float]) -> dict[str, float]:
    n = s["n"]
    return {"rmse": (s["se"] / n) ** 0.5, "deviance": s["dev"] / n, "mae": s["ae"] / n, "n": n}


def score(df: pl.DataFrame, col: str, eps: float) -> dict[str, float]:
    return finalize(metric_sums(df["y"].to_numpy(), df[col].to_numpy(), eps))


def common_subset(df: pl.DataFrame) -> pl.DataFrame:
    """Rows where both baselines exist."""
    return df.filter(pl.col("bl_1w").is_not_null() & pl.col("bl_4w").is_not_null())


def improvement(model: float, baseline: float) -> float:
    return 100.0 * (1.0 - model / baseline)


def comparison(df: pl.DataFrame, model_cols: dict[str, str], eps: float) -> pl.DataFrame:
    """Head-to-head table on the common subset, plus the model(s) on all rows.

    ``model_cols`` maps a display name to a prediction column. Each model gets a
    "(common subset)" row with % improvement over the best baseline per metric, and an
    "(all rows)" row. Rows with a null prediction are dropped for that model only.
    """
    common = common_subset(df)
    rows: list[dict[str, Any]] = []
    cov = common.height / df.height
    base = {c: score(common, c, eps) for c in ("p_zero", *BASELINES)}
    best = {m: min(base[b][m] for b in BASELINES) for m in METRICS}
    labels = {
        "p_zero": "Predict zero (reference)",
        "bl_1w": "Baseline: same hour last week",
        "bl_4w": "Baseline: mean of same hour, last 4 weeks",
    }
    for col, label in labels.items():
        rows.append({"model": label, "subset": "common", **base[col], "coverage": cov})
    for name, col in model_cols.items():
        sub = common.filter(pl.col(col).is_not_null())
        s = score(sub, col, eps)
        rows.append(
            {
                "model": name,
                "subset": "common",
                **s,
                "coverage": cov,
                **{f"{m}_vs_best_baseline_pct": improvement(s[m], best[m]) for m in METRICS},
            }
        )
        full = df.filter(pl.col(col).is_not_null())
        rows.append(
            {
                "model": name,
                "subset": "all rows",
                **score(full, col, eps),
                "coverage": full.height / df.height,
            }
        )
    return pl.DataFrame(rows, infer_schema_length=None)


def add_slice_columns(df: pl.DataFrame, stations: pl.DataFrame, new_hours: int) -> pl.DataFrame:
    """Attach slice labels. All use the target hour except history length (at the origin)."""
    bucket = pl.lit(None, dtype=pl.String)
    for lo, hi, name in HOUR_BUCKETS:
        bucket = (
            pl.when(pl.col("hour_of_day").is_between(lo, hi)).then(pl.lit(name)).otherwise(bucket)
        )
    return df.join(stations.select("station_id", "tier"), on="station_id", how="left").with_columns(
        bucket.alias("hour_bucket"),
        pl.when(pl.col("precip_mm_tgt").is_null())
        .then(None)
        .when(pl.col("precip_mm_tgt") > 0)
        .then(pl.lit("rain"))
        .otherwise(pl.lit("no rain"))
        .alias("rain"),
        pl.when(pl.col("is_holiday") == 1)
        .then(pl.lit("holiday"))
        .when(pl.col("day_of_week") >= 6)
        .then(pl.lit("weekend"))
        .otherwise(pl.lit("weekday"))
        .alias("day_type"),
        pl.when(pl.col("station_age_h") < new_hours)
        .then(pl.lit("new (<8 weeks history)"))
        .otherwise(pl.lit("established"))
        .alias("history"),
    )


def slice_table(df: pl.DataFrame, model_col: str, eps: float) -> pl.DataFrame:
    """Model vs best baseline per slice value on the common subset.

    ``win_<metric>`` is True when the model is strictly better than the best baseline *in that
    slice* (the best baseline is chosen per slice and metric).
    """
    common = common_subset(df).filter(pl.col(model_col).is_not_null())
    rows: list[dict[str, Any]] = []
    for dim in ("tier", "hour_bucket", "rain", "day_type", "history"):
        for value in sorted(v for v in common[dim].unique().to_list() if v is not None):
            part = common.filter(pl.col(dim) == value)
            m = score(part, model_col, eps)
            b = {c: score(part, c, eps) for c in BASELINES}
            row: dict[str, Any] = {"slice": dim, "value": value, "n": int(m["n"])}
            for k in METRICS:
                best = min(b[c][k] for c in BASELINES)
                row[f"model_{k}"] = m[k]
                row[f"best_baseline_{k}"] = best
                row[f"{k}_vs_best_baseline_pct"] = improvement(m[k], best)
                row[f"win_{k}"] = m[k] < best
            rows.append(row)
    return pl.DataFrame(rows)
