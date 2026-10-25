"""Seasonal-naive baselines. Both read demand at t+h-168k, which is always <= t for h <= 168."""

import polars as pl


def same_hour_last_week() -> pl.Expr:
    """Demand at the target hour one week earlier. Null for stations younger than a week."""
    return pl.col("same_how_1w")


def mean_same_hour_last_4_weeks() -> pl.Expr:
    """Mean demand at the same hour-of-week over the last four weeks; null unless all four exist."""
    weeks = [pl.col(f"same_how_{k}w") for k in (1, 2, 3, 4)]
    return pl.when(pl.all_horizontal(w.is_not_null() for w in weeks)).then(
        pl.sum_horizontal(weeks) / 4
    )


def add_baselines(df: pl.DataFrame) -> pl.DataFrame:
    """Add baseline predictions ``bl_1w`` and ``bl_4w`` plus the predict-zero reference."""
    return df.with_columns(
        same_hour_last_week().alias("bl_1w"),
        mean_same_hour_last_4_weeks().alias("bl_4w"),
        pl.lit(0.0, dtype=pl.Float32).alias("p_zero"),
    )
