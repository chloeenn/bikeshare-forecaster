import math

import numpy as np
import polars as pl
import pytest

from bikeshare.evaluate import comparison, finalize, metric_sums, slice_table


def test_rmse_mae_deviance_on_known_values() -> None:
    y = np.array([0.0, 2.0, 4.0])
    p = np.array([1.0, 2.0, 2.0])
    m = finalize(metric_sums(y, p, eps=1e-3))
    assert m["mae"] == pytest.approx(1.0)
    assert m["rmse"] == pytest.approx(math.sqrt(5 / 3))
    # unit deviance 2*(y*log(y/mu) - (y-mu)): y=0,mu=1 -> 2; y=2,mu=2 -> 0; y=4,mu=2 -> 2*(4*ln2-2)
    expected = (2.0 + 0.0 + 2 * (4 * math.log(2) - 2)) / 3
    assert m["deviance"] == pytest.approx(expected)


def test_deviance_clips_zero_predictions_at_eps() -> None:
    m = finalize(metric_sums(np.array([1.0]), np.array([0.0]), eps=1e-3))
    assert m["deviance"] == pytest.approx(2 * (math.log(1 / 1e-3) - (1 - 1e-3)))
    assert math.isfinite(m["deviance"])


def test_folds_pool_exactly_via_sufficient_statistics() -> None:
    rng = np.random.default_rng(0)
    y, p = rng.poisson(2, 100).astype(float), rng.uniform(0, 4, 100)
    whole = finalize(metric_sums(y, p, 1e-3))
    parts = [metric_sums(y[:30], p[:30], 1e-3), metric_sums(y[30:], p[30:], 1e-3)]
    from bikeshare.evaluate import combine

    pooled = finalize(combine(parts))
    assert pooled == pytest.approx(whole)


def _frame() -> pl.DataFrame:
    n = 8
    return pl.DataFrame(
        {
            "y": [0.0, 1.0, 2.0, 3.0] * 2,
            "bl_1w": [0.0, 0.0, 4.0, 0.0, 1.0, 1.0, 1.0, 1.0],
            "bl_4w": [0.5, 0.5, 0.5, 0.5, 2.0, 2.0, 2.0, None],
            "p_lgbm": [0.1, 1.0, 2.0, 3.0, 3.0, 3.0, 3.0, 3.0],
            "p_zero": [0.0] * n,
            "tier": ["top"] * 4 + ["low"] * 4,
            "hour_bucket": ["night 00-05"] * n,
            "rain": ["no rain"] * n,
            "day_type": ["weekday"] * n,
            "history": ["established"] * n,
        }
    )


def test_comparison_uses_common_subset_and_reports_coverage() -> None:
    t = comparison(_frame(), {"LightGBM": "p_lgbm"}, eps=1e-3)
    common = t.filter((pl.col("subset") == "common") & (pl.col("model") == "LightGBM"))
    assert common["n"][0] == 7  # the row with a null bl_4w is excluded
    assert common["coverage"][0] == pytest.approx(7 / 8)
    all_rows = t.filter((pl.col("subset") == "all rows") & (pl.col("model") == "LightGBM"))
    assert all_rows["n"][0] == 8
    zero = t.filter(pl.col("model").str.starts_with("Predict zero"))
    assert zero["mae"][0] == pytest.approx(np.mean([0, 1, 2, 3, 0, 1, 2]))


def test_slice_table_flags_slices_where_model_loses() -> None:
    s = slice_table(_frame(), "p_lgbm", eps=1e-3)
    top = s.filter((pl.col("slice") == "tier") & (pl.col("value") == "top")).row(0, named=True)
    assert top["win_rmse"] is True
    low = s.filter((pl.col("slice") == "tier") & (pl.col("value") == "low")).row(0, named=True)
    assert low["win_rmse"] is False  # RMSE 2.2 vs best baseline 0.8: flagged as a loss
