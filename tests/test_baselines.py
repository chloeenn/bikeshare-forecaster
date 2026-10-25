import polars as pl

from bikeshare.baselines import add_baselines


def _df(**cols: list) -> pl.DataFrame:
    base = {
        "same_how_1w": [None],
        "same_how_2w": [None],
        "same_how_3w": [None],
        "same_how_4w": [None],
    }
    return pl.DataFrame({**base, **cols}, schema_overrides={k: pl.Float32 for k in base})


def test_same_hour_last_week_is_the_one_week_lag() -> None:
    out = add_baselines(_df(same_how_1w=[3.0]))
    assert out["bl_1w"][0] == 3.0


def test_four_week_mean_averages_the_four_weekly_lags() -> None:
    out = add_baselines(
        _df(same_how_1w=[4.0], same_how_2w=[0.0], same_how_3w=[2.0], same_how_4w=[2.0])
    )
    assert out["bl_4w"][0] == 2.0


def test_four_week_mean_is_undefined_unless_all_four_weeks_exist() -> None:
    out = add_baselines(_df(same_how_1w=[4.0], same_how_2w=[1.0], same_how_3w=[1.0]))
    assert out["bl_4w"][0] is None
    assert out["bl_1w"][0] == 4.0  # the 1-week baseline still exists


def test_predict_zero_reference() -> None:
    assert add_baselines(_df())["p_zero"][0] == 0.0
