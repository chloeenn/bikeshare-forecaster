from datetime import datetime, timedelta

import polars as pl

from bikeshare.config import load_config
from bikeshare.train import final_fold, make_folds, split

CFG = load_config()
GAP = timedelta(hours=CFG["modeling"]["gap_hours"])


def _grid() -> pl.DataFrame:
    hours = pl.datetime_range(datetime(2023, 1, 1), datetime(2026, 9, 30, 23), "1h", eager=True)
    return pl.DataFrame({"target_hour": hours, "y": 1.0})


def test_backtest_fold_dates_match_the_spec() -> None:
    starts = [(f.val_start, f.val_end) for f in make_folds(CFG)]
    assert starts == [
        (datetime(2024, 10, 1), datetime(2025, 1, 1)),
        (datetime(2025, 1, 1), datetime(2025, 4, 1)),
        (datetime(2025, 4, 1), datetime(2025, 7, 1)),
        (datetime(2025, 7, 1), datetime(2025, 10, 1)),
    ]


def test_no_training_target_inside_or_after_its_validation_block() -> None:
    df = _grid()
    for fold in make_folds(CFG):
        train, val = split(df, fold)
        assert train["target_hour"].max() < fold.val_start - GAP + timedelta(seconds=1)
        assert train["target_hour"].max() <= fold.val_start - GAP
        assert train["target_hour"].min() == datetime(2023, 1, 1)
        assert val["target_hour"].min() >= fold.val_start
        assert val["target_hour"].max() < fold.val_end
        assert train["target_hour"].max() < val["target_hour"].min()


def test_final_trains_on_2023_2025_and_tests_2026_jan_to_sep() -> None:
    train, val = split(_grid(), final_fold(CFG))
    assert train["target_hour"].max() == datetime(2025, 12, 31, 23)
    assert val["target_hour"].min() == datetime(2026, 1, 1)
    assert val["target_hour"].max() == datetime(2026, 9, 30, 23)
