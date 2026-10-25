"""Folds, LightGBM fits, grid search, backtest and the one-shot final evaluation.

Every fit is logged to MLflow (local file store) as soon as it finishes.
"""

import json
import logging
import os
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import lightgbm as lgb
import mlflow
import numpy as np
import polars as pl

from bikeshare.baselines import add_baselines
from bikeshare.evaluate import (
    add_slice_columns,
    combine,
    comparison,
    finalize,
    metric_sums,
    slice_table,
)
from bikeshare.features import ABLATION, CATEGORICAL, FEATURES

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Fold:
    name: str
    val_start: datetime
    val_end: datetime  # exclusive
    train_end: datetime  # training targets must be strictly before this


def _dt(s: str) -> datetime:
    return datetime.fromisoformat(s)


def make_folds(cfg: dict[str, Any]) -> list[Fold]:
    """Backtest folds: training targets end ``gap_hours`` before each validation block."""
    m = cfg["modeling"]
    gap = timedelta(hours=m["gap_hours"])
    return [
        Fold(f["name"], _dt(f["start"]), _dt(f["end"]), _dt(f["start"]) - gap) for f in m["folds"]
    ]


def final_fold(cfg: dict[str, Any]) -> Fold:
    f = cfg["modeling"]["final"]
    return Fold("final", _dt(f["start"]), _dt(f["end"]), _dt(f["train_end"]))


def split(df: pl.DataFrame, fold: Fold) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Time-based split on the TARGET hour. Origins may precede the block by up to h hours."""
    train = df.filter(pl.col("target_hour") < fold.train_end)
    val = df.filter(
        (pl.col("target_hour") >= fold.val_start) & (pl.col("target_hour") < fold.val_end)
    )
    return train, val


ABL = "LightGBM + weather at t+h (ablation, upper bound)"
MODELS = {"LightGBM": "p_lgbm", ABL: "p_abl"}
KEYS = {
    "Predict zero (reference)": "zero",
    "Baseline: same hour last week": "bl_1w",
    "Baseline: mean of same hour, last 4 weeks": "bl_4w",
    "LightGBM": "lgbm",
    ABL: "abl",
}


def flat_metrics(table: pl.DataFrame) -> dict[str, float]:
    """One MLflow-safe metric per (model, subset, metric), e.g. ``lgbm_common_rmse``."""
    return {
        f"{KEYS[r['model']]}_{r['subset'].replace(' ', '_')}_{k}": r[k]
        for r in table.iter_rows(named=True)
        for k in ("rmse", "deviance", "mae")
    }


# ---------------------------------------------------------------- MLflow and bookkeeping


def git_info() -> dict[str, str]:
    def run(*args: str) -> str:
        return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()

    return {
        "git_hash": run("rev-parse", "HEAD"),
        "git_dirty": str(bool(run("status", "--porcelain"))),
    }


def init_mlflow(cfg: dict[str, Any]) -> None:
    # MLflow 3.x puts the file store in maintenance mode; the project requires it, so opt in.
    os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
    root = cfg["paths"]["reports"].parent / cfg["mlflow"]["tracking_uri"]
    mlflow.set_tracking_uri(root.resolve().as_uri())
    mlflow.set_experiment(cfg["mlflow"]["experiment"])


def log_run(
    name: str,
    tags: dict[str, str],
    params: dict[str, Any],
    metrics: dict[str, float],
    features: list[str],
) -> None:
    with mlflow.start_run(run_name=name):
        mlflow.set_tags({**git_info(), **tags})
        mlflow.log_params(params)
        mlflow.log_metrics({k: float(v) for k, v in metrics.items()})
        mlflow.log_dict({"features": features, "categorical": CATEGORICAL}, "features.json")


def load_features(cfg: dict[str, Any], h: int) -> pl.DataFrame:
    return pl.read_parquet(cfg["paths"]["processed"] / f"features_h{h}.parquet")


# ---------------------------------------------------------------- fitting


def lgbm_params(cfg: dict[str, Any], num_leaves: int) -> dict[str, Any]:
    p = cfg["modeling"]["lgbm"]
    return {
        "objective": "poisson",
        "learning_rate": p["learning_rate"],
        "num_leaves": num_leaves,
        "min_data_in_leaf": p["min_data_in_leaf"],
        "feature_fraction": p["feature_fraction"],
        "bagging_fraction": p["bagging_fraction"],
        "bagging_freq": p["bagging_freq"],
        "seed": cfg["seed"],
        "num_threads": 8,
        "deterministic": True,
        "force_row_wise": True,
        "verbose": -1,
    }


def to_matrix(df: pl.DataFrame, features: list[str]) -> np.ndarray:
    return df.select([pl.col(c).cast(pl.Float32) for c in features]).to_numpy()


def fit(
    train: pl.DataFrame,
    features: list[str],
    cfg: dict[str, Any],
    num_leaves: int,
    n_estimators: int,
    presampled: bool = False,
) -> lgb.Booster:
    """Fit on a uniform random sample of the training rows (or ``train`` itself if presampled)."""
    m = cfg["modeling"]
    sample = (
        train if presampled else train.sample(fraction=m["train_sample_frac"], seed=cfg["seed"])
    )
    ds = lgb.Dataset(
        to_matrix(sample, features),
        label=sample["y"].to_numpy(),
        feature_name=features,
        categorical_feature=[c for c in CATEGORICAL if c in features],
    )
    return lgb.train(lgbm_params(cfg, num_leaves), ds, num_boost_round=n_estimators)


def fold_data(cfg: dict[str, Any], h: int, fold: Fold) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sampled training rows and all validation rows for one fold, read straight from Parquet.

    The sample is identical to the one ``fit`` draws from the full training split (same row
    order, same seed), but the full split is never kept in memory next to it.
    """
    lf = pl.scan_parquet(cfg["paths"]["processed"] / f"features_h{h}.parquet")
    train = lf.filter(pl.col("target_hour") < fold.train_end).collect()
    sample = train.sample(fraction=cfg["modeling"]["train_sample_frac"], seed=cfg["seed"])
    del train
    in_block = (pl.col("target_hour") >= fold.val_start) & (pl.col("target_hour") < fold.val_end)
    return sample, add_baselines(lf.filter(in_block).collect())


def predict(booster: lgb.Booster, df: pl.DataFrame, features: list[str], n_iter: int) -> np.ndarray:
    return booster.predict(to_matrix(df, features), num_iteration=n_iter).astype(np.float32)


def _pred_path(cfg: dict[str, Any], h: int, fold: str, leaves: int, n: int) -> Path:
    return cfg["paths"]["models"] / f"preds_h{h}_{fold}_l{leaves}_n{n}.parquet"


def _model_path(cfg: dict[str, Any], h: int, fold: str, leaves: int) -> Path:
    return cfg["paths"]["models"] / f"model_h{h}_{fold}_l{leaves}.txt"


def _write_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


# ---------------------------------------------------------------- baselines


def run_baselines(cfg: dict[str, Any]) -> None:
    """Score the predict-zero reference and both baselines per fold (common subset)."""
    init_mlflow(cfg)
    eps = cfg["modeling"]["deviance_eps"]
    out = []
    for h in cfg["modeling"]["horizons"]:
        feats = add_baselines(load_features(cfg, h))
        for fold in make_folds(cfg):
            _, val = split(feats, fold)
            table = comparison(val, {}, eps).with_columns(
                pl.lit(h).alias("horizon"), pl.lit(fold.name).alias("fold")
            )
            out.append(table)
            metrics = flat_metrics(table)
            metrics["coverage"] = table["coverage"][0]
            log_run(
                f"baselines-h{h}-{fold.name}",
                {"stage": "baselines", "horizon": str(h), "fold": fold.name},
                {"horizon": h, "fold": fold.name, "deviance_eps": eps},
                metrics,
                [],
            )
    res = pl.concat(out)
    reports = cfg["paths"]["reports"]
    reports.mkdir(exist_ok=True)
    res.write_csv(reports / "baselines_by_fold.csv")
    print(res.select("horizon", "fold", "model", "rmse", "deviance", "mae", "n", "coverage"))


# ---------------------------------------------------------------- grid search


def run_grid(cfg: dict[str, Any]) -> None:
    """Fit the grid on every backtest fold and pick a config per horizon by pooled deviance."""
    init_mlflow(cfg)
    m, eps = cfg["modeling"], cfg["modeling"]["deviance_eps"]
    cfg["paths"]["models"].mkdir(parents=True, exist_ok=True)
    n_max = max(m["grid"]["n_estimators"])
    results: list[dict[str, Any]] = []
    t_all = time.time()
    for h in m["horizons"]:
        feats = load_features(cfg, h)
        for fold in make_folds(cfg):
            train, val = split(feats, fold)
            for leaves in m["grid"]["num_leaves"]:
                t0 = time.time()
                booster = fit(train, FEATURES, cfg, leaves, n_max)
                secs = time.time() - t0
                booster.save_model(str(_model_path(cfg, h, fold.name, leaves)))
                for n in m["grid"]["n_estimators"]:
                    p = predict(booster, val, FEATURES, n)
                    sums = metric_sums(val["y"].to_numpy(), p, eps)
                    val.select("station_id", "target_hour").with_columns(
                        pl.Series("p", p)
                    ).write_parquet(_pred_path(cfg, h, fold.name, leaves, n))
                    res = {
                        "h": h,
                        "fold": fold.name,
                        "num_leaves": leaves,
                        "n_estimators": n,
                        "fit_seconds": secs,
                        **sums,
                    }
                    results.append(res)
                    log_run(
                        f"grid-h{h}-{fold.name}-l{leaves}-n{n}",
                        {"stage": "grid", "horizon": str(h), "fold": fold.name},
                        {
                            "horizon": h,
                            "fold": fold.name,
                            "num_leaves": leaves,
                            "n_estimators": n,
                            "train_sample_frac": m["train_sample_frac"],
                            "n_train_rows": train.height,
                            "n_train_sampled": int(train.height * m["train_sample_frac"]),
                            "val_start": fold.val_start,
                            "train_targets_before": fold.train_end,
                            "seed": cfg["seed"],
                            "deviance_eps": eps,
                            **lgbm_params(cfg, leaves),
                        },
                        {**{f"val_{k}": v for k, v in finalize(sums).items()}, "fit_seconds": secs},
                        FEATURES,
                    )
                    log.info(
                        "h=%d %s leaves=%d n=%d rmse=%.4f dev=%.4f (%.0fs fit)",
                        h,
                        fold.name,
                        leaves,
                        n,
                        finalize(sums)["rmse"],
                        finalize(sums)["deviance"],
                        secs,
                    )
    _write_json(cfg["paths"]["reports"] / "grid_results.json", results)
    best = pick_best(results)
    _write_json(cfg["paths"]["reports"] / "best_params.json", best)
    for h, b in best.items():
        log.info("best for h=%s: %s", h, b)
    _write_json(cfg["paths"]["reports"] / "runtime_grid.json", {"seconds": time.time() - t_all})


def pick_best(results: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Per horizon, the config with the lowest pooled Poisson deviance over the backtest folds."""
    best: dict[str, dict[str, Any]] = {}
    for h in sorted({r["h"] for r in results}):
        configs = sorted({(r["num_leaves"], r["n_estimators"]) for r in results if r["h"] == h})
        scored = []
        for leaves, n in configs:
            parts = [
                r for r in results if (r["h"], r["num_leaves"], r["n_estimators"]) == (h, leaves, n)
            ]
            pooled = finalize(combine(parts))
            scored.append((pooled["deviance"], leaves, n, pooled))
        dev, leaves, n, pooled = min(scored)
        best[str(h)] = {
            "num_leaves": leaves,
            "n_estimators": n,
            "pooled": pooled,
            "all_configs": [{"num_leaves": s[1], "n_estimators": s[2], **s[3]} for s in scored],
        }
    return best


# ---------------------------------------------------------------- backtest and final


def _attach_preds(val: pl.DataFrame, path: Path, name: str) -> pl.DataFrame:
    p = pl.read_parquet(path).rename({"p": name})
    return val.join(p, on=["station_id", "target_hour"], how="left")


def _ablation_preds(
    train: pl.DataFrame, val: pl.DataFrame, cfg: dict[str, Any], leaves: int, n: int
) -> np.ndarray:
    feats = [*FEATURES, *ABLATION]
    return predict(fit(train, feats, cfg, leaves, n, presampled=True), val, feats, n)


def run_backtest(cfg: dict[str, Any]) -> None:
    """Per-fold and pooled comparison, slices and the weather-at-t+h ablation.

    ``train`` below is the already-sampled training rows (see ``fold_data``).
    """
    init_mlflow(cfg)
    eps = cfg["modeling"]["deviance_eps"]
    reports = cfg["paths"]["reports"]
    best = json.loads((reports / "best_params.json").read_text())
    stations = pl.read_parquet(cfg["paths"]["processed"] / "stations.parquet")
    models = MODELS
    t_all = time.time()
    for h in cfg["modeling"]["horizons"]:
        b = best[str(h)]
        pooled_parts, fold_tables = [], []
        for fold in make_folds(cfg):
            log.info("backtest h=%d %s: loading fold data", h, fold.name)
            train, val = fold_data(cfg, h, fold)
            val = _attach_preds(
                val, _pred_path(cfg, h, fold.name, b["num_leaves"], b["n_estimators"]), "p_lgbm"
            )
            t0 = time.time()
            val = val.with_columns(
                pl.Series(
                    "p_abl", _ablation_preds(train, val, cfg, b["num_leaves"], b["n_estimators"])
                )
            )
            secs = time.time() - t0
            log.info("backtest h=%d %s: ablation fit+predict %.0fs", h, fold.name, secs)
            del train
            table = comparison(val, models, eps).with_columns(
                pl.lit(h).alias("horizon"), pl.lit(fold.name).alias("fold")
            )
            fold_tables.append(table)
            pooled_parts.append(val)
            log_run(
                f"ablation-h{h}-{fold.name}",
                {"stage": "backtest", "horizon": str(h), "fold": fold.name},
                {
                    "horizon": h,
                    "fold": fold.name,
                    "num_leaves": b["num_leaves"],
                    "n_estimators": b["n_estimators"],
                    "ablation_features": "weather at t+h (upper bound, not a real forecast)",
                },
                {**flat_metrics(table), "ablation_fit_seconds": secs},
                [*FEATURES, *ABLATION],
            )
        pooled = pl.concat(pooled_parts)
        pooled_table = comparison(pooled, models, eps).with_columns(pl.lit(h).alias("horizon"))
        slices = slice_table(
            add_slice_columns(pooled, stations, cfg["modeling"]["new_station_hours"]), "p_lgbm", eps
        ).with_columns(pl.lit(h).alias("horizon"))
        pl.concat(fold_tables).write_csv(reports / f"backtest_h{h}_by_fold.csv")
        pooled_table.write_csv(reports / f"backtest_h{h}_pooled.csv")
        slices.write_csv(reports / f"backtest_h{h}_slices.csv")
        pooled.select(
            "station_id",
            "hour",
            "target_hour",
            "y",
            "bl_1w",
            "bl_4w",
            "p_lgbm",
            "p_abl",
            "hour_of_day",
        ).write_parquet(cfg["paths"]["models"] / f"backtest_h{h}.parquet")
        log_run(
            f"backtest-h{h}-pooled",
            {"stage": "backtest-pooled", "horizon": str(h)},
            {"horizon": h, "num_leaves": b["num_leaves"], "n_estimators": b["n_estimators"]},
            flat_metrics(pooled_table),
            FEATURES,
        )
        print(f"\n=== horizon {h}: pooled over 4 folds ===")
        print(
            pooled_table.select(
                "model",
                "subset",
                "rmse",
                "deviance",
                "mae",
                "n",
                "coverage",
                "rmse_vs_best_baseline_pct",
                "deviance_vs_best_baseline_pct",
            )
        )
    _write_json(reports / "runtime_backtest.json", {"seconds": time.time() - t_all})


def run_final(cfg: dict[str, Any], force: bool = False) -> None:
    """Train on 2023-2025, evaluate 2026-01..09 exactly once. No tuning after this."""
    reports = cfg["paths"]["reports"]
    guard = reports / "final_done.json"
    if guard.exists() and not force:
        raise SystemExit(f"{guard} exists: the final test set has already been evaluated once.")
    init_mlflow(cfg)
    eps = cfg["modeling"]["deviance_eps"]
    best = json.loads((reports / "best_params.json").read_text())
    stations = pl.read_parquet(cfg["paths"]["processed"] / "stations.parquet")
    models = {"LightGBM": "p_lgbm"}  # no ablation on the final test
    fold = final_fold(cfg)
    info = git_info()  # captured before any output file is written, so the hash stays clean
    done: dict[str, Any] = {"started": datetime.now().isoformat(), **info}
    t_all = time.time()
    for h in cfg["modeling"]["horizons"]:
        b = best[str(h)]
        train, val = fold_data(cfg, h, fold)
        booster = fit(train, FEATURES, cfg, b["num_leaves"], b["n_estimators"], presampled=True)
        booster.save_model(str(_model_path(cfg, h, "final", b["num_leaves"])))
        val = val.with_columns(
            pl.Series("p_lgbm", predict(booster, val, FEATURES, b["n_estimators"]))
        )
        established = val.filter(pl.col("station_code").is_not_null())
        new = val.filter(pl.col("station_code").is_null())
        head = comparison(established, models, eps).with_columns(pl.lit(h).alias("horizon"))
        newt = comparison(new, models, eps).with_columns(pl.lit(h).alias("horizon"))
        slices = slice_table(
            add_slice_columns(established, stations, cfg["modeling"]["new_station_hours"]),
            "p_lgbm",
            eps,
        ).with_columns(pl.lit(h).alias("horizon"))
        head.write_csv(reports / f"final_h{h}_headline.csv")
        newt.write_csv(reports / f"final_h{h}_new_in_2026.csv")
        slices.write_csv(reports / f"final_h{h}_slices.csv")
        established.select(
            "station_id",
            "hour",
            "target_hour",
            "y",
            "bl_1w",
            "bl_4w",
            "p_lgbm",
            "hour_of_day",
        ).write_parquet(cfg["paths"]["models"] / f"final_h{h}.parquet")
        with mlflow.start_run(run_name=f"final-h{h}"):
            mlflow.set_tags({**info, "stage": "final", "horizon": str(h)})
            mlflow.log_params(
                {
                    "horizon": h,
                    "num_leaves": b["num_leaves"],
                    "n_estimators": b["n_estimators"],
                    "train_targets_before": fold.train_end,
                    "test_start": fold.val_start,
                    "test_end": fold.val_end,
                    "train_sample_frac": cfg["modeling"]["train_sample_frac"],
                    "seed": cfg["seed"],
                    "n_train_sampled": train.height,
                }
            )
            mlflow.log_metrics({k: float(v) for k, v in flat_metrics(head).items()})
            mlflow.log_dict({"features": FEATURES}, "features.json")
        print(f"\n=== FINAL horizon {h} (established stations, 2026-01..09) ===")
        print(
            head.select(
                "model",
                "subset",
                "rmse",
                "deviance",
                "mae",
                "n",
                "coverage",
                "rmse_vs_best_baseline_pct",
                "deviance_vs_best_baseline_pct",
            )
        )
        print(f"--- new in 2026 ({new.height} rows) ---")
        print(newt.select("model", "subset", "rmse", "deviance", "mae", "n", "coverage"))
    done["finished"] = datetime.now().isoformat()
    done["seconds"] = time.time() - t_all
    _write_json(guard, done)
