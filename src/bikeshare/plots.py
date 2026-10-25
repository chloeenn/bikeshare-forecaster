"""The three README figures, drawn from the saved backtest predictions and the fold-4 models."""

import json
from datetime import datetime
from pathlib import Path
from typing import Any

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import polars as pl  # noqa: E402

from bikeshare.evaluate import common_subset  # noqa: E402
from bikeshare.train import _model_path  # noqa: E402

SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e3e2de"
BLUE, ORANGE = "#2a78d6", "#eb6834"  # series 1 (LightGBM) and 2 (baseline)
WEEK = (datetime(2025, 8, 4), datetime(2025, 8, 11))


def _style(ax: plt.Axes) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.tick_params(colors=MUTED, labelsize=9, length=0)


def _fig(nrows: int, ncols: int, size: tuple[float, float], **kw: Any):
    fig, axes = plt.subplots(nrows, ncols, figsize=size, facecolor=SURFACE, **kw)
    for ax in axes.flat if hasattr(axes, "flat") else [axes]:
        _style(ax)
    return fig, axes


def _sample_stations(cfg: dict[str, Any]) -> list[tuple[str, int]]:
    """The median-volume station of each tier (by 2023-2025 departures per active hour)."""
    st = pl.read_parquet(cfg["paths"]["processed"] / "stations.parquet").filter(
        ~pl.col("is_new_2026")
    )
    out = []
    for tier in ("top", "mid", "low"):
        t = st.filter(pl.col("tier") == tier).sort("rate")
        out.append((tier, int(t["station_id"][t.height // 2])))
    return out


def plot_actual_vs_predicted(cfg: dict[str, Any], out: Path) -> None:
    df = pl.read_parquet(cfg["paths"]["models"] / "backtest_h1.parquet")
    df = df.filter((pl.col("target_hour") >= WEEK[0]) & (pl.col("target_hour") < WEEK[1]))
    stations = _sample_stations(cfg)
    fig, axes = _fig(3, 1, (9, 7.5), sharex=True)
    for ax, (tier, sid) in zip(axes, stations, strict=True):
        d = df.filter(pl.col("station_id") == sid).sort("target_hour")
        x = d["target_hour"].to_list()
        ax.plot(x, d["y"], color=INK, linewidth=2, label="Actual")
        ax.plot(x, d["bl_4w"], color=ORANGE, linewidth=2, label="Baseline: 4-week same-hour mean")
        ax.plot(x, d["p_lgbm"], color=BLUE, linewidth=2, label="LightGBM")
        ax.set_title(f"Station {sid} ({tier} volume tier)", loc="left", fontsize=10, color=INK)
        ax.set_ylabel("departures / hour", fontsize=9, color=MUTED)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        frameon=False,
        fontsize=9,
        ncol=3,
        loc="upper left",
        labelcolor=INK,
        bbox_to_anchor=(0.005, 0.955),
    )
    fig.suptitle(
        "Actual vs predicted, 1 hour ahead, 4-11 Aug 2025 (backtest fold 4)",
        x=0.01,
        y=0.985,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.925))
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_error_by_hour(cfg: dict[str, Any], out: Path) -> None:
    fig, axes = _fig(1, 2, (11, 4), sharey=True)
    for ax, h in zip(axes, cfg["modeling"]["horizons"], strict=True):
        df = common_subset(pl.read_parquet(cfg["paths"]["models"] / f"backtest_h{h}.parquet"))
        for col, color, label in (
            ("bl_1w", "#9a9893", "Baseline: same hour last week"),
            ("bl_4w", ORANGE, "Baseline: 4-week same-hour mean"),
            ("p_lgbm", BLUE, "LightGBM"),
        ):
            rmse = (
                df.with_columns(((pl.col("y") - pl.col(col)) ** 2).alias("se"))
                .group_by("hour_of_day")
                .agg(pl.col("se").mean().sqrt().alias("rmse"))
                .sort("hour_of_day")
            )
            ax.plot(rmse["hour_of_day"], rmse["rmse"], color=color, linewidth=2, label=label)
        ax.set_title(f"{h} hour{'s' if h > 1 else ''} ahead", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("hour of day of the target hour", fontsize=9, color=MUTED)
        ax.set_xticks(range(0, 24, 3))
    axes[0].set_ylabel("RMSE (departures / hour)", fontsize=9, color=MUTED)
    axes[0].legend(frameon=False, fontsize=9, loc="upper left", labelcolor=INK)
    fig.suptitle(
        "Error by hour of day, pooled over the 4 backtest folds",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def plot_importance(cfg: dict[str, Any], out: Path, top: int = 12) -> None:
    best = json.loads((cfg["paths"]["reports"] / "best_params.json").read_text())
    fig, axes = _fig(1, 2, (11, 4.6))
    for ax, h in zip(axes, cfg["modeling"]["horizons"], strict=True):
        leaves = best[str(h)]["num_leaves"]
        booster = lgb.Booster(model_file=str(_model_path(cfg, h, "f4", leaves)))
        imp = (
            pl.DataFrame(
                {"feature": booster.feature_name(), "gain": booster.feature_importance("gain")}
            )
            .with_columns((pl.col("gain") / pl.col("gain").sum() * 100).alias("share"))
            .sort("share", descending=True)
            .head(top)
            .sort("share")
        )
        ax.barh(imp["feature"], imp["share"], color=BLUE, height=0.6)
        ax.grid(axis="y", visible=False)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_title(f"{h} hour{'s' if h > 1 else ''} ahead", loc="left", fontsize=10, color=INK)
        ax.set_xlabel("share of total gain (%)", fontsize=9, color=MUTED)
    fig.suptitle(
        "LightGBM feature importance (gain), fold-4 model",
        x=0.01,
        ha="left",
        fontsize=11,
        color=INK,
    )
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor=SURFACE)
    plt.close(fig)


def make_plots(cfg: dict[str, Any]) -> None:
    fig_dir = cfg["paths"]["reports"] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_actual_vs_predicted(cfg, fig_dir / "actual_vs_predicted.png")
    plot_error_by_hour(cfg, fig_dir / "error_by_hour.png")
    plot_importance(cfg, fig_dir / "feature_importance.png")
