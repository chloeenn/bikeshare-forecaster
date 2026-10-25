"""Render result CSVs into markdown tables and splice them into the README."""

import json
from typing import Any

import polars as pl

START, END = "<!-- phase2:start -->", "<!-- phase2:end -->"
ABL_PREFIX = "LightGBM + weather"


def _f(x: float | None, nd: int = 3) -> str:
    return "" if x is None else f"{x:.{nd}f}"


def _pct(x: float | None) -> str:
    return "" if x is None else f"{x:+.1f}%"


def md(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def results_table(df: pl.DataFrame) -> str:
    """Pooled/final head-to-head table: baselines on the common subset, LightGBM on both subsets."""
    rows = []
    for r in df.iter_rows(named=True):
        abl = r["model"].startswith(ABL_PREFIX)
        is_model = r["model"].startswith("LightGBM")
        name = r["model"] + (f" [{r['subset']}]" if is_model else "")
        rows.append(
            [
                f"*{name}*" if abl else name,
                _f(r["rmse"]),
                _f(r["deviance"]),
                _f(r["mae"]),
                f"{int(r['n']):,}",
                _f(r["coverage"], 3),
                _pct(r.get("rmse_vs_best_baseline_pct")),
                _pct(r.get("deviance_vs_best_baseline_pct")),
                _pct(r.get("mae_vs_best_baseline_pct")),
            ]
        )
    return md(
        [
            "Model",
            "RMSE",
            "Poisson dev.",
            "MAE",
            "Rows",
            "Coverage",
            "RMSE vs best baseline",
            "Dev. vs best baseline",
            "MAE vs best baseline",
        ],
        rows,
    )


def fold_table(df: pl.DataFrame) -> str:
    rows = []
    for fold in df["fold"].unique(maintain_order=True):
        d = df.filter(pl.col("fold") == fold)
        b1 = d.filter(pl.col("model").str.starts_with("Baseline: same")).row(0, named=True)
        b4 = d.filter(pl.col("model").str.starts_with("Baseline: mean")).row(0, named=True)
        m = d.filter((pl.col("model") == "LightGBM") & (pl.col("subset") == "common")).row(
            0, named=True
        )
        rows.append(
            [
                fold,
                _f(b1["rmse"]),
                _f(b4["rmse"]),
                _f(m["rmse"]),
                _pct(m["rmse_vs_best_baseline_pct"]),
                _f(b1["deviance"]),
                _f(b4["deviance"]),
                _f(m["deviance"]),
                _pct(m["deviance_vs_best_baseline_pct"]),
                _pct(m["mae_vs_best_baseline_pct"]),
            ]
        )
    return md(
        [
            "Fold",
            "RMSE 1w",
            "RMSE 4w",
            "RMSE LGBM",
            "vs best",
            "Dev. 1w",
            "Dev. 4w",
            "Dev. LGBM",
            "vs best",
            "MAE vs best",
        ],
        rows,
    )


def slice_table(df: pl.DataFrame) -> tuple[str, list[str]]:
    """Slice table plus a list of human-readable losses (RMSE or deviance; MAE reported apart)."""
    rows, losses = [], []
    for r in df.iter_rows(named=True):
        lose = [k for k in ("rmse", "deviance", "mae") if not r[f"win_{k}"]]
        flag = "" if not lose else " ✗ " + "/".join(lose)
        if "rmse" in lose or "deviance" in lose:
            losses.append(f"{r['slice']} = {r['value']} ({'/'.join(lose)})")
        rows.append(
            [
                r["slice"],
                r["value"],
                f"{r['n']:,}",
                _f(r["model_rmse"]),
                _f(r["best_baseline_rmse"]),
                _pct(r["rmse_vs_best_baseline_pct"]),
                _pct(r["deviance_vs_best_baseline_pct"]),
                _pct(r["mae_vs_best_baseline_pct"]),
                flag.strip(),
            ]
        )
    return md(
        [
            "Slice",
            "Value",
            "Rows",
            "RMSE LGBM",
            "RMSE best base",
            "RMSE",
            "Deviance",
            "MAE",
            "Loses on",
        ],
        rows,
    ), losses


def write_report(cfg: dict[str, Any]) -> None:
    rep = cfg["paths"]["reports"]
    parts = ["## Results", ""]
    best_path = rep / "best_params.json"
    if best_path.exists():
        best = json.loads(best_path.read_text())
        parts += [
            "Chosen config (lowest pooled backtest Poisson deviance): "
            + ", ".join(
                f"h={h}: num_leaves={b['num_leaves']}, n_estimators={b['n_estimators']}"
                for h, b in best.items()
            ),
            "",
        ]
    for h in cfg["modeling"]["horizons"]:
        pooled = rep / f"backtest_h{h}_pooled.csv"
        if not pooled.exists():
            continue
        parts += [
            f"### Backtest, {h} hour{'s' if h > 1 else ''} ahead (pooled over 4 folds)",
            "",
            results_table(pl.read_csv(pooled)),
            "",
            f"Per fold (common subset), h={h}:",
            "",
            fold_table(pl.read_csv(rep / f"backtest_h{h}_by_fold.csv")),
            "",
        ]
        st, losses = slice_table(pl.read_csv(rep / f"backtest_h{h}_slices.csv"))
        parts += [
            f"Slices, h={h} (common subset; % = improvement over the best baseline in the slice):",
            "",
            st,
            "",
            "**Slices where LightGBM does not beat the best baseline on RMSE or deviance:** "
            + ("; ".join(losses) if losses else "none."),
            "",
        ]
    for h in cfg["modeling"]["horizons"]:
        head = rep / f"final_h{h}_headline.csv"
        if not head.exists():
            continue
        parts += [
            f"### FINAL test, 2026-01..09, {h} hour{'s' if h > 1 else ''} ahead "
            "(established stations; evaluated once)",
            "",
            results_table(pl.read_csv(head)),
            "",
            f"New in 2026 stations, h={h} (reported separately):",
            "",
            results_table(pl.read_csv(rep / f"final_h{h}_new_in_2026.csv")),
            "",
        ]
        st, losses = slice_table(pl.read_csv(rep / f"final_h{h}_slices.csv"))
        parts += [
            f"Final slices, h={h}:",
            "",
            st,
            "",
            "**Final slices where LightGBM does not beat the best baseline on RMSE or deviance:** "
            + ("; ".join(losses) if losses else "none."),
            "",
        ]
    text = "\n".join(parts)
    (rep / "results.md").write_text(text)
    readme = cfg["paths"]["reports"].parent / "README.md"
    s = readme.read_text()
    if START in s and END in s:
        a, b = s.index(START), s.index(END)
        readme.write_text(s[: a + len(START)] + "\n" + text + "\n" + s[b:])
