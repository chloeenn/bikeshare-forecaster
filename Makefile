.PHONY: setup ingest process features baselines train backtest final test

setup:
	uv sync

ingest:  ## download trips + weather into data/raw
	uv run python -m bikeshare.pipeline ingest

process:  ## clean + weather + aggregate -> data/processed
	uv run python -m bikeshare.pipeline process

features:  ## station table + feature tables for each horizon
	uv run python -m bikeshare.run features

baselines:  ## score predict-zero and both baselines on the backtest folds
	uv run python -m bikeshare.run baselines

train:  ## grid search on the 4 backtest folds (logs every fit to MLflow)
	uv run python -m bikeshare.run train

backtest:  ## comparison tables, slices, ablation and README figures
	uv run python -m bikeshare.run backtest
	uv run python -m bikeshare.run plots
	uv run python -m bikeshare.run report

final:  ## ONE-SHOT: train 2023-2025, evaluate 2026-01..09. Refuses to run twice.
	uv run python -m bikeshare.run final
	uv run python -m bikeshare.run report

test:  ## lint, unit tests and data tests (fails if data/processed is missing)
	uv run ruff check .
	uv run pytest
