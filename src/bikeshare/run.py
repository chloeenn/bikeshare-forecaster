import argparse
import logging
import time

from bikeshare.config import load_config
from bikeshare.features import build_all
from bikeshare.plots import make_plots
from bikeshare.report import write_report
from bikeshare.train import run_backtest, run_baselines, run_final, run_grid

STEPS = {
    "features": build_all,
    "baselines": run_baselines,
    "train": run_grid,
    "backtest": run_backtest,
    "plots": make_plots,
    "report": write_report,
}


def main() -> None:
    parser = argparse.ArgumentParser(description="Modelling steps")
    parser.add_argument("step", choices=(*STEPS, "final"))
    parser.add_argument("--force", action="store_true", help="re-run final (discouraged)")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    t0 = time.time()
    if args.step == "final":
        run_final(cfg, force=args.force)
    else:
        STEPS[args.step](cfg)
    logging.info("%s finished in %.0fs", args.step, time.time() - t0)


if __name__ == "__main__":
    main()
