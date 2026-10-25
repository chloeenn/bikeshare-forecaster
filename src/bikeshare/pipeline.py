import argparse
import logging
from datetime import date

from bikeshare.aggregate import build_panel
from bikeshare.clean import clean_year, write_summary
from bikeshare.config import load_config
from bikeshare.ingest import download_trips, download_weather
from bikeshare.weather import build_weather

STEPS = ("download", "clean", "weather", "aggregate")
GROUPS = {"ingest": ("download",), "process": ("clean", "weather", "aggregate"), "all": STEPS}


def main() -> None:
    parser = argparse.ArgumentParser(description="data/raw -> data/interim -> data/processed")
    parser.add_argument("step", choices=(*STEPS, *GROUPS))
    step = parser.parse_args().step
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    cfg = load_config()
    run = GROUPS.get(step, (step,))
    if "download" in run:
        download_trips(cfg)
        download_weather(cfg, date.today())
    if "clean" in run:
        counts = {y: clean_year(y, cfg) for y in cfg["trips"]["years"]}
        print(write_summary(counts, cfg["paths"]["interim"] / "cleaning_summary.json"))
    if "weather" in run:
        build_weather(cfg)
    if "aggregate" in run:
        build_panel(cfg)


if __name__ == "__main__":
    main()
