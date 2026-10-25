import logging
import time
import zipfile
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)


def download(url: str, dest: Path, retries: int = 3) -> Path:
    """Stream ``url`` to ``dest`` via a .part file; skip if ``dest`` already exists."""
    if dest.exists():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    part = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            part.rename(dest)
            return dest
        except requests.RequestException as e:
            log.warning("download failed (%d/%d): %s", attempt, retries, e)
            time.sleep(2**attempt)
    raise RuntimeError(f"could not download {url}")


def download_trips(cfg: dict[str, Any]) -> list[Path]:
    """Download and extract each configured year of trips into ``data/raw/trips/<year>``."""
    root = cfg["paths"]["raw"] / "trips"
    out = []
    for year in cfg["trips"]["years"]:
        zpath = download(
            cfg["trips"]["url_template"].format(year=year),
            root / f"bikeshare-ridership-{year}.zip",
        )
        target = root / str(year)
        if not target.exists():
            with zipfile.ZipFile(zpath) as z:
                z.extractall(target)
        out.append(target)
    return out


def month_range(years: list[int], today: Any) -> list[tuple[int, int]]:
    """All (year, month) pairs for ``years`` that are not in the future relative to ``today``."""
    return [(y, m) for y in years for m in range(1, 13) if (y, m) <= (today.year, today.month)]


def download_weather(cfg: dict[str, Any], today: Any) -> list[Path]:
    """Download one ECCC hourly bulk CSV per month. The current month is re-fetched."""
    root = cfg["paths"]["raw"] / "weather"
    w = cfg["weather"]
    out = []
    for year, month in month_range(cfg["trips"]["years"], today):
        dest = root / f"{year}-{month:02d}.csv"
        if (year, month) == (today.year, today.month):
            dest.unlink(missing_ok=True)
        url = w["url_template"].format(station_id=w["station_id"], year=year, month=month)
        out.append(download(url, dest))
    return out
