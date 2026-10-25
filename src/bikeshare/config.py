from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = REPO_ROOT / "config" / "config.yaml"


def load_config(path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    """Load the YAML config and resolve the ``paths`` section against the repo root."""
    with open(path) as f:
        cfg = yaml.safe_load(f)
    cfg["paths"] = {k: REPO_ROOT / v for k, v in cfg["paths"].items()}
    return cfg
