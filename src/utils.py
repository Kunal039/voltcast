"""Shared helpers: config loading, paths and logging."""
import logging
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def load_params(path: Path | str = ROOT / "configs" / "params.yaml") -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve(rel: str) -> Path:
    p = ROOT / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def get_logger(name: str) -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
    )
    return logging.getLogger(name)
