"""Load and validate config.yaml; seed helpers."""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.yaml"

REQUIRED_TOP = [
    "frozen", "seeds", "model", "prompt_versions", "retrieval", "state",
    "classifier", "calibration", "abstention", "sizes", "bootstrap", "cv", "paths", "data",
]


def load_config(path: Path = CONFIG_PATH) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    missing = [k for k in REQUIRED_TOP if k not in cfg]
    if missing:
        raise KeyError(f"config.yaml is missing keys: {missing}")
    return cfg


def require_frozen(cfg: dict) -> None:
    """Evaluation entry points call this first."""
    if cfg.get("frozen") is not True:
        raise RuntimeError("config.yaml has frozen: false. Freeze (Phase 4) before running evaluation.")
    unset = [name for name, v in (
        ("state.m", cfg["state"]["m"]),
        ("classifier.type", cfg["classifier"]["type"]),
        ("calibration.type", cfg["calibration"]["type"]),
        ("abstention.tau", cfg["abstention"]["tau"]),
    ) if v is None]
    if unset:
        raise RuntimeError(f"Frozen config still has unset values: {unset}")


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def resolve(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]
