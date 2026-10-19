"""Config loading, paths and small I/O helpers shared by all steps."""

import argparse
import json
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent.parent
SPLITS = ("train", "val", "test")


REPLACE_KEYS = {"datasets"}  # an override lists the full set of datasets, it does not extend it


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        nested = isinstance(v, dict) and isinstance(out.get(k), dict) and k not in REPLACE_KEYS
        out[k] = _merge(out[k], v) if nested else v
    return out


def load_config(path: str) -> dict:
    path = Path(path)
    cfg = yaml.safe_load(path.read_text())
    if "base" in cfg:
        base = Path(cfg.pop("base"))
        cfg = _merge(load_config(base if base.is_absolute() else path.parent / base), cfg)
    return cfg


def parse_args(description: str, extra=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config", default=str(ROOT / "configs" / "main.yaml"))
    if extra:
        extra(parser)
    return parser.parse_args()


def data_dir(cfg: dict) -> Path:
    return ROOT / cfg["data_dir"]


def results_dir(cfg: dict) -> Path:
    return ROOT / cfg["results_dir"]


def mask_configs(cfg: dict) -> list[str]:
    return [f"{p}_{k}" for k in cfg["hidden_counts"] for p in cfg["patterns"]]


def load_series(cfg: dict, dataset: str) -> dict:
    """Scaled series [T, C], split boundaries and train-only scaling statistics."""
    d = np.load(data_dir(cfg) / f"{dataset}.npz", allow_pickle=True)
    return {k: d[k] for k in d.files}


def load_masks(cfg: dict, split: str, dataset: str) -> dict:
    d = np.load(results_dir(cfg) / "masks" / split / f"{dataset}.npz")
    return {k: d[k] for k in d.files}


def pred_path(cfg: dict, split: str, dataset: str, model: str) -> Path:
    p = results_dir(cfg) / "predictions" / split / dataset / f"{model}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def window_values(series: dict, masks: dict, key: str) -> np.ndarray:
    """Ground-truth windows [N, L] (scaled) for one mask config."""
    x = series["x"]
    L = masks["window"].item()
    ch, st = masks[f"{key}/channel"], masks[f"{key}/start"]
    return np.stack([x[s:s + L, c] for c, s in zip(ch, st)])


def masked_windows(series: dict, masks: dict, key: str) -> tuple[np.ndarray, np.ndarray]:
    """Windows with hidden positions set to NaN, and hidden indices [N, k]."""
    y = window_values(series, masks, key)
    idx = masks[f"{key}/idx"]
    x = y.copy()
    np.put_along_axis(x, idx, np.nan, axis=1)
    return x, idx


def save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))
