"""Step 2: sample eligible complete windows and save matched masks for every split.

Each window gets all mask configs (point/block x hidden counts), so scattered and contiguous
masks with the same hidden count are scored on identical windows. The first and last `edge`
points of every window stay observed.

Usage: python -m tsi.masks --config configs/main.yaml
"""

import zlib

import numpy as np

from .common import SPLITS, load_config, load_series, parse_args, results_dir


def eligible_starts(x: np.ndarray, lo: int, hi: int, L: int) -> list[tuple[int, int]]:
    """All (channel, start) with a NaN-free window fully inside [lo, hi)."""
    out = []
    for c in range(x.shape[1]):
        bad = np.isnan(x[lo:hi, c]).astype(int)
        csum = np.concatenate([[0], np.cumsum(bad)])
        n = hi - lo - L + 1
        if n <= 0:
            continue
        ok = np.nonzero(csum[L:L + n] - csum[:n] == 0)[0]
        out += [(c, lo + s) for s in ok]
    return out


def sample_windows(series: dict, split: str, L: int, n: int, rng) -> tuple[np.ndarray, np.ndarray]:
    b = series["bounds"]
    i = SPLITS.index(split)
    cand = eligible_starts(series["x"], b[i], b[i + 1], L)
    pick = rng.choice(len(cand), size=min(n, len(cand)), replace=False)
    ch, st = np.array([cand[j] for j in sorted(pick)]).T
    return ch, st


def hidden_index(pattern: str, k: int, L: int, edge: int, rng) -> np.ndarray:
    if pattern == "point":
        return np.sort(rng.choice(np.arange(edge, L - edge), size=k, replace=False))
    start = rng.integers(edge, L - edge - k + 1)
    return np.arange(start, start + k)


def random_mask(L: int, edge: int, rng, max_hidden: int) -> np.ndarray:
    """Training mask from the point/block mixture (1 = observed, 0 = hidden)."""
    k = int(rng.integers(1, max_hidden + 1))
    idx = hidden_index("point" if rng.random() < 0.5 else "block", k, L, edge, rng)
    m = np.ones(L, dtype=np.float32)
    m[idx] = 0
    return m


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    L, edge = cfg["window"], cfg["edge"]
    for split in ("val", "test"):
        for name in cfg["datasets"]:
            rng = np.random.default_rng([cfg["seeds"][split], zlib.crc32(name.encode())])
            series = load_series(cfg, name)
            ch, st = sample_windows(series, split, L, cfg["n_eval_windows"], rng)
            out = {"window": np.array(L)}
            for k in cfg["hidden_counts"]:
                for p in cfg["patterns"]:
                    key = f"{p}_{k}"
                    out[f"{key}/channel"], out[f"{key}/start"] = ch, st
                    out[f"{key}/idx"] = np.stack([hidden_index(p, k, L, edge, rng) for _ in ch])
            path = results_dir(cfg) / "masks" / split / f"{name}.npz"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.savez(path, **out)
            print(f"{split}/{name}: {len(ch)} windows from {len(np.unique(ch))} channels")


if __name__ == "__main__":
    main()
