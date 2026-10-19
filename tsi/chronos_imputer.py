"""Step 3b: frozen Chronos-2-small with a two-sided adaptation for imputation.

For every maximal run of hidden positions inside a window:
  * left:  forecast the run from the window prefix;
  * right: reverse the window suffix, forecast, reverse the forecast back;
  * blend: average matching quantiles with linear weights, w_left(j) = (k - j) / (k + 1) for
    position j = 0..k-1 of a k-step run, so the nearer side gets more weight.
Other hidden runs stay NaN in the context. The blend is a heuristic, not conditional inference.

Usage: python -m tsi.chronos_imputer --config configs/main.yaml
"""

import time
from collections import defaultdict

import numpy as np
import torch
from chronos import Chronos2Pipeline

from .common import load_config, load_masks, load_series, mask_configs, masked_windows, parse_args, pred_path


def hidden_runs(idx: np.ndarray) -> list[tuple[int, int]]:
    """Maximal contiguous runs [start, end) of a sorted index array."""
    breaks = np.nonzero(np.diff(idx) != 1)[0]
    starts = np.concatenate([[0], breaks + 1])
    ends = np.concatenate([breaks + 1, [len(idx)]])
    return [(idx[a], idx[b - 1] + 1) for a, b in zip(starts, ends)]


def forecast(pipe, contexts: list[np.ndarray], horizon: int, qs: list[float], batch: int) -> np.ndarray:
    out = []
    for i in range(0, len(contexts), batch):
        ctx = [torch.tensor(c, dtype=torch.float32) for c in contexts[i:i + batch]]
        q, _ = pipe.predict_quantiles(ctx, prediction_length=horizon, quantile_levels=qs)
        out += [t[0].float().numpy() for t in q]
    return np.stack(out)  # [n, horizon, Q]


def impute(pipe, x: np.ndarray, idx: np.ndarray, qs: list[float], batch: int) -> dict:
    N, k = idx.shape
    res = {s: np.empty((N, k, len(qs)), dtype=np.float32) for s in ("left", "right", "blend")}
    jobs = defaultdict(list)  # run length -> [(window, run start, position in idx)]
    for n in range(N):
        pos = 0
        for s, e in hidden_runs(idx[n]):
            jobs[e - s].append((n, s, pos))
            pos += e - s
    for h, items in jobs.items():
        left = forecast(pipe, [x[n, :s] for n, s, _ in items], h, qs, batch)
        right = forecast(pipe, [x[n, s + h:][::-1].copy() for n, s, _ in items], h, qs, batch)[:, ::-1]
        w = ((h - np.arange(h)) / (h + 1))[None, :, None]
        blend = w * left + (1 - w) * right
        for j, (n, _, pos) in enumerate(items):
            res["left"][n, pos:pos + h] = left[j]
            res["right"][n, pos:pos + h] = right[j]
            res["blend"][n, pos:pos + h] = blend[j]
    return res


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    ccfg = cfg["chronos"]
    pipe = Chronos2Pipeline.from_pretrained(ccfg["model"], device_map=ccfg["device"])
    qs = list(cfg["quantiles"])
    for split in ("val", "test"):
        for name in cfg["datasets"]:
            series, masks = load_series(cfg, name), load_masks(cfg, split, name)
            outs = {s: {} for s in ("left", "right", "blend")}
            t0 = time.time()
            for key in mask_configs(cfg):
                x, idx = masked_windows(series, masks, key)
                for side, q in impute(pipe, x, idx, qs, ccfg["batch_size"]).items():
                    outs[side][f"{key}/quantiles"] = q
            for side, out in outs.items():
                np.savez(pred_path(cfg, split, name, f"chronos_{side}"), **out)
            print(f"{split}/{name}: chronos done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
