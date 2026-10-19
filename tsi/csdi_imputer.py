"""Step 3c: conditional diffusion imputer CSDI via PyPOTS.

One model per dataset, pooled over channels (univariate windows). PyPOTS "mix" strategy: half of
the training masks are our point/block masks with 1..max(hidden_counts) hidden values, half are
PyPOTS random-ratio scattered masks. Early stopping uses masked validation windows. Prediction keeps `n_samples` draws per window; the point estimate is the
empirical sample median and quantiles are empirical sample quantiles.

Usage: python -m tsi.csdi_imputer --config configs/main.yaml [--skip-train | --resume]
--resume reuses saved checkpoints and predictions and only computes what is missing.
"""

import json
import time
import zlib

import numpy as np
from pypots.imputation import CSDI

from .common import (load_config, load_masks, load_series, mask_configs, masked_windows, parse_args,
                     pred_path, results_dir, save_json)
from .masks import random_mask, sample_windows


def training_windows(series: dict, split: str, n: int, L: int, edge: int, max_hidden: int, rng) -> dict:
    ch, st = sample_windows(series, split, L, n, rng)
    X = np.stack([series["x"][s:s + L, c] for c, s in zip(ch, st)])[..., None]
    M = np.stack([random_mask(L, edge, rng, max_hidden) for _ in ch])[..., None]
    return {"X": X, "mask": M}


def draw_samples(model: CSDI, x: np.ndarray, n_samples: int, batch_size: int) -> np.ndarray:
    """Independent draws [N, S, L]. Each window is repeated S times inside large batches, which is
    much faster than PyPOTS' sequential `n_sampling_times` loop over small batches."""
    N, L = x.shape
    rep = np.repeat(x[:, None, :], n_samples, axis=1).reshape(N * n_samples, L, 1)
    train_batch, model.batch_size = model.batch_size, batch_size
    try:
        imp = model.predict({"X": rep}, n_sampling_times=1)["imputation"]
    finally:
        model.batch_size = train_batch
    return imp.reshape(N, n_samples, L)


def main() -> None:
    def extra(p):
        p.add_argument("--skip-train", action="store_true")
        p.add_argument("--resume", action="store_true")
    args = parse_args(__doc__, extra)
    cfg = load_config(args.config)
    c = cfg["csdi"]
    L, edge, max_hidden = cfg["window"], cfg["edge"], max(cfg["hidden_counts"])
    qs = np.array(cfg["quantiles"])
    log = {}
    for name in cfg["datasets"]:
        series = load_series(cfg, name)
        ckpt = results_dir(cfg) / "csdi" / name / "model.pypots"
        model = CSDI(n_steps=L, n_features=1, n_layers=c["n_layers"], n_heads=c["n_heads"],
                     n_channels=c["n_channels"], d_time_embedding=c["d_time_embedding"],
                     d_feature_embedding=c["d_feature_embedding"],
                     d_diffusion_embedding=c["d_diffusion_embedding"],
                     n_diffusion_steps=c["n_diffusion_steps"], target_strategy="mix",
                     batch_size=c["batch_size"], epochs=c["epochs"], patience=c["patience"],
                     device=c["device"], saving_path=str(ckpt.parent / "runs"),
                     model_saving_strategy="best", verbose=True)
        todo = [s for s in ("val", "test") if not (args.resume and pred_path(cfg, s, name, "csdi").exists())]
        if not todo:
            continue
        if args.skip_train or (args.resume and ckpt.exists()):
            model.load(str(ckpt))
        else:
            rng = np.random.default_rng([cfg["seeds"]["train"], zlib.crc32(name.encode())])
            tr = training_windows(series, "train", c["n_train_windows"], L, edge, max_hidden, rng)
            va = training_windows(series, "val", c["n_val_windows"], L, edge, max_hidden, rng)
            va_X = np.where(va["mask"] == 1, va["X"], np.nan)
            t0 = time.time()
            model.fit({"X": tr["X"], "for_pattern_mask": tr["mask"]},
                      {"X": va_X, "X_ori": va["X"]})
            hours = (time.time() - t0) / 3600
            model.save(str(ckpt), overwrite=True)
            log[name] = {"train_hours": hours, "n_train_windows": len(tr["X"]),
                         "best_val_loss": float(model.best_loss)}
            print(f"{name}: CSDI trained in {hours:.2f} h")
        for split in todo:
            masks = load_masks(cfg, split, name)
            out = {}
            t0 = time.time()
            for key in mask_configs(cfg):
                x, idx = masked_windows(series, masks, key)
                samples = draw_samples(model, x, c["n_samples"], c["sample_batch_size"])
                samples = np.take_along_axis(samples, idx[:, None, :], axis=2)  # [N, S, k]
                out[f"{key}/samples"] = samples.astype(np.float32)
                out[f"{key}/quantiles"] = np.moveaxis(np.quantile(samples, qs, axis=1), 0, -1)
            np.savez(pred_path(cfg, split, name, "csdi"), **out)
            log.setdefault(name, {})[f"{split}_sampling_hours"] = (time.time() - t0) / 3600
    if log:
        log_path = results_dir(cfg) / "csdi" / ("log_skip_train.json" if args.skip_train else "log.json")
        if args.resume and log_path.exists():
            old = json.loads(log_path.read_text())
            for k, v in log.items():
                old.setdefault(k, {}).update(v)
            log = old
        save_json(log, log_path)
    total = sum(v.get("train_hours", 0) for v in log.values())
    if total > c["gpu_hour_cap"]:
        print(f"WARNING: CSDI training used {total:.2f} h, above the {c['gpu_hour_cap']} h cap")


if __name__ == "__main__":
    main()
