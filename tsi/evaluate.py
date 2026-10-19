"""Step 4: score saved predictions and rebuild every table and figure without rerunning models.

Outputs (under results_dir):
  tables/metrics_{split}.csv, tables/metrics_{split}.md   one row per dataset x mask x model
  tables/csdi_draws_{split}.csv                           CSDI 50 vs 100 draw sensitivity
  figures/reliability_{split}_{dataset}.png               reliability curves per mask config
  figures/along_gap_{split}_{dataset}.png                 coverage/width along contiguous gaps
  figures/bridge.png                                      analytical Gaussian illustration

Usage: python -m tsi.evaluate --config configs/main.yaml
"""

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .common import (load_config, load_masks, load_series, mask_configs, parse_args, pred_path,
                     results_dir, window_values)
from .metrics import all_metrics, along_gap, reliability

MODELS = {"kalman": "Kalman smoother", "csdi": "CSDI", "chronos_blend": "Chronos-2 blend",
          "chronos_left": "Chronos-2 left-only", "chronos_right": "Chronos-2 right-only"}
COLORS = {"kalman": "#2a6fdb", "csdi": "#d1495b", "chronos_blend": "#e08e0b",
          "chronos_left": "#7a7a7a", "chronos_right": "#3c9d5d"}
METRIC_COLS = ["MAE", "MSE", "RMSE", "MRE", "pinball", "CRPS", "NLL", "PCE", "CCE", "cov80", "width80", "SIW"]


def load_pred(cfg, split, name, model):
    p = pred_path(cfg, split, name, model)
    if not p.exists():
        return None
    d = np.load(p)
    return {k: d[k] for k in d.files}


def targets(series, masks, key):
    y = np.take_along_axis(window_values(series, masks, key), masks[f"{key}/idx"], 1)
    ch = masks[f"{key}/channel"]
    mu, sd = series["mean"][ch][:, None], series["std"][ch][:, None]
    return y, mu, sd


def score_split(cfg, split):
    qs = np.array(cfg["quantiles"])
    rows, draws, curves, gaps = [], [], {}, {}
    for name in cfg["datasets"]:
        series, masks = load_series(cfg, name), load_masks(cfg, split, name)
        for key in mask_configs(cfg):
            pattern, k = key.split("_")
            y, mu, sd = targets(series, masks, key)
            for model in MODELS:
                pred = load_pred(cfg, split, name, model)
                if pred is None:
                    continue
                Q = pred[f"{key}/quantiles"]
                m = Q[..., list(qs).index(0.5)]
                extra = {}
                if f"{key}/mean" in pred:
                    extra = {"mean": pred[f"{key}/mean"], "std": pred[f"{key}/std"]}
                elif f"{key}/samples" in pred:
                    extra = {"samples": pred[f"{key}/samples"]}
                met = all_metrics(y, Q, qs, y * sd + mu, m * sd + mu, **extra)
                rows.append({"split": split, "dataset": name, "pattern": pattern, "hidden": int(k),
                             "model": model, "n_windows": len(y), **met})
                curves[(name, key, model)] = reliability(y, Q, qs)
                if pattern == "block":
                    gaps[(name, key, model)] = along_gap(y, Q, qs)
                if "samples" in extra and extra["samples"].shape[1] >= 100:
                    s = extra["samples"]
                    half = np.moveaxis(np.quantile(s[:, :50], qs, axis=1), 0, -1)
                    m50 = all_metrics(y, half, qs, y * sd + mu, half[..., list(qs).index(0.5)] * sd + mu,
                                      samples=s[:, :50])
                    draws.append({"dataset": name, "pattern": pattern, "hidden": int(k),
                                  "mean_abs_quantile_diff": float(np.abs(half - Q).mean()),
                                  **{f"{c}_50": m50[c] for c in ("CRPS", "PCE", "cov80", "width80")},
                                  **{f"{c}_100": met[c] for c in ("CRPS", "PCE", "cov80", "width80")}})
    return pd.DataFrame(rows), pd.DataFrame(draws), curves, gaps


def plot_reliability(cfg, split, name, curves, out):
    qs = np.array(cfg["quantiles"])
    keys = mask_configs(cfg)
    fig, axes = plt.subplots(1, len(keys), figsize=(4 * len(keys), 4), sharey=True)
    for ax, key in zip(axes, keys):
        ax.plot([0, 1], [0, 1], ls="--", c="black", lw=0.8)
        for model in ("kalman", "csdi", "chronos_blend"):
            if (name, key, model) in curves:
                ax.plot(qs, curves[(name, key, model)], marker="o", ms=3, c=COLORS[model], label=MODELS[model])
        ax.set_title(f"{name}: {key.replace('_', ', ')} hidden")
        ax.set_xlabel("nominal quantile level q")
    axes[0].set_ylabel("observed frequency  y ≤ Q_q")
    axes[0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"reliability_{split}_{name}.png", dpi=150)
    plt.close(fig)


def plot_along_gap(cfg, split, name, gaps, out):
    keys = [k for k in mask_configs(cfg) if k.startswith("block")]
    fig, axes = plt.subplots(2, len(keys), figsize=(5.5 * len(keys), 7), squeeze=False)
    for col, key in enumerate(keys):
        for model in MODELS:
            if (name, key, model) not in gaps:
                continue
            cov, width = gaps[(name, key, model)]
            pos = np.arange(1, len(cov) + 1)
            ls = ":" if model in ("chronos_left", "chronos_right") else "-"
            axes[0, col].plot(pos, cov, ls=ls, c=COLORS[model], label=MODELS[model])
            axes[1, col].plot(pos, width, ls=ls, c=COLORS[model])
        axes[0, col].axhline(0.8, c="black", lw=0.8, ls="--")
        axes[0, col].set_title(f"{name}: contiguous gap of {key.split('_')[1]}")
        axes[0, col].set_ylabel("80% coverage")
        axes[1, col].set_ylabel("80% width (scaled units)")
        axes[1, col].set_xlabel("position in the gap")
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out / f"along_gap_{split}_{name}.png", dpi=150)
    plt.close(fig)


def plot_bridge(out, gap=24):
    """Random walk, noiseless endpoints: left forecast vs quantile blend vs exact bridge (std)."""
    j = np.arange(1, gap + 1)
    n = gap + 1
    left, right = np.sqrt(j), np.sqrt(n - j)
    w = (gap - (j - 1)) / (gap + 1)
    blend = w * left + (1 - w) * right  # quantile averaging of Gaussians averages the std
    bridge = np.sqrt(j * (n - j) / n)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(j, left, ls=":", c="#7a7a7a", label="left forecast")
    ax.plot(j, blend, c="#e08e0b", label="quantile blend")
    ax.plot(j, bridge, c="#2a6fdb", label="exact Gaussian bridge")
    ax.set_xlabel(f"position in a {gap}-step gap")
    ax.set_ylabel("standard deviation (innovation units)")
    ax.set_title(f"Blend vs exact bridge: midpoint ratio {blend[gap // 2 - 1] / bridge[gap // 2 - 1]:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out / "bridge.png", dpi=150)
    plt.close(fig)


def to_markdown(df: pd.DataFrame) -> str:
    cols = ["dataset", "pattern", "hidden", "model"] + METRIC_COLS
    d = df[cols].copy()
    d["model"] = d["model"].map(MODELS)
    for c in METRIC_COLS:
        d[c] = d[c].map(lambda v: "—" if pd.isna(v) else f"{v:.3f}")
    head = "| " + " | ".join(cols) + " |\n|" + "---|" * len(cols) + "\n"
    return head + "\n".join("| " + " | ".join(map(str, r)) + " |" for r in d.values) + "\n"


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    tables, figs = results_dir(cfg) / "tables", results_dir(cfg) / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)
    plot_bridge(figs)
    for split in ("val", "test"):
        df, draws, curves, gaps = score_split(cfg, split)
        if df.empty:
            continue
        df.to_csv(tables / f"metrics_{split}.csv", index=False)
        (tables / f"metrics_{split}.md").write_text(to_markdown(df))
        if not draws.empty:
            draws.to_csv(tables / f"csdi_draws_{split}.csv", index=False)
        for name in cfg["datasets"]:
            plot_reliability(cfg, split, name, curves, figs)
            plot_along_gap(cfg, split, name, gaps, figs)
        print(f"{split}: {len(df)} rows -> {tables / f'metrics_{split}.csv'}")


if __name__ == "__main__":
    main()
