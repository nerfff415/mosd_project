"""Step 1: download datasets with tsdb, build univariate channels, split and scale on train only.

Usage: python -m tsi.data --config configs/main.yaml
"""

import numpy as np
import pandas as pd
import tsdb

from .common import data_dir, load_config, parse_args


def max_zero_run(col: np.ndarray) -> int:
    best = run = 0
    for v in col == 0:
        run = run + 1 if v else 0
        best = max(best, run)
    return best


def load_frame(name: str, spec: dict, split: list[float]) -> pd.DataFrame:
    raw = tsdb.load(spec["source"])
    if name == "ETT_h1":
        return raw[spec["subset"]].astype(float)
    if name == "Electricity":
        df = raw["X"].loc[spec["start"]:].resample(spec["resample"]).mean()
        train_end = int(len(df) * split[0])
        # Channel choice looks at the train part only.
        keep = [c for c in sorted(df.columns)
                if max_zero_run(df[c].values[:train_end]) <= spec["max_zero_run"]]
        return df[keep[:spec["n_channels"]]]
    if name == "BeijingAir":
        df = raw["X"]
        df["time"] = pd.to_datetime(df[["year", "month", "day", "hour"]])
        return df.pivot(index="time", columns="station", values=spec["variable"]).astype(float)
    raise ValueError(name)


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    out = data_dir(cfg)
    out.mkdir(parents=True, exist_ok=True)
    for name, spec in cfg["datasets"].items():
        df = load_frame(name, spec, cfg["split"])
        x = df.values.astype(np.float64)
        T = len(x)
        train_end = int(T * cfg["split"][0])
        val_end = int(T * (cfg["split"][0] + cfg["split"][1]))
        mean = np.nanmean(x[:train_end], axis=0)
        std = np.nanstd(x[:train_end], axis=0)
        std[std == 0] = 1.0
        np.savez(out / f"{name}.npz", x=((x - mean) / std).astype(np.float32), mean=mean, std=std,
                 bounds=np.array([0, train_end, val_end, T]), channels=np.array(df.columns, dtype=str),
                 time=df.index.astype(str).values)
        print(f"{name}: T={T}, channels={x.shape[1]}, missing={np.isnan(x).mean():.3%}, "
              f"split={train_end}/{val_end - train_end}/{T - val_end}")


if __name__ == "__main__":
    main()
