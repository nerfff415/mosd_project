"""Point, probabilistic and calibration metrics on hidden targets.

Shapes: y [N, k] targets, Q [N, k, n_q] predictive quantiles at levels `qs` (sorted).
"""

import numpy as np
from scipy.special import ndtr

PINBALL_LEVELS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]


def central_pairs(qs: np.ndarray) -> list[tuple[float, int, int]]:
    """(nominal level s, lower index, upper index) for symmetric quantile pairs."""
    out = []
    for i, lo in enumerate(qs):
        j = int(np.argmin(np.abs(qs - (1 - lo))))
        if lo < 0.5 and np.isclose(qs[j], 1 - lo):
            out.append((round(1 - 2 * lo, 6), i, j))
    return out


def level_index(qs: np.ndarray, level: float) -> int:
    i = int(np.argmin(np.abs(qs - level)))
    assert np.isclose(qs[i], level), f"quantile {level} not available"
    return i


def pinball(y: np.ndarray, Q: np.ndarray, qs: np.ndarray, levels=PINBALL_LEVELS) -> float:
    vals = []
    for q in levels:
        e = y - Q[..., level_index(qs, q)]
        vals.append(np.maximum(q * e, (q - 1) * e).mean())
    return float(np.mean(vals))


def crps_gaussian(y, mean, std) -> float:
    z = (y - mean) / std
    return float((std * (z * (2 * ndtr(z) - 1) + 2 * _phi_pdf(z) - 1 / np.sqrt(np.pi))).mean())


def crps_samples(y: np.ndarray, samples: np.ndarray) -> float:
    """y [N, k], samples [N, S, k]. E|X - y| - 0.5 E|X - X'| with the sorted-sample identity."""
    S = samples.shape[1]
    term1 = np.abs(samples - y[:, None, :]).mean(axis=1)
    xs = np.sort(samples, axis=1)
    w = (2 * np.arange(1, S + 1) - S - 1)[None, :, None]
    term2 = (w * xs).sum(axis=1) / S ** 2  # = 0.5 E|X - X'|
    return float((term1 - term2).mean())


def crps_quantiles(y, Q, qs) -> float:
    """Quantile-score approximation: CRPS ~ 2 * mean pinball over all available levels."""
    return 2 * pinball(y, Q, qs, levels=list(qs))


def nll_gaussian(y, mean, std) -> float:
    return float((0.5 * np.log(2 * np.pi * std ** 2) + 0.5 * ((y - mean) / std) ** 2).mean())


def _phi_pdf(z):
    return np.exp(-0.5 * z ** 2) / np.sqrt(2 * np.pi)


def reliability(y, Q, qs) -> np.ndarray:
    """Empirical c(q) = mean 1[y <= Q_q] for every level."""
    return (y[..., None] <= Q).reshape(-1, len(qs)).mean(axis=0)


def all_metrics(y, Q, qs, y_orig, m_orig, mean=None, std=None, samples=None) -> dict:
    qs = np.asarray(qs)
    m = Q[..., level_index(qs, 0.5)]
    err = y - m
    out = {"MAE": np.abs(err).mean(), "MSE": (err ** 2).mean()}
    out["RMSE"] = np.sqrt(out["MSE"])
    denom = np.abs(y_orig).sum()
    out["MRE"] = np.abs(y_orig - m_orig).sum() / denom if denom > 1e-8 * y_orig.size else np.nan
    out["pinball"] = pinball(y, Q, qs)
    if mean is not None:
        out["CRPS"], out["CRPS_method"] = crps_gaussian(y, mean, std), "analytic"
        out["NLL"] = nll_gaussian(y, mean, std)
    elif samples is not None:
        out["CRPS"], out["CRPS_method"] = crps_samples(y, samples), "samples"
        out["NLL"] = np.nan
    else:
        out["CRPS"], out["CRPS_method"] = crps_quantiles(y, Q, qs), "quantile approx"
        out["NLL"] = np.nan
    out["PCE"] = float(np.abs(qs - reliability(y, Q, qs)).mean())
    yflat = y.reshape(-1)
    cce, siw = [], []
    for s, lo, hi in central_pairs(qs):
        inside = (Q[..., lo] <= y) & (y <= Q[..., hi])
        width = Q[..., hi] - Q[..., lo]
        cce.append(s - inside.mean())
        iqr = np.quantile(yflat, (1 + s) / 2) - np.quantile(yflat, (1 - s) / 2)
        siw.append(width.mean() / iqr if iqr > 0 else np.nan)
        if np.isclose(s, 0.8):
            out["cov80"], out["width80"] = inside.mean(), width.mean()
    out["CCE"], out["SIW"] = float(np.mean(cce)), float(np.nanmean(siw))
    return {k: (float(v) if not isinstance(v, str) else v) for k, v in out.items()}


def along_gap(y, Q, qs) -> tuple[np.ndarray, np.ndarray]:
    """80% coverage and width at each position of a contiguous gap, averaged over windows."""
    lo, hi = level_index(np.asarray(qs), 0.1), level_index(np.asarray(qs), 0.9)
    inside = (Q[..., lo] <= y) & (y <= Q[..., hi])
    return inside.mean(axis=0), (Q[..., hi] - Q[..., lo]).mean(axis=0)
