"""Step 3a: Gaussian Kalman smoother (local-level model) as the classical probabilistic baseline.

    y_t = mu_t + eps_t,    eps_t ~ N(0, r)     (observation noise)
    mu_t = mu_{t-1} + eta_t, eta_t ~ N(0, q)   (random-walk level)

Process and observation variances are fitted by maximum likelihood (coarse-to-fine grid) on train
data, one pair per channel. Inside a window the smoother conditions on all observed positions. The predictive
distribution of a missing *measurement* is N(mu_{t|T}, P_{t|T} + r), so observation variance is
included. Fitted-parameter uncertainty is excluded.

Usage: python -m tsi.kalman --config configs/main.yaml
"""

import numpy as np
from scipy.stats import norm

from .common import load_config, load_masks, load_series, mask_configs, masked_windows, parse_args, pred_path

P0 = 1e6  # near-diffuse prior variance of the initial level (data are standardised)


def kalman_filter(y: np.ndarray, q: np.ndarray, r: np.ndarray):
    """Batched filter over rows of y [N, T] with NaN for missing values."""
    N, T = y.shape
    a_f, P_f = np.empty((N, T)), np.empty((N, T))
    a, P = np.zeros(N), np.full(N, P0)
    loglik = np.zeros(N)
    seen = np.zeros(N, dtype=bool)
    for t in range(T):
        if t > 0:
            P = P + q
        obs = ~np.isnan(y[:, t])
        F = P + r
        v = np.where(obs, y[:, t] - a, 0.0)
        # Skip the likelihood term of the first observation (diffuse initialisation).
        use = obs & seen
        loglik += np.where(use, -0.5 * (np.log(2 * np.pi * F) + v ** 2 / F), 0.0)
        K = np.where(obs, P / F, 0.0)
        a = a + K * v
        P = P * (1 - K)
        seen |= obs
        a_f[:, t], P_f[:, t] = a, P
    return a_f, P_f, loglik


def kalman_smoother(y: np.ndarray, q: np.ndarray, r: np.ndarray):
    a_f, P_f, _ = kalman_filter(y, q, r)
    a_s, P_s = a_f.copy(), P_f.copy()
    for t in range(y.shape[1] - 2, -1, -1):
        P_pred = P_f[:, t] + q
        J = P_f[:, t] / P_pred
        a_s[:, t] = a_f[:, t] + J * (a_s[:, t + 1] - a_f[:, t])
        P_s[:, t] = P_f[:, t] + J ** 2 * (P_s[:, t + 1] - P_pred)
    return a_s, P_s


def fit_variances(train: np.ndarray, n_grid: int = 15, rounds: int = 3) -> np.ndarray:
    """MLE of (q, r) for every channel of train [T, C] by a coarse-to-fine log grid.

    All channels and grid points run in one batched filter, which is much faster than a
    per-channel optimiser loop. Returns params [C, 2].
    """
    C = train.shape[1]
    lo = np.tile(np.log([1e-5, 1e-6]), (C, 1))
    hi = np.tile(np.log([10.0, 10.0]), (C, 1))
    for _ in range(rounds):
        gq = np.linspace(lo[:, 0], hi[:, 0], n_grid).T  # [C, G]
        gr = np.linspace(lo[:, 1], hi[:, 1], n_grid).T
        q = np.repeat(gq, n_grid, axis=1).reshape(-1)  # [C * G * G]
        r = np.tile(gr, (1, n_grid)).reshape(-1)
        y = np.repeat(train.T, n_grid * n_grid, axis=0)
        ll = kalman_filter(y, np.exp(q), np.exp(r))[2].reshape(C, -1)
        best = ll.argmax(axis=1)
        bq, br = q.reshape(C, -1)[np.arange(C), best], r.reshape(C, -1)[np.arange(C), best]
        sq, sr = (hi - lo)[:, 0] / (n_grid - 1), (hi - lo)[:, 1] / (n_grid - 1)
        lo, hi = np.stack([bq - sq, br - sr], 1), np.stack([bq + sq, br + sr], 1)
    return np.exp(np.stack([bq, br], 1))


def main() -> None:
    cfg = load_config(parse_args(__doc__).config)
    qs = np.array(cfg["quantiles"])
    for name in cfg["datasets"]:
        series = load_series(cfg, name)
        train = series["x"][:series["bounds"][1]]
        params = fit_variances(train[-cfg["kalman"]["fit_points"]:])
        print(f"{name}: q/r per channel (median) = {np.median(params[:, 0]):.4f} / {np.median(params[:, 1]):.4f}")
        for split in ("val", "test"):
            masks = load_masks(cfg, split, name)
            out = {"params": params}
            for key in mask_configs(cfg):
                x, idx = masked_windows(series, masks, key)
                ch = masks[f"{key}/channel"]
                a_s, P_s = kalman_smoother(x, params[ch, 0], params[ch, 1])
                mean = np.take_along_axis(a_s, idx, 1)
                std = np.sqrt(np.take_along_axis(P_s, idx, 1) + params[ch, 1][:, None])
                out[f"{key}/mean"], out[f"{key}/std"] = mean, std
                out[f"{key}/quantiles"] = mean[..., None] + std[..., None] * norm.ppf(qs)
            np.savez(pred_path(cfg, split, name, "kalman"), **out)


if __name__ == "__main__":
    main()
