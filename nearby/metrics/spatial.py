import json
import warnings

import numpy as np
import scipy
import pandas as pd


def distance_angle(w1, w2):
    return np.arccos(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)))


def get_random_unit_vec(d, rng=None):
    rng = np.random.default_rng(rng)
    v = rng.normal(size=d)
    n = np.linalg.norm(v)
    if n == 0:
        return get_random_unit_vec(d, rng)
    return v / n


def mean_extrinsic(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    U = X / norms

    S = U.sum(axis=0)
    s = np.linalg.norm(S)

    return S / s


def mean_frechet_p2(X,
                    max_iter=1000,
                    lr=2.0,
                    alpha=1e-4,
                    beta=0.5,
                    tol=1e-8,
                    eps=1e-12,
                    init="extrinsic",
                    seed=42):
    X = np.array(X)

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    if init == "extrinsic":
        m = mean_extrinsic(X)
    elif init == "random":
        rng = np.random.default_rng(seed)
        m = get_random_unit_vec(d=X.shape[1], rng=rng)
    else:
        m = np.array(init)
        m /= np.linalg.norm(m)

    # m = X.mean(axis=0)
    # m /= np.linalg.norm(m)
    # print(m)

    def f_val(m):
        # ci = np.clip(X @ m, -1.0 + eps, 1.0 - eps)
        ci = X @ m
        return np.sum(np.arccos(ci) ** 2)

    f_prev = f_val(m)

    for _ in range(max_iter):

        # ci = np.clip(X @ m, -1.0 + eps, 1.0 - eps)
        ci = X @ m
        theta = np.arccos(ci)
        den = np.sqrt(np.maximum(eps, 1.0 - ci ** 2))

        Ui = (X - np.outer(ci, m)) / den[:, None]

        grad = -((2.0 * theta)[:, None] * Ui).sum(axis=0)

        g2 = np.dot(grad, grad)
        if g2 < tol ** 2:
            break

        eta = lr
        while True:
            m_cand = m - eta * grad
            m_cand /= np.linalg.norm(m_cand)

            f_cand = f_val(m_cand)
            if f_cand <= f_prev - alpha * eta * g2:
                m, f_prev = m_cand, f_cand
                # lr = max(lr, eta)
                break
            eta *= beta
            if eta < 1e-12:
                return m

    else:
        warnings.warn("Convergence not reached")

    return m


def spatial_dispersion(tfrs, baseline=[-1.0, 0.0], fmin=7, fmax=13, tmin=0.5, tmax=4.5, metrics="angle", average=True,
                       additional_values=None):
    tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
    # data.shape: n_epochs, n_channels, n_freqs, n_times

    data = data.mean(2)
    # data.shape: n_epochs, n_channels, n_times

    data = data.mean(2)
    # data.shape: n_epochs, n_channels

    n_epochs, _ = data.shape

    distances = []

    if metrics == "cosine":
        mean = mean_extrinsic(data)

        for n in range(n_epochs):
            d = scipy.spatial.distance.cosine(mean, data[n, :])
            distances.append(d)

    elif metrics == "angle":
        mean = mean_frechet_p2(data)

        for n in range(n_epochs):
            d = distance_angle(mean, data[n, :])
            distances.append(d)

    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    if average:
        distances = np.array(distances)
        distances = distances.mean()
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["spatial_dispersion"] = [distances]
    df["metrics"] = [metrics]

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df
