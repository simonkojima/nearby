import json
import warnings

import numpy as np
import scipy
import pandas as pd

import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2


def atsv(tfrs, baseline=[-1.0, 0.0], fmin=7, fmax=13, tmin=0.5, tmax=4.5, metrics="angle", average=True,
         additional_values=None):
    if baseline is not None:
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
        mean = mean_frechet_p2(data, verbosity=0)

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
    if metrics == "angle":
        df["atsva"] = [distances]
    elif metrics == "cosine":
        df["atsvc"] = [distances]
    # df["metrics"] = [metrics]

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df
