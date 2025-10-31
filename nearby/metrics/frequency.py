import json
import numpy as np
import pandas as pd

import scipy

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic


def frequency_dispersion(tfrs, baseline=[-1.0, 0.0], tmin=0.5, tmax=4.5, fmin=None, fmax=None, average=True,
                         additional_values=None,
                         picks=None):
    # import warnings
    # warnings.deprecated("frequency_dispersion is deprecated. Use atfv instead.")
    return atfv(tfrs, baseline, tmin, tmax, fmin, fmax, "angle", average, additional_values, picks)


def atfv(tfrs, baseline=[-1.0, 0.0], tmin=0.5, tmax=4.5, fmin=None, fmax=None, metrics="angle", average=True,
         additional_values=None,
         picks=None):
    # Spectral Angle Mapper
    # Angular distance between two frequency representations

    if picks is not None:
        ch_names = picks
    else:
        ch_names = tfrs.ch_names

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch).squeeze()
        # data.shape: n_epochs, n_freqs, n_times

        print(data)

        exit()

        data = data.mean(2)
        # data.shape: n_epochs, n_freqs

        n_epochs, _ = data.shape

        distances = []

        if metrics == "angle":
            mean = mean_frechet_p2(data, verbosity=0)

            for n in range(n_epochs):
                d = distance_angle(mean, data[n, :])
                distances.append(d)
        elif metrics == "cosine":
            mean = mean_extrinsic(data)

            for n in range(n_epochs):
                d = scipy.spatial.distance.cosine(mean, data[n, :])
                distances.append(d)

        else:
            raise ValueError("metrics must be 'angle' or 'cosine'")

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df = pd.DataFrame()
        # df["atfv"] = [distances]
        if metrics == "angle":
            df["atfva"] = [distances]
        elif metrics == "cosine":
            df["atfvc"] = [distances]
        df["channel"] = [ch]

        if additional_values is not None:
            for key, value in additional_values.items():
                df[key] = [value for m in range(df.shape[0])]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df
