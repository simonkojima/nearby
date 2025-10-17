import json
import numpy as np
import pandas as pd

from ..distances import distance_angle
from ..mean import mean_frechet_p2


def frequency_dispersion(tfrs, baseline=[-1.0, 0.0], tmin=0.5, tmax=4.5, average=True, additional_values=None,
                         picks=None):
    # Spectral Angle Mapper
    # Angular distance between two frequency representations

    if picks is not None:
        ch_names = picks
    else:
        ch_names = tfrs.ch_names

    tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = tfrs.get_data(tmin=tmin, tmax=tmax, picks=ch).squeeze()
        # data.shape: n_epochs, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_freqs

        n_epochs, _ = data.shape

        mean = mean_frechet_p2(data)

        distances = []
        for n in range(n_epochs):
            d = distance_angle(mean, data[n, :])
            distances.append(d)

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df = pd.DataFrame()
        df["spatial_dispersion"] = [distances]
        df["channel"] = [ch]

        if additional_values is not None:
            for key, value in additional_values.items():
                df[key] = [value for m in range(df.shape[0])]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    return df
