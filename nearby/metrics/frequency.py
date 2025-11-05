import json
import numpy as np
import pandas as pd

import scipy

import mne

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic

from ..utils import check_obj_type, add_values_to_df


def atfv(tfrs,
         baseline=None,
         tmin=None,
         tmax=None,
         fmin=None,
         fmax=None,
         metrics="angle",
         picks=None,
         average=True,
         additional_values=None):
    type_obj = check_obj_type(tfrs)

    if type_obj != "tfr":
        raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.BaseTFR")

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    if metrics == "angle":
        df_dict = {"channel": [], "atfva": []}
    elif metrics == "cosine":
        df_dict = {"channel": [], "atfvc": []}
    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    for ch in tfrs.ch_names:
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_epochs, n_freqs, n_times

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

        df_dict["channel"].append(ch)
        if metrics == "angle":
            df_dict["atfva"].append(distances)
        elif metrics == "cosine":
            df_dict["atfvc"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
