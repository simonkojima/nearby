import json
import warnings

import numpy as np
import scipy
import pandas as pd

import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2

from ..utils import check_obj_type, add_values_to_df


def atsv(obj,
         baseline=None,
         fmin=None,
         fmax=None,
         tmin=None,
         tmax=None,
         metrics="angle",
         picks=None,
         average=True,
         additional_values=None):
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    if type_obj == "tfr":
        data = obj.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels
    elif type_obj == "epochs":
        data = obj.get_data(tmin=tmin, tmax=tmax, units="uV")
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

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
