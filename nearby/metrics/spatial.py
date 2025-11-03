import json
import warnings

import numpy as np
import scipy
import pandas as pd

import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2


def spatial_dispersion_cross_user_array(data, metrics="angle", average=True, additional_values=None):
    n_users, _ = data.shape

    distances = []

    if metrics == "cosine":
        mean = mean_extrinsic(data)

        for n in range(n_users):
            d = scipy.spatial.distance.cosine(mean, data[n, :])
            distances.append(d)

    elif metrics == "angle":
        mean = mean_frechet_p2(data)

        for n in range(n_users):
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


def spatial_dispersion_cross_user(tfrs_list, baseline=[-1.0, 0.0], fmin=7, fmax=13, tmin=0.5, tmax=4.5, metrics="angle",
                                  average=True,
                                  additional_values=None):
    data_list = []
    for tfrs in tfrs_list:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels

        data = data.mean(0)
        # data.shape: n_channels

        data_list.append(data)

    data = np.stack(data_list, axis=0)

    df = spatial_dispersion_cross_user_array(data, metrics=metrics, average=average,
                                             additional_values=additional_values)

    return df


def spatial_dispersion(tfrs, baseline=[-1.0, 0.0], fmin=7, fmax=13, tmin=0.5, tmax=4.5, metrics="angle", average=True,
                       additional_values=None):
    return atsv(tfrs, baseline, fmin, fmax, tmin, tmax, metrics, average, additional_values)


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


def atsv_cross_user(tfrs_list, fmin=7, fmax=13, tmin=0.5, tmax=4.5, metrics="angle", average=True,
                    additional_values=None):
    data_list = []
    for tfrs in tfrs_list:
        if not isinstance(tfrs, mne.time_frequency.tfr.AverageTFR):
            raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.AverageTFR. use tfrs.average()")

        data_list.append(tfrs.get_data())

    info = tfrs_list[0].info
    times = tfrs_list[0].times
    freqs = tfrs_list[0].freqs

    tfrs = mne.time_frequency.EpochsTFRArray(
        info=info,
        data=np.array(data_list),
        times=times,
        freqs=freqs,
    )

    return atsv(tfrs, baseline=None, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, metrics=metrics, average=True,
                additional_values=additional_values)
