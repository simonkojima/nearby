import json
import warnings

import numpy as np
import scipy
import pandas as pd

import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2

from ..utils import (
    check_obj_type,
    add_values_to_df,
    auto_windows,
    check_averaged_obj_type,
)


def atsv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        metrics="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Spatial Variability values.

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.BaseTFR``
        The MNE object containing either epochs or time–frequency data.

    baseline : tuple | list | None, optional
        Time interval used for baseline correction. If `None`, no baseline is applied.
        If `obj` is an MNE TFR object, the baseline is applied using `obj.copy().apply_baseline(baseline, mode="percent")`.
        If `obj` is an MNE Epochs object, it is applied using `obj.copy().apply_baseline(baseline)`.

    fmin, fmax : float | None, optional
        Frequency range of interest. Only used when `obj` is a TFR object.

    tmin, tmax : float | None, optional
        Time range of interest, in seconds.

    metrics : {"angle", "cosine"}, default="angle"
        If "angle", the distance between the mean and each epoch is computed using
        angular distance. The mean is estimated using the Fréchet mean (p = 2).

        If "cosine", the distance between the mean and each epoch is computed using
        cosine distance. The mean is estimated using an extrinsic mean.

    picks : str | array_like | None, optional
        Channels to include.

    average : bool, default=True
        If `True`, values are averaged across epochs. If `False`, a value
        is returned for each epoch.
        If ``average=False``, the DataFrame contains one entry per channel,
        where each entry consists of a JSON-encoded list representing the
        per-epoch ERD/S values.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing atsv values

    """
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


def wtsv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        mode="windows",
        windows=None,
        auto_window_size=0.1,
        auto_window_step=0.1,
        tmin=None,
        tmax=None,
        metric="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Within-Trial Spatial Variability values.

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.BaseTFR``
        The MNE object containing either epochs or time–frequency data.

    baseline : tuple | list | None, optional
        Time interval used for baseline correction. If `None`, no baseline is applied.
        If `obj` is an MNE TFR object, the baseline is applied using `obj.copy().apply_baseline(baseline, mode="percent")`.
        If `obj` is an MNE Epochs object, it is applied using `obj.copy().apply_baseline(baseline)`.

    fmin, fmax : float | None, optional
        Frequency range of interest. Only used when `obj` is a TFR object.

    tmin, tmax : float | None, optional
        Time range of interest, in seconds.

    metric : {"angle", "cosine"}, default="angle"
        If "angle", the distance between the mean and each epoch is computed using
        angular distance. The mean is estimated using the Fréchet mean (p = 2).

        If "cosine", the distance between the mean and each epoch is computed using
        cosine distance. The mean is estimated using an extrinsic mean.

    picks : str | array_like | None, optional
        Channels to include.

    average : bool, default=True
        If `True`, values are averaged across epochs. If `False`, a value
        is returned for each epoch.
        If ``average=False``, the DataFrame contains one entry per channel,
        where each entry consists of a JSON-encoded list representing the
        per-epoch ERD/S values.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing atsv values

    """
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    if (tmin is not None) or (tmax is not None):
        obj = obj.crop(tmin=tmin, tmax=tmax)

    if type_obj == "tfr":
        if (fmin is not None) or (fmax is not None):
            obj = obj.crop(fmin=fmin, fmax=fmax)

    data_list = []
    if mode == "windows":
        if windows is None:
            windows = auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=obj.tmin,
                tmax=obj.tmax,
            )

        for window in windows:
            data = obj.get_data(tmin=window[0], tmax=window[1])
            # tfrs.shape: n_epochs, n_channels, n_freqs, n_times
            # epochs.shape: n_epochs, n_channels, n_times

            if type_obj == "tfr":
                data = data.mean(2)
                # data.shape: n_epochs, n_channels, n_times

            data = data.mean(2)
            # data.shape: n_epochs, n_channels

            data_list.append(data)

        data = np.array(data_list)
        # data.shape: n_windows, n_epochs, n_channels

        data = data.transpose(1, 0, 2)
        # data.shape: n_epochs, n_windows, n_channels

    elif mode == "all_samples":
        data = obj.get_data()
        # tfrs.shape: n_epochs, n_channels, n_freqs, n_times
        # epochs.shape: n_epochs, n_channels, n_times

        if type_obj == "tfr":
            data = data.mean(2)
            # data.shape: n_epochs, n_channels, n_times
        # In all_samples mode, n_times is equal to the number of windows
        # data.shape: n_epochs, n_channels, n_windows

        data = data.transpose(0, 2, 1)
        # data.shape: n_epochs, n_windows, n_channels

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    n_epochs, n_windows, _ = data.shape

    distances = []

    for idx_trial in range(n_epochs):
        data_trial = data[idx_trial, :, :].squeeze()
        # data_trial.shape: n_windows, n_channels

        if metric == "angle":
            mean = mean_frechet_p2(data_trial, verbosity=0)

            distances_trial = []
            for n in range(n_windows):
                d = distance_angle(mean, data_trial[n, :])
                distances_trial.append(d)

        elif metric == "cosine":
            mean = mean_extrinsic(data_trial)

            distances_trial = []
            for n in range(n_windows):
                d = scipy.spatial.distance.cosine(mean, data_trial[n, :])
                distances_trial.append(d)

        dispersion = np.mean(distances_trial)

        distances.append(dispersion)

    if average:
        distances = np.mean(distances)
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    if metric == "angle":
        df["wtsva"] = [distances]
    elif metric == "cosine":
        df["wtsvc"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def sg_atsv(
        obj_A,
        obj_B,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        metric="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Spatial Variability values.

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.BaseTFR``
        The MNE object containing either epochs or time–frequency data.

    baseline : tuple | list | None, optional
        Time interval used for baseline correction. If `None`, no baseline is applied.
        If `obj` is an MNE TFR object, the baseline is applied using `obj.copy().apply_baseline(baseline, mode="percent")`.
        If `obj` is an MNE Epochs object, it is applied using `obj.copy().apply_baseline(baseline)`.

    fmin, fmax : float | None, optional
        Frequency range of interest. Only used when `obj` is a TFR object.

    tmin, tmax : float | None, optional
        Time range of interest, in seconds.

    metrics : {"angle", "cosine"}, default="angle"
        If "angle", the distance between the mean and each epoch is computed using
        angular distance. The mean is estimated using the Fréchet mean (p = 2).

        If "cosine", the distance between the mean and each epoch is computed using
        cosine distance. The mean is estimated using an extrinsic mean.

    picks : str | array_like | None, optional
        Channels to include.

    average : bool, default=True
        If `True`, values are averaged across epochs. If `False`, a value
        is returned for each epoch.
        If ``average=False``, the DataFrame contains one entry per channel,
        where each entry consists of a JSON-encoded list representing the
        per-epoch ERD/S values.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing atsv values

    """
    type_obj = check_averaged_obj_type(obj_A)

    if picks is not None:
        obj_A = obj_A.pick(picks=picks)
        for n in range(len(obj_B)):
            obj_B[n] = obj_B[n].pick(picks=picks)

    def proc_tfr(tfr):
        data = tfr.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        # data.shape: n_channels, n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_channels, n_times

        data = data.mean(1)
        # data.shape: n_times

        return data

    if type_obj == "tfr":

        data_A = proc_tfr(obj_A)
        data_B = [proc_tfr(obj_B[n]) for n in range(len(obj_B))]

    elif type_obj == "epochs":
        raise NotImplementedError("SG_ATTV is not implemented for epochs objects.")
        data = obj.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels

    distances = []

    if metric == "cosine":
        for B in data_B:
            d = scipy.spatial.distance.cosine(data_A, B)
            distances.append(d)

    elif metric == "angle":
        for B in data_B:
            d = distance_angle(data_A, B)
            distances.append(d)

    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    if average:
        distances = np.array(distances)
        distances = distances.mean()
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    if metric == "angle":
        df["sg_atsva"] = [distances]
    elif metric == "cosine":
        df["sg_atsvc"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
