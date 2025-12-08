import json
import numpy as np
import pandas as pd

import scipy

import mne

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic

from ..utils import (
    check_obj_type,
    add_values_to_df,
    auto_windows,
    check_averaged_obj_type,
)


def atfv(
        tfrs,
        baseline=None,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metric="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Frequency Variability values.

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
        DataFrame containing atfv values for each channel.

    """
    type_obj = check_obj_type(tfrs)

    if type_obj != "tfr":
        raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.BaseTFR")

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.copy().pick(picks=picks)

    if metric == "angle":
        df_dict = {"channel": [], "atfva": []}
    elif metric == "cosine":
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

        if metric == "angle":
            mean = mean_frechet_p2(data, verbosity=0)

            for n in range(n_epochs):
                d = distance_angle(mean, data[n, :])
                distances.append(d)
        elif metric == "cosine":
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
        if metric == "angle":
            df_dict["atfva"].append(distances)
        elif metric == "cosine":
            df_dict["atfvc"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def wtfv(
        tfrs,
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
    Compute Within-Trial Frequency Variability values.

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
    type_obj = check_obj_type(tfrs)

    if type_obj != "tfr":
        raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.BaseTFR")

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.copy().pick(picks=picks)

    if (tmin is not None) or (tmax is not None):
        tfrs = tfrs.copy().crop(tmin=tmin, tmax=tmax)

    if (fmin is not None) or (fmax is not None):
        tfrs = tfrs.copy().crop(fmin=fmin, fmax=fmax)

    if metric == "angle":
        df_dict = {"channel": [], "wtfva": []}
    elif metric == "cosine":
        df_dict = {"channel": [], "wtfvc": []}
    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    data_dict = {}
    if mode == "windows":
        if windows is None:
            windows = auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=tfrs.tmin,
                tmax=tfrs.tmax,
            )

        for ch in tfrs.ch_names:
            data_dict[ch] = []
            for window in windows:
                data = tfrs.get_data(tmin=window[0], tmax=window[1], picks=ch)
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                data = data.squeeze()
                # data.shape: n_epochs, n_freqs, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_freqs

                data_dict[ch].append(data)

            data_dict[ch] = np.array(data_dict[ch])
            # data.shape: n_windows, n_epochs, n_freqs

            data_dict[ch] = data_dict[ch].transpose(1, 0, 2)
            # data.shape: n_epochs, n_windows, n_freqs

    elif mode == "all_samples":
        for ch in tfrs.ch_names:
            data = tfrs.get_data(picks=ch)
            # tfrs.shape: n_epochs, n_channels, n_freqs, n_times

            data = data.squeeze()
            # data.shape: n_epochs, n_freqs, n_times
            # In all_samples mode, n_times is equal to the number of windows
            # data.shape: n_epochs, n_freqs, n_windows

            data = data.transpose(0, 2, 1)
            # data.shape: n_epochs, n_windows, n_freqs

            data_dict[ch] = data

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    for ch in tfrs.ch_names:
        n_epochs, n_windows, _ = data_dict[ch].shape

        dispersion = []
        for idx_trial in range(n_epochs):
            data_trial = data_dict[ch][idx_trial, :, :].squeeze()
            # data_trial.shape: n_windows, n_freqs

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

            dispersion_within_trial = np.mean(distances_trial)

            dispersion.append(dispersion_within_trial)

        if average:
            dispersion = np.mean(dispersion)
        else:
            dispersion = json.dumps(dispersion)

        df_dict["channel"].append(ch)
        if metric == "angle":
            df_dict["wtfva"].append(dispersion)
        elif metric == "cosine":
            df_dict["wtfvc"].append(dispersion)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def sgfv(
        tfr_A,  # subject
        tfrs_B,  # group
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metric="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Subject-to-Group Frequency Variability values.

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
        DataFrame containing atfv values for each channel.

    """
    # type_obj = check_averaged_obj_type(tfrs_A)

    if not isinstance(tfr_A, mne.time_frequency.tfr.AverageTFR):
        raise ValueError(
            "tfr_A must be an instance of mne.time_frequency.tfr.AverageTFR"
        )

    if not isinstance(tfrs_B, list):
        raise ValueError("tfrs_B must be an list")

    if not isinstance(tfrs_B[0], mne.time_frequency.tfr.AverageTFR):
        raise ValueError("tfrs_B must be an list of mne.time_frequency.tfr.AverageTFR")
    """
    if not isinstance(tfrs_B, mne.time_frequency.tfr.EpochsTFR):
        if isinstance(tfrs_B, list) and (
            not isinstance(tfrs_B[0], mne.time_frequency.tfr.AverageTFR)
        ):
            raise ValueError(
                "tfr_B must be an instance of mne.time_frequency.tfr.EpochsTFR or list of mne.time_frequency.tfr.AverageTFR"
            )
    """

    # make it list of tfr if tfrs_B is EpochsTFR
    if isinstance(tfrs_B, mne.time_frequency.tfr.EpochsTFR):
        tfrs_B = [tfrs_B[n].average() for n in range(len(tfrs_B))]

    if picks is not None:
        tfr_A = tfr_A.copy().pick(picks=picks)
        for n in range(len(tfrs_B)):
            tfrs_B[n] = tfrs_B[n].copy().pick(picks=picks)

    if metric == "angle":
        df_dict = {"channel": [], "sgfva": []}
    elif metric == "cosine":
        df_dict = {"channel": [], "sgfvc": []}
    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    def proc(tfrs, ch):
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_freqs

        return data

    for ch in tfr_A.ch_names:

        data_A = proc(tfr_A, ch=ch)
        data_B = [proc(tfrs_B[n], ch=ch) for n in range(len(tfrs_B))]

        distances = []

        if metric == "angle":
            for B in data_B:
                d = distance_angle(data_A, B)
                distances.append(d)
        elif metric == "cosine":
            for B in data_B:
                d = scipy.spatial.distance.cosine(data_A, B)
                distances.append(d)

        else:
            raise ValueError("metrics must be 'angle' or 'cosine'")

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df_dict["channel"].append(ch)
        if metric == "angle":
            df_dict["sgfva"].append(distances)
        elif metric == "cosine":
            df_dict["sgfvc"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
