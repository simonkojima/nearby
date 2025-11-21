import json

import numpy as np
import pandas as pd

import mne

from ..utils import check_obj_type, add_values_to_df
from ..mean import mean_frechet_p2, mean_extrinsic
from ..distances import distance_angle


def auto_windows(size, step, tmin, tmax):
    start = np.arange(tmin, tmax, step)
    end = start + size

    windows = np.vstack([start, end]).T

    return windows


def attv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Temporal Variability values.

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
        DataFrame containing attv values for each channel.

    """
    try:
        import tslearn.metrics
    except:
        raise RuntimeError("tslearn is not installed. pip install tslearn")

    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "attv": []}
    for ch in obj.ch_names:
        if type_obj == "tfr":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times

        elif type_obj == "epochs":
            data = obj.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # data.shape: n_epochs, n_channels, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_times

        mean = data.mean(0)
        # mean.shape: n_times

        scores = []
        for m in range(data.shape[0]):
            score = tslearn.metrics.dtw(mean, data[m, :])
            scores.append(score)

        if average:
            scores = np.array(scores)
            scores = scores.mean()
        else:
            scores = json.dumps(scores)

        df_dict["channel"].append(ch)
        df_dict["attv"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def wttv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
        average=True,
):
    """
    Compute Within-Trial Temporal Variability values.

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
        DataFrame containing wttv values for each channel.

    """
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "wttv": []}
    for ch in obj.ch_names:
        if type_obj == "tfr":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times
        elif type_obj == "epochs":
            data = obj.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # data.shape: n_epochs, n_channels, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_times

        sd = np.std(data, axis=1)

        if average:
            sd = sd.mean(0)
        else:
            sd = sd.tolist()
            sd = json.dumps(sd)

        df_dict["channel"].append(ch)
        df_dict["wttv"].append(sd)

    df = pd.DataFrame(df_dict)

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
        metrics="angle",
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

    if type_obj == "epochs":
        raise NotImplementedError("epochs is not implemented yet.")

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

    print(obj)

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
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels

                data_list.append(data)

    elif mode == "all_samples":
        raise NotImplementedError("all_samples is not implemented yet.")
    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    print(data_list)

    data = np.array(data_list)
    # data.shape: n_windows, n_epochs, n_channels

    data = data.transpose(1, 0, 2)
    # data.shape: n_epochs, n_windows, n_channels

    n_epochs, n_windows, _ = data.shape

    print(data.shape)

    distances = []

    for idx_trial in range(n_epochs):
        data_trial = data[idx_trial, :, :].squeeze()
        # data_trial.shape: n_windows, n_channels

        if metrics == "angle":
            mean = mean_frechet_p2(data_trial, verbosity=0)

            distances_trial = []
            for n in range(n_windows):
                d = distance_angle(mean, data_trial[n, :])
                distances_trial.append(d)

        elif metrics == "cosine":
            mean = mean_extrinsic(data_trial)
            raise NotImplementedError("cosine is not implemented yet.")

        dispersion = np.mean(distances_trial)

        distances.append(dispersion)

    if average:
        distances = np.mean(distances)
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["atsv"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    print(df)
