import json
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
    get_obj_type,
)


def within_trial_spatial(
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
        DataFrame containing variability values

    """
    type_obj = get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError("obj must be an instance of EpochsTFR or Epochs objects.")

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    """
    if (tmin is not None) or (tmax is not None):
        obj = obj.crop(tmin=tmin, tmax=tmax)

    if type_obj == "EpochsTFR":
        if (fmin is not None) or (fmax is not None):
            obj = obj.crop(fmin=fmin, fmax=fmax)
    """

    data_list = []
    if mode == "windows":
        if windows is None:
            windows = auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=tmin,
                tmax=tmax,
            )

            if (windows.ndim != 2) or (windows.shape[0] == 1):
                raise ValueError(f"Two or more windows must be specified. {windows=}")

        for window in windows:
            if type_obj == "EpochsTFR":
                data = obj.get_data(tmin=window[0], tmax=window[1])
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels
            elif type_obj == "Epochs":
                data = obj.get_data(tmin=window[0], tmax=window[1], units="uV")
                # data.shape: n_epochs, n_channels, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels

            data_list.append(data)

        data = np.array(data_list)
        # data.shape: n_windows, n_epochs, n_channels

        data = data.transpose(1, 0, 2)
        # data.shape: n_epochs, n_windows, n_channels

    elif mode == "all_samples":
        if type_obj == "EpochsTFR":
            data = obj.get_data()
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = data.mean(2)
            # data.shape: n_epochs, n_channels, n_times
        elif type_obj == "Epochs":
            data = obj.get_data(units="uV")
            # data.shape: n_epochs, n_channels, n_times

        # In all_samples mode, n_times is equal to the number of windows
        # data.shape: n_epochs, n_channels, n_windows = n_epochs, n_channels, n_times

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
    """
    if metric == "angle":
        df["wtsva"] = [distances]
    elif metric == "cosine":
        df["wtsvc"] = [distances]
    """

    df["within_trial_spatial"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_trial_spatial(
        obj,
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
        DataFrame containing variability values

    """
    type_obj = get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError("obj must be an instance of EpochsTFR or Epochs objects.")

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    if type_obj == "EpochsTFR":
        data = obj.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels
    elif type_obj == "Epochs":
        data = obj.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels

    n_epochs, _ = data.shape

    distances = []

    if metric == "cosine":
        mean = mean_extrinsic(data)

        for n in range(n_epochs):
            d = scipy.spatial.distance.cosine(mean, data[n, :])
            distances.append(d)

    elif metric == "angle":
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
    """
    if metric == "angle":
        df["atsva"] = [distances]
    elif metric == "cosine":
        df["atsvc"] = [distances]
    """
    df["between_trial_spatial"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_average_spatial(
        objs,
        centroid="mean",
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        metric="angle",
        picks=None,
        average=True,
        weights="nave",
        additional_values=None,
):
    """
    Compute Subject-to-Group Spatial Variability values.

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
        DataFrame containing variability values

    """

    type_objs = get_obj_type(objs)

    if not ((type_objs == "list_Evoked") or (type_objs == "list_AverageTFR")):
        raise ValueError(
            "objs must be an instance of list of Evoked or AverageTFR objects."
        )

    if isinstance(centroid, str):
        match centroid:
            case "mean":
                if type_objs == "list_Evoked":
                    obj_centroid = mne.combine_evoked(objs, weights=weights)
                elif type_objs == "list_AverageTFR":
                    obj_centroid = mne.time_frequency.combine_tfr(objs, weights=weights)
            case _:
                raise NotImplementedError(f"centroid={centroid} is not implemented.")
    else:
        obj_centroid = centroid

    type_obj_centroid = get_obj_type(obj_centroid)
    if not ((type_obj_centroid == "Evoked") or (type_obj_centroid == "AverageTFR")):
        raise ValueError(
            "centroid must be an instance of Evoked or AverageTFR objects."
        )

    if type_obj_centroid not in type_objs:
        raise ValueError("centroid and objs must be of the same type.")

    obj_centroid = obj_centroid.copy()
    objs = [objs[n].copy() for n in range(len(objs))]

    if picks is not None:
        obj_centroid = obj_centroid.pick(picks=picks)
        for n in range(len(objs)):
            objs[n] = objs[n].pick(picks=picks)

    def proc_tfr(tfr):
        data = tfr.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        # data.shape: n_channels, n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_channels, n_times

        data = data.mean(1)
        # data.shape: n_channels

        return data

    def proc_evoked(evoked):
        data = evoked.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_channels, n_times

        data = data.mean(1)
        # data.shape: n_channels

        return data

    if type_obj_centroid == "AverageTFR":

        data_centroid = proc_tfr(obj_centroid)
        data_list = [proc_tfr(objs[n]) for n in range(len(objs))]

    elif type_obj_centroid == "Evoked":
        data_centroid = proc_evoked(obj_centroid)
        data_list = [proc_evoked(objs[n]) for n in range(len(objs))]

    distances = []

    if metric == "cosine":
        for data in data_list:
            d = scipy.spatial.distance.cosine(data_centroid, data)
            distances.append(d)

    elif metric == "angle":
        for data in data_list:
            d = distance_angle(data_centroid, data)
            distances.append(d)

    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    if average:
        distances = np.array(distances)
        distances = distances.mean()
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["between_average_spatial"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
