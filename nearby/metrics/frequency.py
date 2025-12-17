import json
import numpy as np
import pandas as pd
import scipy
import mne

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic

from ..utils import (
    add_values_to_df,
    auto_windows,
    get_obj_type,
)


def within_trial_frequency(
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
        DataFrame containing variability values

    """
    type_obj = get_obj_type(tfrs)

    if not (type_obj == "EpochsTFR"):
        raise ValueError("obj must be an instance of EpochsTFR objects.")

    tfrs = tfrs.copy()

    if baseline is not None:
        tfrs = tfrs.apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    """
    if (tmin is not None) or (tmax is not None):
        tfrs = tfrs.crop(tmin=tmin, tmax=tmax)

    if (fmin is not None) or (fmax is not None):
        tfrs = tfrs.crop(fmin=fmin, fmax=fmax)
    """

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "within_trial_frequency": []}

    data_dict = {}
    if mode == "windows":
        if windows is None:
            windows = auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=tmin,
                tmax=tmax,
            )

        for ch in tfrs.ch_names:
            data_dict[ch] = []
            for window in windows:
                data = tfrs.get_data(
                    tmin=window[0], tmax=window[1], fmin=fmin, fmax=fmax, picks=ch
                )
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
            data = tfrs.get_data(fmin=fmin, fmax=fmax, picks=ch)
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
        """
        if metric == "angle":
            df_dict["wtfva"].append(dispersion)
        elif metric == "cosine":
            df_dict["wtfvc"].append(dispersion)
        """
        df_dict["within_trial_frequency"].append(dispersion)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_trial_frequency(
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
        DataFrame containing variability values for each channel.

    """
    type_obj = get_obj_type(tfrs)

    if not (type_obj == "EpochsTFR"):
        raise ValueError("obj must be an instance of EpochsTFR objects.")

    tfrs = tfrs.copy()

    if baseline is not None:
        tfrs = tfrs.apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "between_trial_frequency": []}

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
        """
        if metric == "angle":
            df_dict["atfva"].append(distances)
        elif metric == "cosine":
            df_dict["atfvc"].append(distances)
        """
        df_dict["between_trial_frequency"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_trial_group_frequency(
        tfrs,
        centroid="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metric="angle",
        picks=None,
        average=True,
        weights="nave",
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
        DataFrame containing variability values for each channel.

    """
    # type_obj = check_averaged_obj_type(tfrs_A)

    if isinstance(centroid, str):
        match centroid:
            case "mean":
                tfr_centroid = mne.time_frequency.combine_tfr(tfrs, weights=weights)
            case _:
                raise NotImplementedError(f"centroid={centroid} is not implemented.")
    else:
        tfr_centroid = centroid

    type_tfr_centroid = get_obj_type(tfr_centroid)
    type_tfrs = get_obj_type(tfrs)

    if not (type_tfr_centroid == "AverageTFR"):
        raise ValueError("centroid must be an instance of AverageTFR objects.")

    if not (type_tfrs == "list_AverageTFR"):
        raise ValueError("tfrs must be an instance of list of AverageTFR objects.")

    tfr_centroid = tfr_centroid.copy()
    tfrs = [tfrs[n].copy() for n in range(len(tfrs))]

    if picks is not None:
        tfr_centroid = tfr_centroid.pick(picks=picks)
        for n in range(len(tfrs)):
            tfrs[n] = tfrs[n].pick(picks=picks)

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "between_average_frequency": []}

    def proc(tfrs, ch):
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_freqs

        return data

    for ch in tfr_centroid.ch_names:

        data_centroid = proc(tfr_centroid, ch=ch)
        data_list = [proc(tfrs[n], ch=ch) for n in range(len(tfrs))]

        distances = []

        if metric == "angle":
            for data in data_list:
                d = distance_angle(data_centroid, data)
                distances.append(d)
        elif metric == "cosine":
            for data in data_list:
                d = scipy.spatial.distance.cosine(data_centroid, data)
                distances.append(d)

        else:
            raise ValueError("metrics must be 'angle' or 'cosine'")

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df_dict["channel"].append(ch)
        df_dict["between_average_frequency"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
