import json
import numpy as np
import pandas as pd
import tslearn.metrics
import mne

from ..utils import (
    check_obj_type,
    add_values_to_df,
    check_averaged_obj_type,
    get_obj_type,
)


def within_trial_temporal(
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
    Compute Within-Trial Temporal (WTTemp) Variability values.

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.EpochsTFR``
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
        DataFrame containing variability values for each channel.

    """
    type_obj = get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError(
            "within_trial_temporal is only implemented for EpochsTFR or Epochs objects."
        )

    if baseline is not None:
        if type_obj == "EpochsTFR":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "Epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.copy().pick(picks=picks)

    df_dict = {"channel": [], "within_trial_temporal": []}
    for ch in obj.ch_names:
        if type_obj == "EpochsTFR":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times
        elif type_obj == "Epochs":
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
        df_dict["within_trial_temporal"].append(sd)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_trial_temporal(
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
    Compute Between-Trial Temporal (BTTemp) Variability values.

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
        DataFrame containing variability values for each channel.

    """

    type_obj = get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError(
            "within_trial_temporal is only implemented for EpochsTFR or Epochs objects."
        )

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.copy().pick(picks=picks)

    df_dict = {"channel": [], "between_trial_temporal": []}
    for ch in obj.ch_names:
        if type_obj == "EpochsTFR":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times

        elif type_obj == "Epochs":
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
        df_dict["between_trial_temporal"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def between_trial_group_temporal(
        objs,
        centroid="mean",
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
        weights="nave",
        average=True,
):
    """
    Compute Between-Average Temporal (BATemp) values.

    Parameters
    ----------

    obj : ``mne.evoked.Evoked`` | ``mne.time_frequency.tfr.AverageTFR``
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
        DataFrame containing variability values for each channel.

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
        data = tfr.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        # data.shape: n_channels, n_freqs, n_times

        data = np.squeeze(data)
        # data.shape: n_freqs, n_times

        data = data.mean(0)
        # data.shape: n_times

        return data

    def proc_evoked(evoked):
        data = evoked.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
        # data.shape: n_channels, n_times

        data = np.squeeze(data)
        # data.shape: n_times

        return data

    df_dict = {"channel": [], "between_average_temporal": []}
    for ch in obj_centroid.ch_names:
        if type_obj_centroid == "AverageTFR":

            data_centroid = proc_tfr(obj_centroid)
            data_list = [proc_tfr(objs[n]) for n in range(len(objs))]

        elif type_obj_centroid == "Evoked":

            data_centroid = proc_evoked(obj_centroid)
            data_list = [proc_evoked(objs[n]) for n in range(len(objs))]

        scores = []
        for data in data_list:
            score = tslearn.metrics.dtw(data_centroid, data)
            scores.append(score)

        if average:
            scores = np.array(scores)
            scores = scores.mean()
        else:
            scores = json.dumps(scores)

        df_dict["channel"].append(ch)
        df_dict["between_average_temporal"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
