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


def between_average_temporal(
        obj_A,
        objs_B,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
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

    type_obj_A = get_obj_type(obj_A)
    type_objs_B = get_obj_type(objs_B)

    if not ((type_obj_A == "Evoked") or (type_obj_A == "AverageTFR")):
        raise ValueError("obj_A must be an instance of Evoked or AverageTFR objects.")

    if not ((type_objs_B == "list_Evoked") or (type_objs_B == "list_AverageTFR")):
        raise ValueError(
            "objs_B must be an instance of list of Evoked or AverageTFR objects."
        )

    if type_obj_A not in type_objs_B:
        raise ValueError("obj_A and objs_B must be of the same type.")

    obj_A = obj_A.copy()
    objs_B = [objs_B[n].copy() for n in range(len(objs_B))]

    if picks is not None:
        obj_A = obj_A.pick(picks=picks)
        for n in range(len(objs_B)):
            objs_B[n] = objs_B[n].pick(picks=picks)

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
    for ch in obj_A.ch_names:
        if type_obj_A == "AverageTFR":

            data_A = proc_tfr(obj_A)
            data_B = [proc_tfr(objs_B[n]) for n in range(len(objs_B))]

        elif type_obj_A == "Evoked":

            data_A = proc_evoked(obj_A)
            data_B = [proc_evoked(objs_B[n]) for n in range(len(objs_B))]

        scores = []
        for B in data_B:
            score = tslearn.metrics.dtw(data_A, B)
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
        obj = obj.copy().pick(picks=picks)

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
        obj = obj.copy().pick(picks=picks)

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


def sgtv(
        obj_A,
        objs_B,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
        average=True,
):
    """
    Compute SG_ATTV (Subject-to-Group Within-Trial Temporal Variability) values.

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

    try:
        import tslearn.metrics
    except:
        raise RuntimeError("tslearn is not installed. pip install tslearn")

    type_obj = check_averaged_obj_type(obj_A)

    if not isinstance(objs_B, list):
        raise ValueError("objs_B must be an list")

    if (not isinstance(objs_B[0], mne.time_frequency.tfr.AverageTFR)) and (
            not isinstance(objs_B[0], mne.evoked.Evoked)
    ):
        raise ValueError(
            "tfr_B must be an list of mne.time_frequency.tfr.AverageTFR or mne.evoked.Evoked objects"
        )

    if picks is not None:
        obj_A = obj_A.copy().pick(picks=picks)
        for n in range(len(objs_B)):
            objs_B[n] = objs_B[n].copy().pick(picks=picks)

    def proc_tfr(tfr):
        data = tfr.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        # data.shape: n_channels, n_freqs, n_times

        data = np.squeeze(data)
        # data.shape: n_freqs, n_times

        data = data.mean(0)
        # data.shape: n_times

        return data

    df_dict = {"channel": [], "sgtv": []}
    for ch in obj_A.ch_names:
        if type_obj == "tfr":

            data_A = proc_tfr(obj_A)
            data_B = [proc_tfr(objs_B[n]) for n in range(len(objs_B))]

        elif type_obj == "epochs":
            raise NotImplementedError("SG_ATTV is not implemented for epochs objects.")
            data = obj.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # data.shape: n_epochs, n_channels, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_times

        scores = []
        for B in data_B:
            score = tslearn.metrics.dtw(data_A, B)
            scores.append(score)

        if average:
            scores = np.array(scores)
            scores = scores.mean()
        else:
            scores = json.dumps(scores)

        df_dict["channel"].append(ch)
        df_dict["sgtv"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
