import json
import numpy as np
import pandas as pd
import tslearn.metrics
import mne

from ..utils.core import (
    _add_values_to_df,
    _get_obj_type,
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
    Compute Within-Trial Temporal (WiTrialTemp) Variability values.

    This metric quantifies how much the time course varies within each epoch.
    For each channel, it computes the standard deviation over time within an epoch,
    then optionally averages those values across epochs.

    - If ``obj`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, power values are
      first restricted to the specified frequency band (``fmin``–``fmax``) and then
      averaged over frequency, yielding a single time course per epoch.
    - If ``obj`` is an :class:`mne.epochs.Epochs`, the epoched signal is used
      directly (in microvolts).

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE object containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied.

        For :class:`mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline, mode="percent")

        For :class:`mne.epochs.Epochs`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline)

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). Only used when ``obj`` is an
        :class:`mne.time_frequency.tfr.EpochsTFR`. If ``None``, MNE defaults are used.

    tmin, tmax : float | None, optional
        Time window of interest (in seconds). If provided, data are cropped to
        ``[tmin, tmax]`` before computing variability.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` / :meth:`mne.time_frequency.EpochsTFR.pick`.
        If ``None``, all channels are used.

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging the
        per-epoch within-trial temporal standard deviations across epochs.

        If ``False``, returns a per-channel JSON-encoded list containing one value
        per epoch (order follows the epoch order in ``obj``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``within_trial_temporal`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-epoch variability values.

    Notes
    -----
    For each channel and epoch, this function computes the standard deviation over
    the time axis within the selected time window. For time–frequency input, the
    time course is first reduced by averaging over the frequency axis within
    ``[fmin, fmax]``.

    This function is implemented only for :class:`mne.epochs.Epochs` and
    :class:`mne.time_frequency.tfr.EpochsTFR`.
    """
    type_obj = _get_obj_type(obj)

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
    Compute Between-Trial Temporal (BtwTrialTemp) Variability values.

    This metric quantifies how much the time course varies across epochs.
    For each channel, it first computes a representative (mean) time course across
    epochs, then measures the dissimilarity (distance) between each epoch's time course and
    this mean time course using Dynamic Time Warping (DTW). The final channel-level
    score is the average DTW distance across epochs.

    - If ``obj`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, power values are
      first restricted to the specified frequency band (``fmin``–``fmax``) and then
      averaged over frequency, yielding a single time course per epoch.
    - If ``obj`` is an :class:`mne.epochs.Epochs`, the epoched signal is used
      directly (in microvolts).

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE object containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied.

        For :class:`mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline, mode="percent")

        For :class:`mne.epochs.Epochs`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline)

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). Only used when ``obj`` is an
        :class:`mne.time_frequency.tfr.EpochsTFR`. If ``None``, MNE defaults are used.

    tmin, tmax : float | None, optional
        Time window of interest (in seconds). If provided, data are cropped to
        ``[tmin, tmax]`` before computing variability.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` / :meth:`mne.time_frequency.EpochsTFR.pick`.
        If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging
        DTW distances across epochs.

        If ``False``, returns a per-channel JSON-encoded list containing one DTW
        distance per epoch (in the same epoch order as ``obj``).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``between_trial_temporal`` : float | str
            If ``average=True``, a scalar variability value (mean DTW distance).
            If ``average=False``, a JSON-encoded list of per-epoch DTW distances.

    Notes
    -----
    The representative time course is computed as a simple point-wise mean over epochs.
    DTW is then used to compare each epoch's time course to this mean while allowing
    non-linear temporal alignment.

    This function is implemented only for :class:`mne.epochs.Epochs` and
    :class:`mne.time_frequency.tfr.EpochsTFR`.
    """

    type_obj = _get_obj_type(obj)

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
    Compute Between-Trial-Group Temporal (BtwTrialGrpTemp) values.

    This metric quantifies how much the group-level time courses differ from
    each other. Each element of ``objs`` is assumed to represent a *trial-group
    representative* time course (e.g., an averaged waveform such as an
    :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`).

    For each channel, the function builds (or accepts) a centroid time course and
    computes the Dynamic Time Warping (DTW) distance between the centroid and each
    group representative. The final channel-level score is the mean DTW distance
    across groups (or a per-group list).

    - If the inputs are :class:`mne.time_frequency.tfr.AverageTFR`, power values are
      restricted to ``fmin``–``fmax`` and averaged over frequency to obtain a
      single time course per group.
    - If the inputs are :class:`mne.Evoked`, the evoked waveform is used directly
      (in microvolts).

    Parameters
    ----------
    objs : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        List of group-representative objects. All entries must be the same type:
        either :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.Evoked | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) waveform used for DTW comparisons.

        - ``"mean"``: the centroid is computed as the weighted average of ``objs``
          using :func:`mne.combine_evoked` (for :class:`mne.Evoked`) or
          :func:`mne.time_frequency.combine_tfr` (for :class:`mne.time_frequency.tfr.AverageTFR`).
        - If an object is provided, it is used directly as the centroid.

        The centroid must be of the same type as the elements of ``objs``.

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). Only used when inputs are
        :class:`mne.time_frequency.tfr.AverageTFR`. If ``None``, MNE defaults are used.

    tmin, tmax : float | None, optional
        Time window of interest (in seconds). If provided, data are cropped to
        ``[tmin, tmax]`` before computing DTW distances.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to both the centroid and all objects in ``objs``.
        If ``None``, all channels are used.

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``, ``group_id``).

    weights : str | array_like of float, default="nave"
        Weights used when ``centroid="mean"``.
        Passed to :func:`mne.combine_evoked` or :func:`mne.time_frequency.combine_tfr`.
        The default ``"nave"`` uses the number of averaged epochs (when available).

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging DTW
        distances across groups.

        If ``False``, returns a per-channel JSON-encoded list containing one DTW
        distance per group (order follows ``objs``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``between_trial_group_temporal`` : float | str
            If ``average=True``, a scalar variability value (mean DTW distance).
            If ``average=False``, a JSON-encoded list of per-group DTW distances.

    Notes
    -----
    This function measures variability *between group-level averages* rather than
    between single-trial epochs. DTW is used to compare time courses while allowing
    non-linear temporal alignment.

    This function is implemented only for lists of :class:`mne.Evoked` or lists of
    :class:`mne.time_frequency.tfr.AverageTFR`.
    """

    type_objs = _get_obj_type(objs)

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

    type_obj_centroid = _get_obj_type(obj_centroid)
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

    df_dict = {"channel": [], "between_trial_group_temporal": []}
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
        df_dict["between_trial_group_temporal"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df
