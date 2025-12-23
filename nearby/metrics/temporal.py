import json
import numpy as np
import pandas as pd
import mne

from tslearn.metrics import dtw, soft_dtw, gamma_soft_dtw
from tslearn.barycenters import (
    euclidean_barycenter,
    dtw_barycenter_averaging,
    dtw_barycenter_averaging_subgradient,
)

from ..utils.core import (
    _add_values_to_df,
    _get_data_type,
)


def within_trial_temporal(
        data,
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

    - If ``data`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, power values are
      first restricted to the specified frequency band (``fmin``–``fmax``) and then
      averaged over frequency, yielding a single time course per epoch.
    - If ``data`` is an :class:`mne.epochs.Epochs`, the epoched signal is used
      directly (in microvolts).

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE object containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time-frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied.

        For :class:`mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied as::

            data.copy().apply_baseline(baseline, mode="percent")

        For :class:`mne.epochs.Epochs`, baseline correction is applied as::

            data.copy().apply_baseline(baseline)

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
    type_data = _get_data_type(data)

    if not ((type_data == "EpochsTFR") or (type_data == "Epochs")):
        raise ValueError(
            "within_trial_temporal is only implemented for EpochsTFR or Epochs objects."
        )

    data = data.copy()

    if baseline is not None:
        if type_data == "EpochsTFR":
            data = data.apply_baseline(baseline, "percent")
        elif type_data == "Epochs":
            data = data.apply_baseline(baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    df_dict = {"channel": [], "within_trial_temporal": []}
    for ch in data.ch_names:
        if type_data == "EpochsTFR":
            X = data.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # X.shape: n_epochs, n_channels, n_freqs, n_times

            X = np.squeeze(X)
            # X.shape: n_epochs, n_freqs, n_times

            X = X.mean(1)
            # X.shape: n_epochs, n_times
        elif type_data == "Epochs":
            X = data.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # X.shape: n_epochs, n_channels, n_times

            X = np.squeeze(X)
            # X.shape: n_epochs, n_times

        sd = np.std(X, axis=1)

        if average:
            sd = sd.mean(0)
        else:
            sd = sd.tolist()
            sd = json.dumps(sd)

        df_dict["channel"].append(ch)
        df_dict["within_trial_temporal"].append(sd)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_temporal(
        data,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        barycenter="dba",
        params_barycenter=None,
        params_dtw=None,
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

    - If ``data`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, power values are
      first restricted to the specified frequency band (``fmin``–``fmax``) and then
      averaged over frequency, yielding a single time course per epoch.
    - If ``data`` is an :class:`mne.epochs.Epochs`, the epoched signal is used
      directly (in microvolts).

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE dataect containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied.

        For :class:`mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied as::

            data.copy().apply_baseline(baseline, mode="percent")

        For :class:`mne.epochs.Epochs`, baseline correction is applied as::

            data.copy().apply_baseline(baseline)

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). Only used when ``data`` is an
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
        distance per epoch (in the same epoch order as ``data``).

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

    if params_barycenter is None:
        params_barycenter = {}
    else:
        params_barycenter = params_barycenter.copy()

    if params_dtw is None:
        params_dtw = {}
    else:
        params_dtw = params_dtw.copy()

    type_data = _get_data_type(data)

    if not ((type_data == "EpochsTFR") or (type_data == "Epochs")):
        raise ValueError(
            "between_trial_temporal is only implemented for EpochsTFR or Epochs objects."
        )

    data = data.copy()

    if baseline is not None:
        if type_data == "EpochsTFR":
            data = data.apply_baseline(baseline, "percent")
        elif type_data == "Epochs":
            data = data.apply_baseline(baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    df_dict = {"channel": [], "between_trial_temporal": []}
    for ch in data.ch_names:
        if type_data == "EpochsTFR":
            X = data.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # X.shape: n_epochs, n_channels, n_freqs, n_times

            X = np.squeeze(X)
            # X.shape: n_epochs, n_freqs, n_times

            X = X.mean(1)
            # X.shape: n_epochs, n_times

        elif type_data == "Epochs":
            X = data.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # X.shape: n_epochs, n_channels, n_times

            X = np.squeeze(X)
            # X.shape: n_epochs, n_times

        match barycenter:
            case "dba":
                mean = dtw_barycenter_averaging(X, **params_barycenter)
            case "dba_subgradient":
                mean = dtw_barycenter_averaging_subgradient(X, **params_barycenter)
            case "euclidean":
                mean = euclidean_barycenter(X, **params_barycenter)
            case _:
                raise ValueError(f"Invalid barycenter method: {barycenter}")

            # mean.shape: n_times, 1

        n_epochs = X.shape[0]

        scores = []
        for m in range(n_epochs):
            # X[m, None].shape: n_times, 1
            # mean.shape: n_times, 1
            score = dtw(X[m, None].T, mean, **params_dtw)
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
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_group_temporal(
        data,
        centroid="mean",
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        barycenter="dba",
        params_barycenter=None,
        params_dtw=None,
        additional_values=None,
        average=True,
):
    """
    Compute Between-Trial-Group Temporal (BtwTrialGrpTemp) values.

    This metric quantifies how much the group-level time courses differ from
    each other. Each element of ``data`` is assumed to represent a *trial-group
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
    data : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        List of group-representative dataects. All entries must be the same type:
        either :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.Evoked | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) waveform used for DTW comparisons.

        - ``"mean"``: the centroid is computed as the weighted average of ``data``
          using :func:`mne.combine_evoked` (for :class:`mne.Evoked`) or
          :func:`mne.time_frequency.combine_tfr` (for :class:`mne.time_frequency.tfr.AverageTFR`).
        - If an dataect is provided, it is used directly as the centroid.

        The centroid must be of the same type as the elements of ``data``.

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

    if params_barycenter is None:
        params_barycenter = {}
    else:
        params_barycenter = params_barycenter.copy()

    if params_dtw is None:
        params_dtw = {}
    else:
        params_dtw = params_dtw.copy()

    def proc_tfr(tfr, tmin, tmax, fmin, fmax, ch):
        X = tfr.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # X.shape: n_channels, n_freqs, n_times

        X = np.squeeze(X)
        # X.shape: n_freqs, n_times

        X = X.mean(0)
        # X.shape: n_times

        return X

    def proc_evoked(evoked, tmin, tmax, ch):
        X = evoked.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
        # X.shape: n_channels, n_times

        X = np.squeeze(X)
        # X.shape: n_times

        return X

    type_data = _get_data_type(data)

    if not ((type_data == "list_Evoked") or (type_data == "list_AverageTFR")):
        raise ValueError(
            "data must be an instance of list of Evoked or AverageTFR objects."
        )

    if not isinstance(centroid, str):
        data_centroid = centroid

        type_data_centroid = _get_data_type(data_centroid)
        if not (
                (type_data_centroid == "Evoked") or (type_data_centroid == "AverageTFR")
        ):
            raise ValueError(
                "centroid must be an instance of Evoked or AverageTFR objects."
            )

        if type_data_centroid not in type_data:
            # type_data expect "list_Evoked" or "list_AverageTFR"
            # while type_data_centroid expect "Evoked" or "AverageTFR"
            raise ValueError("centroid and data must be of the same type.")

        data_centroid = data_centroid.copy()
        if picks is not None:
            data_centroid = data_centroid.pick(picks=picks)
    else:
        if centroid not in ["mean"]:
            raise ValueError(f"Invalid centroid: {centroid}")
        data_centroid = None

    data = [data[n].copy() for n in range(len(data))]

    if picks is not None:
        for n in range(len(data)):
            data[n] = data[n].pick(picks=picks)

    df_dict = {"channel": [], "between_trial_group_temporal": []}
    for ch in data[0].ch_names:
        if type_data == "list_AverageTFR":
            X_list = [
                proc_tfr(data[n], tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, ch=ch)
                for n in range(len(data))
            ]
            if data_centroid is not None:
                X_centroid = proc_tfr(
                    data_centroid, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, ch=ch
                )

        elif type_data == "list_Evoked":

            X_list = [
                proc_evoked(data[n], tmin=tmin, tmax=tmax, ch=ch)
                for n in range(len(data))
            ]

            if data_centroid is not None:
                X_centroid = proc_evoked(data_centroid, tmin=tmin, tmax=tmax, ch=ch)

        if data_centroid is None:
            match barycenter:
                case "dba":
                    X_centroid = dtw_barycenter_averaging(
                        np.array(X_list), **params_barycenter
                    )
                case "dba_subgradient":
                    X_centroid = dtw_barycenter_averaging_subgradient(
                        np.array(X_list), **params_barycenter
                    )
                case "euclidean":
                    X_centroid = euclidean_barycenter(
                        np.array(X_list), **params_barycenter
                    )
                case _:
                    raise ValueError(f"Invalid barycenter method: {barycenter}")

        scores = []
        for X in X_list:
            # np.atleast_2d(X).T: n_times, 1
            # X_centroid.shape: n_times, 1

            X = np.atleast_2d(X).T
            score = dtw(X, X_centroid, **params_dtw)
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
