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

from ..distances import distance_squared_euclidean

from ..utils.core import (
    _add_values_to_df,
    _auto_windows,
    _get_data_type,
)


def within_trial_temporal(
        data,
        baseline=None,
        fmin=None,
        fmax=None,
        mode="windows",
        windows=None,
        auto_window_size=0.1,
        auto_window_step=0.1,
        tmin=None,
        tmax=None,
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Within-Trial Temporal (WiTrialTemp) Variability values.

    This metric quantifies how much the *temporal pattern over time* varies within each
    epoch. For each channel and epoch, the signal is converted into a 1D sequence
    across time (one value per time window or one per time sample). A reference value
    is estimated for that epoch (the mean across windows/samples), and the dispersion
    of values around the reference is computed. The final output is the mean dispersion
    across epochs, or a per-epoch list.

    The input can be time-domain epochs (:class:`mne.epochs.Epochs`) or time–frequency
    epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input, values are
    optionally restricted to ``fmin``–``fmax`` and averaged over frequency before
    forming temporal sequences. Variability is computed independently for each channel.

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object. Only these two types are supported.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(-1, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing variability.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline is applied with
        ``mode="percent"``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` is an
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If provided, TFR values are cropped
        in frequency to ``[fmin, fmax]`` and then averaged over frequency to yield a
        single time course per epoch and channel.

    mode : {"windows", "all_samples"}, default="windows"
        How to represent within-epoch temporal evolution.

        - ``"windows"``: average data within time windows; each window yields one
          scalar per epoch and channel.
        - ``"all_samples"``: use each time sample as one scalar (no window averaging).
          For TFR input, frequency is still averaged first.

    windows : array_like of shape (n_windows, 2) | None, optional
        Explicit time windows in seconds: ``[[tmin_0, tmax_0], ..., [tmin_k, tmax_k]]``.
        Only used when ``mode="windows"``. If ``None``, windows are generated
        automatically using ``auto_window_size`` and ``auto_window_step`` between
        ``tmin`` and ``tmax``.

    auto_window_size : float, default=0.1
        Window length in seconds used when ``windows=None`` and ``mode="windows"``.

    auto_window_step : float, default=0.1
        Step in seconds used when ``windows=None`` and ``mode="windows"``.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` when extracting window averages (``mode="windows"``) and also
        when extracting sample-wise values (``mode="all_samples"``).

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a single scalar (mean dispersion across epochs).
        - If ``False``, return a JSON-encoded list of per-epoch dispersion values
          (epoch order matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``within_trial_temporal`` : float | str
            If ``average=True``, the mean dispersion across epochs.
            If ``average=False``, a JSON-encoded list of dispersions with length
            ``n_epochs``.

    Notes
    -----
    - For each channel and epoch, a temporal sequence is constructed either by
      averaging within time windows (``mode="windows"``) or by using individual
      time samples (``mode="all_samples"``).
    - The function operates on a copy of the input object (``data.copy()``) and does
      not modify ``data`` in place.
    - This metric captures *within-trial temporal variability*, complementary to
      spatial and frequency variability metrics.

    """
    type_data = _get_data_type(data)

    if not ((type_data == "EpochsTFR") or (type_data == "Epochs")):
        raise ValueError("data must be an instance of EpochsTFR or Epochs objects.")

    data = data.copy()

    if baseline is not None:
        if type_data == "EpochsTFR":
            data = data.apply_baseline(baseline, "percent")
        elif type_data == "Epochs":
            data = data.apply_baseline(baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    X_dict = {}
    if mode == "windows":
        if windows is None:
            windows = _auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=tmin,
                tmax=tmax,
            )

            if (windows.ndim != 2) or (windows.shape[0] == 1):
                raise ValueError(f"Two or more windows must be specified. {windows=}")

        for ch in data.ch_names:
            X_dict[ch] = []
            for window in windows:
                if type_data == "EpochsTFR":
                    X = data.get_data(
                        tmin=window[0], tmax=window[1], fmin=fmin, fmax=fmax, picks=ch
                    )
                    # X.shape: n_epochs, n_channels, n_freqs, n_times

                    X = X.squeeze()
                    # X.shape: n_epochs, n_freqs, n_times

                    X = X.mean(1)
                    # X.shape: n_epochs, n_times

                    X = X.mean(1)
                    # X.shape: n_epochs

                    X_dict[ch].append(X)

                elif type_data == "Epochs":
                    X = data.get_data(
                        tmin=window[0], tmax=window[1], units="uV", picks=ch
                    )
                    # X.shape: n_epochs, n_channels, n_times

                    X = X.squeeze()
                    # X.shape: n_epochs, n_times

                    X = X.mean(1)
                    # X.shape: n_epochs

                    X_dict[ch].append(X)

            X_dict[ch] = np.array(X_dict[ch])
            # X.shape: n_windows, n_epochs

            X_dict[ch] = X_dict[ch].transpose(1, 0)
            # X.shape: n_epochs, n_windows

    elif mode == "all_samples":
        for ch in data.ch_names:
            X_dict[ch] = []
            if type_data == "EpochsTFR":
                X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                X = X.squeeze()
                # X.shape: n_epochs, n_freqs, n_times

                X = X.mean(1)
                # X.shape: n_epochs, n_times

            elif type_data == "Epochs":

                X = data.get_data(tmin=tmin, tmax=tmax, units="uV", picks=ch)
                # X.shape: n_epochs, n_channels, n_times

                X = X.squeeze()
                # X.shape: n_epochs, n_times

            # In all_samples mode, n_times is equal to the number of windows
            # X.shape: n_epochs, n_windows
            X_dict[ch] = X

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    df_dict = {"channel": [], "within_trial_temporal": []}
    for ch in data.ch_names:
        n_epochs, n_windows = X_dict[ch].shape

        dispersion = []
        for idx_trial in range(n_epochs):
            X_trial = X_dict[ch][idx_trial, :].squeeze()
            # X_trial.shape: n_windows

            mean = X_trial.mean(0)

            distances_trial = []
            for n in range(n_windows):
                d = distance_squared_euclidean(X_trial[n], mean)
                distances_trial.append(d)

            dispersion_within_trial = np.mean(distances_trial)

            dispersion.append(dispersion_within_trial)

        if average:
            dispersion = np.mean(dispersion)
        else:
            dispersion = json.dumps(dispersion)

        df_dict["channel"].append(ch)
        df_dict["within_trial_temporal"].append(dispersion)

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
    Compute between-trial temporal variability using DTW distances to a mean.

    For each channel, this function forms a set of 1D time courses (one per epoch),
    computes a representative time course (the mean or barycenter) across epochs, then
    measures how much each epoch deviates from the mean using Dynamic Time
    Warping (DTW). The channel-level score is the mean DTW distance across epochs.

    DTW is computed using the `tslearn.metrics.dtw <https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw>`_ module.

    The input can be either time-domain epochs (:class:`mne.epochs.Epochs`) or
    time–frequency power epochs (:class:`mne.time_frequency.tfr.EpochsTFR`):

    - For :class:`~mne.time_frequency.tfr.EpochsTFR`, power is optionally cropped in
      frequency (``fmin``–``fmax``), then averaged over frequency to obtain a single
      time course per epoch.
    - For :class:`~mne.epochs.Epochs`, the epoched signal is used directly (optionally
      cropped in time; returned in microvolts).

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object. Only these two types are supported.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(-1, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing variability.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline is applied with
        ``mode="percent"``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` is an
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If either is ``None``, no
        frequency cropping is applied.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` prior to mean/DTW computations.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` or
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    barycenter : {"dba", "dba_subgradient", "euclidean"}, default="dba"
        Method used to compute the representative time course across epochs.

        - ``"dba"``: DTW Barycenter Averaging (DBA). See `tslearn.barycenters.dtw_barycenter_averaging <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html#tslearn.barycenters.dtw_barycenter_averaging>`_
        - ``"dba_subgradient"``: DBA using a subgradient-based optimization. See `tslearn.barycenters.dtw_barycenter_averaging_subgradient <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging_subgradient.html#tslearn.barycenters.dtw_barycenter_averaging_subgradient>`_
        - ``"euclidean"``: Euclidean mean over epochs (no DTW alignment). See `tslearn.barycenters.euclidean_barycenter <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.euclidean_barycenter.html#tslearn.barycenters.euclidean_barycenter>`_

    params_barycenter : dict | None, optional
        Parameters passed to the selected barycenter routine. If ``None``, an empty
        dict is used.

    params_dtw : dict | None, optional
        Parameters passed to the DTW distance function used to compare each epoch to
        the barycenter. If ``None``, an empty dict is used.

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a scalar per channel (mean DTW distance across epochs).
        - If ``False``, return a JSON-encoded list of per-epoch DTW distances (epoch
          order matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``between_trial_temporal`` : float | str
            If ``average=True``, the mean DTW distance across epochs.
            If ``average=False``, a JSON-encoded list of DTW distances with length
            ``n_epochs``.
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
            X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
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
    Compute between-trial-group temporal variability using DTW distances to a centroid.

    This metric quantifies how much *group-representative* time courses differ from
    each other. Each element of ``data`` is expected to be a representative waveform
    for a trial-group (e.g., an averaged response), and the variability is computed
    by comparing each group waveform to a centroid using Dynamic Time Warping (DTW).

    DTW is computed using the `tslearn.metrics.dtw <https://tslearn.readthedocs.io/en/stable/gen_modules/metrics/tslearn.metrics.dtw.html#tslearn.metrics.dtw>`_ module.

    For each channel:

    1. Convert each group object to a 1D time course (optionally cropped in time, and
       for TFR optionally averaged within ``fmin``–``fmax``).
    2. Define a centroid time course:
       - If ``centroid`` is an MNE object, use it directly.
       - If ``centroid="mean"``, compute a barycenter from the group time courses using ``barycenter``.
    3. Compute DTW distance between each group time course and the centroid.
    4. Aggregate distances across groups (mean or per-group list).

    Parameters
    ----------
    data : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        List of group-representative objects. All entries must have the same type:
        either :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.Evoked | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid waveform used as DTW reference.

        - ``"mean"``: compute a centroid time course from ``data`` using the method
          specified by ``barycenter``.
        - If an MNE object is provided, it is used directly as the centroid (after
          optional picking and cropping).

        The centroid object must be of the same type as the elements of ``data``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` contains
        :class:`~mne.time_frequency.tfr.AverageTFR`. If either is ``None``, no
        frequency cropping is applied for that bound.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, all objects are cropped to
        ``[tmin, tmax]`` prior to centroid/DTW computations.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to the centroid (if provided as an object) and
        to every element in ``data``. If ``None``, all channels are used.

    barycenter : {"dba", "dba_subgradient", "euclidean"}, default="dba"
        Method used to compute the centroid time course when ``centroid="mean"``.

        - ``"dba"``: DTW Barycenter Averaging (DBA). See `tslearn.barycenters.dtw_barycenter_averaging <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging.html#tslearn.barycenters.dtw_barycenter_averaging>`_
        - ``"dba_subgradient"``: DBA using a subgradient-based optimization. See `tslearn.barycenters.dtw_barycenter_averaging_subgradient <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.dtw_barycenter_averaging_subgradient.html#tslearn.barycenters.dtw_barycenter_averaging_subgradient>`_
        - ``"euclidean"``: Euclidean mean over epochs (no DTW alignment). See `tslearn.barycenters.euclidean_barycenter <https://tslearn.readthedocs.io/en/stable/gen_modules/barycenters/tslearn.barycenters.euclidean_barycenter.html#tslearn.barycenters.euclidean_barycenter>`_

    params_barycenter : dict | None, optional
        Parameters passed to the selected barycenter routine. If ``None``, an empty
        dict is used.

    params_dtw : dict | None, optional
        Parameters passed to the DTW distance function used to compare each group
        time course to the centroid. If ``None``, an empty dict is used.

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a scalar per channel (mean DTW distance across groups).
        - If ``False``, return a JSON-encoded list of per-group DTW distances (group
          order matches ``data``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``between_trial_group_temporal`` : float | str
            If ``average=True``, the mean DTW distance across groups.
            If ``average=False``, a JSON-encoded list of DTW distances with length
            ``len(data)``.

    Notes
    -----
    - This function measures variability between *group-level representatives*
      (e.g., averages) rather than between single-trial epochs.
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
