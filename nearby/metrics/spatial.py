import json
import numpy as np
import scipy
import pandas as pd
import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2

from ..utils.core import (
    _add_values_to_df,
    _auto_windows,
    _get_data_type,
)


def within_trial_spatial(
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
    metric="angle",
    picks=None,
    average=True,
    additional_values=None,
):
    """
    Compute within-trial spatial variability from time-resolved channel patterns.

    This metric quantifies how much the spatial pattern changes
    within each epoch. For every epoch, the data are converted into a sequence of
    spatial vectors (one per time window or one per time sample). A reference spatial
    vector is estimated for that epoch, and the dispersion of spatial vectors around
    the reference is computed. The final output is the mean dispersion across epochs,
    or a per-epoch list.

    The input can be time-domain epochs (:class:`mne.epochs.Epochs`) or time–frequency
    epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input, values are
    optionally restricted to ``fmin``–``fmax`` and averaged over frequency before
    forming spatial vectors.

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
        in frequency to ``[fmin, fmax]`` and then averaged over frequency.

    mode : {"windows", "all_samples"}, default="windows"
        How to represent within-epoch temporal evolution of spatial patterns.

        - ``"windows"``: average data within time windows; each window yields one
          spatial vector of shape ``(n_channels,)`` per epoch.
        - ``"all_samples"``: use each time sample as a spatial vector (no window
          averaging). For TFR input, frequency is still averaged first.

    windows : array_like of shape (n_windows, 2) | None, optional
        Explicit time windows in seconds: ``[[tmin_0, tmax_0], ..., [tmin_k, tmax_k]]``.
        Only used when ``mode="windows"``. If ``None``, windows are generated
        automatically using ``auto_window_size`` and ``auto_window_step`` between
        ``tmin`` and ``tmax``. At least two windows are required.

    auto_window_size : float, default=0.1
        Window length in seconds used when ``windows=None`` and ``mode="windows"``.

    auto_window_step : float, default=0.1
        Step in seconds used when ``windows=None`` and ``mode="windows"``.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` when extracting window averages (``mode="windows"``) and also
        when extracting sample-wise patterns (``mode="all_samples"``).

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition used to compute within-epoch spatial dispersion.

        - ``"angle"``: angular distance from the Fréchet mean (p=2) of the spatial
          vectors in an epoch. See :func:`nearby.mean.mean_frechet_p2`.
        - ``"cosine"``: cosine distance from the extrinsic mean of the spatial vectors. See :func:`nearby.mean.mean_extrinsic`:.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format.

        - If ``True``, return a single scalar (mean dispersion across epochs).
        - If ``False``, return a JSON-encoded list of per-epoch dispersion values
          (epoch order matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with a single row.

        ``within_trial_spatial`` : float | str
            If ``average=True``, the mean dispersion across epochs.
            If ``average=False``, a JSON-encoded list of dispersions with length
            ``n_epochs``.

    Notes
    -----
    - When ``mode="windows"`` and ``windows=None``, automatic windows are generated
      using ``tmin``, ``tmax``, ``auto_window_size``, and ``auto_window_step``. Two or
      more windows are required.
    - When ``mode="all_samples"``, the number of spatial vectors per epoch equals the
      number of time samples in the cropped interval ``[tmin, tmax]`` (or the full epoch
      if both are ``None``).
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

    X_list = []
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

        for window in windows:
            if type_data == "EpochsTFR":
                X = data.get_data(tmin=window[0], tmax=window[1], fmin=fmin, fmax=fmax)
                # X.shape: n_epochs, n_channels, n_freqs, n_times

                X = X.mean(2)
                # X.shape: n_epochs, n_channels, n_times

                X = X.mean(2)
                # X.shape: n_epochs, n_channels
            elif type_data == "Epochs":
                X = data.get_data(tmin=window[0], tmax=window[1], units="uV")
                # X.shape: n_epochs, n_channels, n_times

                X = X.mean(2)
                # X.shape: n_epochs, n_channels

            X_list.append(X)

        X = np.array(X_list)
        # X.shape: n_windows, n_epochs, n_channels

        X = X.transpose(1, 0, 2)
        # X.shape: n_epochs, n_windows, n_channels

    elif mode == "all_samples":
        if type_data == "EpochsTFR":
            X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
            # X.shape: n_epochs, n_channels, n_freqs, n_times

            X = X.mean(2)
            # X.shape: n_epochs, n_channels, n_times
        elif type_data == "Epochs":
            X = data.get_data(tmin=tmin, tmax=tmax, units="uV")
            # X.shape: n_epochs, n_channels, n_times

        # In all_samples mode, n_times is equal to the number of windows
        # X.shape: n_epochs, n_channels, n_windows = n_epochs, n_channels, n_times

        X = X.transpose(0, 2, 1)
        # X.shape: n_epochs, n_windows, n_channels

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    n_epochs, n_windows, _ = X.shape

    dispersions_list = []

    for idx_trial in range(n_epochs):
        X_trial = X[idx_trial, :, :].squeeze()
        # X_trial.shape: n_windows, n_channels

        distances_trial = []
        if metric == "angle":
            mean = mean_frechet_p2(X_trial, verbosity=0)

            for n in range(n_windows):
                d = distance_angle(mean, X_trial[n, :])
                distances_trial.append(d)

        elif metric == "cosine":
            mean = mean_extrinsic(X_trial)

            for n in range(n_windows):
                d = scipy.spatial.distance.cosine(mean, X_trial[n, :])
                distances_trial.append(d)

        dispersion = np.mean(distances_trial)

        dispersions_list.append(dispersion)

    if average:
        dispersions = np.mean(dispersions_list)
    else:
        dispersions = json.dumps(dispersions_list)

    df = pd.DataFrame()
    df["within_trial_spatial"] = [dispersions]

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_spatial(
    data,
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
    Compute between-trial spatial variability from epoch-wise channel patterns.

    This metric quantifies how much the spatial pattern varies
    across epochs. Each epoch is reduced to a single spatial vector (one value per
    channel) by averaging over time (and over frequency for TFR input). A reference
    spatial pattern is then estimated across epochs, and distances from this mean
    to each epoch’s spatial vector are computed. The final output is the mean
    distance across epochs, or a per-epoch list.

    Two distance/mean definitions are supported via ``metric``:

    - ``"angle"``: angular distance to a Fréchet mean (p=2) on the unit sphere.
    - ``"cosine"``: cosine distance to an extrinsic mean.

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
        in frequency to ``[fmin, fmax]`` prior to averaging.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition used to measure dispersion across epochs.

        - ``"angle"``: estimate the across-epoch mean pattern using a Fréchet mean
          (p=2), then compute angular distances to each epoch pattern. See :func:`nearby.mean.mean_frechet_p2`
        - ``"cosine"``: estimate the mean pattern using an extrinsic mean, then compute
          cosine distances to each epoch pattern. See :func:`nearby.mean.mean_extrinsic`.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format.

        - If ``True``, return a single scalar (mean distance across epochs).
        - If ``False``, return a JSON-encoded list of per-epoch distances (epoch order
          matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with a single row.

        ``between_trial_spatial`` : float | str
            If ``average=True``, the mean distance across epochs.
            If ``average=False``, a JSON-encoded list of distances with length
            ``n_epochs``.
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

    if type_data == "EpochsTFR":
        X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # X.shape: n_epochs, n_channels, n_freqs, n_times

        X = X.mean(2)
        # X.shape: n_epochs, n_channels, n_times

        X = X.mean(2)
        # X.shape: n_epochs, n_channels
    elif type_data == "Epochs":
        X = data.get_data(tmin=tmin, tmax=tmax, units="uV")
        # X.shape: n_epochs, n_channels, n_times

        X = X.mean(2)
        # X.shape: n_epochs, n_channels

    n_epochs, _ = X.shape

    distances = []

    if metric == "cosine":
        mean = mean_extrinsic(X)

        for n in range(n_epochs):
            d = scipy.spatial.distance.cosine(mean, X[n, :])
            distances.append(d)

    elif metric == "angle":
        mean = mean_frechet_p2(X, verbosity=0)

        for n in range(n_epochs):
            d = distance_angle(mean, X[n, :])
            distances.append(d)

    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    if average:
        distances = np.array(distances)
        distances = distances.mean()
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["between_trial_spatial"] = [distances]

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_group_spatial(
    data,
    centroid="mean",
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
    Compute between-trial-group spatial variability from group-representative patterns.

    This metric quantifies how much group-level spatial patterns differ from one
    another. Each element of ``data`` is expected to be a trial-group representative
    object (e.g., an averaged response), and each group object is reduced to a single
    spatial vector (one value per channel) by averaging over time (and over frequency
    for TFR input). Distances between each group spatial vector and a centroid spatial
    vector are then computed. The final output is the mean distance across groups, or
    a per-group list.

    The centroid can be computed from the group vectors (``centroid="mean"``) or
    provided explicitly as an MNE object. The centroid computation depends on
    ``metric``:

    - ``metric="angle"`` uses a Fréchet mean (p=2) on the unit sphere.
    - ``metric="cosine"`` uses an extrinsic (Euclidean) mean.

    Parameters
    ----------
    data : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        List of group-representative objects. All entries must have the same type:
        either :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.Evoked | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) pattern used for comparisons.

        - ``"mean"``: compute the centroid spatial vector from the group spatial
          vectors. The mean definition depends on ``metric`` (Fréchet mean for
          ``"angle"``, extrinsic mean for ``"cosine"``).
        - If an MNE object is provided, it is reduced to a centroid spatial vector
          using the same averaging operations as for ``data``.

        The centroid object must be of the same type as the elements of ``data``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` contains
        :class:`~mne.time_frequency.tfr.AverageTFR`. If provided, TFR values are cropped
        in frequency to ``[fmin, fmax]`` prior to averaging.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, objects are cropped to
        ``[tmin, tmax]`` prior to averaging over time.

    metric : {"angle", "cosine"}, default="angle"
        Distance definition used to measure dispersion across groups.

        - ``"angle"``: angular distance between spatial patterns.
        - ``"cosine"``: cosine distance between spatial patterns.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to each element in ``data`` and to the centroid
        object (if provided). If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format.

        - If ``True``, return a single scalar (mean distance across groups).
        - If ``False``, return a JSON-encoded list of per-group distances (group order
          matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with a single row.

        ``between_trial_group_spatial`` : float | str
            If ``average=True``, the mean distance across groups.
            If ``average=False``, a JSON-encoded list of distances with length
            ``len(data)``.

    Notes
    -----
    - For :class:`~mne.time_frequency.tfr.AverageTFR`, each group spatial vector is
      computed by averaging over frequency and then over time within the selected
      window.
    - For :class:`~mne.Evoked`, each group spatial vector is computed by averaging
      over time within the selected window (returned in microvolts).
    - This function measures variability between *group-level representatives*
      rather than between single-trial epochs.
    """

    # ----- Define processing functions -----
    def proc_tfr(tfr, tmin, tmax, fmin, fmax):
        X = tfr.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # X.shape: n_channels, n_freqs, n_times

        X = X.mean(1)
        # X.shape: n_channels, n_times

        X = X.mean(1)
        # X.shape: n_channels

        return X

    def proc_evoked(evoked, tmin, tmax):
        X = evoked.get_data(tmin=tmin, tmax=tmax, units="uV")
        # X.shape: n_channels, n_times

        X = X.mean(1)
        # X.shape: n_channels

        return X

    # ----- END: Define processing functions -----

    type_data = _get_data_type(data)

    if not ((type_data == "list_Evoked") or (type_data == "list_AverageTFR")):
        raise ValueError(
            "data must be an instance of list of Evoked or AverageTFR objects."
        )

    data = [data[n].copy() for n in range(len(data))]
    if picks is not None:
        for n in range(len(data)):
            data[n] = data[n].pick(picks=picks)

    if type_data == "list_AverageTFR":
        X_list = [
            proc_tfr(data[n], tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
            for n in range(len(data))
        ]

    elif type_data == "list_Evoked":
        X_list = [proc_evoked(data[n], tmin=tmin, tmax=tmax) for n in range(len(data))]

    if isinstance(centroid, str):
        data_centroid = None
        match centroid:
            case "mean":
                if metric == "angle":
                    X_centroid = mean_frechet_p2(X_list, verbosity=0)
                elif metric == "cosine":
                    X_centroid = mean_extrinsic(X_list)
                else:
                    raise ValueError("metrics must be 'angle' or 'cosine'")
            case _:
                raise NotImplementedError(f"centroid={centroid} is not implemented.")
    else:
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

        if type_data == "list_AverageTFR":
            X_centroid = proc_tfr(
                data_centroid, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax
            )
        elif type_data == "list_Evoked":
            X_centroid = proc_evoked(data_centroid, tmin=tmin, tmax=tmax)

    distances = []

    if metric == "cosine":
        for X in X_list:
            d = scipy.spatial.distance.cosine(X_centroid, X)
            distances.append(d)

    elif metric == "angle":
        for X in X_list:
            d = distance_angle(X_centroid, X)
            distances.append(d)

    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    if average:
        distances = np.array(distances)
        distances = distances.mean()
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["between_trial_group_spatial"] = [distances]

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df
