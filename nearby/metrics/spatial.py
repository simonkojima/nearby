from functools import partial
import numpy as np
import mne

from ..distances.distances import distance_angle
from ..mean import frechet_mean_angular

from ..utils.core import (
    _auto_windows,
    _get_data_type,
)

from ..utils.proc import _get_data, _apply_baseline


def within_trial_spatial(
        data,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        callback_distance=None,
        callback_centroid=None,
        mode="windows",
        window_size=0.1,
        window_step=0.1,
        picks=None,
        baseline=None,
        average=True,
        return_centroid=False,
):
    """
    Compute within-trial spatial variability.

    This metric quantifies how much the *spatial pattern across channels* varies
    within each epoch over time. For each epoch, the signal is converted into a
    sequence of spatial vectors (one vector per time window in ``mode="windows"``,
    or one vector per time sample in ``mode="all_samples"``). A within-epoch centroid
    spatial pattern is estimated using ``callback_centroid`` and the dispersion is
    computed as the mean distance (``callback_distance``) between each spatial vector
    and the centroid. The final output aggregates dispersions across epochs.

    The input can be either time-domain epochs (:class:`mne.epochs.Epochs`) or
    time–frequency epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input,
    power may be restricted to ``fmin``–``fmax`` (Hz) and reduced to a single time
    course per epoch/channel (e.g., by averaging across frequency) before computing
    spatial vectors.

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(-1.0, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). Only used when ``data`` is
        :class:`~mne.time_frequency.tfr.EpochsTFR`.

    mode : {"windows", "all_samples"}, default="windows"
        Temporal sampling strategy for spatial patterns.

        - ``"windows"``: average within windows; each window yields one spatial vector
          of shape ``(n_channels,)`` per epoch.
        - ``"all_samples"``: use each time sample as a spatial vector (no window
          averaging). For TFR input, frequency is still averaged first.

    window_size : float, default=0.1
        Window length (seconds) used when ``mode="windows"``.

    window_step : float, default=0.1
        Step size (seconds) used when ``mode="windows"``.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to select the region of interest. Applied during
        data extraction (and thus affects the number of windows/samples).

    callback_distance : callable | None, optional
        Distance function comparing two spatial vectors of shape ``(n_channels,)``.
        If ``None``, uses :func:`nearby.distances.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid spatial vector from an epoch-specific matrix
        ``X_trial`` of shape ``(n_windows, n_channels)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, return a scalar averaged across epochs. If ``False``, return the
        per-epoch dispersion values as an array of shape ``(n_epochs,)``.

    return_centroid : bool, default=False
        If ``True``, also return the per-epoch centroid spatial vectors.

    Returns
    -------
    dispersion : float | ndarray of shape (n_epochs,)
        Within-trial spatial dispersion. If ``average=True``, a scalar (mean across
        epochs). Otherwise, an array with one value per epoch.

    centroid : ndarray of shape (n_epochs, n_channels), optional
        Returned only when ``return_centroid=True``. The within-epoch centroid spatial
        vectors.

    Notes
    -----
    - The function operates on a copy of the input object (``data.copy()``) and does
      not modify ``data`` in place.
    - For ``mode="windows"``, at least two windows must be generated; otherwise a
      :class:`ValueError` is raised.
    - The exact meaning of distance and centroid depends on
      ``callback_distance`` / ``callback_centroid``.
    """
    type_data = _get_data_type(data)

    if type_data not in ["EpochsTFR", "Epochs"]:
        raise ValueError("data must be an instance of EpochsTFR or Epochs objects.")

    if callback_distance is None:
        callback_distance = distance_angle

    if callback_centroid is None:
        callback_centroid = partial(frechet_mean_angular, verbosity=0)

    data = data.copy()

    if picks is not None:
        data = data.pick(picks=picks)

    if baseline is not None:
        data = _apply_baseline(data, baseline)

    if mode == "windows":
        windows = _auto_windows(
            size=window_size,
            step=window_step,
            tmin=tmin,
            tmax=tmax,
        )

        if (windows.ndim != 2) or (windows.shape[0] == 1):
            raise ValueError(f"Two or more windows must be specified. {windows=}")

        X = _get_data(
            data=data,
            domain="spatial",
            windows=windows,
            tmin=None,
            tmax=None,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )
        # X.shape: n_windows, n_epochs, n_channels

        X = X.transpose(1, 0, 2)
        # X.shape: n_epochs, n_windows, n_channels

    elif mode == "all_samples":
        if type_data == "EpochsTFR":
            X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
            # X.shape: n_epochs, n_channels, n_freqs, n_times

            X = X.mean(2)
            # X.shape: n_epochs, n_channels, n_times
        elif type_data == "Epochs":
            X = data.get_data(tmin=tmin, tmax=tmax, units="uV", picks=picks)
            # X.shape: n_epochs, n_channels, n_times

        # In all_samples mode, n_times is equal to the number of windows
        # X.shape: n_epochs, n_channels, n_windows = n_epochs, n_channels, n_times

        X = X.transpose(0, 2, 1)
        # X.shape: n_epochs, n_windows, n_channels
    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    n_epochs, n_windows, _ = X.shape

    dispersions_list, centroid_list = [], []

    for idx_trial in range(n_epochs):
        X_trial = X[idx_trial, :, :].squeeze()
        # X_trial.shape: n_windows, n_channels

        distances_trial = []

        # shape: n_windows, n_channels
        centroid = callback_centroid(X_trial)
        centroid_list.append(centroid)

        for n in range(n_windows):
            # mean.shape: n_channels,
            # X_trial[n, :].shape: n_channels,
            d = callback_distance(X_trial[n, :], centroid)
            distances_trial.append(d)

        dispersion = np.mean(distances_trial)

        dispersions_list.append(dispersion)

    if average:
        dispersion = np.mean(dispersions_list)
    else:
        dispersion = np.array(dispersions_list)

    if return_centroid:
        return dispersion, np.array(centroid_list)

    return dispersion


def between_trial_spatial(
        data,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        callback_distance=None,
        callback_centroid=None,
        picks=None,
        baseline=None,
        average=True,
        return_centroid=False,
):
    """
    Compute between-trial spatial variability.

    This metric quantifies how much the spatial pattern across channels varies across
    epochs. Each epoch is reduced to a single spatial vector by averaging over time
    (and additionally over frequency for TFR input). A centroid spatial pattern is
    then estimated across epochs using ``callback_centroid``, and the dispersion is
    computed as the mean distance (``callback_distance``) between each epoch spatial
    vector and the centroid. The final output aggregates distances across epochs.

    The input can be either time-domain epochs (:class:`mne.epochs.Epochs`) or
    time–frequency epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input,
    power may be restricted to ``fmin``–``fmax`` (Hz) before the time-averaging step.

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(-1.0, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). Only used when ``data`` is
        :class:`~mne.time_frequency.tfr.EpochsTFR`.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop the data to ``[tmin, tmax]`` prior to
        averaging over time.

    callback_distance : callable | None, optional
        Distance function comparing two spatial vectors of shape ``(n_channels,)``.
        If ``None``, uses :func:`nearby.distances.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid spatial vector from the stacked epoch vectors
        ``X`` of shape ``(n_epochs, n_channels)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, return a scalar mean distance across epochs. If ``False``, return
        per-epoch distances as an array of shape ``(n_epochs,)``.

    return_centroid : bool, default=False
        If ``True``, also return the centroid spatial vector.

    Returns
    -------
    dispersion : float | ndarray of shape (n_epochs,)
        Between-trial spatial dispersion. If ``average=True``, a scalar (mean across
        epochs). Otherwise, an array with one value per epoch.

    centroid : ndarray of shape (n_channels,), optional
        Returned only when ``return_centroid=True``. The centroid spatial vector.

    Notes
    -----
    - The function operates on a copy of the input object (``data.copy()``) and does
      not modify ``data`` in place.
    - The exact meaning of distance and centroid depends on
      ``callback_distance`` / ``callback_centroid``.
    """
    type_data = _get_data_type(data)

    if type_data not in ["EpochsTFR", "Epochs"]:
        raise ValueError("data must be an instance of EpochsTFR or Epochs objects.")

    if callback_distance is None:
        callback_distance = distance_angle

    if callback_centroid is None:
        callback_centroid = partial(frechet_mean_angular, verbosity=0)

    data = data.copy()

    if picks is not None:
        data = data.pick(picks=picks)

    if baseline is not None:
        data = _apply_baseline(data, baseline)

    X = _get_data(
        data=data,
        domain="spatial",
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
    )

    n_epochs, _ = X.shape

    distances = []

    centroid = callback_centroid(X)
    # X.shape: n_epochs, n_channels
    # centroid.shape: n_channels,

    for n in range(n_epochs):
        # centroid.shape: n_channels,
        # X[n, :].shape: n_channels,
        d = callback_distance(X[n, :], centroid)
        distances.append(d)

    if average:
        dispersion = np.mean(distances)
    else:
        dispersion = np.array(distances)

    if return_centroid:
        return dispersion, centroid

    return dispersion


def between_trial_group_spatial(
        data,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        callback_distance=None,
        callback_centroid=None,
        centroid=None,
        picks=None,
        baseline=None,
        average=True,
        return_centroid=False,
):
    """
    Compute between-trial-group spatial variability.

    This metric quantifies how much group-representative spatial patterns differ from
    one another. Each element of ``data`` is expected to be a group-level
    representative MNE object (either :class:`mne.Evoked` or
    :class:`mne.time_frequency.tfr.AverageTFR`). Each object is reduced to a single
    spatial vector by averaging over time (and additionally over frequency for TFR
    input). A centroid spatial vector is then defined (either computed from the group
    vectors or extracted from an explicit centroid object), and the dispersion is the
    mean distance (``callback_distance``) between each group vector and the centroid.

    Parameters
    ----------
    data : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        Group-representative MNE objects. All entries must have the same type.

    centroid : mne.Evoked | mne.time_frequency.tfr.AverageTFR | None, optional
        Reference centroid object. If ``None``, the centroid spatial vector is
        computed from the group spatial vectors via ``callback_centroid``.
        If provided, it must be of the same type as the entries in ``data`` and will
        be reduced to a spatial vector using the same extraction procedure as for
        group objects.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). Only used when objects are
        :class:`~mne.time_frequency.tfr.AverageTFR`.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop objects prior to averaging over time.

    callback_distance : callable | None, optional
        Distance function comparing two spatial vectors of shape ``(n_channels,)``.
        If ``None``, uses :func:`nearby.distances.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid spatial vector from a stack of group spatial
        vectors with shape ``(n_groups, n_channels)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to each element in ``data`` and to the centroid
        object (if provided). If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, return a scalar mean distance across groups. If ``False``, return
        per-group distances as an array of shape ``(n_groups,)``.

    return_centroid : bool, default=False
        If ``True``, also return the centroid spatial vector.

    Returns
    -------
    dispersion : float | ndarray of shape (n_groups,)
        Between-trial-group spatial dispersion. If ``average=True``, a scalar (mean
        across groups). Otherwise, an array with one value per group.

    centroid_vector : ndarray of shape (n_channels,), optional
        Returned only when ``return_centroid=True``. The centroid spatial vector used
        as the reference.

    Notes
    -----
    - The function operates on copies of the input objects (``.copy()``) and does not
      modify the originals in place.
    - The exact meaning of distance and centroid depends on
      ``callback_distance`` / ``callback_centroid``.
    """
    type_data = _get_data_type(data)

    if type_data not in ["list_Evoked", "list_AverageTFR"]:
        raise ValueError(
            "data must be an instance of list of Evoked or AverageTFR objects."
        )

    if callback_distance is None:
        callback_distance = distance_angle

    if callback_centroid is None:
        callback_centroid = partial(frechet_mean_angular, verbosity=0)

    data = [data[n].copy() for n in range(len(data))]

    if picks is not None:
        for n in range(len(data)):
            data[n] = data[n].pick(picks=picks)

    if baseline is not None:
        for n in range(len(data)):
            data[n] = _apply_baseline(data[n], baseline)

    X_list = []
    for n in range(len(data)):
        x = _get_data(
            data=data[n],
            domain="spatial",
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )
        X_list.append(x)

    if centroid is None:
        # np.array(X_list).shape: (n_groups, n_channels)
        X_centroid = callback_centroid(np.array(X_list))
    else:
        type_data_centroid = _get_data_type(centroid)
        if type_data_centroid not in ["Evoked", "AverageTFR"]:
            raise ValueError(
                "centroid must be an instance of Evoked or AverageTFR objects."
            )

        if type_data_centroid not in type_data:
            # type_data expect "list_Evoked" or "list_AverageTFR"
            # while type_data_centroid expect "Evoked" or "AverageTFR"
            raise ValueError("centroid and data must be of the same type.")

        centroid = centroid.copy()

        if picks is not None:
            centroid = centroid.pick(picks=picks)

        if baseline is not None:
            centroid = _apply_baseline(centroid, baseline)

        X_centroid = _get_data(
            data=centroid,
            domain="spatial",
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )

    distances = []

    for X in X_list:
        d = callback_distance(X, X_centroid)
        distances.append(d)

    if average:
        dispersion = np.mean(distances)
    else:
        dispersion = np.array(distances)

    if return_centroid:
        return dispersion, X_centroid

    return dispersion
