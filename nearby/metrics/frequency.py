import numpy as np
import mne

from functools import partial

from ..distances import distance_angle
from ..mean import frechet_mean_angular

from ..utils.core import (
    _auto_windows,
    _get_data_type,
)

from ..utils.proc import _get_data, _apply_baseline


def within_trial_frequency(
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
    Compute within-trial frequency variability.

    This metric quantifies how much the spectral pattern across frequencies varies
    *within each epoch* over time. Each epoch is represented as a sequence of
    frequency vectors (one vector per time window in ``mode="windows"``, or one vector
    per time sample in ``mode="all_samples"``). A within-epoch centroid frequency
    pattern is estimated using ``callback_centroid`` and the dispersion is computed
    as the mean distance (``callback_distance``) between each frequency vector and
    the centroid. The final output aggregates dispersions across epochs.

    Only time–frequency input (:class:`mne.time_frequency.tfr.EpochsTFR`) is supported.

    Parameters
    ----------
    data : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to define the region of interest. In ``mode="windows"``
        these bounds control the window generation. In ``mode="all_samples"``, these
        bounds are passed to :meth:`mne.time_frequency.EpochsTFR.get_data`.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). If provided, spectra are restricted to
        ``[fmin, fmax]`` prior to computing the metric.

    callback_distance : callable | None, optional
        Distance function comparing two frequency vectors of shape ``(n_freqs,)``.
        If ``None``, uses :func:`nearby.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid frequency vector from an epoch-specific matrix
        ``X_trial`` of shape ``(n_windows, n_freqs)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    mode : {"windows", "all_samples"}, default="windows"
        Temporal sampling strategy for frequency patterns.

        - ``"windows"``: average within time windows; each window yields one frequency
          vector of shape ``(n_freqs,)`` per epoch.
        - ``"all_samples"``: use each time sample as a frequency vector (no time
          window averaging).

    window_size : float, default=0.1
        Window length (seconds) used when ``mode="windows"``.

    window_step : float, default=0.1
        Step size (seconds) used when ``mode="windows"``.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.time_frequency.EpochsTFR.pick`.
        If ``None``, all channels are used.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.

    average : bool, default=True
        If ``True``, return a scalar averaged across epochs. If ``False``, return the
        per-epoch dispersion values as an array of shape ``(n_epochs,)``.

    return_centroid : bool, default=False
        If ``True``, also return the per-epoch centroid frequency vectors.

    Returns
    -------
    dispersion : float | ndarray of shape (n_epochs,)
        Within-trial frequency dispersion. If ``average=True``, a scalar (mean across
        epochs). Otherwise, an array with one value per epoch.

    centroid : ndarray of shape (n_epochs, n_freqs), optional
        Returned only when ``return_centroid=True``. The within-epoch centroid
        frequency vectors.

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

    if not (type_data == "EpochsTFR"):
        raise ValueError("data must be an instance of EpochsTFR objects.")

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
            domain="frequency",
            tmin=None,
            tmax=None,
            windows=windows,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )

        X = np.array(X)
        # X.shape: n_windows, n_epochs, n_freqs

        X = X.transpose(1, 0, 2)
        # X.shape: n_epochs, n_windows, n_freqs

    elif mode == "all_samples":
        X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        X = X.mean(axis=1)
        # X.shape: n_epochs, n_freqs, n_times
        # In all_samples mode, n_times is equal to the number of windows
        # X.shape: n_epochs, n_freqs, n_windows

        X = X.transpose(0, 2, 1)
        # X.shape: n_epochs, n_windows, n_freqs
    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    n_epochs, n_windows, _ = X.shape

    dispersion, centroid_list = [], []
    for idx_trial in range(n_epochs):
        X_trial = X[idx_trial, :, :].squeeze()
        # X_trial.shape: n_windows, n_freqs

        centroid = callback_centroid(X_trial)
        centroid_list.append(centroid)

        distances_trial = []
        for n in range(n_windows):
            d = callback_distance(X_trial[n, :], centroid)
            distances_trial.append(d)

        dispersion_within_trial = np.mean(distances_trial)

        dispersion.append(dispersion_within_trial)

    if average:
        dispersion = np.mean(dispersion)
    else:
        dispersion = np.array(dispersion)

    if return_centroid:
        return dispersion, np.array(centroid_list)

    return dispersion


def between_trial_frequency(
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
    Compute between-trial frequency variability.

    This metric quantifies how much the spectral pattern across frequencies varies
    across epochs. Each epoch is reduced to a single frequency vector by averaging
    the time–frequency representation over time within ``[tmin, tmax]`` (and optionally
    restricting frequencies to ``[fmin, fmax]``). A centroid frequency pattern is then
    estimated across epochs using ``callback_centroid``, and the dispersion is
    computed as the mean distance (``callback_distance``) between each epoch spectrum
    and the centroid. The final output aggregates distances across epochs.

    Only time–frequency input (:class:`mne.time_frequency.tfr.EpochsTFR`) is supported.

    Parameters
    ----------
    data : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop the data to ``[tmin, tmax]`` prior to
        averaging over time.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). If provided, spectra are restricted to
        ``[fmin, fmax]`` prior to computing the metric.

    callback_distance : callable | None, optional
        Distance function comparing two frequency vectors of shape ``(n_freqs,)``.
        If ``None``, uses :func:`nearby.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid frequency vector from the stacked epoch spectra
        ``X`` of shape ``(n_epochs, n_freqs)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.time_frequency.EpochsTFR.pick`.
        If ``None``, all channels are used.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.

    average : bool, default=True
        If ``True``, return a scalar mean distance across epochs. If ``False``, return
        per-epoch distances as an array of shape ``(n_epochs,)``.

    return_centroid : bool, default=False
        If ``True``, also return the centroid frequency vector.

    Returns
    -------
    dispersion : float | ndarray of shape (n_epochs,)
        Between-trial frequency dispersion. If ``average=True``, a scalar (mean across
        epochs). Otherwise, an array with one value per epoch.

    centroid : ndarray of shape (n_freqs,), optional
        Returned only when ``return_centroid=True``. The centroid frequency vector.

    Notes
    -----
    - The function operates on a copy of the input object (``data.copy()``) and does
      not modify ``data`` in place.
    - The exact meaning of distance and centroid depends on
      ``callback_distance`` / ``callback_centroid``.
    """
    type_data = _get_data_type(data)

    if not (type_data == "EpochsTFR"):
        raise ValueError("data must be an instance of EpochsTFR objects.")

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
        domain="frequency",
        tmin=tmin,
        tmax=tmax,
        fmin=fmin,
        fmax=fmax,
        picks=picks,
    )

    n_epochs, _ = X.shape

    distances = []

    centroid = callback_centroid(X)

    for n in range(n_epochs):
        d = callback_distance(X[n, :], centroid)
        distances.append(d)

    if average:
        dispersion = np.mean(distances)
    else:
        dispersion = np.array(distances)

    if return_centroid:
        return dispersion, centroid

    return dispersion


def between_trial_group_frequency(
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
    Compute between-trial-group frequency variability.

    This metric quantifies how much group-representative spectral patterns differ
    from one another. Each element of ``data`` is expected to be a group-level
    representative time–frequency object (:class:`mne.time_frequency.tfr.AverageTFR`).
    Each group object is reduced to a single frequency vector by averaging the TFR
    over time within ``[tmin, tmax]`` (and optionally restricting frequencies to
    ``[fmin, fmax]``). A centroid spectrum is then defined (either computed from the
    group spectra or extracted from an explicit centroid object), and the dispersion
    is computed as the mean distance (``callback_distance``) between each group
    spectrum and the centroid.

    Parameters
    ----------
    data : list of mne.time_frequency.tfr.AverageTFR
        Group-representative time–frequency objects. All entries must be
        :class:`~mne.time_frequency.tfr.AverageTFR`.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop objects prior to averaging over time.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). If provided, spectra are restricted to
        ``[fmin, fmax]`` prior to computing the metric.

    callback_distance : callable | None, optional
        Distance function comparing two frequency vectors of shape ``(n_freqs,)``.
        If ``None``, uses :func:`nearby.distances.distance_angle`.

    callback_centroid : callable | None, optional
        Function computing a centroid frequency vector from a stack of group spectra
        with shape ``(n_groups, n_freqs)``.
        If ``None``, uses an angular Fréchet mean via
        ``partial(frechet_mean_angular, verbosity=0)``.

    centroid : mne.time_frequency.tfr.AverageTFR | None, optional
        Reference centroid object. If ``None``, the centroid spectrum is computed from
        the group spectra via ``callback_centroid``. If provided, it must be an
        :class:`~mne.time_frequency.tfr.AverageTFR` and will be reduced to a spectrum
        using the same extraction procedure as for group objects.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to each element in ``data`` and to the centroid
        object (if provided). If ``None``, all channels are used.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.

    average : bool, default=True
        If ``True``, return a scalar mean distance across groups. If ``False``, return
        per-group distances as an array of shape ``(n_groups,)``.

    return_centroid : bool, default=False
        If ``True``, also return the centroid frequency vector used as the reference.

    Returns
    -------
    dispersion : float | ndarray of shape (n_groups,)
        Between-trial-group frequency dispersion. If ``average=True``, a scalar (mean
        across groups). Otherwise, an array with one value per group.

    centroid_vector : ndarray of shape (n_freqs,), optional
        Returned only when ``return_centroid=True``. The centroid frequency vector
        used as the reference.

    Notes
    -----
    - The function operates on copies of the input objects (``.copy()``) and does not
      modify the originals in place.
    - The exact meaning of distance and centroid depends on
      ``callback_distance`` / ``callback_centroid``.
    """
    type_data = _get_data_type(data)

    if not (type_data == "list_AverageTFR"):
        raise ValueError("data must be an instance of list of AverageTFR objects.")

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
            domain="frequency",
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )
        X_list.append(x)

    if centroid is None:
        X_centroid = callback_centroid(np.array(X_list))
    else:
        type_data_centroid = _get_data_type(centroid)

        if type_data_centroid != "AverageTFR":
            raise ValueError("centroid must be an instance of AverageTFR objects.")

        centroid = centroid.copy()

        if picks is not None:
            centroid = centroid.pick(picks=picks)
        if baseline is not None:
            centroid = _apply_baseline(centroid, baseline)

        X_centroid = _get_data(
            data=centroid,
            domain="frequency",
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
