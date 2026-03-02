import json
import numpy as np
import pandas as pd
import scipy
import mne

from functools import partial

from ..distances import distance_angle
from ..mean import extrinsic_mean, frechet_mean_angular

from ..utils.core import (
    _add_values_to_df,
    _auto_windows,
    _get_data_type,
)

from ..utils.proc import _get_data, _apply_baseline


def within_trial_frequency(
        data,
        baseline=None,
        fmin=None,
        fmax=None,
        mode="windows",
        auto_window_size=0.1,
        auto_window_step=0.1,
        tmin=None,
        tmax=None,
        callback_distance=None,
        callback_centroid=None,
        picks=None,
        average=True,
        return_centroid=False,
):
    """
    Compute within-trial frequency variability from time-resolved spectral patterns.

    This metric quantifies how much the *spectral pattern across frequencies* changes
    within each epoch. For each channel and epoch, the time–frequency representation
    (TFR) is converted into a sequence of frequency vectors (one per time window or
    one per time sample). A reference frequency vector is estimated for that epoch,
    and the dispersion of frequency vectors around the reference is computed. The
    final output is the mean dispersion across epochs, or a per-epoch list.

    Only time–frequency input (:class:`mne.time_frequency.tfr.EpochsTFR`) is supported.
    For each channel, variability is computed independently.

    Parameters
    ----------
    data : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data. This function is implemented for TFR input only.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` using ``mode="percent"``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. If provided, spectra are restricted to
        frequencies in ``[fmin, fmax]`` before computing variability. If ``None``,
        all frequencies are used.

    mode : {"windows", "all_samples"}, default="windows"
        How to represent within-epoch temporal evolution of frequency patterns.

        - ``"windows"``: average the TFR within time windows; each window yields one
          frequency vector of shape ``(n_freqs,)`` per epoch.
        - ``"all_samples"``: use each time sample as a frequency vector (no temporal
          window averaging).

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
        Time range of interest in seconds. Used only to generate automatic windows
        when ``windows=None`` and ``mode="windows"``. These values are not applied
        to cropping in ``mode="all_samples"``.

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition used to compute within-epoch spectral dispersion.

        - ``"angle"``: angular distance from the Fréchet mean (p=2) of the frequency
          vectors within an epoch.
        - ``"cosine"``: cosine distance from the extrinsic mean of the frequency
          vectors within an epoch.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to
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

        ``within_trial_frequency`` : float | str
            If ``average=True``, the mean dispersion across epochs.
            If ``average=False``, a JSON-encoded list of dispersions with length
            ``n_epochs``.

    Notes
    -----
    - For each channel and epoch, frequency vectors are constructed either by
      averaging over time windows (``mode="windows"``) or by using individual
      time samples (``mode="all_samples"``).
    - The function operates on a copy of the input object (``data.copy()``) and does
      not modify ``data`` in place.
    - This metric captures *within-trial frequency variability*, complementary to
      spatial and temporal variability metrics.
    """
    type_data = _get_data_type(data)

    if not (type_data == "EpochsTFR"):
        raise ValueError("data must be an instance of EpochsTFR objects.")

    if callback_distance is None:
        callback_distance = distance_angle

    if callback_centroid is None:
        callback_centroid = partial(frechet_mean_angular, verbosity=0)

    data = data.copy()

    if baseline is not None:
        data = _apply_baseline(data, baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    if mode == "windows":
        windows = _auto_windows(
            size=auto_window_size,
            step=auto_window_step,
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

    dispersion, mean_list = [], []
    for idx_trial in range(n_epochs):
        X_trial = X[idx_trial, :, :].squeeze()
        # X_trial.shape: n_windows, n_freqs

        mean = callback_centroid(X_trial)
        mean_list.append(mean)

        distances_trial = []
        for n in range(n_windows):
            d = callback_distance(mean, X_trial[n, :])
            distances_trial.append(d)

        dispersion_within_trial = np.mean(distances_trial)

        dispersion.append(dispersion_within_trial)

    if average:
        dispersion = np.mean(dispersion)
    else:
        dispersion = np.array(dispersion)

    if return_centroid:
        return dispersion, np.array(mean_list)

    return dispersion


def between_trial_frequency(
        data,
        baseline=None,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        callback_distance=None,
        callback_centroid=None,
        picks=None,
        average=True,
        return_centroid=False,
):
    """
    Compute between-trial frequency variability from epoch-wise spectra.

    This metric quantifies how much the *spectral pattern across frequencies* varies
    across epochs. For each channel, each epoch is reduced to a single frequency
    vector by averaging the TFR over time within the selected window (``tmin``–``tmax``).
    A reference spectrum is then estimated across epochs, and distances from this
    mean to each epoch’s spectrum are computed. The final output is the mean distance
    across epochs (or a per-epoch list) for each channel.

    Two distance/mean definitions are supported via ``metric``:

    - ``"angle"``: angular distance to a Fréchet mean (p=2) on the unit sphere.
    - ``"cosine"``: cosine distance to an extrinsic (Euclidean) mean.

    Parameters
    ----------
    data : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data. This function is implemented for TFR input only.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` using ``mode="percent"``.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. If provided, spectra are restricted to
        frequencies in ``[fmin, fmax]`` before computing variability. If ``None``,
        all frequencies are used.

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition used to measure dispersion across epochs.

        - ``"angle"``: estimate the across-epoch mean spectrum using a Fréchet mean
          (p=2), then compute angular distances to each epoch spectrum.
        - ``"cosine"``: estimate the mean spectrum using an extrinsic mean, then compute
          cosine distances to each epoch spectrum.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a single scalar (mean distance across epochs).
        - If ``False``, return a JSON-encoded list of per-epoch distances (epoch order
          matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``between_trial_frequency`` : float | str
            If ``average=True``, the mean distance across epochs.
            If ``average=False``, a JSON-encoded list of distances with length
            ``n_epochs``.

    Notes
    -----
    For each channel, the per-epoch spectrum is computed by averaging TFR values over
    time within ``[tmin, tmax]`` (after optional frequency restriction). Variability is
    measured as the dispersion of these spectra around a mean spectrum estimated across
    epochs. The function operates on a copy of the input object (``data.copy()``) and
    does not modify ``data`` in place.
    """
    type_data = _get_data_type(data)

    if not (type_data == "EpochsTFR"):
        raise ValueError("data must be an instance of EpochsTFR objects.")

    if callback_distance is None:
        callback_distance = distance_angle

    if callback_centroid is None:
        callback_centroid = partial(frechet_mean_angular, verbosity=0)

    data = data.copy()

    if baseline is not None:
        data = _apply_baseline(data, baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    df_dict = {"channel": [], "between_trial_frequency": []}

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

    mean = callback_centroid(X, verbosity=0)

    for n in range(n_epochs):
        d = callback_distance(mean, X[n, :])
        distances.append(d)

    if average:
        dispersion = np.mean(distances)
    else:
        dispersion = np.array(distances)

    if return_centroid:
        return dispersion, mean

    return dispersion


def between_trial_group_frequency(
        data,
        centroid=None,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        picks=None,
        callback_distance=None,
        callback_centroid=None,
        average=True,
        return_centroid=False,
):
    """
    Compute between-trial-group frequency variability from group-representative spectra.

    This metric quantifies how much *group-level spectral patterns* differ from one
    another. Each element of ``data`` is expected to be a group-representative TFR
    (:class:`mne.time_frequency.tfr.AverageTFR`). For each channel, each group object
    is reduced to a single frequency vector by averaging the TFR over time within
    ``[tmin, tmax]`` (after optional frequency restriction to ``[fmin, fmax]``).
    Distances between each group spectrum and a centroid spectrum are then computed.
    The final output is the mean distance across groups (or a per-group list) for
    each channel.

    The centroid can be computed from the group spectra (``centroid="mean"``) or
    provided explicitly as an :class:`~mne.time_frequency.tfr.AverageTFR`. When
    ``centroid="mean"``, the centroid computation depends on ``metric``:

    - ``metric="angle"`` uses a Fréchet mean (p=2) on the unit sphere.
    - ``metric="cosine"`` uses an extrinsic (Euclidean) mean.

    Parameters
    ----------
    data : list of mne.time_frequency.tfr.AverageTFR
        List of group-representative TFR objects. Only lists of
        :class:`~mne.time_frequency.tfr.AverageTFR` are supported.

    centroid : {"mean"} | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) spectrum used for comparisons.

        - ``"mean"``: compute the centroid spectrum from the group spectra. The mean
          definition depends on ``metric`` (Fréchet mean for ``"angle"``, extrinsic
          mean for ``"cosine"``).
        - If an :class:`~mne.time_frequency.tfr.AverageTFR` is provided, it is reduced
          to a centroid spectrum using the same averaging operations as for ``data``.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. If provided, spectra are restricted to
        frequencies in ``[fmin, fmax]`` before computing distances. If ``None``,
        all frequencies are used.

    metric : {"angle", "cosine"}, default="angle"
        Distance definition used to compare spectra across groups.

        - ``"angle"``: angular distance between spectra (via ``distance_angle``).
        - ``"cosine"``: cosine distance between spectra.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to each element in ``data`` and to the centroid
        object (if provided). If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a single scalar (mean distance across groups).
        - If ``False``, return a JSON-encoded list of per-group distances (group order
          matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``between_trial_group_frequency`` : float | str
            If ``average=True``, the mean distance across groups.
            If ``average=False``, a JSON-encoded list of distances with length
            ``len(data)``.

    Notes
    -----
    - For each group :class:`~mne.time_frequency.tfr.AverageTFR`, the per-channel
      spectrum is computed by averaging TFR values over time within ``[tmin, tmax]``
      (after optional frequency restriction).
    - This function measures variability between *group-level representatives*
      rather than between single-trial epochs.
    - The function operates on copies of the input objects (``copy()``) and does not
      modify inputs in place.
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

    if centroid is None:
        data_centroid = None
    else:
        data_centroid = centroid
        type_data_centroid = _get_data_type(data_centroid)

        if type_data_centroid != "AverageTFR":
            raise ValueError("centroid must be an instance of AverageTFR objects.")

        data_centroid = data_centroid.copy()
        if picks is not None:
            data_centroid = data_centroid.pick(picks=picks)

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

    if data_centroid is not None:
        X_centroid = _get_data(
            data=data_centroid,
            domain="frequency",
            tmin=tmin,
            tmax=tmax,
            fmin=fmin,
            fmax=fmax,
            picks=picks,
        )
    else:
        X_centroid = callback_centroid(np.array(X_list))

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
