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
    _get_obj_type,
)


def within_trial_spatial(
        obj,
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
    Compute Within-Trial Spatial (WiTrialSpat) Variability values.

    This metric quantifies how much the *spatial pattern across channels* changes
    within a single epoch. For each epoch, the signal is summarized into a sequence
    of spatial vectors (one vector per time window, or per sample). A reference
    spatial vector (mean pattern) is then estimated for that epoch, and the
    dispersion of spatial vectors around that mean is computed. The function returns
    the epoch-wise dispersion averaged across epochs (or a per-epoch list).

    The spatial dispersion is computed using one of two metrics:

    - ``metric="angle"``: angular distance to a Fréchet mean (p = 2) on the unit sphere.
    - ``metric="cosine"``: cosine distance to an extrinsic mean.

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched time-domain data (:class:`mne.epochs.Epochs`) or epoched time–frequency
        data (:class:`mne.time_frequency.tfr.EpochsTFR`).

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied.

        For :class:`mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline, mode="percent")

        For :class:`mne.epochs.Epochs`, baseline correction is applied as::

            obj.copy().apply_baseline(baseline)

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). Only used when ``obj`` is an
        :class:`mne.time_frequency.tfr.EpochsTFR`. If provided, the data are reduced by
        averaging over frequencies in ``[fmin, fmax]`` before computing spatial patterns.

    mode : {"windows", "all_samples"}, default="windows"
        How to represent the within-epoch time evolution of spatial patterns.

        - ``"windows"``: average the data within time windows and treat each window as one
          spatial vector.
        - ``"all_samples"``: treat each time sample as one spatial vector (no temporal
          window averaging). For TFR input, frequency is still averaged first.

    windows : array_like of shape (n_windows, 2) | None, optional
        Explicit time windows ``[[tmin_0, tmax_0], ..., [tmin_k, tmax_k]]`` in seconds
        used when ``mode="windows"``. If ``None``, windows are generated automatically
        with ``auto_window_size`` and ``auto_window_step`` between ``tmin`` and ``tmax``.

    auto_window_size : float, default=0.1
        Window size (in seconds) used to generate windows when ``windows=None`` and
        ``mode="windows"``.

    auto_window_step : float, default=0.1
        Step size (in seconds) used to generate windows when ``windows=None`` and
        ``mode="windows"``.

    tmin, tmax : float | None, optional
        Time range of interest (in seconds). Used to generate automatic windows when
        ``windows=None``. In ``mode="all_samples"``, these are not currently applied in
        the implementation (data are taken from the full epoch).

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition for measuring spatial dispersion within each epoch.

        - ``"angle"``: angular distance between the epoch mean pattern and each window
          pattern. The mean is estimated using the Fréchet mean (p = 2) after
          normalizing patterns to unit norm.
        - ``"cosine"``: cosine distance to an extrinsic mean pattern.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels in ``obj`` are used.

    average : bool, default=True
        If ``True``, returns a single scalar obtained by averaging epoch-wise dispersions
        across epochs.

        If ``False``, returns a JSON-encoded list containing one dispersion value per
        epoch (order follows the epoch order in ``obj``).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with a single row.

        - ``within_trial_spatial`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-epoch variability values.

    Notes
    -----
    Implementation outline (per epoch):

    1. Build a sequence of spatial vectors over time (windows or samples).
    2. Estimate an epoch-specific mean spatial pattern (Fréchet mean for ``"angle"``,
       extrinsic mean for ``"cosine"``).
    3. Compute distances from the mean to each spatial vector and average them to obtain
       an epoch dispersion.
    4. Average epoch dispersions across epochs (or return per-epoch values).

    This function is implemented only for :class:`mne.epochs.Epochs` and
    :class:`mne.time_frequency.tfr.EpochsTFR`.
    """
    type_obj = _get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError("obj must be an instance of EpochsTFR or Epochs objects.")

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    """
    if (tmin is not None) or (tmax is not None):
        obj = obj.crop(tmin=tmin, tmax=tmax)

    if type_obj == "EpochsTFR":
        if (fmin is not None) or (fmax is not None):
            obj = obj.crop(fmin=fmin, fmax=fmax)
    """

    data_list = []
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
            if type_obj == "EpochsTFR":
                data = obj.get_data(
                    tmin=window[0], tmax=window[1], fmin=fmin, fmax=fmax
                )
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels
            elif type_obj == "Epochs":
                data = obj.get_data(tmin=window[0], tmax=window[1], units="uV")
                # data.shape: n_epochs, n_channels, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_channels

            data_list.append(data)

        data = np.array(data_list)
        # data.shape: n_windows, n_epochs, n_channels

        data = data.transpose(1, 0, 2)
        # data.shape: n_epochs, n_windows, n_channels

    elif mode == "all_samples":
        if type_obj == "EpochsTFR":
            data = obj.get_data()
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = data.mean(2)
            # data.shape: n_epochs, n_channels, n_times
        elif type_obj == "Epochs":
            data = obj.get_data(units="uV")
            # data.shape: n_epochs, n_channels, n_times

        # In all_samples mode, n_times is equal to the number of windows
        # data.shape: n_epochs, n_channels, n_windows = n_epochs, n_channels, n_times

        data = data.transpose(0, 2, 1)
        # data.shape: n_epochs, n_windows, n_channels

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    n_epochs, n_windows, _ = data.shape

    distances = []

    for idx_trial in range(n_epochs):
        data_trial = data[idx_trial, :, :].squeeze()
        # data_trial.shape: n_windows, n_channels

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

        dispersion = np.mean(distances_trial)

        distances.append(dispersion)

    if average:
        distances = np.mean(distances)
    else:
        distances = json.dumps(distances)

    df = pd.DataFrame()
    df["within_trial_spatial"] = [distances]

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_spatial(
        obj,
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
    Compute Between-Trial Spatial (BtwTrialSpat) Variability values.

    This metric quantifies how much the *spatial pattern across channels* varies
    across epochs. For each epoch, the data are reduced to a single spatial vector
    (one value per channel) by averaging over time (and over frequency for TFR input).
    A reference spatial pattern (mean across epochs) is then estimated, and distances
    from this mean to each epoch's spatial vector are computed. The function returns
    the mean distance across epochs (or a per-epoch list).

    The mean pattern and distances are computed using one of two metrics:

    - ``metric="angle"``: angular distance to a Fréchet mean (p = 2) on the unit sphere.
    - ``metric="cosine"``: cosine distance to an extrinsic mean.

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched time-domain data (:class:`mne.epochs.Epochs`) or epoched time–frequency
        data (:class:`mne.time_frequency.tfr.EpochsTFR`).

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
        ``[tmin, tmax]`` before averaging.

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition for measuring dispersion across epochs.

        - ``"angle"``: angular distance between the across-epoch mean pattern and each
          epoch pattern. The mean is estimated using the Fréchet mean (p = 2) after
          normalizing patterns to unit norm.
        - ``"cosine"``: cosine distance to an extrinsic mean pattern.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels in ``obj`` are used.

    average : bool, default=True
        If ``True``, returns a single scalar obtained by averaging epoch-wise distances
        across epochs.

        If ``False``, returns a JSON-encoded list containing one distance value per
        epoch (order follows the epoch order in ``obj``).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with a single row.

        - ``between_trial_spatial`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-epoch variability values.

    Notes
    -----
    For :class:`mne.time_frequency.tfr.EpochsTFR`, the spatial vector for each epoch is
    computed by averaging over the frequency axis and then averaging over time within
    the selected window. For :class:`mne.epochs.Epochs`, the spatial vector is computed
    by averaging over time only.

    This function is implemented only for :class:`mne.epochs.Epochs` and
    :class:`mne.time_frequency.tfr.EpochsTFR`.
    """
    type_obj = _get_obj_type(obj)

    if not ((type_obj == "EpochsTFR") or (type_obj == "Epochs")):
        raise ValueError("obj must be an instance of EpochsTFR or Epochs objects.")

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    if type_obj == "EpochsTFR":
        data = obj.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels
    elif type_obj == "Epochs":
        data = obj.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels

    n_epochs, _ = data.shape

    distances = []

    if metric == "cosine":
        mean = mean_extrinsic(data)

        for n in range(n_epochs):
            d = scipy.spatial.distance.cosine(mean, data[n, :])
            distances.append(d)

    elif metric == "angle":
        mean = mean_frechet_p2(data, verbosity=0)

        for n in range(n_epochs):
            d = distance_angle(mean, data[n, :])
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
        objs,
        centroid="mean",
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        metric="angle",
        picks=None,
        average=True,
        weights="nave",
        additional_values=None,
):
    """
    Compute Between-Trial-Group Spatial (BtwTrialGrpSpat) Variability values.

    This metric quantifies how much *group-level spatial patterns* differ from one
    another. Each element of ``objs`` is assumed to represent a *trial-group
    representative* spatial pattern (e.g., an averaged ERD/S pattern such as an
    :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`).

    For each group, the data are reduced to a single spatial vector (one value per
    channel) by averaging over time (and over frequency for TFR input). A centroid
    pattern is then built (or provided), and distances between the centroid and each
    group pattern are computed. The function returns the mean distance across groups
    (or a per-group list).

    Parameters
    ----------
    objs : list of mne.Evoked | list of mne.time_frequency.tfr.AverageTFR
        List of group-representative objects. All entries must be the same type:
        either :class:`mne.Evoked` or :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.Evoked | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) pattern used for comparisons.

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
        ``[tmin, tmax]`` before averaging.

    metric : {"angle", "cosine"}, default="angle"
        Distance definition for measuring dispersion across groups.

        - ``"angle"``: angular distance between spatial patterns. In this implementation,
          the centroid and group patterns are compared directly with ``distance_angle``.
        - ``"cosine"``: cosine distance between centroid and group patterns.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to both the centroid and all objects in ``objs``.
        If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, returns a single scalar obtained by averaging distances across
        groups.

        If ``False``, returns a JSON-encoded list containing one distance value per
        group (order follows ``objs``).

    weights : str | array_like of float, default="nave"
        Weights used when ``centroid="mean"``.
        Passed to :func:`mne.combine_evoked` or :func:`mne.time_frequency.combine_tfr`.
        The default ``"nave"`` uses the number of averaged epochs (when available).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``, ``group_id``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with a single row.

        - ``between_trial_group_spatial`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-group variability values.

    Notes
    -----
    For :class:`mne.time_frequency.tfr.AverageTFR`, the spatial vector is computed by
    averaging over the frequency axis and then averaging over time within the selected
    window. For :class:`mne.Evoked`, the spatial vector is computed by averaging over
    time only.

    This function measures variability *between group-level averages* rather than
    between single-trial epochs.
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
        data = tfr.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax)
        # data.shape: n_channels, n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_channels, n_times

        data = data.mean(1)
        # data.shape: n_channels

        return data

    def proc_evoked(evoked):
        data = evoked.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_channels, n_times

        data = data.mean(1)
        # data.shape: n_channels

        return data

    if type_obj_centroid == "AverageTFR":

        data_centroid = proc_tfr(obj_centroid)
        data_list = [proc_tfr(objs[n]) for n in range(len(objs))]

    elif type_obj_centroid == "Evoked":
        data_centroid = proc_evoked(obj_centroid)
        data_list = [proc_evoked(objs[n]) for n in range(len(objs))]

    distances = []

    if metric == "cosine":
        for data in data_list:
            d = scipy.spatial.distance.cosine(data_centroid, data)
            distances.append(d)

    elif metric == "angle":
        for data in data_list:
            d = distance_angle(data_centroid, data)
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
