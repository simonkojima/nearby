import json
import numpy as np
import pandas as pd
import scipy
import mne

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic

from ..utils.core import (
    _add_values_to_df,
    _auto_windows,
    _get_obj_type,
)


def within_trial_frequency(
        tfrs,
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
    Compute Within-Trial Frequency (WiTrialFreq) Variability values.

    This metric quantifies how much the *spectral pattern across frequencies* changes
    within a single epoch. For each channel and epoch, the TFR is summarized into a
    sequence of frequency vectors (one vector per time window, or per time sample).
    A reference frequency vector (mean spectrum) is then estimated for that epoch,
    and the dispersion of frequency vectors around that mean is computed. The function
    returns the epoch-wise dispersion averaged across epochs (or a per-epoch list).

    The spectral dispersion is computed using one of two metrics:

    - ``metric="angle"``: angular distance to a Fréchet mean (p = 2) on the unit sphere.
    - ``metric="cosine"``: cosine distance to an extrinsic mean.

    Parameters
    ----------
    tfrs : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).
        This function is implemented for TFR input only.

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied. Baseline correction is applied as::

            tfrs.copy().apply_baseline(baseline, mode="percent")

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). If provided, the returned spectra are
        restricted to frequencies in ``[fmin, fmax]`` before computing variability.
        If ``None``, MNE defaults are used.

    mode : {"windows", "all_samples"}, default="windows"
        How to represent the within-epoch time evolution of frequency patterns.

        - ``"windows"``: average the TFR within time windows and treat each window as one
          frequency vector.
        - ``"all_samples"``: treat each time sample as one frequency vector (no temporal
          window averaging).

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
        Distance/mean definition for measuring spectral dispersion within each epoch.

        - ``"angle"``: angular distance between the epoch mean spectrum and each window
          spectrum. The mean is estimated using the Fréchet mean (p = 2) after
          normalizing spectra to unit norm.
        - ``"cosine"``: cosine distance to an extrinsic mean spectrum.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels in ``tfrs`` are used.

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging
        epoch-wise dispersions across epochs.

        If ``False``, returns a per-channel JSON-encoded list containing one dispersion
        value per epoch (order follows the epoch order in ``tfrs``).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``within_trial_frequency`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-epoch variability values.

    Notes
    -----
    Implementation outline (per channel, per epoch):

    1. Build a sequence of frequency vectors over time (windows or samples).
    2. Estimate an epoch-specific mean spectrum (Fréchet mean for ``"angle"``,
       extrinsic mean for ``"cosine"``).
    3. Compute distances from the mean to each frequency vector and average them to
       obtain an epoch dispersion.
    4. Average epoch dispersions across epochs (or return per-epoch values).
    """
    type_obj = _get_obj_type(tfrs)

    if not (type_obj == "EpochsTFR"):
        raise ValueError("obj must be an instance of EpochsTFR objects.")

    tfrs = tfrs.copy()

    if baseline is not None:
        tfrs = tfrs.apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "within_trial_frequency": []}

    data_dict = {}
    if mode == "windows":
        if windows is None:
            windows = _auto_windows(
                size=auto_window_size,
                step=auto_window_step,
                tmin=tmin,
                tmax=tmax,
            )

        for ch in tfrs.ch_names:
            data_dict[ch] = []
            for window in windows:
                data = tfrs.get_data(
                    tmin=window[0], tmax=window[1], fmin=fmin, fmax=fmax, picks=ch
                )
                # data.shape: n_epochs, n_channels, n_freqs, n_times

                data = data.squeeze()
                # data.shape: n_epochs, n_freqs, n_times

                data = data.mean(2)
                # data.shape: n_epochs, n_freqs

                data_dict[ch].append(data)

            data_dict[ch] = np.array(data_dict[ch])
            # data.shape: n_windows, n_epochs, n_freqs

            data_dict[ch] = data_dict[ch].transpose(1, 0, 2)
            # data.shape: n_epochs, n_windows, n_freqs

    elif mode == "all_samples":
        for ch in tfrs.ch_names:
            data = tfrs.get_data(fmin=fmin, fmax=fmax, picks=ch)
            # tfrs.shape: n_epochs, n_channels, n_freqs, n_times

            data = data.squeeze()
            # data.shape: n_epochs, n_freqs, n_times
            # In all_samples mode, n_times is equal to the number of windows
            # data.shape: n_epochs, n_freqs, n_windows

            data = data.transpose(0, 2, 1)
            # data.shape: n_epochs, n_windows, n_freqs

            data_dict[ch] = data

    else:
        raise ValueError("mode must be 'windows' or 'all_samples'")

    for ch in tfrs.ch_names:
        n_epochs, n_windows, _ = data_dict[ch].shape

        dispersion = []
        for idx_trial in range(n_epochs):
            data_trial = data_dict[ch][idx_trial, :, :].squeeze()
            # data_trial.shape: n_windows, n_freqs

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

            dispersion_within_trial = np.mean(distances_trial)

            dispersion.append(dispersion_within_trial)

        if average:
            dispersion = np.mean(dispersion)
        else:
            dispersion = json.dumps(dispersion)

        df_dict["channel"].append(ch)
        df_dict["within_trial_frequency"].append(dispersion)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_frequency(
        tfrs,
        baseline=None,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metric="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Between-Trial Frequency (BetTrialFreq) Variability values.

    This metric quantifies how much the *spectral pattern across frequencies* varies
    across epochs. For each channel and epoch, the TFR is reduced to a single
    frequency vector by averaging over time within the selected window
    (``tmin``–``tmax``). A reference frequency pattern (mean spectrum across epochs)
    is then estimated, and distances from this mean to each epoch's spectrum are
    computed. The function returns the mean distance across epochs (or a per-epoch list)
    for each channel.

    The mean pattern and distances are computed using one of two metrics:

    - ``metric="angle"``: angular distance to a Fréchet mean (p = 2) on the unit sphere.
    - ``metric="cosine"``: cosine distance to an extrinsic mean.

    Parameters
    ----------
    tfrs : mne.time_frequency.tfr.EpochsTFR
        Epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).
        This function is implemented for TFR input only.

    baseline : tuple of float | list of float | None, optional
        Time interval (in seconds) used for baseline correction. If ``None``, no baseline
        is applied. Baseline correction is applied as::

            tfrs.copy().apply_baseline(baseline, mode="percent")

    tmin, tmax : float | None, optional
        Time window of interest (in seconds). If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). If provided, the spectra are restricted
        to frequencies in ``[fmin, fmax]`` before computing variability.
        If ``None``, MNE defaults are used.

    metric : {"angle", "cosine"}, default="angle"
        Distance/mean definition for measuring dispersion across epochs.

        - ``"angle"``: angular distance between the across-epoch mean spectrum and each
          epoch spectrum. The mean is estimated using the Fréchet mean (p = 2) after
          normalizing spectra to unit norm.
        - ``"cosine"``: cosine distance to an extrinsic mean spectrum.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels in ``tfrs`` are used.

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging
        epoch-wise distances across epochs.

        If ``False``, returns a per-channel JSON-encoded list containing one distance
        value per epoch (order follows the epoch order in ``tfrs``).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``between_trial_frequency`` : float | str
            If ``average=True``, a scalar variability value.
            If ``average=False``, a JSON-encoded list of per-epoch variability values.

    Notes
    -----
    For each channel, the per-epoch spectrum is computed by averaging TFR values over
    time within the selected window (``tmin``–``tmax``). Variability is then measured
    as the dispersion of these spectra around a mean spectrum estimated across epochs.
    """
    type_obj = _get_obj_type(tfrs)

    if not (type_obj == "EpochsTFR"):
        raise ValueError("obj must be an instance of EpochsTFR objects.")

    tfrs = tfrs.copy()

    if baseline is not None:
        tfrs = tfrs.apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "between_trial_frequency": []}

    for ch in tfrs.ch_names:
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_epochs, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_freqs

        n_epochs, _ = data.shape

        distances = []

        if metric == "angle":
            mean = mean_frechet_p2(data, verbosity=0)

            for n in range(n_epochs):
                d = distance_angle(mean, data[n, :])
                distances.append(d)
        elif metric == "cosine":
            mean = mean_extrinsic(data)

            for n in range(n_epochs):
                d = scipy.spatial.distance.cosine(mean, data[n, :])
                distances.append(d)

        else:
            raise ValueError("metrics must be 'angle' or 'cosine'")

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df_dict["channel"].append(ch)
        df_dict["between_trial_frequency"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df


def between_trial_group_frequency(
        tfrs,
        centroid="mean",
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metric="angle",
        picks=None,
        average=True,
        weights="nave",
        additional_values=None,
):
    """
    Compute Between-Trial-Group Frequency (BtwTrialGrpFreq) Variability values.

    This metric quantifies how much *group-level spectral patterns* differ from one
    another. Each element of ``tfrs`` is assumed to represent a group-representative
    spectrum (e.g., an averaged TFR such as :class:`mne.time_frequency.tfr.AverageTFR`).

    For each channel, the function builds (or accepts) a centroid spectrum and then
    measures the dissimilarity between the centroid and each group spectrum across
    frequencies. The final channel-level score is the mean distance across groups
    (or a per-group list).

    Parameters
    ----------
    tfrs : list of mne.time_frequency.tfr.AverageTFR
        List of group-representative TFR objects. All entries must be
        :class:`mne.time_frequency.tfr.AverageTFR`.

    centroid : {"mean"} | mne.time_frequency.tfr.AverageTFR, default="mean"
        Definition of the centroid (reference) spectrum used for comparisons.

        - ``"mean"``: the centroid is computed as the weighted average of ``tfrs`` via
          :func:`mne.time_frequency.combine_tfr`.
        - If an :class:`mne.time_frequency.tfr.AverageTFR` is provided, it is used
          directly as the centroid.

    tmin, tmax : float | None, optional
        Time window of interest (in seconds). If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    fmin, fmax : float | None, optional
        Frequency band of interest (in Hz). If provided, spectra are restricted to
        frequencies in ``[fmin, fmax]`` before computing distances.
        If ``None``, MNE defaults are used.

    metric : {"angle", "cosine"}, default="angle"
        Distance definition for comparing spectra across groups.

        - ``"angle"``: angular distance between centroid and group spectra via
          ``distance_angle``.
        - ``"cosine"``: cosine distance between centroid and group spectra.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Applied to both the centroid and all objects in ``tfrs``.
        If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging
        distances across groups.

        If ``False``, returns a per-channel JSON-encoded list containing one distance
        per group (order follows ``tfrs``).

    weights : str | array_like of float, default="nave"
        Weights used when ``centroid="mean"``. Passed to
        :func:`mne.time_frequency.combine_tfr`. The default ``"nave"`` uses the number
        of averaged epochs (when available).

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for
        adding metadata (e.g., ``subject``, ``session``, ``run``, ``group_id``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``between_trial_group_frequency`` : float | str
            If ``average=True``, a scalar variability value (mean distance).
            If ``average=False``, a JSON-encoded list of per-group distance values.

    Notes
    -----
    For each group TFR, the per-channel spectrum is computed by averaging TFR values
    over time within ``[tmin, tmax]``. Distances are then computed between each group
    spectrum and the centroid spectrum.

    This function measures variability *between group-level averages* rather than
    between single-trial epochs.

    """

    if isinstance(centroid, str):
        match centroid:
            case "mean":
                tfr_centroid = mne.time_frequency.combine_tfr(tfrs, weights=weights)
            case _:
                raise NotImplementedError(f"centroid={centroid} is not implemented.")
    else:
        tfr_centroid = centroid

    type_tfr_centroid = _get_obj_type(tfr_centroid)
    type_tfrs = _get_obj_type(tfrs)

    if not (type_tfr_centroid == "AverageTFR"):
        raise ValueError("centroid must be an instance of AverageTFR objects.")

    if not (type_tfrs == "list_AverageTFR"):
        raise ValueError("tfrs must be an instance of list of AverageTFR objects.")

    tfr_centroid = tfr_centroid.copy()
    tfrs = [tfrs[n].copy() for n in range(len(tfrs))]

    if picks is not None:
        tfr_centroid = tfr_centroid.pick(picks=picks)
        for n in range(len(tfrs)):
            tfrs[n] = tfrs[n].pick(picks=picks)

    if not ((metric == "angle") or (metric == "cosine")):
        raise ValueError("metrics must be 'angle' or 'cosine'")

    df_dict = {"channel": [], "between_trial_group_frequency": []}

    def proc(tfrs, ch):
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_freqs

        return data

    for ch in tfr_centroid.ch_names:

        data_centroid = proc(tfr_centroid, ch=ch)
        data_list = [proc(tfrs[n], ch=ch) for n in range(len(tfrs))]

        distances = []

        if metric == "angle":
            for data in data_list:
                d = distance_angle(data_centroid, data)
                distances.append(d)
        elif metric == "cosine":
            for data in data_list:
                d = scipy.spatial.distance.cosine(data_centroid, data)
                distances.append(d)

        else:
            raise ValueError("metrics must be 'angle' or 'cosine'")

        if average:
            distances = np.array(distances)
            distances = distances.mean()
        else:
            distances = json.dumps(distances)

        df_dict["channel"].append(ch)
        df_dict["between_trial_group_frequency"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = _add_values_to_df(df, additional_values)

    return df
