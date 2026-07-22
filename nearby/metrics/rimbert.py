import numpy as np

from ..utils.core import _get_data_type
from ..utils.proc import _get_data, _apply_baseline


def mean_erds(
    data,
    tmin=None,
    tmax=None,
    fmin=None,
    fmax=None,
    picks=None,
    baseline=None,
    average=True,
):
    """
    Compute mean ERD/S amplitude per channel.

    This function computes the channel-wise mean ERD/S amplitude as defined as
    "The means of the ERD" in Rimbert et al. (MetroXRAINE 2022). It is **not** a
    variability metric.

    For each channel, an epoch-wise ERD/S time course is extracted, each epoch is
    summarized by its mean over time within ``[tmin, tmax]``, and (optionally) these
    per-epoch values are averaged across epochs.

    Supported inputs are time-domain epochs (:class:`mne.epochs.Epochs`) and
    time–frequency epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input,
    values can be restricted to ``[fmin, fmax]`` and reduced to a single time course
    per epoch/channel by the internal data extraction routine.

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the mean.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied
        with ``mode="percent"`` (via :meth:`mne.time_frequency.EpochsTFR.apply_baseline`).

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). Only used when ``data`` is
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If provided, values are restricted
        to ``[fmin, fmax]`` before frequency reduction.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop the extracted time course(s) to
        ``[tmin, tmax]`` before averaging over time.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, return a scalar mean across epochs (mean of per-epoch means).
        If ``False``, return the per-epoch mean values as an array of shape
        ``(n_epochs,)``.

    Returns
    -------
    mean : float | ndarray of shape (n_epochs,)
        Channel-wise mean ERD/S amplitude aggregated across epochs (scalar) or
        per-epoch values (array), depending on ``average``.

    Notes
    -----
    The per-epoch value is computed by averaging the extracted time course over time
    within ``[tmin, tmax]``. The function operates on a copy of the input object
    (``data.copy()``) and does not modify inputs in place.

    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_data = _get_data_type(data)

    if not ((type_data == "EpochsTFR") or (type_data == "Epochs")):
        raise ValueError(
            "mean_erds is only implemented for EpochsTFR or Epochs objects."
        )

    data = data.copy()

    if baseline is not None:
        if type_data == "EpochsTFR":
            data = data.apply_baseline(baseline, "percent")
        elif type_data == "Epochs":
            data = data.apply_baseline(baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    X = _get_data(
        data,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        domain="temporal",
    )
    # X.shape: n_epochs, n_times

    X = X.mean(1)
    # X.shape: n_epochs

    if average:
        mean = X.mean(0)
    else:
        mean = np.asarray(X)

    return mean


def standard_deviation_erds(
    data,
    tmin=None,
    tmax=None,
    fmin=None,
    fmax=None,
    ddof=1,
    picks=None,
    baseline=None,
):
    """

    Compute the standard deviation of per-epoch mean ERD/S amplitudes.

    This metric quantifies *between-trial amplitude variability* of ERD/S by taking
    the standard deviation across epochs of a per-epoch ERD/S amplitude summary.

    For each channel:
    1) extract an epoch-wise ERD/S time course,
    2) summarize each epoch by its mean over time within ``[tmin, tmax]``,
       (and over frequency for TFR input, as handled by the internal extraction),
    3) compute the standard deviation of these per-epoch means across epochs.

    Supported inputs are time-domain epochs (:class:`mne.epochs.Epochs`) and
    time–frequency epochs (:class:`mne.time_frequency.tfr.EpochsTFR`). For TFR input,
    values can be restricted to ``[fmin, fmax]`` and reduced to a single time course
    per epoch/channel by the internal data extraction routine.

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0.0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing the metric.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline correction is applied
        with ``mode="percent"``.

    fmin, fmax : float | None, optional
        Lower and upper frequency bounds (Hz). Only used when ``data`` is
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If provided, values are restricted
        to ``[fmin, fmax]`` before frequency reduction.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used to crop the extracted time course(s) to
        ``[tmin, tmax]`` before averaging over time.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    ddof : int, default=1
        Delta degrees of freedom passed to :func:`numpy.std`. The standard deviation
        is computed as ``np.std(values, ddof=ddof)`` over epochs.

    Returns
    -------
    std : float
        Standard deviation of the per-epoch mean ERD/S amplitudes across epochs.

    Notes
    -----
    The per-epoch value is computed by averaging the extracted time course over time
    within ``[tmin, tmax]``. The function operates on a copy of the input object
    (``data.copy()``) and does not modify inputs in place.

    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_data = _get_data_type(data)

    if not ((type_data == "EpochsTFR") or (type_data == "Epochs")):
        raise ValueError(
            "standard_deviation_erds is only implemented for EpochsTFR or Epochs objects."
        )

    data = data.copy()

    data = _apply_baseline(data, baseline)

    if picks is not None:
        data = data.pick(picks=picks)

    X = _get_data(
        data,
        fmin=fmin,
        fmax=fmax,
        tmin=tmin,
        tmax=tmax,
        picks=picks,
        domain="temporal",
    )
    # X.shape: n_epochs, n_times

    X = X.mean(1)
    # X.shape: n_epochs

    std = np.std(X, ddof=ddof)

    return std
