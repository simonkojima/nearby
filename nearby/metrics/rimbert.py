import numpy as np
import pandas as pd

from ..utils.core import _get_data_type, _add_values_to_df
from ..utils.proc import _get_data, _apply_baseline


def mean_erds(
        data,
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
    Compute mean ERD/S values for each channel.

    For each channel, this function extracts an epoch-wise ERD/S time course,
    summarizes each epoch by its mean over time within ``[tmin, tmax]``, and then
    optionally averages these per-epoch values across epochs.

    This metrics is **NOT a variability metric**, and it is called **The means of the ERD** in the original
    paper by Rimbert et al. [1]_.

    - If ``data`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, values are optionally
      restricted to ``[fmin, fmax]`` and averaged over frequency, yielding one time
      course per epoch.
    - If ``data`` is an :class:`mne.epochs.Epochs`, the epoched signal is used directly
      (in microvolts).

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object. Only these two types are supported.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing means.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline is applied with
        ``mode="percent"``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` is an
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If provided, values are restricted
        to frequencies in ``[fmin, fmax]`` before averaging over frequency.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` before averaging over time.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    average : bool, default=True
        Controls the output format for each channel.

        - If ``True``, return a single scalar per channel (mean of per-epoch means).
        - If ``False``, return a JSON-encoded list of per-epoch mean values (epoch order
          matches ``data``).

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``mean_erds`` : float | str
            If ``average=True``, a scalar mean value.
            If ``average=False``, a JSON-encoded list of per-epoch mean values with
            length ``n_epochs``.

    Notes
    -----
    The per-epoch value is computed by averaging the extracted time course over time
    within ``[tmin, tmax]``. For TFR input, the time course is first obtained by
    averaging over frequencies within ``[fmin, fmax]``.
    The function operates on a copy of the input object (``data.copy()``) and does not
    modify inputs in place.


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

    return X


def standard_deviation_erds(
        data,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        ddof=1,
        additional_values=None,
):
    """
        Compute the standard deviation of mean ERD/S values across trials.

    This metric quantifies *between-trial amplitude variability* of ERD/S.
    For each channel, each epoch is first summarized by a single mean ERD/S
    value (averaged over time, and over frequency for TFR input). The standard
    deviation of these per-epoch mean values is then computed across epochs.

    - If ``data`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, values are
      optionally restricted to ``[fmin, fmax]``, averaged over frequency, and
      then averaged over time.
    - If ``data`` is an :class:`mne.epochs.Epochs`, the epoched signal is averaged
      over time directly (in microvolts).

    Parameters
    ----------
    data : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        Epoched MNE object. Only these two types are supported.

    baseline : tuple of float | None, optional
        Baseline interval in seconds (e.g., ``(None, 0)``). If provided, baseline
        correction is applied on a copy of ``data`` before computing values.
        For :class:`~mne.time_frequency.tfr.EpochsTFR`, baseline is applied with
        ``mode="percent"``.

    fmin, fmax : float | None, optional
        Frequency band of interest in Hz. Only used when ``data`` is an
        :class:`~mne.time_frequency.tfr.EpochsTFR`. If provided, values are restricted
        to frequencies in ``[fmin, fmax]`` before averaging.

    tmin, tmax : float | None, optional
        Time window of interest in seconds. If provided, data are cropped to
        ``[tmin, tmax]`` before computing per-epoch means.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. Passed to :meth:`mne.Epochs.pick` /
        :meth:`mne.time_frequency.EpochsTFR.pick`. If ``None``, all channels are used.

    ddof : int, default=1
        Delta degrees of freedom passed to :func:`numpy.std`.
        This parameter is forwarded directly to ``np.std(X, ddof=ddof)`` when
        computing the standard deviation across epochs.

    additional_values : dict | None, optional
        Extra columns to append to the returned dataframe (e.g., ``{"subject": 1}``).

    Returns
    -------
    df : pandas.DataFrame
        Dataframe with one row per channel.

        ``channel`` : str
            Channel name.

        ``standard_deviation_erds`` : float
            Standard deviation of per-epoch mean ERD/S values across epochs.

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
