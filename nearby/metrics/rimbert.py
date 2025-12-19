import numpy as np
import pandas as pd

from ..utils.core import _get_obj_type, _add_values_to_df


def mean_erds(
        obj,
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

    This function extracts ERD/S time courses from an epoched MNE object and
    summarizes each epoch by its mean value over time (and over frequency for TFR
    input). It then optionally averages these per-epoch values across epochs to
    produce a single scalar per channel.

    - If ``obj`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, values are first
      restricted to the frequency band ``[fmin, fmax]`` and averaged over frequency,
      yielding one time course per epoch.
    - If ``obj`` is an :class:`mne.epochs.Epochs`, the epoched signal is used
      directly (in microvolts).

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE object containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

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
        ``[tmin, tmax]`` before computing means.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels are used.

    average : bool, default=True
        If ``True``, returns a single scalar per channel obtained by averaging the
        per-epoch mean ERD/S values across epochs.

        If ``False``, returns the per-epoch mean ERD/S values for each channel (as a
        numpy array). Note that in this case, the dataframe cell will contain an array-like
        object.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe. This is useful for adding
        metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``mean_erds`` : float | numpy.ndarray
            If ``average=True``, a scalar mean ERD/S value.
            If ``average=False``, a vector of per-epoch mean ERD/S values.

    Notes
    -----
    The per-epoch ERD/S value is computed by averaging the ERD/S time course over time
    within ``[tmin, tmax]``. For TFR input, the time course is first obtained by
    averaging over frequencies within ``[fmin, fmax]``.


    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_obj = check_obj_type(obj)

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "mean_erds": []}
    for ch in obj.ch_names:

        if type_obj == "tfr":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times

        elif type_obj == "epochs":
            data = obj.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # data.shape: n_epochs, n_channels, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_times

        mean = data.mean(1)
        # mean.shape: n_epochs

        if average:
            mean = mean.mean(0)

        df_dict["channel"].append(ch)
        df_dict["mean_erds"].append(mean)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def standard_deviation_erds(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
):
    """
    Compute standard deviation of mean ERD/S values across trials.

    This function measures *across-trial amplitude variability* of ERD/S by first
    summarizing each epoch with a single mean ERD/S value (mean over time, and over
    frequency for TFR input) and then computing the standard deviation of these
    per-epoch means across epochs.

    - If ``obj`` is an :class:`mne.time_frequency.tfr.EpochsTFR`, values are restricted
      to ``[fmin, fmax]``, averaged over frequency, and then averaged over time.
    - If ``obj`` is an :class:`mne.epochs.Epochs`, the epoched signal is averaged over
      time directly (in microvolts).

    Parameters
    ----------
    obj : mne.epochs.Epochs | mne.time_frequency.tfr.EpochsTFR
        The MNE object containing either epoched time-domain data (:class:`mne.epochs.Epochs`)
        or epoched time–frequency data (:class:`mne.time_frequency.tfr.EpochsTFR`).

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
        ``[tmin, tmax]`` before computing per-epoch means.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include. If ``None``, all channels are used.

    additional_values : dict | None, optional
        Additional columns to append to the returned dataframe. This is useful for adding
        metadata (e.g., ``subject``, ``session``, ``run``).

    Returns
    -------
    df : pandas.DataFrame
        Output dataframe with one row per channel.

        - ``channel`` : str
            Channel name.
        - ``standard_deviation_erds`` : float
            Standard deviation of per-epoch mean ERD/S values across epochs.

    Notes
    -----
    This metric corresponds to the standard deviation across trials of the mean ERD/S
    amplitude within the selected time window (and frequency band for TFR input).
    In other words, it captures *between-trial dispersion of mean ERD/S values* rather
    than temporal-shape variability.

    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_obj = check_obj_type(obj)

    obj = obj.copy()

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "standard_deviation_erds": []}
    for ch in obj.ch_names:
        if type_obj == "tfr":
            data = obj.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
            # data.shape: n_epochs, n_channels, n_freqs, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_freqs, n_times

            data = data.mean(1)
            # data.shape: n_epochs, n_times

            data = data.mean(1)
            # data.shape: n_epochs
        elif type_obj == "epochs":
            data = obj.get_data(tmin=tmin, tmax=tmax, picks=ch, units="uV")
            # data.shape: n_epochs, n_channels, n_times

            data = np.squeeze(data)
            # data.shape: n_epochs, n_times

            data = data.mean(1)
            # data.shape: n_epochs

        std = np.std(data)

        df_dict["channel"].append(ch)
        df_dict["standard_deviation_erds"].append(std)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
