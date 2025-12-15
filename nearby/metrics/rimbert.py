import numpy as np
import pandas as pd

from ..utils import check_obj_type, add_values_to_df


def me(
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
    alias for old function 'me'

    It will be deprecated in the future.
    Use 'mean_erds' instead.
    """
    return mean_erds(
        obj, baseline, fmin, fmax, tmin, tmax, picks, average, additional_values
    ).rename(columns={"mean_erds": "me"})


def smeat(
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
    alias for old function 'smeat'

    It will be deprecated in the future.
    Use 'standard_deviation_erds' instead.
    """
    return standard_deviation_erds(
        obj, baseline, fmin, fmax, tmin, tmax, picks, additional_values
    ).rename(columns={"standard_deviation_erds": "smeat"})


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
    Compute mean ERD/S values.

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.BaseTFR``
        The MNE object containing either epochs or time–frequency data.

    baseline : tuple | list | None, optional
        Time interval used for baseline correction. If `None`, no baseline is applied.
        If `obj` is an MNE TFR object, the baseline is applied using `obj.copy().apply_baseline(baseline, mode="percent")`.
        If `obj` is an MNE Epochs object, it is applied using `obj.copy().apply_baseline(baseline)`.

    fmin, fmax : float | None, optional
        Frequency range of interest. Only used when `obj` is a TFR object.

    tmin, tmax : float | None, optional
        Time range of interest, in seconds.

    picks : str | array_like | None, optional
        Channels to include.

    average : bool, default=True
        If `True`, values are averaged across epochs. If `False`, a value
        is returned for each epoch.
        If ``average=False``, the DataFrame contains one entry per channel,
        where each entry consists of a JSON-encoded list representing the
        per-epoch ERD/S values.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing mean ERD/S values for each channel.

    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.copy().pick(picks=picks)

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

    Parameters
    ----------

    obj : ``mne.epochs.Epochs`` | ``mne.time_frequency.tfr.BaseTFR``
        The MNE object containing either epochs or time–frequency data.

    baseline : tuple | list | None, optional
        Time interval used for baseline correction. If `None`, no baseline is applied.
        If `obj` is an MNE TFR object, the baseline is applied using `obj.copy().apply_baseline(baseline, mode="percent")`.
        If `obj` is an MNE Epochs object, it is applied using `obj.copy().apply_baseline(baseline)`.

    fmin, fmax : float | None, optional
        Frequency range of interest. Only used when `obj` is a TFR object.

    tmin, tmax : float | None, optional
        Time range of interest, in seconds.

    picks : str | array_like | None, optional
        Channels to include.

    additional_values : dict | None, optional
        Additional columns to append to the output dataframe.

    Returns
    -------
    df : pandas.DataFrame
        DataFrame containing standard deviations of ERD/S for each channel.

    References
    ----------
    .. [1] S. Rimbert, D. Trocellier, F. Lotte,
       "Is Event-Related Desynchronization variability correlated with BCI performance?",
       MetroXRAINE 2022, DOI: `10.1109/MetroXRAINE54828.2022.9967551 <https://doi.org/10.1109/MetroXRAINE54828.2022.9967551>`_
    """
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.copy().pick(picks=picks)

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
