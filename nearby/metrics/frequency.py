import json
import numpy as np
import pandas as pd

import scipy

import mne

from ..distances import distance_angle
from ..mean import mean_frechet_p2, mean_extrinsic

from ..utils import check_obj_type, add_values_to_df


def atfv(
        tfrs,
        baseline=None,
        tmin=None,
        tmax=None,
        fmin=None,
        fmax=None,
        metrics="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Frequency Variability values.

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

    metrics : {"angle", "cosine"}, default="angle"
        If "angle", the distance between the mean and each epoch is computed using
        angular distance. The mean is estimated using the Fréchet mean (p = 2).

        If "cosine", the distance between the mean and each epoch is computed using
        cosine distance. The mean is estimated using an extrinsic mean.

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
        DataFrame containing atfv values for each channel.

    """
    type_obj = check_obj_type(tfrs)

    if type_obj != "tfr":
        raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.BaseTFR")

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, "percent")

    if picks is not None:
        tfrs = tfrs.pick(picks=picks)

    if metrics == "angle":
        df_dict = {"channel": [], "atfva": []}
    elif metrics == "cosine":
        df_dict = {"channel": [], "atfvc": []}
    else:
        raise ValueError("metrics must be 'angle' or 'cosine'")

    for ch in tfrs.ch_names:
        data = tfrs.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=ch)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.squeeze()
        # data.shape: n_epochs, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_freqs

        n_epochs, _ = data.shape

        distances = []

        if metrics == "angle":
            mean = mean_frechet_p2(data, verbosity=0)

            for n in range(n_epochs):
                d = distance_angle(mean, data[n, :])
                distances.append(d)
        elif metrics == "cosine":
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
        if metrics == "angle":
            df_dict["atfva"].append(distances)
        elif metrics == "cosine":
            df_dict["atfvc"].append(distances)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
