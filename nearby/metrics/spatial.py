import json
import warnings

import numpy as np
import scipy
import pandas as pd

import mne

from ..distances import distance_angle
from ..mean import mean_extrinsic, mean_frechet_p2

from ..utils import check_obj_type, add_values_to_df


def atsv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        metrics="angle",
        picks=None,
        average=True,
        additional_values=None,
):
    """
    Compute Across-Trial Spatial Variability values.

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
        DataFrame containing atsv values

    """
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    if type_obj == "tfr":
        data = obj.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax)
        # data.shape: n_epochs, n_channels, n_freqs, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels
    elif type_obj == "epochs":
        data = obj.get_data(tmin=tmin, tmax=tmax, units="uV")
        # data.shape: n_epochs, n_channels, n_times

        data = data.mean(2)
        # data.shape: n_epochs, n_channels

    n_epochs, _ = data.shape

    distances = []

    if metrics == "cosine":
        mean = mean_extrinsic(data)

        for n in range(n_epochs):
            d = scipy.spatial.distance.cosine(mean, data[n, :])
            distances.append(d)

    elif metrics == "angle":
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
    if metrics == "angle":
        df["atsva"] = [distances]
    elif metrics == "cosine":
        df["atsvc"] = [distances]

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
