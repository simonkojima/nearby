import json

import numpy as np
import pandas as pd

import mne

from ..utils import check_obj_type, add_values_to_df


def attv(
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
    try:
        import tslearn.metrics
    except:
        raise RuntimeError("tslearn is not installed. pip install tslearn")

    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "attv": []}
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

        mean = data.mean(0)
        # mean.shape: n_times

        scores = []
        for m in range(data.shape[0]):
            score = tslearn.metrics.dtw(mean, data[m, :])
            scores.append(score)

        if average:
            scores = np.array(scores)
            scores = scores.mean()
        else:
            scores = json.dumps(scores)

        df_dict["channel"].append(ch)
        df_dict["attv"].append(scores)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df


def wttv(
        obj,
        baseline=None,
        fmin=None,
        fmax=None,
        tmin=None,
        tmax=None,
        picks=None,
        additional_values=None,
        average=True,
):
    type_obj = check_obj_type(obj)

    if baseline is not None:
        if type_obj == "tfr":
            obj = obj.copy().apply_baseline(baseline, "percent")
        elif type_obj == "epochs":
            obj = obj.copy().apply_baseline(baseline)

    if picks is not None:
        obj = obj.pick(picks=picks)

    df_dict = {"channel": [], "wttv": []}
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

        sd = np.std(data, axis=1)

        if average:
            sd = sd.mean(0)
        else:
            sd = sd.tolist()
            sd = json.dumps(sd)

        df_dict["channel"].append(ch)
        df_dict["wttv"].append(sd)

    df = pd.DataFrame(df_dict)

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
