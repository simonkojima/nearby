import json

import numpy as np
import pandas as pd

import mne


def mean_erds(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        average=True,
        additional_values=None,
):
    ch_names = tfrs.ch_names

    tfrs = tfrs.apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        # data.shape: (n_trials, n_timesamples)
        data = np.squeeze(
            tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        ).mean(1)

        mean = data.mean(1)
        sd = np.std(mean)

        if average:
            mean = mean.mean(0)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["mean_erds_amplitude"] = [mean]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df


def dtw_dispersion(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        average=True,
        additional_values=None,
):
    df = attv(tfrs, baseline, fmin, fmax, tmin, tmax, average, additional_values)
    df["dtw_dispersion"] = df["attv"].tolist()

    return df


def attv(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        average=True,
        additional_values=None,
):
    try:
        import tslearn.metrics
    except:
        raise RuntimeError("tslearn is not installed. pip install tslearn")

    ch_names = tfrs.ch_names

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = np.squeeze(
            tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        ).mean(1)

        mean = data.mean(0)

        scores = []
        for m in range(data.shape[0]):
            score = tslearn.metrics.dtw(mean, data[m, :])
            scores.append(score)

        if average:
            scores = np.array(scores)
            scores = scores.mean()
        else:
            scores = json.dumps(scores)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["attv"] = [scores]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df


def attv_cross_user(tfrs_list, fmin=7, fmax=13, tmin=0.5, tmax=4.5, average=True, additional_values=None):
    data_list = []
    for tfrs in tfrs_list:
        if not isinstance(tfrs, mne.time_frequency.tfr.AverageTFR):
            raise ValueError("tfrs must be an instance of mne.time_frequency.tfr.AverageTFR. use tfrs.average()")

        data_list.append(tfrs.get_data())

    info = tfrs_list[0].info
    times = tfrs_list[0].times
    freqs = tfrs_list[0].freqs

    tfrs = mne.time_frequency.EpochsTFRArray(
        info=info,
        data=np.array(data_list),
        times=times,
        freqs=freqs,
    )

    return attv(tfrs, baseline=None, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, average=True,
                additional_values=additional_values)


def temporal_variability(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        additional_values=None,
        average=True,
):
    df = wttv(tfrs, baseline, fmin, fmax, tmin, tmax, average, additional_values)
    df["temporal_variability"] = df["wttv"].tolist()

    return df


def wttv(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        additional_values=None,
        average=True,
):
    ch_names = tfrs.ch_names

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = np.squeeze(
            tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        ).mean(1)

        sd = np.std(data, axis=1)

        if average:
            sd = sd.mean(0)
        else:
            sd = sd.tolist()
            sd = json.dumps(sd)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["wttv"] = [sd]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df
