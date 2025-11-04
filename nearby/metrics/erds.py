import numpy as np
import pandas as pd

import mne


def me(
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

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = np.squeeze(tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch))
        # data.shape: n_epochs, n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_epochs, n_times

        mean = data.mean(1)
        # mean.shape: n_epochs

        if average:
            mean = mean.mean(0)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["me"] = [mean]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df


def smeat(
        tfrs,
        baseline=[-1.0, 0.0],
        fmin=7,
        fmax=13,
        tmin=0.5,
        tmax=4.5,
        additional_values=None,
):
    ch_names = tfrs.ch_names

    if baseline is not None:
        tfrs = tfrs.copy().apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = np.squeeze(tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch))
        # data.shape: n_epochs, n_freqs, n_times

        data = data.mean(1)
        # data.shape: n_epochs, n_times

        data = data.mean(1)
        # data.shape: n_epochs

        std = np.std(data)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["smeat"] = [std]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df
