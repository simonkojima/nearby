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


def me_cross_user(tfrs_list, fmin=7, fmax=13, tmin=0.5, tmax=4.5, average=True,
                  additional_values=None):
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

    return me(tfrs, baseline=None, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, average=average,
              additional_values=additional_values)


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


def smeat_cross_user(tfrs_list, fmin=7, fmax=13, tmin=0.5, tmax=4.5, additional_values=None):
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

    return smeat(tfrs, baseline=None, fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, additional_values=additional_values)
