import numpy as np
import pandas as pd


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
    try:
        import tslearn.metrics
    except:
        raise RuntimeError("tslearn is not installed. pip install tslearn")

    ch_names = tfrs.ch_names

    tfrs = tfrs.apply_baseline(baseline, mode="percent")

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

        scores = np.array(scores)

        if average:
            scores = scores.mean()

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["dtw_dispersion"] = [scores]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df


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

    ch_names = tfrs.ch_names

    tfrs = tfrs.apply_baseline(baseline, mode="percent")

    df_list = []
    for ch in ch_names:
        data = np.squeeze(
            tfrs.get_data(fmin=fmin, fmax=fmax, tmin=tmin, tmax=tmax, picks=ch)
        ).mean(1)

        sd = np.std(data, axis=1)

        if average:
            sd = sd.mean(0)

        df = pd.DataFrame()
        df["channel"] = [ch]
        df["temporal_variability"] = [sd]

        df_list.append(df)

    df = pd.concat(df_list, axis=0, ignore_index=True)

    if additional_values is not None:
        for key, value in additional_values.items():
            df[key] = [value for m in range(df.shape[0])]

    return df

