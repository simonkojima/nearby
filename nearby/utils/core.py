import numpy as np
import mne


def auto_windows(size, step, tmin, tmax):
    start = np.arange(tmin, tmax, step)
    end = start + size

    windows = np.vstack([start, end]).T

    I = np.where(end < tmax)[0]
    windows = windows[I, :]

    return windows


def add_values_to_df(df, values):
    for key, value in values.items():
        df[key] = [value for _ in range(len(df))]
    return df


def check_obj_type(obj):
    if isinstance(obj, mne.time_frequency.tfr.EpochsTFR):
        type_obj = "tfr"
    elif isinstance(obj, mne.epochs.BaseEpochs):
        type_obj = "epochs"
    else:
        raise ValueError(
            "obj must be an instance of mne.time_frequency.tfr.BaseTFR or mne.epochs.BaseEpochs"
        )
    return type_obj
