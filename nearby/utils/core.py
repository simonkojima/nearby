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


def get_obj_type(obj):
    if isinstance(obj, mne.time_frequency.tfr.EpochsTFR):
        type_obj = "EpochsTFR"
    elif isinstance(obj, mne.epochs.BaseEpochs):
        type_obj = "Epochs"
    elif isinstance(obj, mne.time_frequency.tfr.AverageTFR):
        type_obj = "AverageTFR"
    elif isinstance(obj, mne.evoked.Evoked):
        type_obj = "Evoked"
    elif isinstance(obj, list):
        type_obj = []
        for o in obj:
            type_obj.append(get_obj_type(o))
        type_obj = list(set(type_obj))

        if len(type_obj) > 1:
            raise ValueError("obj must be a list of the same type")

        type_obj = f"list_{type_obj[0]}"
    else:
        type_obj = None

    if type_obj is None:
        raise ValueError(
            "obj must be an instance of mne.time_frequency.tfr.BaseTFR, mne.epochs.BaseEpochs, mne.time_frequency.tfr.AverageTFR, mne.evoked.Evoked or list of these"
        )

    return type_obj


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


def check_averaged_obj_type(obj):
    if isinstance(obj, mne.time_frequency.tfr.AverageTFR):
        type_obj = "tfr"
    elif isinstance(obj, mne.evoked.Evoked):
        type_obj = "evoked"
    else:
        raise ValueError(
            "obj must be an instance of mne.time_frequency.tfr.BaseTFR or mne.epochs.BaseEpochs"
        )
    return type_obj
