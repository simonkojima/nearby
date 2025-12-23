import numpy as np
import mne


def _auto_windows(size, step, tmin, tmax):
    start = np.arange(tmin, tmax, step)
    end = start + size

    windows = np.vstack([start, end]).T

    I = np.where(end <= tmax)[0]
    windows = windows[I, :]

    return windows


def _add_values_to_df(df, values):
    for key, value in values.items():
        df[key] = [value for _ in range(len(df))]
    return df


def _get_data_type(data):
    if isinstance(data, mne.time_frequency.tfr.EpochsTFR):
        type_data = "EpochsTFR"
    elif isinstance(data, mne.epochs.BaseEpochs):
        type_data = "Epochs"
    elif isinstance(data, mne.time_frequency.tfr.AverageTFR):
        type_data = "AverageTFR"
    elif isinstance(data, mne.evoked.Evoked):
        type_data = "Evoked"
    elif isinstance(data, list):
        type_data = []
        for _data in data:
            type_data.append(_get_data_type(_data))
        type_data = list(set(type_data))

        if len(type_data) > 1:
            raise ValueError("obj must be a list of the same type")

        type_data = f"list_{type_data[0]}"
    else:
        type_data = None

    if type_data is None:
        raise ValueError(
            "data must be an instance of mne.time_frequency.tfr.BaseTFR, mne.epochs.BaseEpochs, mne.time_frequency.tfr.AverageTFR, mne.evoked.Evoked or list of these"
        )

    return type_data
