import mne


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
        raise ValueError("obj must be an instance of mne.time_frequency.tfr.BaseTFR or mne.epochs.BaseEpochs")
    return type_obj
