import mne
import pytest
import numpy as np


def dummy_info():
    ch_names_64 = [
        "Fp1",
        "Fp2",
        "AF7",
        "AF3",
        "AFz",
        "AF4",
        "AF8",
        "F7",
        "F5",
        "F3",
        "F1",
        "Fz",
        "F2",
        "F4",
        "F6",
        "F8",
        "FT9",
        "FT7",
        "FC5",
        "FC3",
        "FC1",
        "FCz",
        "FC2",
        "FC4",
        "FC6",
        "FT8",
        "FT10",
        "T7",
        "C5",
        "C3",
        "C1",
        "Cz",
        "C2",
        "C4",
        "C6",
        "T8",
        "TP9",
        "TP7",
        "CP5",
        "CP3",
        "CP1",
        "CPz",
        "CP2",
        "CP4",
        "CP6",
        "TP8",
        "TP10",
        "P7",
        "P5",
        "P3",
        "P1",
        "Pz",
        "P2",
        "P4",
        "P6",
        "P8",
        "PO7",
        "PO3",
        "POz",
        "PO4",
        "PO8",
        "O1",
        "Oz",
        "O2",
    ]
    sfreq = 100

    return mne.create_info(ch_names=ch_names_64, sfreq=sfreq, ch_types="eeg")


def dummy_epochs(n_epochs=240):
    data = np.random.randn(n_epochs, 64, 100 * 5)
    epochs = mne.EpochsArray(data, info=dummy_info())
    return epochs


def dummy_epochs_tfr(n_epochs=240):
    data = np.random.randn(n_epochs, 64, 45, 100 * 5)
    info = dummy_info()
    times = np.arange(0, 5, 1 / info["sfreq"])
    epochs_tfr = mne.time_frequency.EpochsTFRArray(
        info=info, data=data, times=times, freqs=list(range(1, 46))
    )
    return epochs_tfr


def dummy_average_tfr():
    data = np.random.randn(64, 45, 100 * 5)
    info = dummy_info()
    times = np.arange(0, 5, 1 / info["sfreq"])
    epochs_average_tfr = mne.time_frequency.AverageTFRArray(
        info=info, data=data, times=times, freqs=list(range(1, 46))
    )
    return epochs_average_tfr


def dummy_evoked():
    data = np.random.randn(64, 100 * 5)
    info = dummy_info()
    evoked = mne.EvokedArray(data=data, info=info)
    return evoked
