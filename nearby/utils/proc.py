import numpy as np
from .core import _get_data_type


def _get_temporal_epochs_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_epochs, n_channels, n_freqs, n_times

    X = X.mean(axis=(1, 2))
    # X.shape: n_epochs, n_times

    return X


def _get_spatial_epochs_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_epochs, n_channels, n_freqs, n_times

    X = X.mean(axis=(2, 3))
    # X.shape: n_epochs, n_channels

    return X


def _get_frequency_epochs_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_epochs, n_channels, n_freqs, n_times

    X = X.mean(axis=(1, 3))
    # X.shape: n_epochs, n_freqs

    return X


def _get_temporal_epochs(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks, units="uV")
    # X.shape: n_epochs, n_channels, n_times

    X = X.mean(axis=1)
    # X.shape: n_epochs, n_times

    return X


def _get_spatial_epochs(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks, units="uV")
    # X.shape: n_epochs, n_channels, n_times

    X = X.mean(axis=2)
    # X.shape: n_epochs, n_channels

    return X


def _get_temporal_average_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_channels, n_freqs, n_times

    X = X.mean(axis=(0, 1))
    # X.shape: n_times

    return X


def _get_spatial_average_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_channels, n_freqs, n_times

    X = X.mean(axis=(1, 2))
    # X.shape: n_channels

    return X


def _get_frequency_average_tfr(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
    # X.shape: n_channels, n_freqs, n_times

    X = X.mean(axis=(0, 2))
    # X.shape: n_freqs

    return X


def _get_temporal_evoked(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks, units="uV")
    # X.shape: n_channels, n_times

    X = X.mean(axis=0)
    # X.shape: n_times

    return X


def _get_spatial_evoked(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks, units="uV")
    # X.shape: n_channels, n_times

    X = X.mean(axis=1)
    # X.shape: n_channels

    return X


_GETTERS = {
    "EpochsTFR": {
        "temporal": _get_temporal_epochs_tfr,
        "spatial": _get_spatial_epochs_tfr,
        "frequency": _get_frequency_epochs_tfr,
    },
    "AverageTFR": {
        "temporal": _get_temporal_average_tfr,
        "spatial": _get_spatial_average_tfr,
        "frequency": _get_frequency_average_tfr,
    },
    "Epochs": {
        "temporal": _get_temporal_epochs,
        "spatial": _get_spatial_epochs,
    },
    "Evoked": {
        "temporal": _get_temporal_evoked,
        "spatial": _get_spatial_evoked,
    },
}


def _get_data(data, domain, fmin, fmax, picks, tmin=None, tmax=None, windows=None):
    type_data = _get_data_type(data)

    try:
        func = _GETTERS[type_data][domain]
    except KeyError as e:
        raise ValueError(
            f"Unsupported combination: type={type_data}, domain={domain}"
        ) from e

    if windows is not None:
        if (windows.ndim != 2) or (windows.shape[0] == 1):
            raise ValueError(f"Two or more windows must be specified. {windows=}")

        if (tmin is not None) or (tmax is not None):
            raise ValueError(
                "tmin and tmax cannot be specified when windows are specified."
            )

        X_list = []
        for idx, window in enumerate(windows):
            # get window length as number of sample from the first window
            # use the same number of samples for the later windows
            if idx == 0:
                Nt = data.time_as_index(window[1])[0] - data.time_as_index(window[0])[0]

            X = func(
                data=data,
                tmin=window[0],
                tmax=window[1] + (1 / data.info["sfreq"]),
                fmin=fmin,
                fmax=fmax,
                picks=picks,
            )
            X = X[:, 0:Nt]

            X_list.append(X)

        return np.array(X_list)

    if (tmin is not None) or (tmax is not None):
        if (tmin is None) or (tmax is None):
            raise ValueError(
                "tmin and tmax must be specified when windows are not specified."
            )

        return func(data=data, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)


def _apply_baseline(data, baseline):
    type_data = _get_data_type(data)

    accept_types = ["EpochsTFR", "Epochs", "AverageTFR", "Evoked"]

    if not type_data in accept_types:
        raise ValueError(f"data must be an instance of {accept_types}. {type_data=}")

    if type_data in ["EpochsTFR", "AverageTFR"]:
        data = data.apply_baseline(baseline, "percent")
    elif type_data == ["Epochs", "Evoked"]:
        data = data.apply_baseline(baseline)

    return data
