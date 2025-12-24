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

    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks)
    # X.shape: n_epochs, n_channels, n_times

    X = X.mean(axis=1)
    # X.shape: n_epochs, n_times

    return X


def _get_spatial_epochs(data, tmin, tmax, fmin, fmax, picks):
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks)
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
    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks)
    # X.shape: n_channels, n_times

    X = X.mean(axis=0)
    # X.shape: n_times

    return X


def _get_spatial_evoked(data, tmin, tmax, fmin, fmax, picks):

    X = data.get_data(tmin=tmin, tmax=tmax, picks=picks)
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


def _get_data(data, domain, tmin, tmax, fmin, fmax, picks):
    type_data = _get_data_type(data)

    try:
        func = _GETTERS[type_data][domain]
    except KeyError as e:
        raise ValueError(
            f"Unsupported combination: type={type_data}, domain={domain}"
        ) from e

    return func(data=data, tmin=tmin, tmax=tmax, fmin=fmin, fmax=fmax, picks=picks)
