from nearby.utils.proc import _get_data
import conftest


def test_get_data_epochs_tfr():
    data = conftest.dummy_epochs_tfr(30)
    x = _get_data(data, domain="temporal", tmin=1, tmax=3, fmin=7, fmax=13, picks="C4")
    assert (x.shape == (30, 201)) or (x.shape == (30, 200))

    x = _get_data(
        data,
        domain="spatial",
        tmin=1,
        tmax=3,
        fmin=7,
        fmax=13,
        picks=["C3", "Cz", "C4"],
    )
    assert x.shape == (30, 3)

    x = _get_data(data, domain="frequency", tmin=1, tmax=3, fmin=7, fmax=13, picks="C4")
    assert x.shape == (30, 7)


def test_get_data_average_tfr():
    data = conftest.dummy_average_tfr()

    x = _get_data(data, domain="temporal", tmin=1, tmax=3, fmin=7, fmax=13, picks="C4")
    assert (x.shape == (201,)) or (x.shape == (200,))

    x = _get_data(
        data,
        domain="spatial",
        tmin=1,
        tmax=3,
        fmin=7,
        fmax=13,
        picks=["C3", "Cz", "C4"],
    )
    assert x.shape == (3,)

    x = _get_data(data, domain="frequency", tmin=1, tmax=3, fmin=7, fmax=13, picks="C4")
    assert x.shape == (7,)


def test_get_data_epochs():
    data = conftest.dummy_epochs(30)

    x = _get_data(
        data, domain="temporal", tmin=1, tmax=3, fmin=None, fmax=None, picks="C4"
    )
    assert (x.shape == (30, 201)) or (x.shape == (30, 200))

    x = _get_data(
        data,
        domain="spatial",
        tmin=1,
        tmax=3,
        fmin=7,
        fmax=13,
        picks=["C3", "Cz", "C4"],
    )
    assert x.shape == (30, 3)


def test_get_data_evoked():
    data = conftest.dummy_evoked()

    x = _get_data(
        data, domain="temporal", tmin=1, tmax=3, fmin=None, fmax=None, picks="C4"
    )
    assert (x.shape == (201,)) or (x.shape == (200,))

    x = _get_data(
        data,
        domain="spatial",
        tmin=1,
        tmax=3,
        fmin=7,
        fmax=13,
        picks=["C3", "Cz", "C4"],
    )
    assert x.shape == (3,)
