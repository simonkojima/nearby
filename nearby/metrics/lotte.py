import numpy as np
import pyriemann


def class_stability(
    epochs=None,
    covs=None,
    tmin=None,
    tmax=None,
    picks=None,
    centroid=None,
    estimator="scm",
):
    """
    Compute a Riemannian geometry-based class stability score.

    This function measures how tightly a set of symmetric positive definite (SPD)
    covariance matrices cluster on the Riemannian manifold. It computes a centroid
    covariance matrix (by default, the Riemannian mean) and then measures the
    dispersion as the average Riemannian distance from each covariance matrix to the
    centroid. The dispersion is mapped to a stability score in ``(0, 1]`` via::

        stability = 1 / (1 + dispersion)

    where higher values indicate higher stability (i.e., less dispersion).

    Input can be provided either as:
    - ``covs``: precomputed covariance matrices, or
    - ``epochs``: an :class:`mne.epochs.Epochs` object from which covariance matrices
      are estimated using PyRiemann.

    Parameters
    ----------
    epochs : mne.epochs.Epochs | None, optional
        Epoched EEG data used to estimate covariance matrices when ``covs`` is
        ``None``. Covariances are computed from::

            epochs.get_data(units="uV", tmin=tmin, tmax=tmax, picks=picks)

        using :class:`pyriemann.estimation.Covariances` with ``estimator=estimator``.

    covs : array_like of shape (n_matrices, n_channels, n_channels) | None, optional
        Collection of covariance matrices. Must be 3D and square in the last two
        dimensions. Matrices are assumed to be SPD.

    tmin, tmax : float | None, optional
        Time bounds (seconds) used when extracting epoch data from ``epochs`` (passed
        to :meth:`mne.Epochs.get_data`). Ignored when ``covs`` is provided.

    picks : str | array_like of int | array_like of str | None, optional
        Channels to include when extracting data from ``epochs``. Passed to
        :meth:`mne.Epochs.get_data`. Ignored when ``covs`` is provided.

    centroid : ndarray of shape (n_channels, n_channels) | None, optional
        Centroid covariance matrix. If ``None``, the centroid is computed as the
        Riemannian mean using :func:`pyriemann.utils.mean.mean_covariance` with
        ``metric="riemann"``. If an array is provided, it is used directly.

    estimator : str, default="scm"
        Covariance estimator passed to :class:`pyriemann.estimation.Covariances` when
        computing covariances from ``epochs`` (e.g., ``"scm"``, ``"oas"``, ``"lwf"``).

    Returns
    -------
    stability : float
        Class stability score in ``(0, 1]`` (higher means more stable / less
        dispersed).

    Notes
    -----
    - The dispersion is computed as the mean of Riemannian distances obtained via
      :func:`pyriemann.utils.distance.distance` with ``metric="riemann"``.
    - If both ``epochs`` and ``covs`` are provided, ``covs`` is used and ``epochs`` is
      ignored (current behavior follows the implementation logic).

    References
    ----------
    .. [1] F. Lotte & C. Jeunet, "Defining and quantifying users' mental imagery-based BCI skills: a first step",
       Journal of Neural Engineering. DOI: `10.1088/1741-2552/aac577 <https://doi.org/10.1088/1741-2552/aac577>`_
    """

    if (epochs is None) and (covs is None):
        raise ValueError("Either epochs or covs must be provided.")

    if (epochs is not None) and (covs is None):
        covs = pyriemann.estimation.Covariances(estimator=estimator).fit_transform(
            epochs.get_data(units="uV", tmin=tmin, tmax=tmax, picks=picks)
        )

    covs = np.array(covs)

    if covs.ndim != 3:
        raise ValueError("covs must be a 3D array")

    if covs.shape[1] != covs.shape[2]:
        raise ValueError("covs must be a square matrix")

    if centroid is None:
        M = pyriemann.utils.mean.mean_covariance(covs, metric="riemann")
    else:
        M = centroid

    dispersion = []
    for m in range(covs.shape[0]):
        d = pyriemann.utils.distance.distance(
            M, covs[m, :, :], metric="riemann", squared=False
        )
        dispersion.append(d)

    dispersion = np.mean(dispersion)

    stability = 1 / (1 + dispersion)

    return stability
