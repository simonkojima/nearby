import numpy as np
import pyriemann
import pandas as pd

from ..utils.core import _add_values_to_df


def class_stability(
        epochs=None,
        covs=None,
        tmin=None,
        tmax=None,
        picks=None,
        centroid="mean",
        estimator="scm",
        additional_values=None,
):
    """
    Compute a Riemannian geometry-based class stability score.

    This function quantifies how tightly a set of symmetric positive definite (SPD)
    covariance matrices cluster on the Riemannian manifold. It first computes a
    centroid covariance matrix (by default, the Riemannian mean) and then computes
    the average Riemannian distance from each covariance matrix to the centroid. This
    average distance defines a dispersion value that is mapped to a stability score
    in ``(0, 1]`` via ``1 / (1 + dispersion)`` (higher means more stable / less dispersed).

    Input can be provided either as:

    - ``covs``: a precomputed set of covariance matrices, or
    - ``epochs``: an :class:`mne.epochs.Epochs` object from which covariance matrices
      are estimated using PyRiemann.

    Parameters
    ----------
    epochs : mne.epochs.Epochs | None, optional
        Epoched EEG data used to estimate covariance matrices when ``covs`` is ``None``.
        If provided, covariance matrices are computed from::

            epochs.get_data(units="uV", tmin=tmin, tmax=tmax)

        using :class:`pyriemann.estimation.Covariances` with ``estimator=estimator``.

    covs : array_like, shape (n_matrices, n_channels, n_channels) | None, optional
        A collection of covariance matrices. The array must be 3D and square in the
        last two dimensions. Matrices are assumed to be SPD.

    tmin, tmax : float | None, optional
        Time range of interest (in seconds) used when extracting epoch data from
        ``epochs`` (passed to :meth:`mne.Epochs.get_data`).

    centroid : {"mean"} | array_like, shape (n_channels, n_channels), default="mean"
        Definition of the centroid covariance matrix.

        - ``"mean"``: compute the Riemannian mean using
          :func:`pyriemann.utils.mean.mean_covariance` with ``metric="riemann"``.
        - If an array is provided, it is used directly as the centroid.

    estimator : str, default="scm"
        Covariance estimator passed to :class:`pyriemann.estimation.Covariances`
        when computing covariances from ``epochs`` (e.g., ``"scm"``, ``"oas"``,
        ``"lwf"``).

    additional_values : dict | pandas.Series | None, optional
        Additional columns to append to the returned dataframe (e.g., ``subject``,
        ``session``, ``condition``, ``class``). Added via ``add_values_to_df``.

    Returns
    -------
    df : pandas.DataFrame
        Single-row dataframe containing at least:

        - ``class_stability`` : float
            Stability score in ``(0, 1]`` (higher means more stable / less dispersed).

        If ``additional_values`` is provided, additional columns are added.

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

    if isinstance(centroid, str):
        match centroid:
            case "mean":
                M = pyriemann.utils.mean.mean_covariance(covs, metric="riemann")
            case _:
                raise NotImplementedError(f"centroid={centroid} is not implemented.")
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
