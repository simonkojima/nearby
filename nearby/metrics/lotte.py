import numpy as np
import pyriemann
import pandas as pd

from ..utils import add_values_to_df


def class_stability(
    epochs=None,
    covs=None,
    additional_values=None,
    estimator="scm",
    tmin=None,
    tmax=None,
):
    """
    Compute a Riemannian geometry-based class stability score from covariance matrices.

    The input is a collection of SPD covariance matrices. The Riemannian (geometric)
    mean :math:`M` is computed, then the average Riemannian distance from each matrix
    to :math:`M` defines a dispersion value. The stability score is mapped to :math:`(0, 1]`
    via :math:`1/(1+\\text{dispersion})`.

    .. math::

        M = \\operatorname{Mean}_{\\text{Riemann}}(C_1, \\ldots, C_N)

    .. math::

        \\mathrm{dispersion}
        = \\frac{1}{N}\\sum_{i=1}^{N} d_{\\text{Riemann}}(M, C_i)

    .. math::

        \\mathrm{stability} = \\frac{1}{1 + \\mathrm{dispersion}}

    Parameters
    ----------
    covs : array_like, shape (n_matrices, n_channels, n_channels)
        A 3D array of covariance matrices (must be square in the last two axes).
        Matrices are assumed to be symmetric positive definite (SPD).
    additional_values : dict or pandas.Series or None, optional
        Extra values to append to the returned DataFrame via ``add_values_to_df``.
        This is useful for logging metadata (e.g., subject ID, session, condition, class).

    Returns
    -------
    pandas.DataFrame
        A single-row DataFrame with at least the column:

        - ``"class_stability"`` : float
            Stability score in :math:`(0, 1]` (higher means more stable / less dispersed).

        If ``additional_values`` is provided, extra columns are added.

    Raises
    ------
    ValueError
        If ``covs`` is not 3D, or if the last two dimensions are not square.

    Notes
    -----
    This implementation uses:

    - ``pyriemann.utils.mean.mean_covariance(..., metric="riemann")`` to compute the
      Riemannian mean of SPD matrices.
    - ``pyriemann.utils.distance.distance(..., metric="riemann")`` to compute
      Riemannian distances (non-squared).

    References
    ----------
    .. [1] F. Lotte & C. Jeunet, "Defining and quantifying users' mental imagery-based BCI skills: a first step",
       Journal of Neural Engineering. DOI: `10.1088/1741-2552/aac577 <https://doi.org/10.1088/1741-2552/aac577>`_
    """

    if (epochs is None) and (covs is None):
        raise ValueError("Either epochs or covs must be provided.")

    if (epochs is not None) and (covs is None):
        covs = pyriemann.estimation.Covariances(estimator=estimator).fit_transform(
            epochs.get_data(units="uV", tmin=tmin, tmax=tmax)
        )

    covs = np.array(covs)

    if covs.ndim != 3:
        raise ValueError("covs must be a 3D array")

    if covs.shape[1] != covs.shape[2]:
        raise ValueError("covs must be a square matrix")

    M = pyriemann.utils.mean.mean_covariance(covs, metric="riemann")

    dispersion = []
    for m in range(covs.shape[0]):
        d = pyriemann.utils.distance.distance(
            M, covs[m, :, :], metric="riemann", squared=False
        )
        dispersion.append(d)

    dispersion = np.mean(dispersion)

    stability = 1 / (1 + dispersion)

    results = {"class_stability": stability}
    df = pd.DataFrame(results, index=[0])

    if additional_values is not None:
        df = add_values_to_df(df, additional_values)

    return df
