import numpy as np
import autograd.numpy as anp
import pymanopt


def mean_extrinsic(X):
    """
    Compute the **extrinsic mean direction** of a set of vectors.

    This function normalizes each vector in ``X`` to unit length, sums the
    resulting unit vectors, and normalizes the sum to obtain the extrinsic
    mean direction.

    The computation follows:

    .. math::

        S = \\sum_{i=1}^{N} \\frac{X_i}{|X_i|}

    .. math::

        \mu = \\frac{S}{|S|}

    Parameters
    ----------
    X : array_like, shape (n_samples, n_dims)
        Input vectors. Each row corresponds to a vector.

    Returns
    -------
    ndarray, shape (n_dims,)
        The extrinsic mean direction (unit vector).

    Notes
    -----
    The extrinsic mean is commonly used in directional statistics
    (e.g., unit-sphere embeddings, mean direction on :math:`S^{d-1}`).
    """
    norms = np.linalg.norm(X, axis=1, keepdims=True)

    U = X / norms

    S = U.sum(axis=0)
    s = np.linalg.norm(S)

    return S / s


def get_random_unit_vec(d, rng=None):
    rng = np.random.default_rng(rng)
    v = rng.normal(size=d)
    n = np.linalg.norm(v)
    if n == 0:
        return get_random_unit_vec(d, rng)
    return v / n


def mean_frechet_p2(X, init="extrinsic", seed=42, return_all=False, **kwargs):
    """
    Compute the Fréchet mean on the unit hypersphere with the :math:`p = 2`
    squared angular distance (intrinsic mean).

    Optimization is performed on the unit sphere :math:`S^{d-1}` using
    ``pymanopt`` with a steepest descent solver.

    Parameters
    ----------
    X : array_like, shape (n_samples, n_dims)
        Input vectors. They will be normalized to unit length internally.

    init : {"extrinsic", "random", array_like}, optional
        Initialization method for the Fréchet mean.

        - ``"extrinsic"`` : use the extrinsic mean direction.
        - ``"random"`` : sample a random unit vector as initialization.
        - ndarray : directly specify an initial vector (will be normalized).

    seed : int, optional
        Random seed used when ``init="random"``.

    return_all : bool, optional
        If ``True``, return the full optimizer result object from pymanopt.
        Otherwise, only the estimated Fréchet mean point is returned.

    **kwargs :
        Additional keyword arguments passed to
        ``pymanopt.optimizers.SteepestDescent``.

    Returns
    -------
    ndarray or pymanopt.OptimizeResult
        - If ``return_all=False``: the Fréchet mean (unit vector).
        - If ``return_all=True``: the full result returned by pymanopt.

    Notes
    -----
    - The function computes the *intrinsic* mean on the sphere using the
      geodesic distance.
    - If vectors in ``X`` are not unit length, they are normalized internally.
    - The objective is smooth everywhere except for antipodal configurations.
    """
    X = np.array(X)

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    n_vecs, n_dim = X.shape

    if init == "extrinsic":
        initial_point = mean_extrinsic(X)
    elif init == "random":
        rng = np.random.default_rng(seed)
        initial_point = get_random_unit_vec(d=n_dim, rng=rng)
    else:
        initial_point = np.array(init)
        initial_point /= np.linalg.norm(initial_point)

    manifold = pymanopt.manifolds.Sphere(n_dim)

    @pymanopt.function.autograd(manifold)
    def cost(point):
        f = anp.arccos(X @ point)
        f = anp.power(f, 2)
        f = anp.sum(f)
        return f

    problem = pymanopt.Problem(manifold, cost)

    optimizer = pymanopt.optimizers.SteepestDescent(**kwargs)
    result = optimizer.run(problem, initial_point=initial_point)

    if return_all:
        return result
    else:
        return result.point
