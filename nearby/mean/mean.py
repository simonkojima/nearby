import numpy as np
import warnings


def mean_extrinsic(X):
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
    try:
        import autograd.numpy as anp
        import pymanopt
    except:
        raise RuntimeError(
            "autograd and pymanopt are not installed. pip install autograd pymanopt[autograd]"
        )

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
