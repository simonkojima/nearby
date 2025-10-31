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


def mean_frechet_p2(X,
                    init="extrinsic",
                    seed=42,
                    return_all=False,
                    **kwargs):
    try:
        import autograd.numpy as anp
        import pymanopt
    except:
        raise RuntimeError("autograd and pymanopt are not installed. pip install autograd pymanopt[autograd]")

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


def mean_frechet_p2_old(X,
                        max_iter=1000,
                        lr=2.0,
                        alpha=1e-4,
                        beta=0.5,
                        tol=1e-8,
                        eps=1e-12,
                        init="extrinsic",
                        seed=42):
    X = np.array(X)

    X = X / np.linalg.norm(X, axis=1, keepdims=True)

    if init == "extrinsic":
        m = mean_extrinsic(X)
    elif init == "random":
        rng = np.random.default_rng(seed)
        m = get_random_unit_vec(d=X.shape[1], rng=rng)
    else:
        m = np.array(init)
        m /= np.linalg.norm(m)

    # m = X.mean(axis=0)
    # m /= np.linalg.norm(m)
    # print(m)

    def f_val(m):
        # ci = np.clip(X @ m, -1.0 + eps, 1.0 - eps)
        ci = X @ m
        return np.sum(np.arccos(ci) ** 2)

    f_prev = f_val(m)

    for _ in range(max_iter):

        # ci = np.clip(X @ m, -1.0 + eps, 1.0 - eps)
        ci = X @ m
        theta = np.arccos(ci)
        den = np.sqrt(np.maximum(eps, 1.0 - ci ** 2))

        Ui = (X - np.outer(ci, m)) / den[:, None]

        grad = -((2.0 * theta)[:, None] * Ui).sum(axis=0)

        g2 = np.dot(grad, grad)
        if g2 < tol ** 2:
            break

        eta = lr
        while True:
            m_cand = m - eta * grad
            m_cand /= np.linalg.norm(m_cand)

            f_cand = f_val(m_cand)
            if f_cand <= f_prev - alpha * eta * g2:
                m, f_prev = m_cand, f_cand
                # lr = max(lr, eta)
                break
            eta *= beta
            if eta < 1e-12:
                return m

    else:
        warnings.warn("Convergence not reached")

    return m
