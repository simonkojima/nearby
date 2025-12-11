import numpy as np


def distance_angle(w1, w2):
    """
    Compute the angle (in radians) between two vectors.

    The function returns the angle :math:`\\theta` between vectors ``w1`` and ``w2`` using:

    .. math::

        \mathrm{arccos}\\left(\\frac{w_1 \\cdot w_2}{||w_1|| \\cdot ||w_2||} \\right)

    Parameters
    ----------
    w1 : array_like
        First vector.
    w2 : array_like
        Second vector.

    Returns
    -------
    float
        Angle between ``w1`` and ``w2`` in radians.

    Notes
    -----
    The function computes the cosine similarity between the two vectors
    and applies :func:`numpy.arccos` to obtain the angle.

    Examples
    --------
    >>> import numpy as np
    >>> distance_angle(np.array([1, 0]), np.array([0, 1]))
    1.5707963267948966
    """
    return np.arccos(np.dot(w1, w2) / (np.linalg.norm(w1) * np.linalg.norm(w2)))
