"""Implementation of the Beta-Geometric distribution."""

import numpy as np

from scipy import stats
from scipy.special import beta


def beta_geometric_pmf(x, a, b):
    x, a, b = np.broadcast_arrays(x, a, b)
    return beta(a + 1, x + b) / beta(a, b)


class beta_geometric:
    """Implementation to work like scipy distribution classes.

    References:
        https://arxiv.org/pdf/1405.6392

    """

    def __init__(self, a, b):
        self.a = a
        self.b = b

    def rvs(self, size: int, random_state=None):
        a, b = np.broadcast_arrays(self.a, self.b)
        size = np.broadcast_shapes(a.shape, b.shape, size)

        p = stats.beta.rvs(a, b, size=size, random_state=random_state)
        return stats.geom.rvs(p=p, size=size, random_state=random_state)

    def pmf(self, x):
        return beta_geometric_pmf(x, self.a, self.b)
