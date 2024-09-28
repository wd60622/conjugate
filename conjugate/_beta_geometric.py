"""Implementation of the Beta-Geometric distribution."""

import numpy as np

from scipy import stats
from scipy.special import beta


def beta_geometric_pmf(x, a, b, one_start: bool = True):
    x, a, b = np.broadcast_arrays(x, a, b)

    if not one_start:
        x = x + 1

    return beta(a + 1, x + b) / beta(a, b)


class beta_geometric:
    """Implementation to work like scipy distribution classes.

    References:
        https://arxiv.org/pdf/1405.6392

    """

    def __init__(self, a, b, one_start: bool = True):
        self.a = a
        self.b = b
        self.one_start = one_start

    def rvs(
        self,
        size: int | None = None,
        random_state: np.random.Generator | None = None,
    ):
        a, b = np.broadcast_arrays(self.a, self.b)
        size = np.broadcast_shapes(a.shape, b.shape, size or ())

        p = stats.beta.rvs(a, b, size=size, random_state=random_state)
        result = stats.geom.rvs(p=p, size=size, random_state=random_state)

        if not self.one_start:
            result -= 1

        return result

    def pmf(self, x):
        return beta_geometric_pmf(x, self.a, self.b, one_start=self.one_start)
