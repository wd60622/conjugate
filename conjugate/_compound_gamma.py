import numpy as np

from scipy.special import beta
from scipy.stats import gamma


def compound_gamma_pdf(x, a, b, q):
    return (x / q) ** (a - 1) * (1 + (x / q)) ** (-a - b) / (q * beta(a, b))


class compound_gamma:
    """Implementation to work like scipy distribution classes.

    Reference:
    https://en.wikipedia.org/wiki/Beta_prime_distribution#Generalization
    """

    def __init__(self, a, b, q):
        self.a = a
        self.b = b
        self.q = q

    def rvs(
        self,
        size: int | None = None,
        random_state: np.random.Generator | None = None,
    ):
        a, b, q = np.broadcast_arrays(self.a, self.b, self.q)
        size = np.broadcast_shapes(a.shape, b.shape, q.shape, size or ())

        random_b = gamma.rvs(b, scale=1 / q, size=size, random_state=random_state)
        return gamma.rvs(a, scale=1 / random_b, random_state=random_state)

    def pdf(self, x):
        return compound_gamma_pdf(x, self.a, self.b, self.q)
