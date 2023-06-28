from dataclasses import dataclass
from typing import Any, Tuple

import numpy as np

from scipy import stats

from conjugate._typing import NUMERIC
from conjugate.plot import (
    DiscretePlotMixin,
    ContinuousPlotDistMixin,
    SamplePlotDistMixin,
)
from conjugate.slice import SliceMixin


def get_beta_param_from_mean_and_alpha(
    mean: NUMERIC, alpha: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    beta = alpha * ((1 / mean) - 1)

    return alpha, beta


@dataclass
class Beta(ContinuousPlotDistMixin, SliceMixin):
    """Part of the CLV core assumption for a and b."""

    alpha: NUMERIC
    beta: NUMERIC

    def __post_init__(self) -> None:
        self.max_value = 1.0

    @classmethod
    def from_mean(cls, mean: float, alpha: float) -> "Beta":
        """Alternative constructor."""
        beta = get_beta_param_from_mean_and_alpha(mean=mean, alpha=alpha)
        return cls(alpha=alpha, beta=beta)

    @classmethod 
    def from_successes_and_failures(cls, successes: int, failures: int) -> "Beta":
        """Alternative constructor."""
        alpha = successes + 1
        beta = failures + 1
        return cls(alpha=alpha, beta=beta)

    @property
    def dist(self):
        return stats.beta(self.alpha, self.beta)


class VectorizedDist:
    def __init__(self, params: np.ndarray, dist: Any):
        self.params = params
        self.dist = dist

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.stack([self.dist(param).rvs(size=size) for param in self.params])

    def mean(self) -> np.ndarray:
        return np.stack([self.dist(param).mean() for param in self.params])


@dataclass
class Dirichlet(SamplePlotDistMixin):
    alpha: np.ndarray

    def __post_init__(self) -> None:
        self.max_value = 1.0

    @property
    def dist(self):
        if self.alpha.ndim == 1:
            return stats.dirichlet(self.alpha)

        return VectorizedDist(self.alpha, dist=stats.dirichlet)


@dataclass
class Exponential(ContinuousPlotDistMixin, SliceMixin):
    """
    Implementation from:
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.expon.html

    """

    lam: NUMERIC

    @property
    def dist(self):
        return stats.expon(scale=self.lam)


@dataclass
class Gamma(ContinuousPlotDistMixin, SliceMixin):
    """
    https://en.wikipedia.org/wiki/Gamma_distribution
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html

    """

    alpha: NUMERIC
    beta: NUMERIC

    @property
    def dist(self):
        return stats.gamma(a=self.alpha, scale=1 / self.beta)

    def __mul__(self, other):
        return Gamma(alpha=self.alpha * other, beta=self.beta)

    __rmul__ = __mul__


@dataclass
class NegativeBinomial(DiscretePlotMixin, SliceMixin):
    n: NUMERIC
    p: NUMERIC

    @property
    def dist(self):
        return stats.nbinom(n=self.n, p=self.p)

    def __mul__(self, other):
        return NegativeBinomial(n=self.n * other, p=self.p)

    __rmul__ = __mul__


@dataclass
class Poisson(DiscretePlotMixin, SliceMixin):
    lam: NUMERIC

    @property
    def dist(self):
        return stats.poisson(self.lam)

    def __mul__(self, other):
        return Poisson(lam=self.lam * other)

    __rmul__ = __mul__


@dataclass
class BetaBinomial(DiscretePlotMixin, SliceMixin):
    n: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC

    def __post_init__(self):
        if isinstance(self.n, np.ndarray):
            self.max_value = self.n.max()
        else:
            self.max_value = self.n

    @property
    def dist(self):
        return stats.betabinom(self.n, self.alpha, self.beta)


@dataclass
class BetaNegativeBinomial(DiscretePlotMixin, SliceMixin):
    n: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC