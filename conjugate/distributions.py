"""These are the supported distributions based on the conjugate models.

Many have the `dist` attribute which is a <a href=https://docs.scipy.org/doc/scipy/reference/stats.html>scipy.stats distribution</a> object. From there, 
you can use the methods from scipy.stats to get the pdf, cdf, etc.

Distributions can be plotted using the `plot_pmf` or `plot_pdf` methods of the distribution.

```python 
from conjugate.distribution import Beta 

beta = Beta(1, 1)
scipy_dist = beta.dist 

print(scipy_dist.mean())
# 0.5
print(scipy_dist.ppf([0.025, 0.975]))
# [0.025 0.975]

samples = scipy_dist.rvs(100)

beta.plot_pmf(label="beta distribution")
```

Distributions like Poisson can be added with other Poissons or multiplied by numerical values in order to scale rate. For instance, 

```python 
daily_rate = 0.25
daily_pois = Poisson(lam=daily_rate)

two_day_pois = daily_pois + daily_pois
weekly_pois = 7 * daily_pois
```

Below are the currently supported distributions

"""
from dataclasses import dataclass
from typing import Any, Tuple, Union

import numpy as np

from scipy import stats

from conjugate._typing import NUMERIC
from conjugate.plot import (
    DirichletPlotDistMixin,
    DiscretePlotMixin,
    ContinuousPlotDistMixin,
)
from conjugate.slice import SliceMixin


def get_beta_param_from_mean_and_alpha(
    mean: NUMERIC, alpha: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    beta = alpha * ((1 / mean) - 1)

    return alpha, beta


@dataclass
class Beta(ContinuousPlotDistMixin, SliceMixin):
    """Beta distribution.

    Args:
        alpha: shape parameter
        beta: shape parameter

    """

    alpha: NUMERIC
    beta: NUMERIC

    def __post_init__(self) -> None:
        self.max_value = 1.0

    @classmethod
    def from_mean(cls, mean: float, alpha: float) -> "Beta":
        """Alternative constructor from mean and alpha."""
        beta = get_beta_param_from_mean_and_alpha(mean=mean, alpha=alpha)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_successes_and_failures(cls, successes: int, failures: int) -> "Beta":
        """Alternative constructor based on hyperparameter interpretation."""
        alpha = successes + 1
        beta = failures + 1
        return cls(alpha=alpha, beta=beta)

    @property
    def dist(self):
        return stats.beta(self.alpha, self.beta)


@dataclass
class Binomial(DiscretePlotMixin, SliceMixin):
    """Binomial distribution.

    Args:
        n: number of trials
        p: probability of success

    """

    n: NUMERIC
    p: NUMERIC

    def __post_init__(self):
        if isinstance(self.n, np.ndarray):
            self.max_value = self.n.max()
        else:
            self.max_value = self.n

    @property
    def dist(self):
        return stats.binom(n=self.n, p=self.p)


class VectorizedDist:
    def __init__(self, params: np.ndarray, dist: Any):
        self.params = params
        self.dist = dist

    def rvs(self, size: int = 1) -> np.ndarray:
        return np.stack([self.dist(param).rvs(size=size) for param in self.params])

    def mean(self) -> np.ndarray:
        return np.stack([self.dist(param).mean() for param in self.params])


@dataclass
class Dirichlet(DirichletPlotDistMixin):
    """Dirichlet distribution.

    Args:
        alpha: shape parameter

    """

    alpha: NUMERIC

    def __post_init__(self) -> None:
        self.max_value = 1.0

    @property
    def dist(self):
        if self.alpha.ndim == 1:
            return stats.dirichlet(self.alpha)

        return VectorizedDist(self.alpha, dist=stats.dirichlet)


@dataclass
class Exponential(ContinuousPlotDistMixin, SliceMixin):
    """Exponential distribution.

    Args:
        lam: rate parameter

    """

    lam: NUMERIC

    @property
    def dist(self):
        return stats.expon(scale=self.lam)

    def __mul__(self, other):
        return Gamma(alpha=other, beta=1 / self.lam)

    __rmul__ = __mul__


@dataclass
class Gamma(ContinuousPlotDistMixin, SliceMixin):
    """Gamma distribution.

    <a href=https://en.wikipedia.org/wiki/Gamma_distribution>Gamma Distribution</a>
    <a href=https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.gamma.html>Scipy Docmentation</a>

    Args:
        alpha: shape parameter
        beta: rate parameter
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
    """Negative binomial distribution.

    Args:
        n: number of successes
        p: probability of success

    """

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
    """Poisson distribution.

    Args:
        lam: rate parameter

    """

    lam: NUMERIC

    @property
    def dist(self):
        return stats.poisson(self.lam)

    def __mul__(self, other) -> "Poisson":
        return Poisson(lam=self.lam * other)

    __rmul__ = __mul__

    def __add__(self, other) -> "Poisson":
        return Poisson(self.lam + other.lam)

    __radd__ = __add__


@dataclass
class BetaBinomial(DiscretePlotMixin, SliceMixin):
    """Beta binomial distribution.

    Args:
        n: number of trials
        alpha: shape parameter
        beta: shape parameter

    """

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
class BetaNegativeBinomial(SliceMixin):
    """Beta negative binomial distribution.

    Args:
        n: number of successes
        alpha: shape parameter


    """

    n: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC


@dataclass
class Geometric(DiscretePlotMixin, SliceMixin):
    """Geometric distribution.

    Args:
        p: probability of success

    """

    p: NUMERIC

    @property
    def dist(self):
        return stats.geom(self.p)


@dataclass
class Normal(ContinuousPlotDistMixin, SliceMixin):
    """Normal distribution.

    Args:
        mu: mean
        sigma: standard deviation

    """

    mu: NUMERIC
    sigma: NUMERIC

    @property
    def dist(self):
        return stats.norm(self.mu, self.sigma)

    def __mul__(self, other):
        sigma = ((self.sigma**2) * other) ** 0.5
        return Normal(mu=self.mu * other, sigma=sigma)

    __rmul__ = __mul__


@dataclass
class Uniform(ContinuousPlotDistMixin, SliceMixin):
    """Uniform distribution.

    Args:
        low: lower bound
        high: upper bound

    """

    low: NUMERIC
    high: NUMERIC

    def __post_init__(self):
        self.min_value = self.low
        self.max_value = self.high

    @property
    def dist(self):
        return stats.uniform(self.low, self.high)


@dataclass
class NormalInverseGamma:
    """Normal inverse gamma distribution."""

    mu: NUMERIC
    delta_inverse: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC

    def sample_sigma(self, size: int) -> NUMERIC:
        """Sample sigma from the inverse gamma distribution.

        Args:
            size: number of samples

        Returns:
            sigma: samples from the inverse gamma distribution

        """
        return 1 / stats.gamma(a=self.alpha, scale=1 / self.beta).rvs(size=size)

    def sample_beta(
        self, size: int, return_sigma: bool = False
    ) -> Union[NUMERIC, Tuple[NUMERIC, NUMERIC]]:
        """Sample beta from the normal distribution.

        Args:
            size: number of samples
            return_sigma: whether to return sigma as well

        Returns:
            samples from the normal distribution and optionally sigma

        """
        sigma = self.sample_sigma(size=size)

        beta = np.stack(
            [
                stats.multivariate_normal(self.mu, s * self.delta_inverse).rvs(size=1)
                for s in sigma
            ]
        )

        if return_sigma:
            return beta, sigma

        return beta
