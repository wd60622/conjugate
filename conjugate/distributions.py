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

from packaging import version

import numpy as np

from scipy import stats, __version__ as scipy_version

from conjugate._typing import NUMERIC
from conjugate.plot import (
    DirichletPlotDistMixin,
    DiscretePlotMixin,
    ContinuousPlotDistMixin,
)
from conjugate.slice import SliceMixin


def get_beta_param_from_mean_and_alpha(mean: NUMERIC, alpha: NUMERIC) -> NUMERIC:
    beta = alpha * ((1 / mean) - 1)

    return beta


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
    def from_mean(cls, mean: NUMERIC, alpha: NUMERIC) -> "Beta":
        """Alternative constructor from mean and alpha."""
        beta = get_beta_param_from_mean_and_alpha(mean=mean, alpha=alpha)
        return cls(alpha=alpha, beta=beta)

    @classmethod
    def from_successes_and_failures(
        cls, successes: NUMERIC, failures: NUMERIC
    ) -> "Beta":
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
class Multinomial(SliceMixin):
    """Multinomial distribution.

    Args:
        n: number of trials
        p: probability of success

    """

    n: NUMERIC
    p: NUMERIC

    @property
    def dist(self):
        return stats.multinomial(n=self.n, p=self.p)


@dataclass
class DirichletMultinomial(SliceMixin):
    """Dirichlet multinomial distribution.

    Args:
        alpha: shape parameter
        n: number of trials

    """

    alpha: NUMERIC
    n: NUMERIC

    @property
    def dist(self):
        return stats.dirichlet_multinomial(self.alpha, self.n)


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
class Hypergeometric(DiscretePlotMixin, SliceMixin):
    """Hypergeometric distribution.

    Args:
        N: population size
        k: number of successes in the population
        n: number of draws

    """

    N: NUMERIC
    k: NUMERIC
    n: NUMERIC

    def __post_init__(self) -> None:
        if isinstance(self.N, np.ndarray):
            self.max_value = self.N.max()
        else:
            self.max_value = self.N

    @property
    def dist(self):
        return stats.hypergeom(self.N, self.k, self.n)


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
class BetaNegativeBinomial(DiscretePlotMixin, SliceMixin):
    """Beta negative binomial distribution.

    Args:
        n: number of successes
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
        if version.parse(scipy_version).release < version.parse("1.12.0").release:
            msg = "BetaNegativeBinomial.dist requires scipy >= 1.12.0"
            raise NotImplementedError(msg)

        return stats.betanbinom(self.n, self.alpha, self.beta)


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

    @classmethod
    def from_mean_and_variance(cls, mean: NUMERIC, variance: NUMERIC) -> "Normal":
        """Alternative constructor from mean and variance."""
        return cls(mu=mean, sigma=variance**0.5)

    @classmethod
    def from_mean_and_precision(cls, mean: NUMERIC, precision: NUMERIC) -> "Normal":
        """Alternative constructor from mean and precision."""
        return cls(mu=mean, sigma=precision**-0.5)

    def __mul__(self, other):
        sigma = ((self.sigma**2) * other) ** 0.5
        return Normal(mu=self.mu * other, sigma=sigma)

    __rmul__ = __mul__


@dataclass
class MultivariateNormal:
    mu: NUMERIC
    sigma: NUMERIC

    @property
    def dist(self):
        return stats.multivariate_normal(mean=self.mu, cov=self.sigma)


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
class Pareto(ContinuousPlotDistMixin, SliceMixin):
    """Pareto distribution.

    Args:
        x_m: minimum value
        alpha: scale parameter

    """

    x_m: NUMERIC
    alpha: NUMERIC

    @property
    def dist(self):
        return stats.pareto(self.alpha, scale=self.x_m)


@dataclass
class InverseGamma(ContinuousPlotDistMixin, SliceMixin):
    """InverseGamma distribution.

    Args:
        alpha: shape
        beta: scale

    """

    alpha: NUMERIC
    beta: NUMERIC

    @property
    def dist(self):
        return stats.invgamma(a=self.alpha, scale=self.beta)


@dataclass
class NormalInverseGamma:
    """Normal inverse gamma distribution.

    Supports both 1 dimensional and multivariate cases.

    Args:
        mu: mean
        alpha: shape
        beta: scale
        delta_inverse: covariance matrix, 2d array for multivariate case
        nu: alternative precision parameter for 1 dimensional case

    """

    mu: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC
    delta_inverse: NUMERIC = None
    nu: NUMERIC = None

    def __post_init__(self) -> None:
        if self.delta_inverse is None and self.nu is None:
            raise ValueError("Either delta_inverse or nu must be provided.")

        if self.delta_inverse is not None and self.nu is not None:
            raise ValueError("Only one of delta_inverse or nu must be provided.")

    @classmethod
    def from_inverse_gamma(
        cls,
        mu: NUMERIC,
        inverse_gamma: InverseGamma,
        delta_inverse: NUMERIC = None,
        nu: NUMERIC = None,
    ) -> "NormalInverseGamma":
        return cls(
            mu=mu,
            alpha=inverse_gamma.alpha,
            beta=inverse_gamma.beta,
            delta_inverse=delta_inverse,
            nu=nu,
        )

    @property
    def inverse_gamma(self) -> InverseGamma:
        return InverseGamma(alpha=self.alpha, beta=self.beta)

    def sample_variance(self, size: int, random_state=None) -> NUMERIC:
        """Sample variance from the inverse gamma distribution.

        Args:
            size: number of samples
            random_state: random state

        Returns:
            samples from the inverse gamma distribution

        """
        return self.inverse_gamma.dist.rvs(size=size, random_state=random_state)

    def _sample_beta_1d(self, variance, size: int, random_state=None) -> NUMERIC:
        sigma = (variance / self.nu) ** 0.5
        return stats.norm(self.mu, sigma).rvs(size=size, random_state=random_state)

    def _sample_beta_nd(self, variance, size: int, random_state=None) -> NUMERIC:
        return np.stack(
            [
                stats.multivariate_normal(self.mu, v * self.delta_inverse).rvs(
                    size=1, random_state=random_state
                )
                for v in variance
            ]
        )

    def sample_beta(
        self, size: int, return_variance: bool = False, random_state=None
    ) -> Union[NUMERIC, Tuple[NUMERIC, NUMERIC]]:
        """Sample beta from the normal distribution.

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution and optionally variance

        """
        variance = self.sample_variance(size=size, random_state=random_state)

        sample_beta = (
            self._sample_beta_1d if self.delta_inverse is None else self._sample_beta_nd
        )
        beta = sample_beta(variance=variance, size=size, random_state=random_state)

        if return_variance:
            return beta, variance

        return beta


@dataclass
class StudentT(ContinuousPlotDistMixin, SliceMixin):
    """StudentT distribution.

    Args:
        mu: mean
        sigma: standard deviation
        nu: degrees of freedom

    """

    mu: NUMERIC
    sigma: NUMERIC
    nu: NUMERIC

    @property
    def dist(self):
        return stats.t(self.nu, self.mu, self.sigma)


@dataclass
class MultivariateStudentT:
    """MultivariateStudentT distribution.

    Args:
        mu: mean
        sigma: covariance matrix
        nu: degrees of freedom

    """

    mu: NUMERIC
    sigma: NUMERIC
    nu: NUMERIC

    @property
    def dist(self):
        return stats.multivariate_t(loc=self.mu, shape=self.sigma, df=self.nu)


@dataclass
class Lomax(ContinuousPlotDistMixin, SliceMixin):
    """Lomax distribution.

    Args:
        alpha: shape
        lam: scale

    """

    alpha: NUMERIC
    lam: NUMERIC

    @property
    def dist(self):
        return stats.lomax(c=self.alpha, scale=self.lam)
