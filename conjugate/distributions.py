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
from typing import Any, Callable

from packaging import version

import numpy as np

from scipy import stats, __version__ as scipy_version
from scipy.special import gammaln, i0

from conjugate._compound_gamma import compound_gamma
from conjugate._beta_geometric import beta_geometric
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
    def uninformative(cls) -> "Beta":
        return cls(alpha=1, beta=1)

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
    """Vectorized distribution to handle scipy distributions that don't support vectorization."""

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

    @classmethod
    def uninformative(cls, n: int) -> "Dirichlet":
        return cls(alpha=np.ones(n))

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

    @classmethod
    def from_occurrences_in_intervals(cls, occurrences: NUMERIC, intervals: NUMERIC):
        return cls(alpha=occurrences, beta=intervals)

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
        one_start: whether to start at 1 or 0. Default is 1.

    """

    p: NUMERIC
    one_start: bool = True

    @property
    def dist(self):
        loc = 0 if self.one_start else -1
        return stats.geom(self.p, loc=loc)


@dataclass
class BetaGeometric(DiscretePlotMixin, SliceMixin):
    """Beta geometric distribution.

    Args:
        alpha: shape parameter
        beta: shape parameter
        one_start: whether to start at 1 or 0. Default is 1.

    """

    alpha: NUMERIC
    beta: NUMERIC
    one_start: bool = True

    @property
    def dist(self):
        return beta_geometric(self.alpha, self.beta, one_start=self.one_start)


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
    def uninformative(cls, sigma: NUMERIC = 1) -> "Normal":
        """Uninformative normal distribution."""
        return cls(mu=0, sigma=sigma)

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
    """Multivariate normal distribution.

    Args:
        mu: mean
        cov: covariance matrix

    """

    mu: NUMERIC
    cov: NUMERIC

    @property
    def dist(self):
        return stats.multivariate_normal(mean=self.mu, cov=self.cov)

    def __getitem__(self, key):
        if isinstance(key, int):
            return Normal(mu=self.mu[key], sigma=self.cov[key, key] ** 0.5)

        return MultivariateNormal(mu=self.mu[key], cov=self.cov[key][:, key])


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
    delta_inverse: NUMERIC | None = None
    nu: NUMERIC | None = None

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
        delta_inverse: NUMERIC | None = None,
        nu: NUMERIC | None = None,
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
        variance = (self.delta_inverse[None, ...].T * variance).T
        return np.stack(
            [
                stats.multivariate_normal(self.mu, v).rvs(
                    size=1, random_state=random_state
                )
                for v in variance
            ]
        )

    def sample_mean(
        self,
        size: int,
        return_variance: bool = False,
        random_state=None,
    ) -> NUMERIC | tuple[NUMERIC, NUMERIC]:
        """Sample the mean from the normal distribution.

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution and optionally variance

        """
        return self.sample_beta(
            size=size, return_variance=return_variance, random_state=random_state
        )

    def sample_beta(
        self, size: int, return_variance: bool = False, random_state=None
    ) -> NUMERIC | tuple[NUMERIC, NUMERIC]:
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

    def __getitem__(self, key):
        if isinstance(key, int):
            return StudentT(mu=self.mu[key], sigma=self.sigma[key, key], nu=self.nu)

        return MultivariateStudentT(
            mu=self.mu[key], sigma=self.sigma[key][:, key], nu=self.nu
        )


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


@dataclass
class CompoundGamma(ContinuousPlotDistMixin, SliceMixin):
    """Compound gamma distribution.

    Args:
        alpha: shape
        beta: scale
        lam: rate

    """

    alpha: NUMERIC
    beta: NUMERIC
    lam: NUMERIC

    @property
    def dist(self):
        return compound_gamma(a=self.alpha, b=self.beta, q=self.lam)


@dataclass
class GammaKnownRateProportional:
    """Gamma known rate proportional distribution.

    Args:
        a: prod of observations
        b: number of observations
        c: number of observations

    """

    a: NUMERIC
    b: NUMERIC
    c: NUMERIC

    def approx_log_likelihood(
        self, alpha: NUMERIC, beta: NUMERIC, ln=np.log, gammaln=gammaln
    ) -> NUMERIC:
        """Approximate log likelihood.

        Args:
            alpha: shape parameter
            beta: known rate parameter
            ln: log function
            gammaln: log gamma function

        Returns:
            log likelihood up to a constant

        """
        return (
            (alpha - 1) * ln(self.a)
            + alpha * self.c * ln(beta)
            - self.b * gammaln(alpha)
        )


@dataclass
class GammaProportional:
    """Gamma proportional distribution.

    Args:
        p: product of r observations
        q: sum of s observations
        r: number of observations for p
        s: number of observations for q

    """

    p: NUMERIC
    q: NUMERIC
    r: NUMERIC
    s: NUMERIC

    def approx_log_likelihood(
        self, alpha: NUMERIC, beta: NUMERIC, ln=np.log, gammaln=gammaln
    ) -> NUMERIC:
        """Approximate log likelihood.

        Args:
            alpha: shape parameter
            beta: rate parameter
            ln: log function
            gammaln: log gamma function

        Returns:
            log likelihood up to a constant

        """
        return (
            (alpha - 1) * ln(self.p)
            - self.q * beta
            - self.r * gammaln(alpha)
            + self.s * alpha * ln(beta)
        )


@dataclass
class BetaProportional:
    """Beta proportional distribution.

    Args:
        p: product of observations
        q: product of complements
        k: number of observations

    """

    p: NUMERIC
    q: NUMERIC
    k: NUMERIC

    def approx_log_likelihood(
        self, alpha: NUMERIC, beta: NUMERIC, ln=np.log, gammaln=gammaln
    ) -> NUMERIC:
        """Approximate log likelihood.

        Args:
            alpha: shape parameter
            beta: shape parameter
            ln: log function
            gammaln: log gamma function

        Returns:
            log likelihood up to a constant

        """
        return (
            self.k * gammaln(alpha + beta)
            + alpha * ln(self.p)
            + beta * ln(self.q)
            - self.k * gammaln(alpha)
            - self.k * gammaln(beta)
        )


@dataclass
class VonMises(ContinuousPlotDistMixin, SliceMixin):
    """Von Mises distribution.

    Args:
        mu: mean
        kappa: concentration

    """

    mu: NUMERIC
    kappa: NUMERIC

    def __post_init__(self) -> None:
        self.min_value = -np.pi
        self.max_value = np.pi

    @property
    def dist(self):
        return stats.vonmises(loc=self.mu, kappa=self.kappa)


@dataclass
class VonMisesKnownConcentration:
    """Von Mises known concentration distribution.

    Taken from <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 2.13.1</a>.

    Args:
        a: positive value
        b: value between 0 and 2 pi

    """

    a: NUMERIC
    b: NUMERIC

    def log_likelihood(self, mu: NUMERIC, cos=np.cos, ln=np.log, i0=i0) -> NUMERIC:
        """Approximate log likelihood.

        Args:
            mu: mean
            cos: cosine function
            ln: log function
            i0: modified bessel function of order 0

        Returns:
            log likelihood

        """
        return self.a + cos(mu - self.b) - ln(i0(self.a))


@dataclass
class VonMisesKnownDirectionProportional:
    """Von Mises known direction proportional distribution.

    Taken from <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 2.13.2</a>.

    Args:
    c: NUMERIC
    r: NUMERIC
    """

    c: NUMERIC
    r: NUMERIC

    def approx_log_likelihood(self, kappa: NUMERIC, ln=np.log, i0=i0) -> NUMERIC:
        """Approximate log likelihood.

        Args:
            kappa: concentration
            ln: log function
            i0: modified bessel function of order 0

        Returns:
            log likelihood up to a constant

        """
        return kappa * self.r - self.c * ln(i0(kappa))


@dataclass
class ScaledInverseChiSquared(ContinuousPlotDistMixin, SliceMixin):
    """Scaled inverse chi squared distribution.

    Args:
        nu: degrees of freedom
        sigma2: scale parameter

    """

    nu: NUMERIC
    sigma2: NUMERIC

    @classmethod
    def from_inverse_gamma(
        cls,
        inverse_gamma: InverseGamma,
    ) -> "ScaledInverseChiSquared":
        """Alternative constructor from inverse gamma distribution.

        Args:
            inverse_gamma: inverse gamma distribution

        Returns:
            scaled inverse chi squared distribution

        """
        nu = inverse_gamma.alpha * 2
        sigma2 = inverse_gamma.beta * 2 / nu

        return cls(nu=nu, sigma2=sigma2)

    def to_inverse_gamma(self) -> InverseGamma:
        """Convert to inverse gamma distribution.

        Returns:
            inverse gamma distribution

        """
        return InverseGamma(alpha=self.nu / 2, beta=self.nu * self.sigma2 / 2)

    @property
    def dist(self):
        return stats.invgamma(a=self.nu / 2, scale=self.nu * self.sigma2 / 2)


@dataclass
class NormalGamma:
    """Normal gamma distribution.

    Args:
        mu: mean
        lam: precision
        alpha: shape
        beta: scale

    """

    mu: NUMERIC
    lam: NUMERIC
    alpha: NUMERIC
    beta: NUMERIC

    @property
    def gamma(self) -> Gamma:
        return Gamma(alpha=self.alpha, beta=self.beta)

    def sample_variance(self, size: int, random_state=None) -> NUMERIC:
        """Sample precision from gamma distribution and invert.

        Args:
            size: number of samples
            random_state: random state

        Returns:
            samples from the inverse gamma distribution

        """
        precision = self.lam * self.gamma.dist.rvs(size=size, random_state=random_state)

        return 1 / precision

    def sample_mean(
        self,
        size: int,
        return_variance: bool = False,
        random_state=None,
    ) -> NUMERIC | tuple[NUMERIC, NUMERIC]:
        """Sample mean from the normal distribution.

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution

        """
        return self.sample_beta(
            size=size, return_variance=return_variance, random_state=random_state
        )

    def sample_beta(
        self, size: int, return_variance: bool = False, random_state=None
    ) -> NUMERIC | tuple[NUMERIC, NUMERIC]:
        """Sample beta from the normal distribution.

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution

        """
        variance = self.sample_variance(size=size, random_state=random_state)
        sigma = variance**0.5
        beta = stats.norm(loc=self.mu, scale=sigma).rvs(
            size=size, random_state=random_state
        )

        if return_variance:
            return beta, variance

        return beta


@dataclass
class InverseWishart:
    """Inverse Wishart distribution.

    Args:
        nu: degrees of freedom
        psi: scale matrix

    """

    nu: NUMERIC
    psi: NUMERIC

    @property
    def dist(self):
        return stats.invwishart(df=self.nu, scale=self.psi)


@dataclass
class Wishart:
    """Wishart distribution.

    Args:
        nu: degrees of freedom
        V: scale matrix

    """

    nu: NUMERIC
    V: NUMERIC

    @property
    def dist(self):
        return stats.wishart(df=self.nu, scale=self.V)


@dataclass
class NormalWishart:
    """Normal Wishart distribution.

    Parameterization from <a href=https://en.wikipedia.org/wiki/Normal-Wishart_distribution>Wikipedia</a>.

    Args:
        mu: mean
        lam: precision
        W: scale matrix
        nu: degrees of freedom

    """

    mu: NUMERIC
    lam: NUMERIC
    W: NUMERIC
    nu: NUMERIC

    @property
    def wishart(self):
        return Wishart(nu=self.nu, V=self.W)

    def sample_variance(
        self,
        size: int = 1,
        random_state: np.random.Generator | None = None,
        inv: Callable = np.linalg.inv,
    ) -> np.ndarray:
        """Sample variance

        Args:
            size: number of samples
            random_state: random state
            inv: matrix inversion function

        Returns:
            samples from the inverse wishart distribution

        """

        variance = inv(
            self.lam * self.wishart.dist.rvs(size=size, random_state=random_state)
        )

        if size == 1:
            variance = variance[None, ...]

        return variance

    def sample_mean(
        self,
        size: int = 1,
        return_variance: bool = False,
        random_state: np.random.Generator | None = None,
    ) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
        """Sample mean

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution and optionally variance

        """

        variance = self.sample_variance(size=size, random_state=random_state)

        mean = np.stack(
            [
                stats.multivariate_normal(self.mu, cov=cov).rvs(
                    size=1,
                    random_state=random_state,
                )
                for cov in variance
            ]
        )

        if return_variance:
            return mean, variance

        return mean


@dataclass
class NormalInverseWishart:
    """Normal inverse Wishart distribution.

    Args:
        mu: mean
        kappa: precision
        nu: degrees of freedom
        psi: scale matrix

    """

    mu: NUMERIC
    kappa: NUMERIC
    nu: NUMERIC
    psi: NUMERIC

    @property
    def inverse_wishart(self):
        """Inverse wishart distribution."""
        return InverseWishart(nu=self.nu, psi=self.psi)

    @classmethod
    def from_inverse_wishart(
        cls,
        mu: NUMERIC,
        kappa: NUMERIC,
        inverse_wishart: InverseWishart,
    ):
        return cls(mu=mu, kappa=kappa, nu=inverse_wishart.nu, psi=inverse_wishart.psi)

    def sample_variance(
        self, size: int, random_state: np.random.Generator | None = None
    ) -> NUMERIC:
        """Sample precision from gamma distribution and invert.

        Args:
            size: number of samples
            random_state: random state

        Returns:
            samples from the inverse wishart distribution

        """
        variance = (
            self.inverse_wishart.dist.rvs(size=size, random_state=random_state)
            / self.kappa
        )
        if size == 1:
            variance = variance[None, ...]

        return variance

    def sample_mean(
        self,
        size: int,
        return_variance: bool = False,
        random_state: np.random.Generator | None = None,
    ) -> NUMERIC:
        """Sample the mean from the normal distribution.

        Args:
            size: number of samples
            return_variance: whether to return variance as well
            random_state: random state

        Returns:
            samples from the normal distribution and optionally variance

        """
        variance = self.sample_variance(size=size, random_state=random_state)

        mean = np.stack(
            [
                stats.multivariate_normal(self.mu, cov=cov).rvs(
                    size=1, random_state=random_state
                )
                for cov in variance
            ]
        )

        if return_variance:
            return mean, variance

        return mean


@dataclass
class LogNormal(ContinuousPlotDistMixin, SliceMixin):
    """Log normal distribution.

    Args:
        mu: mean
        sigma: standard deviation

    """

    mu: NUMERIC
    sigma: NUMERIC

    @property
    def dist(self):
        return stats.lognorm(s=self.sigma, loc=self.mu)


@dataclass
class Weibull(ContinuousPlotDistMixin, SliceMixin):
    """Weibull distribution.

    Parameterization from Section 2.11 of <a href="https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf">paper</a>.

    Args:
        beta: shape parameter
        theta: scale parameter

    Example:
        Recreation of the plot on <a href=https://en.wikipedia.org/wiki/Weibull_distribution>Wikipedia</a>.

        ```python
        import matplotlib.pyplot as plt
        import numpy as np

        from conjugate.distributions import Weibull

        lam = 1
        k = np.array([0.5, 1.0, 1.5, 5.0])

        beta = k
        theta = lam ** beta

        distribution = Weibull(beta=beta, theta=theta)
        ax = distribution.set_bounds(0, 2.5).plot_pdf(
            label=["k=0.5", "k=1.0", "k=1.5", "k=5.0"],
            color=["blue", "red", "pink", "green"],
        )
        ax.legend()
        ```
        <!--
        plt.savefig("plot-check.png")
        plt.close()
        -->

    """

    beta: NUMERIC
    theta: NUMERIC

    @property
    def dist(self):
        k = self.beta
        lam = self.theta ** (1 / self.beta)
        return stats.weibull_min(c=k, scale=lam)
