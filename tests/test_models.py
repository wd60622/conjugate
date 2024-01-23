import pytest

from dataclasses import dataclass

from pypika import Field

import numpy as np

import matplotlib.pyplot as plt

from conjugate.distributions import (
    Beta,
    BetaProportional,
    CompoundGamma,
    Dirichlet,
    Pareto,
    Gamma,
    Lomax,
    NegativeBinomial,
    NormalInverseGamma,
    InverseGamma,
    Normal,
    StudentT,
    MultivariateStudentT,
    GammaProportional,
    GammaKnownRateProportional,
    NormalInverseWishart,
    MultivariateNormal,
    LogNormal,
)
from conjugate.models import (
    get_binomial_beta_posterior_params,
    binomial_beta,
    multinomial_dirichlet,
    geometric_beta,
    poisson_gamma,
    poisson_gamma_posterior_predictive,
    linear_regression,
    linear_regression_posterior_predictive,
    uniform_pareto,
    pareto_gamma,
    exponential_gamma,
    exponential_gamma_posterior_predictive,
    normal_known_variance,
    normal_known_variance_posterior_predictive,
    normal_known_mean,
    normal_known_mean_posterior_predictive,
    normal_normal_inverse_gamma,
    normal_normal_inverse_gamma_posterior_predictive,
    gamma_known_shape,
    gamma_known_shape_posterior_predictive,
    gamma,
    gamma_known_rate,
    beta,
    multivariate_normal,
    multivariate_normal_posterior_predictive,
    log_normal_normal_inverse_gamma,
)

rng = np.random.default_rng(42)


class MockOperation:
    def __init__(self, left, right, operation: str):
        self.left = left
        self.right = right
        self.operation = operation

    def __str__(self) -> str:
        return f"{self.left} {self.operation} {self.right}"

    def __eq__(self, other) -> bool:
        if isinstance(self, MockOperation) and isinstance(other, MockOperation):
            return self.left == other.left and self.right == other.right

        return False


@dataclass
class Value:
    value: str

    def __str__(self) -> str:
        return str(self.value)

    def __add__(self, other: "Value") -> MockOperation:
        return MockOperation(self, other, "+")

    __radd__ = __add__

    def __sub__(self, other) -> MockOperation:
        return MockOperation(self, other, "-")

    def __rsub__(self, other) -> MockOperation:
        return MockOperation(other, self, "-")

    def __eq__(self, other) -> bool:
        if isinstance(self, Value) and isinstance(other, Value):
            return self.value == other.value

        return False


@pytest.mark.parametrize(
    "alpha_prior, beta_prior, n, x, alpha_post, beta_post",
    [
        (1, 1, 10, 5, 6, 6),
        # Things that work like numbers
        (
            Value("alpha_prior"),
            Value("beta_prior"),
            Value("N"),
            Value("X"),
            "alpha_prior + X",
            "beta_prior + N - X",
        ),
        (
            Field("alpha_prior"),
            Field("beta_prior"),
            Field("N"),
            Field("X"),
            Field("alpha_prior + X"),
            Field("beta_prior + N - X"),
        ),
    ],
)
def test_get_binomial_beta_posterior_params(
    alpha_prior, beta_prior, n, x, alpha_post, beta_post
) -> None:
    values = get_binomial_beta_posterior_params(
        alpha_prior=alpha_prior, beta_prior=beta_prior, n=n, x=x
    )

    def handle_value(value):
        if isinstance(value, int):
            return value
        return str(value)

    alpha_post_result, beta_post_result = [handle_value(value) for value in values]

    assert (alpha_post_result, beta_post_result) == (alpha_post, beta_post)


@pytest.mark.parametrize(
    "alpha, beta, N, x",
    [
        # All scalars
        (1, 1, 100, 50),
        # Vectorized Parameters
        (np.array([1, 10, 100]), 1, 100, 50),
        (1, np.array([1, 10, 100]), 100, 50),
        (np.array([1, 10, 100]), np.array([1, 10, 100]), 100, 50),
        # Vectorized Data
        (1, 1, np.array([10, 50, 100]), 1),
        (1, 1, 100, np.array([25, 50, 75])),
        (1, 1, np.array([100, 200, 300]), np.array([50, 40, 30])),
        # Both
        (np.array([1, 5]), np.array([1, 5]), np.array([10, 5]), np.array([5, 2])),
    ],
)
def test_handle_vectorize_values(alpha, beta, N, x) -> None:
    prior = Beta(alpha=alpha, beta=beta)
    posterior = binomial_beta(n=N, x=x, beta_prior=prior)

    assert isinstance(posterior, Beta)

    ax = posterior.plot_pdf()
    assert isinstance(ax, plt.Axes)


def test_geometric_beta_model() -> None:
    prior = Beta(alpha=1, beta=1)

    N = 10
    X_TOTAL = 12

    posterior_one_start = geometric_beta(x_total=X_TOTAL, n=N, beta_prior=prior)
    poisterior_zero_start = geometric_beta(
        x_total=X_TOTAL, n=N, beta_prior=prior, one_start=False
    )

    assert isinstance(posterior_one_start, Beta)
    assert isinstance(poisterior_zero_start, Beta)
    assert posterior_one_start.dist.mean() > poisterior_zero_start.dist.mean()


def test_poisson_gamma_analysis() -> None:
    prior = Gamma(alpha=1, beta=1)
    posterior = poisson_gamma(n=10, x_total=5, gamma_prior=prior)

    assert isinstance(posterior, Gamma)

    pp = poisson_gamma_posterior_predictive(gamma=posterior)
    assert isinstance(pp, NegativeBinomial)

    pp_7days = poisson_gamma_posterior_predictive(gamma=posterior, n=7)
    assert isinstance(pp_7days, NegativeBinomial)


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([1, 2, 3]),
        np.array([1, 2, 3, 4]),
        np.array(
            [
                [1, 2, 3],
                [4, 5, 6],
            ]
        ),
    ],
)
def test_multinomial_dirichlet_analysis(alpha) -> None:
    prior = Dirichlet(alpha=alpha)
    posterior = multinomial_dirichlet(x=1, dirichlet_prior=prior)

    assert isinstance(posterior, Dirichlet)
    assert isinstance(posterior.alpha, np.ndarray)

    assert posterior.dist.mean().shape == alpha.shape


@pytest.mark.parametrize(
    "intercept, slope, sigma",
    [
        (5, 2, 1),
        (5, 2, 1.1),
    ],
)
def test_linear_regression(intercept, slope, sigma) -> None:
    n_points = 100

    x = np.linspace(-5, 5, n_points)

    y = intercept + slope * x + rng.normal(loc=0, scale=sigma, size=n_points)

    X = np.stack(
        [
            np.ones_like(x),
            x,
        ]
    ).T

    prior = NormalInverseGamma(
        mu=np.array([0, 0]),
        delta_inverse=np.array(
            [
                [5, 0],
                [0, 5],
            ]
        ),
        alpha=1,
        beta=10,
    )

    posterior = linear_regression(X, y, prior)

    beta_samples, variance_samples = posterior.sample_beta(
        size=500, return_variance=True, random_state=rng
    )
    sigma_samples = variance_samples**0.5

    def between(x, lower, upper):
        return lower <= x <= upper

    q = [0.025, 0.975]
    assert between(intercept, *np.quantile(beta_samples[:, 0], q=q))
    assert between(slope, *np.quantile(beta_samples[:, 1], q=q))
    assert between(sigma, *np.quantile(sigma_samples, q=q))

    posterior_predictive = linear_regression_posterior_predictive(
        normal_inverse_gamma=posterior, X=X
    )
    assert isinstance(posterior_predictive, MultivariateStudentT)


def test_uniform_pareto_python_objects() -> None:
    samples = [1, 2, 3, 4, 5]
    n_samples = len(samples)

    prior = Pareto(x_m=1, alpha=1)
    posterior = uniform_pareto(x_max=max(samples), n=n_samples, pareto_prior=prior)

    assert isinstance(posterior, Pareto)
    assert posterior.x_m == 5
    assert posterior.alpha == 6


def test_uniform_pareto_numpy_objects() -> None:
    samples = np.array([1, 2, 3, 4, 5])
    n_samples = len(samples)

    prior = Pareto(x_m=1, alpha=1)
    posterior = uniform_pareto(x_max=samples.max(), n=n_samples, pareto_prior=prior)

    assert isinstance(posterior, Pareto)
    assert posterior.x_m == 5
    assert posterior.alpha == 6


def test_pareto_gamma() -> None:
    samples = np.array([1, 2, 3, 4, 5])
    n_samples = len(samples)

    prior = Gamma(alpha=1, beta=1)

    posterior = pareto_gamma(
        n=n_samples,
        ln_x_total=np.log(samples).sum(),
        x_m=samples.min(),
        gamma_prior=prior,
    )

    assert isinstance(posterior, Gamma)
    assert posterior.alpha == 6
    assert posterior.beta == 5.787491742782046


def test_exponential_gamma() -> None:
    data = [1, 2, 3, 4, 5]

    n = len(data)
    x_total = sum(data)

    prior = Gamma(alpha=1, beta=1)
    posterior = exponential_gamma(x_total=x_total, n=n, gamma_prior=prior)

    assert isinstance(posterior, Gamma)

    posterior_predictive = exponential_gamma_posterior_predictive(gamma=posterior)

    assert isinstance(posterior_predictive, Lomax)


def test_normal_known_variance() -> None:
    known_var = 2.5

    data = np.array([1, 2, 3, 4, 5])

    prior = Normal(0, 1)
    posterior = normal_known_variance(
        x_total=data.sum(), n=len(data), var=known_var, normal_prior=prior
    )

    assert isinstance(posterior, Normal)

    posterior_predictive = normal_known_variance_posterior_predictive(
        var=known_var, normal=posterior
    )
    assert isinstance(posterior_predictive, Normal)


def test_normal_known_mean() -> None:
    known_mu = 0

    data = np.array([1, 2, 3, 4, 5])

    prior = InverseGamma(1, 1)
    posterior = normal_known_mean(
        x_total=data.sum(),
        x2_total=(data**2).sum(),
        n=len(data),
        mu=known_mu,
        inverse_gamma_prior=prior,
    )

    assert isinstance(posterior, InverseGamma)

    posterior_predictive = normal_known_mean_posterior_predictive(
        mu=known_mu, inverse_gamma=posterior
    )
    assert isinstance(posterior_predictive, StudentT)


def test_normal_normal_inverse_gamma() -> None:
    true_mu, true_sigma = 25, 7.5

    true = Normal(true_mu, true_sigma)

    n = 25
    data = true.dist.rvs(size=n, random_state=rng)

    prior = NormalInverseGamma(0.0, alpha=1 / 5, beta=10, nu=1 / 25)
    posterior = normal_normal_inverse_gamma(
        x_total=data.sum(),
        x2_total=(data**2).sum(),
        n=n,
        normal_inverse_gamma_prior=prior,
    )

    assert isinstance(posterior, NormalInverseGamma)

    posterior_predictive = normal_normal_inverse_gamma_posterior_predictive(
        normal_inverse_gamma=posterior,
    )
    assert isinstance(posterior_predictive, StudentT)

    prior_predictive = normal_normal_inverse_gamma_posterior_predictive(
        normal_inverse_gamma=prior,
    )

    assert (
        prior_predictive.dist.logpdf(data).sum()
        < posterior_predictive.dist.logpdf(data).sum()
    )


@pytest.mark.parametrize(
    "shape",
    [
        1,
        np.array([1, 2, 3]),
        np.array([[1, 2, 3], [1, 1, 1]]),
    ],
)
def test_gamma_known_shape(shape) -> None:
    data = np.array([1, 2, 3, 4, 5])

    prior = Gamma(alpha=1, beta=1)
    posterior = gamma_known_shape(
        x_total=data.sum(),
        n=len(data),
        alpha=shape,
        gamma_prior=prior,
    )

    assert isinstance(posterior, Gamma)

    posterior_predictive = gamma_known_shape_posterior_predictive(
        alpha=shape, gamma=posterior
    )
    assert isinstance(posterior_predictive, CompoundGamma)


def test_gamma_proportional_model() -> None:
    true_alpha, true_beta = 2, 6
    true = Gamma(alpha=true_alpha, beta=true_beta)

    n_samples = 15
    samples = true.dist.rvs(size=n_samples, random_state=0)

    prior = GammaProportional(1, 1, 1, 1)
    posterior = gamma(
        x_total=samples.sum(),
        x_prod=np.prod(samples),
        n=n_samples,
        proportional_prior=prior,
    )

    assert isinstance(posterior, GammaProportional)

    # TODO: add a comparison to the true for prior and posterior


def test_gamma_known_rate() -> None:
    true_alpha, true_beta = 2, 6
    true = Gamma(alpha=true_alpha, beta=true_beta)

    n_samples = 15
    samples = true.dist.rvs(size=n_samples, random_state=0)

    prior = GammaKnownRateProportional(1, 1, 1)
    posterior = gamma_known_rate(
        x_prod=np.prod(samples),
        n=n_samples,
        beta=true_beta,
        proportional_prior=prior,
    )

    assert isinstance(posterior, GammaKnownRateProportional)


def test_beta_proportional_model() -> None:
    true_alpha, true_beta = 2, 6
    true = Beta(alpha=true_alpha, beta=true_beta)

    n_samples = 15
    samples = true.dist.rvs(size=n_samples, random_state=0)

    prior = BetaProportional(0.25, 0.25, 1)
    posterior = beta(
        x_prod=np.prod(samples),
        one_minus_x_prod=np.prod(1 - samples),
        n=n_samples,
        proportional_prior=prior,
    )

    assert isinstance(posterior, BetaProportional)


def test_multivariate_normal() -> None:
    mu = np.array([1, 2, 3])
    sigma = np.array(
        [
            [1, 0, 0],
            [0, 2, 0],
            [0, 0, 3],
        ]
    )

    true = MultivariateNormal(mu, sigma)

    n_samples = 15
    X = true.dist.rvs(size=n_samples, random_state=0)

    prior = NormalInverseWishart(
        mu=np.zeros_like(mu),
        kappa=1,
        nu=len(mu),
        psi=np.eye(len(mu)),
    )
    posterior = multivariate_normal(
        X=X,
        normal_inverse_wishart_prior=prior,
    )
    assert isinstance(posterior, NormalInverseWishart)

    prior_predictive = multivariate_normal_posterior_predictive(
        normal_inverse_wishart=prior
    )
    posterior_predictive = multivariate_normal_posterior_predictive(posterior)
    assert isinstance(posterior_predictive, MultivariateStudentT)
    assert prior_predictive.dist.logpdf(mu) < posterior_predictive.dist.logpdf(mu)
    assert (
        prior_predictive.dist.logpdf(X).sum()
        < posterior_predictive.dist.logpdf(X).sum()
    )


def test_log_normal_normal_inverse_gamma() -> None:
    true_mu, true_sigma = 0.25, 2.5

    true = LogNormal(true_mu, true_sigma)

    n = 25
    data = true.dist.rvs(size=n, random_state=rng)
    ln_data = np.log(data)

    prior = NormalInverseGamma(0.0, alpha=1 / 5, beta=10, nu=1 / 25)
    posterior = log_normal_normal_inverse_gamma(
        ln_x_total=ln_data.sum(),
        ln_x2_total=(ln_data**2).sum(),
        n=n,
        normal_inverse_gamma_prior=prior,
    )

    assert isinstance(posterior, NormalInverseGamma)
    assert posterior.inverse_gamma.dist.logpdf(
        true_sigma**2
    ) > prior.inverse_gamma.dist.logpdf(true_sigma**2)
