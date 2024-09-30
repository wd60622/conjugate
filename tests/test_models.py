import pytest

from dataclasses import dataclass

from pypika import Field

import numpy as np

import matplotlib.pyplot as plt

from conjugate.distributions import (
    Beta,
    BetaBinomial,
    BetaGeometric,
    BetaNegativeBinomial,
    BetaProportional,
    CompoundGamma,
    Dirichlet,
    Gamma,
    ScaledInverseChiSquared,
    GammaKnownRateProportional,
    GammaProportional,
    InverseGamma,
    LogNormal,
    Lomax,
    MultivariateNormal,
    MultivariateStudentT,
    NegativeBinomial,
    Normal,
    NormalGamma,
    NormalInverseGamma,
    NormalInverseWishart,
    Pareto,
    StudentT,
)
from conjugate.models import (
    bernoulli_beta,
    bernoulli_beta_predictive,
    beta,
    binomial_beta,
    exponential_gamma,
    exponential_gamma_predictive,
    gamma,
    gamma_known_rate,
    gamma_known_shape,
    gamma_known_shape_predictive,
    geometric_beta,
    geometric_beta_predictive,
    get_binomial_beta_posterior_params,
    hypergeometric_beta_binomial,
    inverse_gamma_known_rate,
    linear_regression,
    linear_regression_predictive,
    log_normal,
    multinomial_dirichlet,
    multivariate_normal,
    multivariate_normal_known_covariance,
    multivariate_normal_known_covariance_predictive,
    multivariate_normal_known_precision,
    multivariate_normal_known_precision_predictive,
    multivariate_normal_predictive,
    negative_binomial_beta,
    negative_binomial_beta_predictive,
    normal,
    normal_predictive,
    normal_known_mean,
    normal_known_mean_predictive,
    normal_known_variance,
    normal_known_variance_predictive,
    normal_normal_inverse_gamma,
    normal_normal_inverse_gamma_predictive,
    pareto_gamma,
    poisson_gamma,
    poisson_gamma_predictive,
    uniform_pareto,
    weibull_inverse_gamma_known_shape,
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
    posterior = binomial_beta(n=N, x=x, prior=prior)

    assert isinstance(posterior, Beta)

    ax = posterior.plot_pdf()
    assert isinstance(ax, plt.Axes)


def test_geometric_beta_model() -> None:
    prior = Beta(alpha=1, beta=1)

    N = 10
    X_TOTAL = 12

    posterior_one_start = geometric_beta(x_total=X_TOTAL, n=N, prior=prior)
    poisterior_zero_start = geometric_beta(
        x_total=X_TOTAL, n=N, prior=prior, one_start=False
    )

    assert isinstance(posterior_one_start, Beta)
    assert isinstance(poisterior_zero_start, Beta)
    assert posterior_one_start.dist.mean() > poisterior_zero_start.dist.mean()


def test_poisson_gamma_analysis() -> None:
    prior = Gamma(alpha=1, beta=1)
    posterior = poisson_gamma(n=10, x_total=5, prior=prior)

    assert isinstance(posterior, Gamma)

    pp = poisson_gamma_predictive(distribution=posterior)
    assert isinstance(pp, NegativeBinomial)

    pp_7days = poisson_gamma_predictive(distribution=posterior, n=7)
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
    posterior = multinomial_dirichlet(x=1, prior=prior)

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

    posterior = linear_regression(X, y, prior=prior)

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

    posterior_predictive = linear_regression_predictive(distribution=posterior, X=X)
    assert isinstance(posterior_predictive, MultivariateStudentT)


def test_uniform_pareto_python_objects() -> None:
    samples = [1, 2, 3, 4, 5]
    n_samples = len(samples)

    prior = Pareto(x_m=1, alpha=1)
    posterior = uniform_pareto(x_max=max(samples), n=n_samples, prior=prior)

    assert isinstance(posterior, Pareto)
    assert posterior.x_m == 5
    assert posterior.alpha == 6


def test_uniform_pareto_numpy_objects() -> None:
    samples = np.array([1, 2, 3, 4, 5])
    n_samples = len(samples)

    prior = Pareto(x_m=1, alpha=1)
    posterior = uniform_pareto(x_max=samples.max(), n=n_samples, prior=prior)

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
        prior=prior,
    )

    assert isinstance(posterior, Gamma)
    assert posterior.alpha == 6
    assert posterior.beta == 5.787491742782046


def test_exponential_gamma() -> None:
    data = [1, 2, 3, 4, 5]

    n = len(data)
    x_total = sum(data)

    prior = Gamma(alpha=1, beta=1)
    posterior = exponential_gamma(x_total=x_total, n=n, prior=prior)

    assert isinstance(posterior, Gamma)

    posterior_predictive = exponential_gamma_predictive(distribution=posterior)

    assert isinstance(posterior_predictive, Lomax)


def test_normal_known_variance() -> None:
    known_var = 2.5

    data = np.array([1, 2, 3, 4, 5])

    prior = Normal(0, 1)
    posterior = normal_known_variance(
        x_total=data.sum(), n=len(data), var=known_var, prior=prior
    )

    assert isinstance(posterior, Normal)

    posterior_predictive = normal_known_variance_predictive(
        var=known_var, distribution=posterior
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
        prior=prior,
    )

    assert isinstance(posterior, InverseGamma)

    posterior_predictive = normal_known_mean_predictive(
        mu=known_mu,
        distribution=posterior,
    )
    assert isinstance(posterior_predictive, StudentT)


def test_normal_normal_inverse_gamma() -> None:
    true_mu, true_sigma = 25, 7.5

    true = Normal(true_mu, true_sigma)

    n = 25
    data = true.dist.rvs(size=n, random_state=rng)

    prior = NormalInverseGamma(0.0, alpha=1 / 5, beta=10, nu=1 / 25)
    posterior = normal(
        x_total=data.sum(),
        x2_total=(data**2).sum(),
        n=n,
        prior=prior,
    )

    assert isinstance(posterior, NormalInverseGamma)

    posterior_predictive = normal_predictive(
        distribution=posterior,
    )
    assert isinstance(posterior_predictive, StudentT)

    prior_predictive = normal_predictive(
        distribution=prior,
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
        prior=prior,
    )

    assert isinstance(posterior, Gamma)

    posterior_predictive = gamma_known_shape_predictive(
        alpha=shape,
        distribution=posterior,
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
        prior=prior,
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
        prior=prior,
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
        prior=prior,
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
        prior=prior,
    )
    assert isinstance(posterior, NormalInverseWishart)

    prior_predictive = multivariate_normal_predictive(distribution=prior)
    posterior_predictive = multivariate_normal_predictive(distribution=posterior)
    assert isinstance(posterior_predictive, MultivariateStudentT)
    assert prior_predictive.dist.logpdf(mu) < posterior_predictive.dist.logpdf(mu)
    assert (
        prior_predictive.dist.logpdf(X).sum()
        < posterior_predictive.dist.logpdf(X).sum()
    )


def test_log_normal() -> None:
    true_mu, true_sigma = 0.25, 2.5

    true = LogNormal(true_mu, true_sigma)

    n = 25
    data = true.dist.rvs(size=n, random_state=rng)
    ln_data = np.log(data)

    prior = NormalInverseGamma(0.0, alpha=1 / 5, beta=10, nu=1 / 25)
    posterior = log_normal(
        ln_x_total=ln_data.sum(),
        ln_x2_total=(ln_data**2).sum(),
        n=n,
        prior=prior,
    )

    assert isinstance(posterior, NormalInverseGamma)
    assert posterior.inverse_gamma.dist.logpdf(
        true_sigma**2
    ) > prior.inverse_gamma.dist.logpdf(true_sigma**2)


def test_bernoulli_beta() -> None:
    prior = Beta(alpha=1, beta=1)
    posterior = bernoulli_beta(
        x=0,
        prior=prior,
    )
    assert posterior == Beta(alpha=1, beta=2)


def test_bernoulli_beta_predictive() -> None:
    prior = Beta(alpha=1, beta=1)
    pp = bernoulli_beta_predictive(distribution=prior)

    assert pp == BetaBinomial(n=1, alpha=1, beta=1)


def test_negative_binomial_beta() -> None:
    prior = Beta(alpha=1, beta=1)
    posterior = negative_binomial_beta(
        r=10,
        n=15,
        x=5,
        prior=prior,
    )

    assert posterior == Beta(
        alpha=151,
        beta=6,
    )


def test_negative_binomial_beta_predictive() -> None:
    prior = Beta(alpha=1, beta=1)
    pp = negative_binomial_beta_predictive(r=10, distribution=prior)

    assert pp == BetaNegativeBinomial(n=10, alpha=1, beta=1)


def test_hypergeometric_beta_binomial() -> None:
    prior = BetaBinomial(n=10, alpha=1, beta=1)
    posterior = hypergeometric_beta_binomial(
        x_total=5,
        n=20,
        prior=prior,
    )

    assert posterior == BetaBinomial(
        n=10,
        alpha=6,
        beta=6,
    )


def test_old_parameter_raises_deprecation_warning_model() -> None:
    beta = Beta(1, 1)
    match = "Parameter 'beta_prior' is deprecated, use 'prior' instead."
    with pytest.warns(DeprecationWarning, match=match):
        posterior = binomial_beta(n=10, x=5, beta_prior=beta)

    assert isinstance(posterior, Beta)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("prior_name", ["prior", "beta_prior"])
def test_wrong_prior_type_raises(prior_name: str) -> None:
    prior = Gamma(1, 1)
    kwargs = {prior_name: prior}
    match = "Expected prior to be of type 'Beta', got 'Gamma' instead."
    with pytest.raises(ValueError, match=match):
        bernoulli_beta(x=0, **kwargs)


def test_old_parameter_raises_deprecation_warning_predictive() -> None:
    distribution = Beta(1, 1)
    match = "Parameter 'beta' is deprecated, use 'distribution' instead."
    with pytest.warns(DeprecationWarning, match=match):
        predictive = bernoulli_beta_predictive(beta=distribution)

    assert isinstance(predictive, BetaBinomial)


@pytest.mark.filterwarnings("ignore")
@pytest.mark.parametrize("distribution_name", ["distribution", "beta"])
def test_wrong_distribution_type_raises(distribution_name: str) -> None:
    distribution = Gamma(1, 1)
    kwargs = {distribution_name: distribution}
    match = "Expected distribution to be of type 'Beta', got 'Gamma' instead."
    with pytest.raises(ValueError, match=match):
        bernoulli_beta_predictive(**kwargs)


def test_inverse_gamma_known_rate() -> None:
    prior = Gamma(1, 1)
    posterior = inverse_gamma_known_rate(
        reciprocal_x_total=1,
        n=1,
        alpha=1,
        prior=prior,
    )
    assert isinstance(posterior, Gamma)


def test_normal_known_mean_alternative_prior() -> None:
    prior = ScaledInverseChiSquared(1, 1)
    posterior = normal_known_mean(
        x_total=0,
        x2_total=1,
        n=1,
        mu=0,
        prior=prior,
    )

    assert isinstance(posterior, ScaledInverseChiSquared)


def test_normal_known_mean_alternative_prior_predictive() -> None:
    distribution = ScaledInverseChiSquared(1, 1)
    predictive = normal_known_mean_predictive(
        mu=0,
        distribution=distribution,
    )

    assert isinstance(predictive, StudentT)


def test_weibull_known_shape() -> None:
    prior = InverseGamma(1, 1)
    posterior = weibull_inverse_gamma_known_shape(n=1, x_beta_total=1, prior=prior)

    assert isinstance(posterior, InverseGamma)


def test_geometric_beta_predictive() -> None:
    distribution = Beta(1, 1)
    predictive = geometric_beta_predictive(distribution=distribution)
    assert isinstance(predictive, BetaGeometric)


def test_multivarate_normal_known_covariance() -> None:
    cov = np.eye(3)
    prior = MultivariateNormal(mu=np.zeros(3), cov=np.eye(3))
    posterior = multivariate_normal_known_covariance(
        n=1,
        x_bar=np.zeros(3),
        cov=cov,
        prior=prior,
    )

    assert isinstance(posterior, MultivariateNormal)


def test_multivariate_normal_known_covariance_predictive() -> None:
    cov = np.eye(3)
    distribution = MultivariateNormal(mu=np.zeros(3), cov=np.eye(3))
    predictive = multivariate_normal_known_covariance_predictive(
        distribution=distribution,
        cov=cov,
    )

    assert isinstance(predictive, MultivariateNormal)


def test_multivariate_normal_known_precision() -> None:
    precision = np.eye(3)
    prior = MultivariateNormal(mu=np.zeros(3), cov=np.eye(3))
    posterior = multivariate_normal_known_precision(
        n=1,
        x_bar=np.zeros(3),
        precision=precision,
        prior=prior,
    )

    assert isinstance(posterior, MultivariateNormal)


def test_multivariate_normal_known_precision_predictive() -> None:
    precision = np.eye(3)
    distribution = MultivariateNormal(mu=np.zeros(3), cov=np.eye(3))
    predictive = multivariate_normal_known_precision_predictive(
        distribution=distribution,
        precision=precision,
    )

    assert isinstance(predictive, MultivariateNormal)


@pytest.mark.parametrize(
    "prior",
    [
        NormalGamma(mu=0, lam=1, alpha=1, beta=1),
        NormalInverseGamma(mu=0, nu=1, alpha=1, beta=1),
    ],
)
def test_normal(prior) -> None:
    mu = 1
    sigma = 0.25
    n = 100

    data = rng.normal(loc=mu, scale=sigma, size=n)

    posterior = normal(
        x_total=data.sum(),
        n=len(data),
        x2_total=(data**2).sum(),
        prior=prior,
    )

    assert isinstance(posterior, prior.__class__)


def test_normal_normal_inverse_gamma_deprecation() -> None:
    prior = NormalInverseGamma(mu=0, nu=1, alpha=1, beta=1)
    match = "This function is deprecated"
    with pytest.warns(DeprecationWarning, match=match):
        posterior = normal_normal_inverse_gamma(
            x_total=0,
            x2_total=1,
            n=1,
            prior=prior,
        )

    assert isinstance(posterior, NormalInverseGamma)


def test_normal_normal_inverse_gamma_predictive_deprecation() -> None:
    distribution = NormalInverseGamma(mu=0, nu=1, alpha=1, beta=1)
    match = "This function is deprecated"
    with pytest.warns(DeprecationWarning, match=match):
        predictive = normal_normal_inverse_gamma_predictive(
            distribution=distribution,
        )

    assert isinstance(predictive, StudentT)
