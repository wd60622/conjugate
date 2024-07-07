import pytest

from packaging import version

import numpy as np

import matplotlib.pyplot as plt

from scipy import __version__ as scipy_version

from conjugate.distributions import (
    Beta,
    BetaBinomial,
    BetaNegativeBinomial,
    Geometric,
    Binomial,
    CompoundGamma,
    Dirichlet,
    Exponential,
    Gamma,
    NormalGamma,
    InverseGamma,
    InverseWishart,
    Hypergeometric,
    LogNormal,
    Lomax,
    MultivariateNormal,
    MultivariateStudentT,
    NegativeBinomial,
    Normal,
    NormalInverseGamma,
    NormalInverseWishart,
    Pareto,
    Poisson,
    ScaledInverseChiSquared,
    StudentT,
    Uniform,
    VectorizedDist,
    VonMises,
    get_beta_param_from_mean_and_alpha,
)


@pytest.mark.parametrize(
    "alpha, beta",
    [
        (1, 1),
        (np.array([1, 10, 100]), 1),
        (1, np.array([1, 10, 100])),
        (np.array([1, 10, 100]), np.array([1, 10, 100])),
    ],
)
def test_beta(alpha, beta) -> None:
    beta = Beta(alpha=alpha, beta=beta)

    assert beta.max_value == 1.0

    ax = beta.plot_pdf()
    assert isinstance(ax, plt.Axes)


def test_beta_uninformative() -> None:
    beta = Beta.uninformative()
    assert beta.alpha == 1.0
    assert beta.beta == 1.0


def test_beta_from_success_and_failures() -> None:
    beta = Beta.from_successes_and_failures(successes=0, failures=0)
    assert beta.alpha == 1.0
    assert beta.beta == 1.0


@pytest.mark.parametrize("mean", [0.025, 0.5, 0.75])
@pytest.mark.parametrize("alpha", [1, 10, 100])
def test_beta_mean_constructor(mean: float, alpha: float) -> None:
    beta = get_beta_param_from_mean_and_alpha(mean, alpha)
    dist = Beta.from_mean(mean=mean, alpha=alpha)

    assert beta > 0
    assert dist.beta == beta
    assert round(dist.dist.mean(), 3) == mean


beta_negative_binomial_dist_supported = (
    version.parse(scipy_version).release >= version.parse("1.12.0").release
)


@pytest.mark.skipif(
    beta_negative_binomial_dist_supported,
    reason="scipy version must be < 1.12.0",
)
def test_beta_negative_binomial_scipy_not_supported() -> None:
    beta = BetaNegativeBinomial(n=10, alpha=1, beta=1)
    # Requires the scipy version to be < 1.12.0
    with pytest.raises(NotImplementedError, match="scipy >= 1.12.0"):
        beta.plot_pmf()


@pytest.mark.skipif(
    not beta_negative_binomial_dist_supported,
    reason="scipy version must be >= 1.12.0",
)
def test_beta_negative_binomial_scipy_supported() -> None:
    beta = BetaNegativeBinomial(n=10, alpha=1, beta=1)
    # Requires the scipy version to be >= 1.12.0
    ax = beta.plot_pmf()
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "alpha, beta, result_alpha, result_beta",
    [
        (1, 1, 1, 1),
        (1.0, 2.0, 1, 2),
        # Array slicing
        (np.array([1, 10, 100]), 1, 1, 1),
        (1, np.array([1, 10, 100]), 1, 1),
        (np.array([1, 10, 100]), np.array([1, 10, 100]), 1, 1),
    ],
)
def test_slicing_single(alpha, beta, result_alpha, result_beta):
    beta = Beta(alpha=alpha, beta=beta)
    beta_slice = beta[0]
    assert isinstance(beta_slice, Beta)
    assert beta_slice.alpha == result_alpha
    assert beta_slice.beta == result_beta


@pytest.mark.parametrize(
    "alpha, beta, result_alpha, result_beta",
    [
        (1, 1, 1, 1),
        (1.0, 2.0, 1, 2),
        # Array slicing
        (np.array([1, 10, 100]), 1, np.array([1, 10]), 1),
        (1, np.array([1, 10, 100]), 1, np.array([1, 10])),
        (
            np.array([1, 10, 100]),
            np.array([1, 10, 100]),
            np.array([1, 10]),
            np.array([1, 10]),
        ),
    ],
)
def test_slicing_array(alpha, beta, result_alpha, result_beta):
    beta = Beta(alpha=alpha, beta=beta)
    beta_slice = beta[:2]
    assert isinstance(beta_slice, Beta)

    np.testing.assert_almost_equal(beta_slice.alpha, result_alpha)
    np.testing.assert_almost_equal(beta_slice.beta, result_beta)


@pytest.mark.parametrize(
    "dist, result_dist",
    [
        (Poisson(1), Poisson(2)),
        (Gamma(1, 1), Gamma(2, 1)),
        (NegativeBinomial(1, 1), NegativeBinomial(2, 1)),
        (Exponential(1), Gamma(2, 1)),
    ],
)
def test_distribution(dist, result_dist) -> None:
    if hasattr(dist, "plot_pdf"):
        with pytest.raises(ValueError):
            dist.plot_pdf()
    else:
        with pytest.raises(ValueError):
            dist.plot_pmf()

    dist.max_value = 10.0
    assert dist.max_value == 10.0

    if hasattr(dist, "plot_pdf"):
        ax = dist.plot_pdf()
    else:
        ax = dist.plot_pmf()
    assert isinstance(ax, plt.Axes)

    if result_dist is not None:
        other = 2 * dist
        assert other == result_dist

    lower, upper = -20, 20

    assert dist.min_value != lower
    assert dist.max_value != upper

    dist.set_bounds(lower, upper)

    assert dist.min_value == lower
    assert dist.max_value == upper


@pytest.mark.parametrize(
    "label",
    [
        "test",
        ["test 1", "test 2"],
        np.array(["test 1", "test 2"]),
        lambda i: f"test {i}",
    ],
)
def test_label(label):
    beta = Beta(alpha=1, beta=np.array([1, 2]))

    ax = beta.plot_pdf(label=label)
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "alpha",
    [
        np.array([1, 2, 3]),
        np.array([[1, 2, 3], [1, 1, 1]]),
        np.array([[1, 1, 1, 1], [2, 3, 1, 2], [1, 1, 1, 1]]),
    ],
)
def test_vectorized(alpha: np.ndarray) -> None:
    d = Dirichlet(alpha=alpha)
    assert d.dist.mean().shape == alpha.shape


def test_slice_multivariate_normal() -> None:
    mu = np.array([0, 1])
    cov = np.array([[1, 0], [0, 3]])

    mvn = MultivariateNormal(mu, cov)

    mvn_slice = mvn[0]
    assert isinstance(mvn_slice, Normal)
    assert mvn_slice.dist.mean() == mu[0]
    assert mvn_slice.dist.var() == cov[0, 0]

    mvn_slice = mvn[[1, 0]]
    assert isinstance(mvn_slice, MultivariateNormal)
    np.testing.assert_allclose(mvn_slice.dist.mean, mu[[1, 0]])
    np.testing.assert_allclose(mvn_slice.dist.cov, cov[[1, 0], :][:, [1, 0]])


def test_slice_multivariate_t() -> None:
    mu = np.array([0, 1])
    cov = np.array([[1, 0], [0, 3]])
    df = 2

    mvn = MultivariateStudentT(mu, cov, df)

    mvn_slice = mvn[0]
    assert isinstance(mvn_slice, StudentT)
    assert mvn_slice.dist.mean() == mu[0]
    assert mvn_slice.sigma == cov[0, 0]

    mvn_slice = mvn[[1, 0]]
    assert isinstance(mvn_slice, MultivariateStudentT)
    np.testing.assert_allclose(mvn_slice.dist.loc, mu[[1, 0]])
    np.testing.assert_allclose(mvn_slice.dist.shape, cov[[1, 0], :][:, [1, 0]])
    assert mvn_slice.dist.df == df


def test_normal_inverse_wishart() -> None:
    distribution = NormalInverseWishart(
        mu=np.array([0, 1]),
        kappa=1,
        nu=2,
        psi=np.array([[1, 0], [0, 1]]),
    )

    assert isinstance(distribution.inverse_wishart, InverseWishart)

    variance = distribution.sample_variance(size=1)
    assert variance.shape == (1, 2, 2)

    mean = distribution.sample_mean(size=1)
    assert mean.shape == (1, 2)

    _, variance = distribution.sample_mean(size=1, return_variance=True)
    assert variance.shape == (1, 2, 2)


@pytest.mark.parametrize("n_features", [1, 2, 3])
@pytest.mark.parametrize("n_samples", [1, 2, 10])
def test_normal_inverse_gamma(n_features, n_samples) -> None:
    mu = np.zeros(n_features)
    delta_inverse = np.eye(n_features)
    distribution = NormalInverseGamma(
        mu=mu,
        alpha=1,
        beta=1,
        delta_inverse=delta_inverse,
    )

    assert isinstance(distribution.inverse_gamma, InverseGamma)

    variance = distribution.sample_variance(size=n_samples)
    assert variance.shape == (n_samples,)

    mean = distribution.sample_mean(size=n_samples)

    if n_features == 1:
        assert mean.shape == (n_samples,)
    else:
        assert mean.shape == (n_samples, n_features)


@pytest.mark.parametrize(
    "a, b, q, size",
    [
        (1, 1, 1, (10,)),
        (np.array([1, 2, 3]), 1, 1, (10, 3)),
        (1, np.array([1, 2, 3]), 1, (10, 3)),
        (np.array([1, 2, 3]), np.array([1, 2, 3]), 1, (10, 3)),
        # Check for broadcasting
        (
            np.array([1, 2, 3]),
            np.array([1, 2])[:, None],
            np.array([1, 2, 3]),
            (10, 2, 3),
        ),
    ],
)
def test_compound_gamma(a, b, q, size) -> None:
    dist = CompoundGamma(alpha=1, beta=1, lam=1)
    assert dist.dist.rvs(size=size).shape == size


def test_binomial_max_value() -> None:
    n = np.array([10, 20, 15])
    p = 0.5
    binomial = Binomial(n=n, p=p)

    assert binomial.max_value == 20


def test_dirichlet_rvs() -> None:
    dirichlet = Dirichlet(alpha=np.array([[1, 2, 3], [1, 1, 1]]))

    dist = dirichlet.dist
    assert isinstance(dist, VectorizedDist)
    samples = dist.rvs(size=10)
    assert isinstance(samples, np.ndarray)


@pytest.mark.parametrize(
    "dist",
    [
        Binomial(n=10, p=0.5),
        Geometric(p=0.5),
        BetaBinomial(n=10, alpha=1, beta=1),
        Poisson(lam=1),
        BetaNegativeBinomial(n=10, alpha=1, beta=1),
        NegativeBinomial(n=10, p=0.5),
        Hypergeometric(N=100, k=5, n=10),
    ],
)
def test_plot_pmf(dist) -> None:
    dist.max_value = 10
    ax = dist.plot_pmf()
    assert isinstance(ax, plt.Axes)


@pytest.mark.parametrize(
    "dist",
    [
        Beta(alpha=1, beta=1),
        Gamma(alpha=1, beta=1),
        CompoundGamma(alpha=1, beta=1, lam=1),
        LogNormal(mu=1, sigma=1),
        ScaledInverseChiSquared(nu=1, sigma2=1),
        VonMises(mu=0, kappa=1),
        Lomax(alpha=1, lam=1),
        StudentT(mu=0, sigma=1, nu=10),
        InverseGamma(alpha=1, beta=1),
        Pareto(x_m=10, alpha=1),
        Uniform(low=10, high=20),
        Normal(mu=0, sigma=1),
        Exponential(lam=1),
    ],
)
def test_plot_pdf(dist) -> None:
    dist.max_value = 10
    dist.min_value = 0
    ax = dist.plot_pdf()
    assert isinstance(ax, plt.Axes)


def test_normal_gamma() -> None:
    normal_gamma = NormalGamma(
        mu=0,
        lam=1,
        alpha=1,
        beta=1,
    )

    assert normal_gamma.gamma == Gamma(alpha=1, beta=1)

    mean = normal_gamma.sample_mean(size=10)
    assert mean.shape == (10,)

    _, variance = normal_gamma.sample_mean(size=1, return_variance=True)
    assert variance.shape == (1,)


def test_scaled_inverse_chi_squared_round_trip() -> None:
    inverse_gamma = InverseGamma(alpha=1, beta=1)
    scaled_inverse_gamma = ScaledInverseChiSquared.from_inverse_gamma(inverse_gamma)
    back_again = scaled_inverse_gamma.to_inverse_gamma()

    assert inverse_gamma == back_again


def test_combining_poisson() -> None:
    poisson_1 = Poisson(lam=1)
    poisson_2 = Poisson(lam=2)
    poisson_3 = poisson_1 + poisson_2
    assert poisson_3 == Poisson(lam=3)


def test_scaling_of_normal() -> None:
    normal = Normal(mu=0, sigma=1)

    scaled_normal = 4 * normal
    assert scaled_normal == Normal(mu=0, sigma=2)


def test_normal_alternative_constructors() -> None:
    assert Normal.uninformative() == Normal(mu=0, sigma=1)
    assert Normal.from_mean_and_variance(mean=0, variance=4) == Normal(mu=0, sigma=2)
    assert Normal.from_mean_and_precision(mean=0, precision=1 / 4) == Normal(
        mu=0, sigma=2
    )
