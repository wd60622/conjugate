import pytest

import numpy as np

import matplotlib.pyplot as plt

from conjugate.distributions import (
    get_beta_param_from_mean_and_alpha, 
    Beta,
    Dirichlet,
    Gamma,
    Exponential,
    NegativeBinomial,
    Poisson,
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

@pytest.mark.parametrize("mean", [0.025, 0.5, 0.75])
@pytest.mark.parametrize("alpha", [1, 10, 100])
def test_beta_mean_constructor(mean: float, alpha: float) -> None: 
    beta = get_beta_param_from_mean_and_alpha(mean, alpha)
    dist = Beta.from_mean(mean=mean, alpha=alpha)

    assert beta > 0
    assert dist.beta == beta


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
