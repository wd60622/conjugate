import pytest

from dataclasses import dataclass

from pypika import Field

import numpy as np

import matplotlib.pyplot as plt

from conjugate.distributions import (
    Beta,
    Dirichlet,
    Gamma,
    NegativeBinomial,
    Poisson,
)
from conjugate.models import (
    get_binomial_beta_posterior_params,
    binomial_beta,
    multinomial_dirichlet,
    get_poisson_gamma_posterior_params,
    poisson_gamma,
    poisson_gamma_posterior_predictive,
    get_exponential_gamma_posterior_params,
)


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
