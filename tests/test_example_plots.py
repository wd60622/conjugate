import pytest

import numpy as np

import matplotlib.pyplot as plt

from conjugate.distributions import Beta, Dirichlet, Gamma, Normal
from conjugate.models import binomial_beta, binomial_beta_posterior_predictive


@pytest.mark.mpl_image_compare
def test_label():
    beta = Beta(1, 1)
    ax = beta.plot_pdf(label="Uniform")
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_multiple_labels_str():
    beta = Beta(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ax = beta.plot_pdf(label="Beta")
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_multiple_with_labels():
    beta = Beta(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ax = beta.plot_pdf(label=["First Beta", "Second Beta", "Third Beta"])
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_skip_label() -> None:
    beta = Beta(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ax = beta.plot_pdf(label=["First Beta", None, "Third Beta"])
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_different_distributions() -> None:
    beta = Beta(1, np.array([1, 2]))
    gamma = Gamma(1, 1)
    normal = Normal(0, 1)

    ax = beta.plot_pdf(label="Beta")
    gamma.set_bounds(0, upper=5).plot_pdf(ax=ax, label="Gamma")
    normal.set_bounds(-5, 5).plot_pdf(ax=ax, label="Normal")

    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_analysis() -> None:
    prior = Beta(1, 1)

    N = 10
    x = 8

    posterior = binomial_beta(n=N, x=x, beta_prior=prior)

    prior_predictive = binomial_beta_posterior_predictive(
        n=5,
        beta=prior,
    )
    posterior_predictive = binomial_beta_posterior_predictive(
        n=5,
        beta=posterior,
    )

    fig, axes = plt.subplots(1, 2, figsize=(15, 7))

    ax = axes[0]
    posterior.plot_pdf(ax=ax, label="Posterior")
    prior.plot_pdf(ax=ax, label="Prior")
    ax.axvline(x=x / N, color="black", linestyle="--", label="MLE", ymax=0.5, lw=2)
    ax.legend(loc="upper left")

    ax = axes[1]
    posterior_predictive.plot_pmf(ax=ax, label="Posterior Predictive")
    prior_predictive.plot_pmf(ax=ax, label="Prior Predictive")
    ax.legend(loc="upper left")

    return fig


@pytest.mark.mpl_image_compare
def test_dirichlet() -> None:
    dirichlet = Dirichlet(np.array([1, 2, 3]))
    ax = dirichlet.plot_pdf(random_state=0)
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dirichlet_labels() -> None:
    dirichlet = Dirichlet(np.array([1, 2, 3]))
    ax = dirichlet.plot_pdf(random_state=0, label="Category")
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_dirichlet_multiple_labels() -> None:
    dirichlet = Dirichlet(np.array([1, 2, 3]))
    ax = dirichlet.plot_pdf(
        random_state=0, label=["First Category", "Second Category", "Third Category"]
    )
    ax.legend()
    return plt.gcf()
