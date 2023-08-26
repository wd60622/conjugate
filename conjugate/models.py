"""For more on these models, check out the <a href=https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions>Conjugate Prior Wikipedia Table</a>

Below are the supported models

"""
from typing import Tuple

import numpy as np

from conjugate.distributions import (
    Beta,
    Dirichlet,
    Gamma,
    NegativeBinomial,
    BetaNegativeBinomial,
    BetaBinomial,
    NormalInverseGamma,
)
from conjugate._typing import NUMERIC


def get_binomial_beta_posterior_params(
    alpha_prior: NUMERIC, beta_prior: NUMERIC, n: NUMERIC, x: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha_prior + x
    beta_post = beta_prior + (n - x)

    return alpha_post, beta_post


def binomial_beta(n: NUMERIC, x: NUMERIC, beta_prior: Beta) -> Beta:
    """Posterior distribution for a binomial likelihood with a beta prior.

    Args:
        n: total number of trials
        x: sucesses from that trials
        beta_prior: Beta distribution prior

    Returns:
        Beta distribution posterior

    """
    alpha_post, beta_post = get_binomial_beta_posterior_params(
        beta_prior.alpha, beta_prior.beta, n, x
    )

    return Beta(alpha=alpha_post, beta=beta_post)


def binomial_beta_posterior_predictive(n: NUMERIC, beta: Beta) -> BetaBinomial:
    """Posterior predictive distribution for a binomial likelihood with a beta prior.

    Args:
        n: number of trials
        beta: Beta distribution

    Returns:
        BetaBinomial posterior predictive distribution

    """
    return BetaBinomial(n=n, alpha=beta.alpha, beta=beta.beta)


def negative_binomial_beta(r, n, x, beta_prior: Beta) -> Beta:
    """Posterior distribution for a negative binomial likelihood with a beta prior.

    Args:

    """
    alpha_post = beta_prior.alpha + (r * n)
    beta_post = beta_prior.beta + x

    return Beta(alpha=alpha_post, beta=beta_post)


def negative_binomial_beta_posterior_predictive(r, beta: Beta) -> BetaNegativeBinomial:
    """Posterior predictive distribution for a negative binomial likelihood with a beta prior"""
    return BetaNegativeBinomial(r=r, alpha=beta.alpha, beta=beta.beta)


def geometric_beta(x_total, n, beta_prior: Beta, one_start: bool = True) -> Beta:
    """Posterior distribution for a geometric likelihood with a beta prior.

    Args:
        x_total: sum of all trials outcomes
        n: total number of trials
        beta_prior: Beta distribution prior
        one_start: whether to outcomes start at 1, defaults to True. False is 0 start.

    Returns:
        Beta distribution posterior

    """
    alpha_post = beta_prior.alpha + n
    beta_post = beta_prior.beta + x_total

    if one_start:
        beta_post = beta_post - n

    return Beta(alpha=alpha_post, beta=beta_post)


def get_dirichlet_posterior_params(alpha_prior: NUMERIC, x: NUMERIC) -> NUMERIC:
    try:
        return alpha_prior + x
    except Exception:
        return [alpha_prior_i + x_i for alpha_prior_i, x_i in zip(alpha_prior, x)]


def get_categorical_dirichlet_posterior_params(
    alpha_prior: NUMERIC, x: NUMERIC
) -> NUMERIC:
    return get_dirichlet_posterior_params(alpha_prior, x)


def categorical_dirichlet(x: NUMERIC, dirichlet_prior: Dirichlet) -> Dirichlet:
    """Posterior distribution of Categorical model with Dirichlet prior."""
    alpha_post = get_dirichlet_posterior_params(dirichlet_prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


def get_multi_categorical_dirichlet_posterior_params(
    alpha_prior: NUMERIC, x: NUMERIC
) -> NUMERIC:
    return get_dirichlet_posterior_params(alpha_prior, x)


def multinomial_dirichlet(x: NUMERIC, dirichlet_prior: Dirichlet) -> Dirichlet:
    """Posterior distribution of Multinomial model with Dirichlet prior.

    Args:
        x: counts
        dirichlet_prior: Dirichlet prior on the counts

    Returns:
        Dirichlet posterior distribution

    """
    alpha_post = get_dirichlet_posterior_params(dirichlet_prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


def get_poisson_gamma_posterior_params(
    alpha: NUMERIC, beta: NUMERIC, x_total: NUMERIC, n: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha + x_total
    beta_post = beta + n

    return alpha_post, beta_post


def poisson_gamma(x_total: NUMERIC, n: NUMERIC, gamma_prior: Gamma) -> Gamma:
    """Posterior distribution for a poisson likelihood with a gamma prior"""
    alpha_post, beta_post = get_poisson_gamma_posterior_params(
        alpha=gamma_prior.alpha, beta=gamma_prior.beta, x_total=x_total, n=n
    )

    return Gamma(alpha=alpha_post, beta=beta_post)


def poisson_gamma_posterior_predictive(
    gamma: Gamma, n: NUMERIC = 1
) -> NegativeBinomial:
    """Posterior predictive distribution for a poisson likelihood with a gamma prior

    Args:
        gamma: Gamma distribution
        n: Number of trials for each sample, defaults to 1.
            Can be used to scale the distributions to a different unit of time.

    Returns:
        NegativeBinomial distribution related to posterior predictive

    """
    n = n * gamma.alpha
    p = gamma.beta / (1 + gamma.beta)

    return NegativeBinomial(n=n, p=p)


# Just happen to be the same as above
get_exponential_gamma_posterior_params = get_poisson_gamma_posterior_params


def exponetial_gamma(x_total: NUMERIC, n: NUMERIC, gamma_prior: Gamma) -> Gamma:
    """Posterior distribution for an exponential likelihood with a gamma prior"""
    alpha_post, beta_post = get_exponential_gamma_posterior_params(
        alpha=gamma_prior.alpha, beta=gamma_prior.beta, x_total=x_total, n=n
    )

    return Gamma(alpha=alpha_post, beta=beta_post)


def linear_regression(
    X: NUMERIC,
    y: NUMERIC,
    normal_inverse_gamma_prior: NormalInverseGamma,
    inv=np.linalg.inv,
) -> NormalInverseGamma:
    """Posterior distribution for a linear regression model with a normal inverse gamma prior.

    Derivation taken from this blog [here](https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/).

    Args:
        X: design matrix
        y: response vector
        normal_inverse_gamma_prior: NormalInverseGamma prior
        inv: function to invert matrix, defaults to np.linalg.inv

    Returns:
        NormalInverseGamma posterior distribution

    """
    N = X.shape[0]

    delta = inv(normal_inverse_gamma_prior.delta_inverse)

    delta_post = (X.T @ X) + delta
    delta_post_inverse = inv(delta_post)

    mu_post = (
        # (B, B)
        delta_post_inverse
        # (B, 1)
        # (B, B) * (B, 1) +  (B, N) * (N, 1)
        @ (delta @ normal_inverse_gamma_prior.mu + X.T @ y)
    )

    alpha_post = normal_inverse_gamma_prior.alpha + (0.5 * N)
    beta_post = normal_inverse_gamma_prior.beta + (
        0.5
        * (
            (y.T @ y)
            # (1, B) * (B, B) * (B, 1)
            + (normal_inverse_gamma_prior.mu.T @ delta @ normal_inverse_gamma_prior.mu)
            # (1, B) * (B, B) * (B, 1)
            - (mu_post.T @ delta_post @ mu_post)
        )
    )

    return NormalInverseGamma(
        mu=mu_post, delta_inverse=delta_post_inverse, alpha=alpha_post, beta=beta_post
    )
