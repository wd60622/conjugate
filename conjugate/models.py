"""Common Conjugate models

Taken from https://en.wikipedia.org/wiki/Conjugate_prior

"""
from typing import Tuple

from conjugate.distributions import (
    Beta,
    Dirichlet,
    Gamma,
    NegativeBinomial,
    BetaBinomial,
)
from conjugate._typing import NUMERIC


def get_binomial_beta_posterior_params(
    alpha_prior: NUMERIC, beta_prior: NUMERIC, n: NUMERIC, x: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha_prior + x
    beta_post = beta_prior + (n - x)

    return alpha_post, beta_post


def binomial_beta(n, x, beta_prior: Beta) -> Beta:
    """Posterior distribution for a binomial likelihood with a beta prior"""
    alpha_post, beta_post = get_binomial_beta_posterior_params(
        beta_prior.alpha, beta_prior.beta, n, x
    )

    return Beta(alpha=alpha_post, beta=beta_post)


def binomial_beta_posterior_predictive(n, beta: Beta) -> BetaBinomial:
    """Posterior predictive distribution for a binomial likelihood with a beta prior"""
    return BetaBinomial(n=n, alpha=beta.alpha, beta=beta.beta)


def get_dirichlet_posterior_params(alpha_prior: NUMERIC, x: NUMERIC) -> NUMERIC:
    try: 
        return alpha_prior + x
    except Exception:
        return [alpha_prior_i + x_i for alpha_prior_i, x_i in zip(alpha_prior, x)]


get_categorical_dirichlet_posterior_params = get_dirichlet_posterior_params


def categorical_dirichlet(x: NUMERIC, dirichlet_prior: Dirichlet) -> Dirichlet:
    alpha_post = get_dirichlet_posterior_params(dirichlet_prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


get_multinomial_dirichlet_posterior_params = get_dirichlet_posterior_params

multinomial_dirichlet = categorical_dirichlet


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
exponential_gamma = poisson_gamma
