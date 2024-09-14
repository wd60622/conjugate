"""For more on these models, check out the <a href=https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions>Conjugate Prior Wikipedia Table</a>

# Supported Likelihoods

## Discrete

- Bernoulli / Binomial
- Negative Binomial
- Geometric
- Hypergeometric
- Poisson
- Categorical / Multinomial

## Continuous

- Normal
- Multivariate Normal
- Linear Regression (Normal)
- Log Normal
- Uniform
- Exponential
- Pareto
- Gamma
- Beta
- Von Mises

# Model Functions

Below are the supported models

"""

from typing import Tuple

import numpy as np

from conjugate.distributions import (
    Beta,
    BetaProportional,
    CompoundGamma,
    Dirichlet,
    DirichletMultinomial,
    Gamma,
    GammaKnownRateProportional,
    GammaProportional,
    NegativeBinomial,
    BetaNegativeBinomial,
    BetaBinomial,
    Pareto,
    InverseGamma,
    Normal,
    NormalInverseGamma,
    StudentT,
    MultivariateStudentT,
    Lomax,
    VonMisesKnownConcentration,
    VonMisesKnownDirectionProportional,
    InverseWishart,
    NormalInverseWishart,
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
        x: successes from that trials
        beta_prior: Beta distribution prior

    Returns:
        Beta distribution posterior

    Examples:
        A / B test example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Beta
        from conjugate.models import binomial_beta

        impressions = np.array([100, 250])
        clicks = np.array([10, 35])

        prior = Beta(1, 1)

        posterior = binomial_beta(
            n=impressions,
            x=clicks,
            beta_prior=prior
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 0.5).plot_pdf(ax=ax, label=["A", "B"])
        prior.set_bounds(0, 0.5).plot_pdf(ax=ax, label="prior")
        ax.legend()
        plt.show()
        ```

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

    Examples:
        A / B test example with 100 new impressions

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Beta
        from conjugate.models import binomial_beta, binomial_beta_posterior_predictive

        impressions = np.array([100, 250])
        clicks = np.array([10, 35])

        prior = Beta(1, 1)
        posterior = binomial_beta(
            n=impressions,
            x=clicks,
            beta_prior=prior
        )
        posterior_predictive = binomial_beta_posterior_predictive(
            n=100,
            beta=posterior
        )


        ax = plt.subplot(111)
        ax.set_title("Posterior Predictive Distribution with 100 new impressions")
        posterior_predictive.set_bounds(0, 50).plot_pmf(
            ax=ax,
            label=["A", "B"],
        )
        plt.show()
        ```
    """
    return BetaBinomial(n=n, alpha=beta.alpha, beta=beta.beta)


def bernoulli_beta(x: NUMERIC, beta_prior: Beta) -> Beta:
    """Posterior distribution for a bernoulli likelihood with a beta prior.

    Args:
        x: successes from a single trial
        beta_prior: Beta distribution prior

    Returns:
        Beta distribution posterior

    Examples:
       Information gain from a single coin flip

        ```python
        from conjugate.distributions import Beta
        from conjugate.models import bernoulli_beta

        prior = Beta(1, 1)

        # Positive outcome
        x = 1
        posterior = bernoulli_beta(
            x=x,
            beta_prior=prior
        )

        posterior.dist.ppf([0.025, 0.975])
        # array([0.15811388, 0.98742088])
        ```

    """
    return binomial_beta(n=1, x=x, beta_prior=beta_prior)


def bernoulli_beta_posterior_predictive(beta: Beta) -> BetaBinomial:
    """Posterior predictive distribution for a bernoulli likelihood with a beta prior.

    Args:
        beta: Beta distribution

    Returns:
        BetaBinomial posterior predictive distribution

    """
    return binomial_beta_posterior_predictive(n=1, beta=beta)


def negative_binomial_beta(
    r: NUMERIC, n: NUMERIC, x: NUMERIC, beta_prior: Beta
) -> Beta:
    """Posterior distribution for a negative binomial likelihood with a beta prior.

    Assumed known number of failures r

    Args:
        r: number of failures
        n: number of trials
        x: number of successes
        beta_prior: Beta distribution prior

    Returns:
        Beta distribution posterior

    """
    alpha_post = beta_prior.alpha + (r * n)
    beta_post = beta_prior.beta + x

    return Beta(alpha=alpha_post, beta=beta_post)


def negative_binomial_beta_posterior_predictive(
    r: NUMERIC, beta: Beta
) -> BetaNegativeBinomial:
    """Posterior predictive distribution for a negative binomial likelihood with a beta prior

    Assumed known number of failures r

    Args:
        r: number of failures
        beta: Beta distribution

    Returns:
        BetaNegativeBinomial posterior predictive distribution

    """
    return BetaNegativeBinomial(n=r, alpha=beta.alpha, beta=beta.beta)


def hypergeometric_beta_binomial(
    x_total: NUMERIC, n: NUMERIC, beta_binomial_prior: BetaBinomial
) -> BetaBinomial:
    """Hypergeometric likelihood with a BetaBinomial prior.

    The total population size is N and is known. Encode it in the BetaBinomial
        prior as n=N

    Args:
        x_total: sum of all trials outcomes
        n: total number of trials
        beta_binomial_prior: BetaBinomial prior
            n is the known N / total population size

    Returns:
        BetaBinomial posterior distribution

    """
    n = beta_binomial_prior.n
    alpha_post = beta_binomial_prior.alpha + x_total
    beta_post = beta_binomial_prior.beta + (n - x_total)

    return BetaBinomial(n=n, alpha=alpha_post, beta=beta_post)


def geometric_beta(x_total, n, beta_prior: Beta, one_start: bool = True) -> Beta:
    """Posterior distribution for a geometric likelihood with a beta prior.

    Args:
        x_total: sum of all trials outcomes
        n: total number of trials
        beta_prior: Beta distribution prior
        one_start: whether to outcomes start at 1, defaults to True. False is 0 start.

    Returns:
        Beta distribution posterior

    Examples:
        Number of usages until user has good experience

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Beta
        from conjugate.models import geometric_beta

        data = np.array([3, 1, 1, 3, 2, 1])

        prior = Beta(1, 1)
        posterior = geometric_beta(
            x_total=data.sum(),
            n=data.size,
            beta_prior=prior
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 1).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, 1).plot_pdf(ax=ax, label="prior")
        ax.legend()
        ax.set(xlabel="chance of good experience")
        plt.show()
        ```

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
    """Posterior distribution of Categorical model with Dirichlet prior.

    Args:
        x: counts
        dirichlet_prior: Dirichlet prior on the counts

    Returns:
        Dirichlet posterior distribution

    """
    alpha_post = get_dirichlet_posterior_params(dirichlet_prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


def categorical_dirichlet_posterior_predictive(
    dirichlet: Dirichlet, n: NUMERIC = 1
) -> DirichletMultinomial:
    """Posterior predictive distribution of Categorical model with Dirichlet prior.

    Args:
        dirichlet: Dirichlet distribution
        n: Number of trials for each sample, defaults to 1.

    """

    return DirichletMultinomial(n=n, alpha=dirichlet.alpha)


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

    Examples:
        Personal preference for ice cream flavors

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Dirichlet
        from conjugate.models import multinomial_dirichlet

        kinds = ["chocolate", "vanilla", "strawberry"]
        data = np.array([
            [5, 2, 1],
            [3, 1, 0],
            [3, 2, 0],
        ])

        prior = Dirichlet([1, 1, 1])
        posterior = multinomial_dirichlet(
            x=data.sum(axis=0),
            dirichlet_prior=prior
        )

        ax = plt.subplot(111)
        posterior.plot_pdf(ax=ax, label=kinds)
        ax.legend()
        ax.set(xlabel="Flavor Preference")
        plt.show()
        ```

    """
    alpha_post = get_dirichlet_posterior_params(dirichlet_prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


def multinomial_dirichlet_posterior_predictive(
    dirichlet: Dirichlet, n: NUMERIC = 1
) -> DirichletMultinomial:
    """Posterior predictive distribution of Multinomial model with Dirichlet prior.

    Args:
        dirichlet: Dirichlet distribution
        n: Number of trials for each sample, defaults to 1.

    """

    return DirichletMultinomial(n=n, alpha=dirichlet.alpha)


def get_poisson_gamma_posterior_params(
    alpha: NUMERIC, beta: NUMERIC, x_total: NUMERIC, n: NUMERIC
) -> Tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha + x_total
    beta_post = beta + n

    return alpha_post, beta_post


def poisson_gamma(x_total: NUMERIC, n: NUMERIC, gamma_prior: Gamma) -> Gamma:
    """Posterior distribution for a poisson likelihood with a gamma prior.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        gamma_prior: Gamma prior

    Returns:
        Gamma posterior distribution

    """
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


def exponential_gamma(x_total: NUMERIC, n: NUMERIC, gamma_prior: Gamma) -> Gamma:
    """Posterior distribution for an exponential likelihood with a gamma prior.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        gamma_prior: Gamma prior

    Returns:
        Gamma posterior distribution

    """
    alpha_post, beta_post = get_exponential_gamma_posterior_params(
        alpha=gamma_prior.alpha, beta=gamma_prior.beta, x_total=x_total, n=n
    )

    return Gamma(alpha=alpha_post, beta=beta_post)


def exponential_gamma_posterior_predictive(gamma: Gamma) -> Lomax:
    """Posterior predictive distribution for an exponential likelihood with a gamma prior

    Args:
        gamma: Gamma distribution

    Returns:
        Lomax distribution related to posterior predictive

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Exponential, Gamma
        from conjugate.models import exponential_gamma, expotential_gamma_posterior_predictive

        true = Exponential(1)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Gamma(1, 1)

        posterior = exponential_gamma(
            n=n_samples,
            x_total=data.sum(),
            gamma_prior=prior
        )

        prior_predictive = expotential_gamma_posterior_predictive(prior)
        posterior_predictive = expotential_gamma_posterior_predictive(posterior)

        ax = plt.subplot(111)
        prior_predictive.set_bounds(0, 2.5).plot_pdf(ax=ax, label="prior predictive")
        true.set_bounds(0, 2.5).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(0, 2.5).plot_pdf(ax=ax, label="posterior predictive")
        ax.legend()
        plt.show()
        ```
    """
    return Lomax(alpha=gamma.beta, lam=gamma.alpha)


def gamma_known_shape(
    x_total: NUMERIC, n: NUMERIC, alpha: NUMERIC, gamma_prior: Gamma
) -> Gamma:
    """Gamma likelihood with a gamma prior.

    The shape parameter of the likelihood is assumed to be known.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        alpha: known shape parameter
        gamma_prior: Gamma prior

    Returns:
        Gamma posterior distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Gamma
        from conjugate.models import gamma_known_shape

        known_shape = 2
        unknown_rate = 5
        true = Gamma(known_shape, unknown_rate)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Gamma(1, 1)

        posterior = gamma_known_shape(
            n=n_samples,
            x_total=data.sum(),
            alpha=known_shape,
            gamma_prior=prior
        )

        bound = 10
        ax = plt.subplot(111)
        posterior.set_bounds(0, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_rate, color="black", linestyle="--", label="true rate")
        ax.legend()
        plt.show()
        ```

    """
    alpha_post = gamma_prior.alpha + n * alpha
    beta_post = gamma_prior.beta + x_total

    return Gamma(alpha=alpha_post, beta=beta_post)


def gamma_known_shape_posterior_predictive(
    gamma: Gamma, alpha: NUMERIC
) -> CompoundGamma:
    """Posterior predictive distribution for a gamma likelihood with a gamma prior

    Args:
        gamma: Gamma distribution
        alpha: known shape parameter

    Returns:
        CompoundGamma distribution related to posterior predictive

    """
    return CompoundGamma(alpha=alpha, beta=gamma.alpha, lam=gamma.beta)


def normal_known_variance(
    x_total: NUMERIC,
    n: NUMERIC,
    var: NUMERIC,
    normal_prior: Normal,
) -> Normal:
    """Posterior distribution for a normal likelihood with known variance and a normal prior on mean.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        var: known variance
        normal_prior: Normal prior for mean

    Returns:
        Normal posterior distribution for the mean

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_variance

        unknown_mu = 0
        known_var = 2.5
        true = Normal(unknown_mu, known_var**0.5)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Normal(0, 10)

        posterior = normal_known_variance(
            n=n_samples,
            x_total=data.sum(),
            var=known_var,
            normal_prior=prior
        )

        bound = 5
        ax = plt.subplot(111)
        posterior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_mu, color="black", linestyle="--", label="true mu")
        ax.legend()
        plt.show()
        ```

    """
    mu_post = ((normal_prior.mu / normal_prior.sigma**2) + (x_total / var)) / (
        (1 / normal_prior.sigma**2) + (n / var)
    )

    var_post = 1 / ((1 / normal_prior.sigma**2) + (n / var))

    return Normal(mu=mu_post, sigma=var_post**0.5)


def normal_known_variance_posterior_predictive(var: NUMERIC, normal: Normal) -> Normal:
    """Posterior predictive distribution for a normal likelihood with known variance and a normal prior on mean.

    Args:
        var: known variance
        normal: Normal posterior distribution for the mean

    Returns:
        Normal posterior predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_variance, normal_known_variance_posterior_predictive

        unknown_mu = 0
        known_var = 2.5
        true = Normal(unknown_mu, known_var**0.5)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Normal(0, 10)

        posterior = normal_known_variance(
            n=n_samples,
            x_total=data.sum(),
            var=known_var,
            normal_prior=prior
        )

        prior_predictive = normal_known_variance_posterior_predictive(
            var=known_var,
            normal=prior
        )
        posterior_predictive = normal_known_variance_posterior_predictive(
            var=known_var,
            normal=posterior
        )

        bound = 5
        ax = plt.subplot(111)
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        ax.legend()
        plt.show()
        ```

    """
    var_posterior_predictive = var + normal.sigma**2
    return Normal(mu=normal.mu, sigma=var_posterior_predictive**0.5)


def normal_known_precision(
    x_total: NUMERIC,
    n: NUMERIC,
    precision: NUMERIC,
    normal_prior: Normal,
) -> Normal:
    """Posterior distribution for a normal likelihood with known precision and a normal prior on mean.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        precision: known precision
        normal_prior: Normal prior for mean

    Returns:
        Normal posterior distribution for the mean

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_precision

        unknown_mu = 0
        known_precision = 0.5
        true = Normal.from_mean_and_precision(unknown_mu, known_precision**0.5)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Normal(0, 10)

        posterior = normal_known_precision(
            n=n_samples,
            x_total=data.sum(),
            precision=known_precision,
            normal_prior=prior
        )

        bound = 5
        ax = plt.subplot(111)
        posterior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_mu, color="black", linestyle="--", label="true mu")
        ax.legend()
        plt.show()
        ```

    """
    return normal_known_variance(
        x_total=x_total,
        n=n,
        var=1 / precision,
        normal_prior=normal_prior,
    )


def normal_known_precision_posterior_predictive(
    precision: NUMERIC, normal: Normal
) -> Normal:
    """Posterior predictive distribution for a normal likelihood with known precision and a normal prior on mean.

    Args:
        precision: known precision
        normal: Normal posterior distribution for the mean

    Returns:
        Normal posterior predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_precision, normal_known_precision_posterior_predictive

        unknown_mu = 0
        known_precision = 0.5
        true = Normal.from_mean_and_precision(unknown_mu, known_precision)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Normal(0, 10)

        posterior = normal_known_precision(
            n=n_samples,
            x_total=data.sum(),
            precision=known_precision,
            normal_prior=prior
        )

        prior_predictive = normal_known_precision_posterior_predictive(
            precision=known_precision,
            normal=prior
        )
        posterior_predictive = normal_known_precision_posterior_predictive(
            precision=known_precision,
            normal=posterior
        )

        bound = 5
        ax = plt.subplot(111)
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        ax.legend()
        plt.show()
        ```

    """
    return normal_known_variance_posterior_predictive(
        var=1 / precision,
        normal=normal,
    )


def normal_known_mean(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    mu: NUMERIC,
    inverse_gamma_prior: InverseGamma,
) -> InverseGamma:
    """Posterior distribution for a normal likelihood with a known mean and a variance prior.

    Args:
        x_total: sum of all outcomes
        x2_total: sum of all outcomes squared
        n: total number of samples in x_total
        mu: known mean
        inverse_gamma_prior: InverseGamma prior for variance

    Returns:
        InverseGamma posterior distribution for the variance

    """
    alpha_post = inverse_gamma_prior.alpha + (n / 2)
    beta_post = inverse_gamma_prior.beta + (
        0.5 * (x2_total - (2 * mu * x_total) + (n * (mu**2)))
    )

    return InverseGamma(alpha=alpha_post, beta=beta_post)


def normal_known_mean_posterior_predictive(
    mu: NUMERIC, inverse_gamma: InverseGamma
) -> StudentT:
    """Posterior predictive distribution for a normal likelihood with a known mean and a variance prior.

    Args:
        mu: known mean
        inverse_gamma: InverseGamma prior

    Returns:
        StudentT posterior predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal, InverseGamma
        from conjugate.models import normal_known_mean, normal_known_mean_posterior_predictive

        unknown_var = 2.5
        known_mu = 0
        true = Normal(known_mu, unknown_var**0.5)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = InverseGamma(1, 1)

        posterior = normal_known_mean(
            n=n_samples,
            x_total=data.sum(),
            x2_total=(data**2).sum(),
            mu=known_mu,
            inverse_gamma_prior=prior
        )

        bound = 5
        ax = plt.subplot(111)
        prior_predictive = normal_known_mean_posterior_predictive(
            mu=known_mu,
            inverse_gamma=prior
        )
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive = normal_known_mean_posterior_predictive(
            mu=known_mu,
            inverse_gamma=posterior
        )
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        ax.legend()
        plt.show()
        ```

    """
    return StudentT(
        mu=mu,
        sigma=(inverse_gamma.beta / inverse_gamma.alpha) ** 0.5,
        nu=2 * inverse_gamma.alpha,
    )


def normal_normal_inverse_gamma(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    normal_inverse_gamma_prior: NormalInverseGamma,
) -> NormalInverseGamma:
    """Posterior distribution for a normal likelihood with a normal inverse gamma prior.

    Derivation from paper [here](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).

    Args:
        x_total: sum of all outcomes
        x2_total: sum of all outcomes squared
        n: total number of samples in x_total and x2_total
        normal_inverse_gamma_prior: NormalInverseGamma prior

    Returns:
        NormalInverseGamma posterior distribution

    """
    x_mean = x_total / n

    nu_0_inv = normal_inverse_gamma_prior.nu

    nu_post_inv = nu_0_inv + n
    mu_post = ((nu_0_inv * normal_inverse_gamma_prior.mu) + (n * x_mean)) / nu_post_inv

    alpha_post = normal_inverse_gamma_prior.alpha + (n / 2)
    beta_post = normal_inverse_gamma_prior.beta + 0.5 * (
        (normal_inverse_gamma_prior.mu**2) * nu_0_inv
        + x2_total
        - (mu_post**2) * nu_post_inv
    )

    return NormalInverseGamma(
        mu=mu_post,
        nu=nu_post_inv,
        alpha=alpha_post,
        beta=beta_post,
    )


def normal_normal_inverse_gamma_posterior_predictive(
    normal_inverse_gamma: NormalInverseGamma,
) -> StudentT:
    """Posterior predictive distribution for a normal likelihood with a normal inverse gamma prior.

    Args:
        normal_inverse_gamma: NormalInverseGamma posterior

    Returns:
        StudentT posterior predictive distribution

    """
    var = (
        normal_inverse_gamma.beta
        * (normal_inverse_gamma.nu + 1)
        / (normal_inverse_gamma.nu * normal_inverse_gamma.alpha)
    )
    return StudentT(
        mu=normal_inverse_gamma.mu,
        sigma=var**0.5,
        nu=2 * normal_inverse_gamma.alpha,
    )


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


def linear_regression_posterior_predictive(
    normal_inverse_gamma: NormalInverseGamma, X: NUMERIC, eye=np.eye
) -> MultivariateStudentT:
    """Posterior predictive distribution for a linear regression model with a normal inverse gamma prior.

    Args:
        normal_inverse_gamma: NormalInverseGamma posterior
        X: design matrix
        eye: function to get identity matrix, defaults to np.eye

    Returns:
        MultivariateStudentT posterior predictive distribution

    """
    mu = X @ normal_inverse_gamma.mu
    sigma = (normal_inverse_gamma.beta / normal_inverse_gamma.alpha) * (
        eye(X.shape[0]) + (X @ normal_inverse_gamma.delta_inverse @ X.T)
    )
    nu = 2 * normal_inverse_gamma.alpha

    return MultivariateStudentT(
        mu=mu,
        sigma=sigma,
        nu=nu,
    )


def uniform_pareto(
    x_max: NUMERIC, n: NUMERIC, pareto_prior: Pareto, max_fn=np.maximum
) -> Pareto:
    """Posterior distribution for a uniform likelihood with a pareto prior.

    Args:
        x_max: maximum value
        n: number of samples
        pareto_prior: Pareto prior
        max_fn: elementwise max function, defaults to np.maximum

    Returns:
        Pareto posterior distribution

    Examples:
        Get the posterior for this model with simulated data:

        ```python
        from conjugate.distributions import Uniform, Pareto
        from conjugate.models import uniform_pareto

        true_max = 5
        true = Uniform(0, true_max)

        n_samples = 10
        data = true.dist.rvs(size=n_samples)

        prior = Pareto(1, 1)

        posterior = uniform_pareto(
            x_max=data.max(),
            n=n_samples,
            pareto_prior=prior
        )
        ```

    """
    alpha_post = pareto_prior.alpha + n
    x_m_post = max_fn(pareto_prior.x_m, x_max)

    return Pareto(x_m=x_m_post, alpha=alpha_post)


def pareto_gamma(
    n: NUMERIC, ln_x_total: NUMERIC, x_m: NUMERIC, gamma_prior: Gamma, ln=np.log
) -> Gamma:
    """Posterior distribution for a pareto likelihood with a gamma prior.

    The parameter x_m is assumed to be known.

    Args:
        n: number of samples
        ln_x_total: sum of the log of all outcomes
        x_m: The known minimum value
        gamma_prior: Gamma prior
        ln: function to take the natural log, defaults to np.log

    Returns:
        Gamma posterior distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Pareto, Gamma
        from conjugate.models import pareto_gamma

        x_m_known = 1
        true = Pareto(x_m_known, 1)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Gamma(1, 1)

        posterior = pareto_gamma(
            n=n_samples,
            ln_x_total=np.log(data).sum(),
            x_m=x_m_known,
            gamma_prior=prior
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 2.5).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, 2.5).plot_pdf(ax=ax, label="prior")
        ax.axvline(x_m_known, color="black", linestyle="--", label="true x_m")
        ax.legend()
        plt.show()
        ```

    """
    alpha_post = gamma_prior.alpha + n
    beta_post = gamma_prior.beta + ln_x_total - n * ln(x_m)

    return Gamma(alpha=alpha_post, beta=beta_post)


def gamma(
    x_total: NUMERIC,
    x_prod: NUMERIC,
    n: NUMERIC,
    proportional_prior: GammaProportional,
) -> GammaProportional:
    """Posterior distribution for a gamma likelihood.

    Inference on alpha and beta

    Args:
        x_total: sum of all outcomes
        x_prod: product of all outcomes
        n: total number of samples in x_total and x_prod
        proportional_prior: GammaProportional prior

    Returns:
        GammaProportional posterior distribution

    """
    p_post = proportional_prior.p * x_prod
    q_post = proportional_prior.q + x_total
    r_post = proportional_prior.r + n
    s_post = proportional_prior.s + n

    return GammaProportional(p=p_post, q=q_post, r=r_post, s=s_post)


def gamma_known_rate(
    x_prod: NUMERIC,
    n: NUMERIC,
    beta: NUMERIC,
    proportional_prior: GammaKnownRateProportional,
) -> GammaKnownRateProportional:
    """Posterior distribution for a gamma likelihood.

    The rate beta is assumed to be known.

    Args:
        x_prod: product of all outcomes
        n: total number of samples in x_prod
        beta: known rate parameter

    Returns:
        GammaKnownRateProportional posterior distribution

    """
    a_post = proportional_prior.a * x_prod
    b_post = proportional_prior.b + n
    c_post = proportional_prior.c + n

    return GammaKnownRateProportional(a=a_post, b=b_post, c=c_post)


def beta(
    x_prod: NUMERIC,
    one_minus_x_prod: NUMERIC,
    n: NUMERIC,
    proportional_prior: BetaProportional,
) -> BetaProportional:
    """Posterior distribution for a Beta likelihood.

    Inference on alpha and beta

    Args:
        x_prod: product of all outcomes
        one_minus_x_prod: product of all (1 - outcomes)
        n: total number of samples in x_prod and one_minus_x_prod
        proportional_prior: BetaProportional prior

    Returns:
        BetaProportional posterior distribution

    """
    p_post = proportional_prior.p * x_prod
    q_post = proportional_prior.q * one_minus_x_prod
    k_post = proportional_prior.k + n

    return BetaProportional(p=p_post, q=q_post, k=k_post)


def von_mises_known_concentration(
    cos_total: NUMERIC,
    sin_total: NUMERIC,
    n: NUMERIC,
    kappa: NUMERIC,
    von_mises_prior: VonMisesKnownConcentration,
    sin=np.sin,
    cos=np.cos,
    arctan2=np.arctan2,
) -> VonMisesKnownConcentration:
    """VonMises likelihood with known concentration parameter.

    Taken from <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 2.13.1</a>.

    Args:
        cos_total: sum of all cosines
        sin_total: sum of all sines
        n: total number of samples in cos_total and sin_total
        kappa: known concentration parameter
        von_mises_prior: VonMisesKnownConcentration prior

    Returns:
        VonMisesKnownConcentration posterior distribution

    """
    sin_total_post = von_mises_prior.a * sin(von_mises_prior.b) + sin_total
    a_post = kappa * sin_total_post

    b_post = arctan2(
        sin_total_post, von_mises_prior.a * cos(von_mises_prior.b) + cos_total
    )

    return VonMisesKnownConcentration(a=a_post, b=b_post)


def von_mises_known_direction(
    centered_cos_total: NUMERIC,
    n: NUMERIC,
    proportional_prior: VonMisesKnownDirectionProportional,
) -> VonMisesKnownDirectionProportional:
    """VonMises likelihood with known direction parameter.

    Taken from <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 2.13.2</a>

    Args:
        centered_cos_total: sum of all centered cosines. sum cos(x - known direction))
        n: total number of samples in centered_cos_total
        proportional_prior: VonMisesKnownDirectionProportional prior

    """

    return VonMisesKnownDirectionProportional(
        c=proportional_prior.c + n,
        r=proportional_prior.r + centered_cos_total,
    )


def multivariate_normal_known_mean(
    X: NUMERIC,
    mu: NUMERIC,
    inverse_wishart_prior: InverseWishart,
) -> InverseWishart:
    """Multivariate normal likelihood with known mean and inverse wishart prior.

    Args:
        X: design matrix
        mu: known mean
        inverse_wishart_prior: InverseWishart prior

    Returns:
        InverseWishart posterior distribution

    """
    nu_post = inverse_wishart_prior.nu + X.shape[0]
    psi_post = inverse_wishart_prior.psi + (X - mu).T @ (X - mu)

    return InverseWishart(
        nu=nu_post,
        psi=psi_post,
    )


def multivariate_normal(
    X: NUMERIC,
    normal_inverse_wishart_prior: NormalInverseWishart,
    outer=np.outer,
) -> NormalInverseWishart:
    """Multivariate normal likelihood with normal inverse wishart prior.

    Args:
        X: design matrix
        mu: known mean
        normal_inverse_wishart_prior: NormalInverseWishart prior
        outer: function to take outer product, defaults to np.outer

    Returns:
        NormalInverseWishart posterior distribution


    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import NormalInverseWishart
        from conjugate.models import multivariate_normal

        true_mean = np.array([1, 5])
        true_cov = np.array([
            [1, 0.5],
            [0.5, 1],
        ])

        n_samples = 100
        rng = np.random.default_rng(42)
        data = rng.multivariate_normal(
            mean=true_mean,
            cov=true_cov,
            size=n_samples,
        )

        prior = NormalInverseWishart(
            mu=np.array([0, 0]),
            kappa=1,
            nu=3,
            psi=np.array([
                [1, 0],
                [0, 1],
            ]),
        )

        posterior = multivariate_normal(
            X=data,
            normal_inverse_wishart_prior=prior,
        )
        ```
    """
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    C = (X - X_mean).T @ (X - X_mean)

    mu_post = (
        normal_inverse_wishart_prior.mu * normal_inverse_wishart_prior.kappa
        + n * X_mean
    ) / (normal_inverse_wishart_prior.kappa + n)

    kappa_post = normal_inverse_wishart_prior.kappa + n
    nu_post = normal_inverse_wishart_prior.nu + n

    mean_difference = X_mean - normal_inverse_wishart_prior.mu
    psi_post = (
        normal_inverse_wishart_prior.psi
        + C
        + outer(mean_difference, mean_difference)
        * n
        * normal_inverse_wishart_prior.kappa
        / kappa_post
    )

    return NormalInverseWishart(
        mu=mu_post,
        kappa=kappa_post,
        nu=nu_post,
        psi=psi_post,
    )


def multivariate_normal_posterior_predictive(
    normal_inverse_wishart: NormalInverseWishart,
) -> MultivariateStudentT:
    """Multivariate normal likelihood with normal inverse wishart prior.

    Args:
        normal_inverse_wishart: NormalInverseWishart posterior

    Returns:
        MultivariateStudentT posterior predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import NormalInverseWishart, MultivariateNormal
        from conjugate.models import multivariate_normal, multivariate_normal_posterior_predictive

        mu_1 = 10
        mu_2 = 5
        sigma_1 = 2.5
        sigma_2 = 1.5
        rho = -0.65
        true_mean = np.array([mu_1, mu_2])
        true_cov = np.array([
            [sigma_1 ** 2, rho * sigma_1 * sigma_2],
            [rho * sigma_1 * sigma_2, sigma_2 ** 2],
        ])
        true = MultivariateNormal(true_mean, true_cov)

        n_samples = 100
        rng = np.random.default_rng(42)
        data = true.dist.rvs(size=n_samples, random_state=rng)

        prior = NormalInverseWishart(
            mu=np.array([0, 0]),
            kappa=1,
            nu=2,
            psi=np.array([
                [5 ** 2, 0],
                [0, 5 ** 2],
            ]),
        )

        posterior = multivariate_normal(
            X=data,
            normal_inverse_wishart_prior=prior,
        )
        prior_predictive = multivariate_normal_posterior_predictive(prior)
        posterior_predictive = multivariate_normal_posterior_predictive(posterior)

        ax = plt.subplot(111)

        xmax = mu_1 + 3 * sigma_1
        ymax = mu_2 + 3 * sigma_2
        x, y = np.mgrid[-xmax:xmax:.1, -ymax:ymax:.1]
        pos = np.dstack((x, y))
        z = true.dist.pdf(pos)
        # z = np.where(z < 0.005, np.nan, z)
        contours = ax.contour(x, y, z, alpha=0.55, color="black")

        for label, dist in zip(["prior", "posterior"], [prior_predictive, posterior_predictive]):
            X = dist.dist.rvs(size=1000)
            ax.scatter(X[:, 0], X[:, 1], alpha=0.15, label=f"{label} predictive")

        ax.axvline(0, color="black", linestyle="--")
        ax.axhline(0, color="black", linestyle="--")
        ax.scatter(data[:, 0], data[:, 1], label="data", alpha=0.5)
        ax.scatter(mu_1, mu_2, color="black", marker="x", label="true mean")

        ax.set(
            xlabel="x1",
            ylabel="x2",
            title=f"Posterior predictive after {n_samples} samples",
            xlim=(-xmax, xmax),
            ylim=(-ymax, ymax),
        )
        ax.legend()
        plt.show()
        ```
    """

    p = normal_inverse_wishart.psi.shape[0]
    mu = normal_inverse_wishart.mu
    nu = normal_inverse_wishart.nu - p + 1
    sigma = (
        (normal_inverse_wishart.kappa + 1)
        * normal_inverse_wishart.psi
        / (normal_inverse_wishart.kappa * (normal_inverse_wishart.nu - p + 1))
    )

    return MultivariateStudentT(mu=mu, sigma=sigma, nu=nu)


def log_normal_normal_inverse_gamma(
    ln_x_total: NUMERIC,
    ln_x2_total: NUMERIC,
    n: NUMERIC,
    normal_inverse_gamma_prior: NormalInverseGamma,
) -> NormalInverseGamma:
    """Log normal likelihood with a normal inverse gamma prior.

    By taking the log of the data, we can use the normal inverse gamma posterior.

    Reference: <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 1.2.1</a>

    Args:
        ln_x_total: sum of the log of all outcomes
        ln_x2_total: sum of the log of all outcomes squared
        n: total number of samples in ln_x_total and ln_x2_total
        normal_inverse_gamma_prior: NormalInverseGamma prior

    Returns:
        NormalInverseGamma posterior distribution

    Example:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import NormalInverseGamma, LogNormal
        from conjugate.models import log_normal_normal_inverse_gamma

        true_mu = 0
        true_sigma = 2.5
        true = LogNormal(true_mu, true_sigma)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        ln_data = np.log(data)

        prior = NormalInverseGamma(1, 1, 1, nu=1)
        posterior = log_normal_normal_inverse_gamma(
            ln_x_total=ln_data.sum(),
            ln_x2_total=(ln_data**2).sum(),
            n=n_samples,
            normal_inverse_gamma_prior=prior
        )

        fig, axes = plt.subplots(ncols=2)
        mean, variance = posterior.sample_mean(4000, return_variance=True, random_state=42)

        ax = axes[0]
        ax.hist(mean, bins=20)
        ax.axvline(true_mu, color="black", linestyle="--", label="true mu")

        ax = axes[1]
        ax.hist(variance, bins=20)
        ax.axvline(true_sigma**2, color="black", linestyle="--", label="true sigma^2")
        plt.show()
        ```
    """

    return normal_normal_inverse_gamma(
        x_total=ln_x_total,
        x2_total=ln_x2_total,
        n=n,
        normal_inverse_gamma_prior=normal_inverse_gamma_prior,
    )
