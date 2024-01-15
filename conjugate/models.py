"""For more on these models, check out the <a href=https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions>Conjugate Prior Wikipedia Table</a>

Below are the supported models

"""
from typing import Tuple

import numpy as np

from conjugate.distributions import (
    Beta,
    Dirichlet,
    DirichletMultinomial,
    Gamma,
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


def bernoulli_beta(x: NUMERIC, beta_prior: Beta) -> Beta:
    """Posterior distribution for a bernoulli likelihood with a beta prior.

    Args:
        x: sucesses from that trials
        beta_prior: Beta distribution prior

    Returns:
        Beta distribution posterior

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
    return BetaNegativeBinomial(r=r, alpha=beta.alpha, beta=beta.beta)


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
