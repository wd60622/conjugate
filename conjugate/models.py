"""For more on these models, check out the <a href=https://en.wikipedia.org/wiki/Conjugate_prior#Table_of_conjugate_distributions>Conjugate Prior Wikipedia Table</a>

# Supported Likelihoods

## Discrete

- Bernoulli / Binomial
- Categorical / Multinomial
- Geometric
- Hypergeometric
- Negative Binomial
- Poisson

## Continuous

- Beta
- Exponential
- Gamma
- Inverse Gamma
- Linear Regression (Normal)
- Log Normal
- Multivariate Normal
- Normal
- Pareto
- Uniform
- Von Mises
- Weibull

# Model Functions

Below are the supported models:

"""

from functools import wraps
from typing import Callable

import numpy as np

import warnings

from conjugate.distributions import (
    Beta,
    BetaBinomial,
    BetaGeometric,
    BetaNegativeBinomial,
    BetaProportional,
    CompoundGamma,
    Dirichlet,
    DirichletMultinomial,
    Gamma,
    GammaKnownRateProportional,
    GammaProportional,
    InverseGamma,
    InverseWishart,
    Lomax,
    MultivariateNormal,
    MultivariateStudentT,
    NegativeBinomial,
    Normal,
    NormalGamma,
    NormalInverseGamma,
    NormalInverseWishart,
    Pareto,
    ScaledInverseChiSquared,
    StudentT,
    VonMisesKnownConcentration,
    VonMisesKnownDirectionProportional,
)
from conjugate._typing import NUMERIC


def validate_type(func, parameter: str):
    expected_type = func.__annotations__.get(parameter, None)

    @wraps(func)
    def wrapper(*args, **kwargs):
        if expected_type is not None:
            parameter_value = kwargs.get(parameter, None)
            recieved_type = type(parameter_value)
            if not isinstance(parameter_value, expected_type):
                msg = (
                    f"Expected {parameter} to be of type {expected_type.__name__!r}, "
                    f"got {recieved_type.__name__!r} instead."
                )
                raise ValueError(msg)

        return func(*args, **kwargs)

    return wrapper


def validate_prior_type(func):
    return validate_type(func, "prior")


def validate_distribution_type(func):
    return validate_type(func, "distribution")


def deprecate_parameter(old_name: str, new_name: str):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if old_name in kwargs:
                msg = (
                    f"Parameter {old_name!r} is deprecated, use {new_name!r} instead. "
                    "It will be removed in a future version."
                )
                warnings.warn(
                    msg,
                    DeprecationWarning,
                    stacklevel=2,
                )
                kwargs[new_name] = kwargs.pop(old_name)
            return func(*args, **kwargs)

        return wrapper

    return decorator


def deprecate_prior_parameter(old_name: str):
    return deprecate_parameter(old_name, "prior")


def deprecate_distribution_parameter(old_name: str):
    return deprecate_parameter(old_name, "distribution")


def get_binomial_beta_posterior_params(
    alpha_prior: NUMERIC, beta_prior: NUMERIC, n: NUMERIC, x: NUMERIC
) -> tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha_prior + x
    beta_post = beta_prior + (n - x)

    return alpha_post, beta_post


@deprecate_prior_parameter("beta_prior")
@validate_prior_type
def binomial_beta(n: NUMERIC, x: NUMERIC, prior: Beta) -> Beta:
    """Posterior distribution for a binomial likelihood with a beta prior.

    Args:
        n: total number of trials
        x: successes from that trials
        prior: Beta distribution prior

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
            prior=prior
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 0.5).plot_pdf(ax=ax, label=["A", "B"])
        prior.set_bounds(0, 0.5).plot_pdf(ax=ax, label="prior")
        ax.legend()
        ```

        <!--
        plt.savefig("./docs/images/docstrings/binomial_beta.png")
        plt.close()
        -->

        ![binomial_beta](./images/docstrings/binomial_beta.png)

    """
    alpha_post, beta_post = get_binomial_beta_posterior_params(
        prior.alpha, prior.beta, n, x
    )

    return Beta(alpha=alpha_post, beta=beta_post)


@deprecate_distribution_parameter("beta")
@validate_distribution_type
def binomial_beta_predictive(n: NUMERIC, distribution: Beta) -> BetaBinomial:
    """Posterior predictive distribution for a binomial likelihood with a beta prior.

    Args:
        n: number of trials
        distribution: Beta distribution

    Returns:
        BetaBinomial predictive distribution

    Examples:
        A / B test example with 100 new impressions

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Beta
        from conjugate.models import binomial_beta, binomial_beta_predictive

        impressions = np.array([100, 250])
        clicks = np.array([10, 35])

        prior = Beta(1, 1)
        posterior = binomial_beta(
            n=impressions,
            x=clicks,
            prior=prior
        )
        posterior_predictive = binomial_beta_predictive(
            n=100,
            distribution=posterior
        )


        ax = plt.subplot(111)
        ax.set_title("Posterior Predictive Distribution with 100 new impressions")
        posterior_predictive.set_bounds(0, 50).plot_pmf(
            ax=ax,
            label=["A", "B"],
        )
        ```

        <!--
        plt.savefig("./docs/images/docstrings/binomial_beta_predictive.png")
        plt.close()
        -->

        ![binomial_beta_predictive](./images/docstrings/binomial_beta_predictive.png)
    """
    return BetaBinomial(n=n, alpha=distribution.alpha, beta=distribution.beta)


@deprecate_prior_parameter("beta_prior")
@validate_prior_type
def bernoulli_beta(x: NUMERIC, prior: Beta) -> Beta:
    """Posterior distribution for a bernoulli likelihood with a beta prior.

    Args:
        x: successes from a single trial
        prior: Beta distribution prior

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
            prior=prior
        )

        posterior.dist.ppf([0.025, 0.975])
        # array([0.15811388, 0.98742088])
        ```

    """
    return binomial_beta(n=1, x=x, prior=prior)


@deprecate_distribution_parameter("beta")
@validate_distribution_type
def bernoulli_beta_predictive(distribution: Beta) -> BetaBinomial:
    """Predictive distribution for a bernoulli likelihood with a beta prior.

    Use for either prior or posterior predictive distribution.

    Args:
        distribution: Beta distribution

    Returns:
        BetaBinomial predictive distribution

    """
    return binomial_beta_predictive(n=1, distribution=distribution)


@deprecate_prior_parameter("beta_prior")
@validate_prior_type
def negative_binomial_beta(r: NUMERIC, n: NUMERIC, x: NUMERIC, prior: Beta) -> Beta:
    """Posterior distribution for a negative binomial likelihood with a beta prior.

    Assumed known number of failures r

    Args:
        r: number of failures
        n: number of trials
        x: number of successes
        prior: Beta distribution prior

    Returns:
        Beta distribution posterior

    """
    alpha_post = prior.alpha + (r * n)
    beta_post = prior.beta + x

    return Beta(alpha=alpha_post, beta=beta_post)


@deprecate_distribution_parameter("beta")
@validate_distribution_type
def negative_binomial_beta_predictive(
    r: NUMERIC,
    distribution: Beta,
) -> BetaNegativeBinomial:
    """Predictive distribution for a negative binomial likelihood with a beta prior

    Assumed known number of failures r

    Args:
        r: number of failures
        distribution: Beta distribution

    Returns:
        BetaNegativeBinomial predictive distribution

    """
    return BetaNegativeBinomial(n=r, alpha=distribution.alpha, beta=distribution.beta)


@deprecate_prior_parameter("beta_binomial_prior")
@validate_prior_type
def hypergeometric_beta_binomial(
    x_total: NUMERIC,
    n: NUMERIC,
    prior: BetaBinomial,
) -> BetaBinomial:
    """Hypergeometric likelihood with a BetaBinomial prior.

    The total population size is N and is known. Encode it in the BetaBinomial
        prior as n=N

    Args:
        x_total: sum of all trials outcomes
        n: total number of trials
        prior: BetaBinomial prior
            n is the known N / total population size

    Returns:
        BetaBinomial posterior distribution

    """
    n = prior.n
    alpha_post = prior.alpha + x_total
    beta_post = prior.beta + (n - x_total)

    return BetaBinomial(n=n, alpha=alpha_post, beta=beta_post)


@deprecate_prior_parameter("beta_prior")
@validate_prior_type
def geometric_beta(x_total, n, prior: Beta, one_start: bool = True) -> Beta:
    """Posterior distribution for a geometric likelihood with a beta prior.

    Args:
        x_total: sum of all trials outcomes
        n: total number of trials
        prior: Beta distribution prior
        one_start: whether to outcomes start at 1, defaults to True. False is 0 start.
            one_start is equivalent to number of Bernoulli trails before
            the first success.

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
            prior=prior
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 1).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, 1).plot_pdf(ax=ax, label="prior")
        ax.legend()
        ax.set(xlabel="chance of good experience")
        ```
        <!--
        plt.savefig("./docs/images/docstrings/geometric_beta.png")
        plt.close()
        -->

        ![geometric_beta](./images/docstrings/geometric_beta.png)

    """
    alpha_post = prior.alpha + n
    beta_post = prior.beta + x_total

    if one_start:
        beta_post = beta_post - n

    return Beta(alpha=alpha_post, beta=beta_post)


@validate_distribution_type
def geometric_beta_predictive(
    distribution: Beta,
    one_start: bool = True,
) -> BetaGeometric:
    """Predictive distribution for a geometric likelihood with a beta prior.

    Args:
        distribution: Beta distribution
        one_start: whether to outcomes start at 1, defaults to True. False is 0 start.
            one_start is equivalent to number of Bernoulli trails before
            the first success.

    Returns:
        BetaGeometric predictive distribution

    """

    return BetaGeometric(
        alpha=distribution.alpha,
        beta=distribution.beta,
        one_start=one_start,
    )


def get_dirichlet_posterior_params(alpha_prior: NUMERIC, x: NUMERIC) -> NUMERIC:
    try:
        return alpha_prior + x
    except Exception:
        return [alpha_prior_i + x_i for alpha_prior_i, x_i in zip(alpha_prior, x)]


def get_categorical_dirichlet_posterior_params(
    alpha_prior: NUMERIC,
    x: NUMERIC,
) -> NUMERIC:
    return get_dirichlet_posterior_params(alpha_prior, x)


@deprecate_prior_parameter("dirichlet_prior")
@validate_prior_type
def categorical_dirichlet(x: NUMERIC, prior: Dirichlet) -> Dirichlet:
    """Posterior distribution of Categorical model with Dirichlet prior.

    Args:
        x: counts
        prior: Dirichlet prior on the counts

    Returns:
        Dirichlet posterior distribution

    """
    alpha_post = get_dirichlet_posterior_params(prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


@deprecate_distribution_parameter("dirichlet")
@validate_distribution_type
def categorical_dirichlet_predictive(
    distribution: Dirichlet,
    n: NUMERIC = 1,
) -> DirichletMultinomial:
    """Predictive distribution of Categorical model with Dirichlet distribution.

    Args:
        distribution: Dirichlet distribution
        n: Number of trials for each sample, defaults to 1.

    Returns:
        DirichletMultinomial distribution related to predictive

    """

    return DirichletMultinomial(n=n, alpha=distribution.alpha)


def get_multi_categorical_dirichlet_posterior_params(
    alpha_prior: NUMERIC,
    x: NUMERIC,
) -> NUMERIC:
    return get_dirichlet_posterior_params(alpha_prior, x)


@deprecate_prior_parameter("dirichlet_prior")
@validate_prior_type
def multinomial_dirichlet(x: NUMERIC, prior: Dirichlet) -> Dirichlet:
    """Posterior distribution of Multinomial model with Dirichlet prior.

    Args:
        x: counts
        prior: Dirichlet prior on the counts

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
            prior=prior
        )

        ax = plt.subplot(111)
        posterior.plot_pdf(ax=ax, label=kinds)
        ax.legend()
        ax.set(xlabel="Flavor Preference")
        ```

        <!--
        plt.savefig("./docs/images/docstrings/multinomial_dirichlet.png")
        plt.close()
        -->

        ![multinomial_dirichlet](./images/docstrings/multinomial_dirichlet.png)

    """
    alpha_post = get_dirichlet_posterior_params(prior.alpha, x)

    return Dirichlet(alpha=alpha_post)


@deprecate_distribution_parameter("dirichlet")
@validate_distribution_type
def multinomial_dirichlet_predictive(
    distribution: Dirichlet,
    n: NUMERIC = 1,
) -> DirichletMultinomial:
    """Predictive distribution of Multinomial model with Dirichlet distribution.

    Args:
        distribution: Dirichlet distribution
        n: Number of trials for each sample, defaults to 1.

    Returns:
        DirichletMultinomial distribution related to predictive

    """

    return DirichletMultinomial(n=n, alpha=distribution.alpha)


def get_poisson_gamma_posterior_params(
    alpha: NUMERIC,
    beta: NUMERIC,
    x_total: NUMERIC,
    n: NUMERIC,
) -> tuple[NUMERIC, NUMERIC]:
    alpha_post = alpha + x_total
    beta_post = beta + n

    return alpha_post, beta_post


@deprecate_prior_parameter("gamma_prior")
@validate_prior_type
def poisson_gamma(x_total: NUMERIC, n: NUMERIC, prior: Gamma) -> Gamma:
    """Posterior distribution for a poisson likelihood with a gamma prior.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        prior: Gamma prior

    Returns:
        Gamma posterior distribution

    """
    alpha_post, beta_post = get_poisson_gamma_posterior_params(
        alpha=prior.alpha, beta=prior.beta, x_total=x_total, n=n
    )

    return Gamma(alpha=alpha_post, beta=beta_post)


@deprecate_distribution_parameter("gamma")
@validate_distribution_type
def poisson_gamma_predictive(distribution: Gamma, n: NUMERIC = 1) -> NegativeBinomial:
    """Predictive distribution for a poisson likelihood with a gamma distribution.

    Args:
        distribution: Gamma distribution
        n: Number of trials for each sample, defaults to 1.
            Can be used to scale the distributions to a different unit of time.

    Returns:
        NegativeBinomial distribution related to predictive

    """
    n = n * distribution.alpha
    p = distribution.beta / (1 + distribution.beta)

    return NegativeBinomial(n=n, p=p)


# Just happen to be the same as above
get_exponential_gamma_posterior_params = get_poisson_gamma_posterior_params


@deprecate_prior_parameter("gamma_prior")
@validate_prior_type
def exponential_gamma(x_total: NUMERIC, n: NUMERIC, prior: Gamma) -> Gamma:
    """Posterior distribution for an exponential likelihood with a gamma prior.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        prior: Gamma prior

    Returns:
        Gamma posterior distribution

    """
    alpha_post, beta_post = get_exponential_gamma_posterior_params(
        alpha=prior.alpha, beta=prior.beta, x_total=x_total, n=n
    )

    return Gamma(alpha=alpha_post, beta=beta_post)


@deprecate_distribution_parameter("gamma")
@validate_distribution_type
def exponential_gamma_predictive(distribution: Gamma) -> Lomax:
    """Predictive distribution for an exponential likelihood with a gamma distribution

    Args:
        distribution: Gamma distribution

    Returns:
        Lomax distribution related to predictive

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Exponential, Gamma
        from conjugate.models import exponential_gamma, exponential_gamma_predictive

        true = Exponential(1)

        n_samples = 15
        data = true.dist.rvs(size=n_samples, random_state=42)

        prior = Gamma(1, 1)

        posterior = exponential_gamma(
            n=n_samples,
            x_total=data.sum(),
            prior=prior
        )

        prior_predictive = exponential_gamma_predictive(distribution=prior)
        posterior_predictive = exponential_gamma_predictive(distribution=posterior)

        ax = plt.subplot(111)
        prior_predictive.set_bounds(0, 2.5).plot_pdf(ax=ax, label="prior predictive")
        true.set_bounds(0, 2.5).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(0, 2.5).plot_pdf(ax=ax, label="posterior predictive")
        ax.legend()
        ```

        <!--
        plt.savefig("./docs/images/docstrings/exponential_gamma_predictive.png")
        plt.close()
        -->

        ![exponential_gamma_predictive](./images/docstrings/exponential_gamma_predictive.png)
    """
    return Lomax(alpha=distribution.beta, lam=distribution.alpha)


@deprecate_prior_parameter("gamma_prior")
@validate_prior_type
def gamma_known_shape(
    x_total: NUMERIC,
    n: NUMERIC,
    alpha: NUMERIC,
    prior: Gamma,
) -> Gamma:
    """Gamma likelihood with a gamma prior.

    The shape parameter of the likelihood is assumed to be known.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        alpha: known shape parameter
        prior: Gamma prior

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
            prior=prior,
        )

        bound = 10
        ax = plt.subplot(111)
        posterior.set_bounds(0, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_rate, color="black", linestyle="--", label="true rate")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/gamma_known_shape.png")
        plt.close()
        -->

        ![gamma_known_shape](./images/docstrings/gamma_known_shape.png)

    """
    alpha_post = prior.alpha + n * alpha
    beta_post = prior.beta + x_total

    return Gamma(alpha=alpha_post, beta=beta_post)


@deprecate_distribution_parameter("gamma")
@validate_distribution_type
def gamma_known_shape_predictive(distribution: Gamma, alpha: NUMERIC) -> CompoundGamma:
    """Predictive distribution for a gamma likelihood with a gamma distribution

    Args:
        distribution: Gamma distribution
        alpha: known shape parameter

    Returns:
        CompoundGamma distribution related to predictive

    """
    return CompoundGamma(alpha=alpha, beta=distribution.alpha, lam=distribution.beta)


@validate_prior_type
def inverse_gamma_known_rate(
    reciprocal_x_total: NUMERIC,
    n: NUMERIC,
    alpha: NUMERIC,
    prior: Gamma,
) -> Gamma:
    """Inverse Gamma likelihood with a known rate and unknown inverse scale.

    Args:
        reciprocal_x_total: sum of all outcomes reciprocals
        n: total number of samples in x_total
        alpha: known rate parameter
        prior: Gamma prior

    Returns:
        Gamma posterior distribution

    """
    alpha_post = prior.alpha + n * alpha
    beta_post = prior.beta + reciprocal_x_total

    return Gamma(alpha=alpha_post, beta=beta_post)


@deprecate_prior_parameter("normal_prior")
@validate_prior_type
def normal_known_variance(
    x_total: NUMERIC,
    n: NUMERIC,
    var: NUMERIC,
    prior: Normal,
) -> Normal:
    """Posterior distribution for a normal likelihood with known variance and a normal prior on mean.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        var: known variance
        prior: Normal prior for mean

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
            prior=prior,
        )

        bound = 5
        ax = plt.subplot(111)
        posterior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_mu, color="black", linestyle="--", label="true mu")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/normal_known_variance.png")
        plt.close()
        -->

        ![normal_known_variance](./images/docstrings/normal_known_variance.png)

    """
    mu_post = ((prior.mu / prior.sigma**2) + (x_total / var)) / (
        (1 / prior.sigma**2) + (n / var)
    )

    var_post = 1 / ((1 / prior.sigma**2) + (n / var))

    return Normal(mu=mu_post, sigma=var_post**0.5)


@deprecate_distribution_parameter("normal")
@validate_distribution_type
def normal_known_variance_predictive(var: NUMERIC, distribution: Normal) -> Normal:
    """Predictive distribution for a normal likelihood with known variance and a normal distribution on mean.

    Args:
        var: known variance
        distribution: Normal distribution for the mean

    Returns:
        Normal predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_variance, normal_known_variance_predictive

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
            prior=prior
        )

        prior_predictive = normal_known_variance_predictive(
            var=known_var,
            distribution=prior,
        )
        posterior_predictive = normal_known_variance_predictive(
            var=known_var,
            distribution=posterior,
        )

        bound = 5
        ax = plt.subplot(111)
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/normal_known_variance_predictive.png")
        plt.close()
        -->

        ![normal_known_variance_predictive](./images/docstrings/normal_known_variance_predictive.png)

    """
    var_posterior_predictive = var + distribution.sigma**2
    return Normal(mu=distribution.mu, sigma=var_posterior_predictive**0.5)


@deprecate_prior_parameter("normal_prior")
@validate_prior_type
def normal_known_precision(
    x_total: NUMERIC,
    n: NUMERIC,
    precision: NUMERIC,
    prior: Normal,
) -> Normal:
    """Posterior distribution for a normal likelihood with known precision and a normal prior on mean.

    Args:
        x_total: sum of all outcomes
        n: total number of samples in x_total
        precision: known precision
        prior: Normal prior for mean

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
            prior=prior
        )

        bound = 5
        ax = plt.subplot(111)
        posterior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior")
        ax.axvline(unknown_mu, color="black", linestyle="--", label="true mu")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/normal_known_precision.png")
        plt.close()
        -->

        ![normal_known_precision](./images/docstrings/normal_known_precision.png)

    """
    return normal_known_variance(
        x_total=x_total,
        n=n,
        var=1 / precision,
        prior=prior,
    )


@deprecate_distribution_parameter("normal")
@validate_distribution_type
def normal_known_precision_predictive(
    precision: NUMERIC,
    distribution: Normal,
) -> Normal:
    """Predictive distribution for a normal likelihood with known precision and a normal prior on mean.

    Args:
        precision: known precision
        distribution: Normal posterior distribution for the mean

    Returns:
        Normal predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal
        from conjugate.models import normal_known_precision, normal_known_precision_predictive

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
            prior=prior
        )

        prior_predictive = normal_known_precision_predictive(
            precision=known_precision,
            distribution=prior,
        )
        posterior_predictive = normal_known_precision_predictive(
            precision=known_precision,
            distribution=posterior,
        )

        bound = 5
        ax = plt.subplot(111)
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/normal_known_precision_predictive.png")
        plt.close()
        -->

        ![normal_known_precision_predictive](./images/docstrings/normal_known_precision_predictive.png)

    """
    return normal_known_variance_predictive(
        var=1 / precision,
        distribution=distribution,
    )


def _normal_known_mean_inverse_gamma_prior(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    mu: NUMERIC,
    prior: InverseGamma,
) -> InverseGamma:
    alpha_post = prior.alpha + (n / 2)
    beta_post = prior.beta + (0.5 * (x2_total - (2 * mu * x_total) + (n * (mu**2))))

    return InverseGamma(alpha=alpha_post, beta=beta_post)


@deprecate_prior_parameter("inverse_gamma_prior")
@validate_prior_type
def normal_known_mean(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    mu: NUMERIC,
    prior: InverseGamma | ScaledInverseChiSquared,
) -> InverseGamma | ScaledInverseChiSquared:
    """Posterior distribution for a normal likelihood with a known mean and a variance prior.

    Args:
        x_total: sum of all outcomes
        x2_total: sum of all outcomes squared
        n: total number of samples in x_total
        mu: known mean
        prior: InverseGamma or ScaledInverseChiSquared prior for variance

    Returns:
        InverseGamma or ScaledInverseChiSquared posterior for variance

    """
    inverse_gamma_input = isinstance(prior, InverseGamma)
    prior = prior if inverse_gamma_input else prior.to_inverse_gamma()

    posterior = _normal_known_mean_inverse_gamma_prior(
        x_total=x_total,
        x2_total=x2_total,
        n=n,
        mu=mu,
        prior=prior,
    )

    if not inverse_gamma_input:
        return ScaledInverseChiSquared.from_inverse_gamma(posterior)

    return posterior


@deprecate_distribution_parameter("inverse_gamma")
@validate_distribution_type
def normal_known_mean_predictive(
    mu: NUMERIC,
    distribution: InverseGamma | ScaledInverseChiSquared,
) -> StudentT:
    """Predictive distribution for a normal likelihood with a known mean and a variance prior.

    Args:
        mu: known mean
        distribution: InverseGamma or ScaledInverseChiSquared prior

    Returns:
        StudentT predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import Normal, InverseGamma
        from conjugate.models import normal_known_mean, normal_known_mean_predictive

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
            prior=prior
        )

        bound = 5
        ax = plt.subplot(111)
        prior_predictive = normal_known_mean_predictive(
            mu=known_mu,
            distribution=prior,
        )
        prior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="prior predictive")
        true.set_bounds(-bound, bound).plot_pdf(ax=ax, label="true distribution")
        posterior_predictive = normal_known_mean_predictive(
            mu=known_mu,
            distribution=posterior,
        )
        posterior_predictive.set_bounds(-bound, bound).plot_pdf(ax=ax, label="posterior predictive")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/normal_known_mean_predictive.png")
        plt.close()
        -->

        ![normal_known_mean_predictive](./images/docstrings/normal_known_mean_predictive.png)

    """
    if isinstance(distribution, ScaledInverseChiSquared):
        distribution = distribution.to_inverse_gamma()

    return StudentT(
        mu=mu,
        sigma=(distribution.beta / distribution.alpha) ** 0.5,
        nu=2 * distribution.alpha,
    )


def _normal(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    mu_0: NUMERIC,
    nu: NUMERIC,
    alpha: NUMERIC,
    beta: NUMERIC,
) -> tuple[NUMERIC, NUMERIC, NUMERIC, NUMERIC]:
    x_mean = x_total / n

    nu_post = nu + n
    mu_post = ((nu * mu_0) + (n * x_mean)) / nu_post

    alpha_post = alpha + (n / 2)
    beta_post = beta + 0.5 * ((mu_0**2) * nu + x2_total - (mu_post**2) * nu_post)

    return mu_post, nu_post, alpha_post, beta_post


@validate_prior_type
def normal(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    prior: NormalInverseGamma | NormalGamma,
) -> NormalInverseGamma | NormalGamma:
    """Posterior distribution for a normal likelihood.

    Args:
        x_total: sum of all outcomes
        x2_total: sum of all outcomes squared
        n: total number of samples in x_total and x2_total
        prior: NormalInverseGamma or NormalGamma prior

    Returns:
        NormalInverseGamma or NormalGamma posterior distribution

    """
    if isinstance(prior, NormalInverseGamma) and prior.nu is None:
        raise ValueError("nu must be provided for the prior")

    mu_post, nu_post, alpha_post, beta_post = _normal(
        x_total=x_total,
        x2_total=x2_total,
        n=n,
        mu_0=prior.mu,
        nu=prior.lam if isinstance(prior, NormalGamma) else prior.nu,  # type: ignore
        alpha=prior.alpha,
        beta=prior.beta,
    )

    kwargs = (
        {
            "mu": mu_post,
            "lam": nu_post,
            "alpha": alpha_post,
            "beta": beta_post,
        }
        if isinstance(prior, NormalGamma)
        else {
            "mu": mu_post,
            "nu": nu_post,
            "alpha": alpha_post,
            "beta": beta_post,
        }
    )

    return prior.__class__(**kwargs)


@deprecate_prior_parameter("normal_inverse_gamma_prior")
@validate_prior_type
def normal_normal_inverse_gamma(
    x_total: NUMERIC,
    x2_total: NUMERIC,
    n: NUMERIC,
    prior: NormalInverseGamma,
) -> NormalInverseGamma:
    """Posterior distribution for a normal likelihood with a normal inverse gamma prior.

    Derivation from paper [here](https://www.cs.ubc.ca/~murphyk/Papers/bayesGauss.pdf).

    Args:
        x_total: sum of all outcomes
        x2_total: sum of all outcomes squared
        n: total number of samples in x_total and x2_total
        prior: NormalInverseGamma prior

    Returns:
        NormalInverseGamma posterior distribution

    """
    warnings.warn(
        "This function is deprecated. Use 'normal' instead.",
        DeprecationWarning,
        stacklevel=1,
    )

    return normal(x_total=x_total, x2_total=x2_total, n=n, prior=prior)  # type: ignore


@validate_distribution_type
def normal_predictive(
    distribution: NormalInverseGamma | NormalGamma,
) -> StudentT:
    """Predictive distribution for Normal likelihood.

    Args:
        distribution: NormalInverseGamma or NormalGamma distribution

    Returns:
        StudentT predictive distribution

    """
    nu = (
        distribution.nu
        if isinstance(distribution, NormalInverseGamma)
        else distribution.lam
    )
    var = distribution.beta * (nu + 1) / (nu * distribution.alpha)
    return StudentT(
        mu=distribution.mu,
        sigma=var**0.5,
        nu=2 * distribution.alpha,
    )


@deprecate_distribution_parameter("normal_inverse_gamma")
@validate_distribution_type
def normal_normal_inverse_gamma_predictive(
    distribution: NormalInverseGamma,
) -> StudentT:
    """Predictive distribution for Normal likelihood.

    Args:
        distribution: NormalInverseGamma distribution

    Returns:
        StudentT predictive distribution

    """
    warnings.warn(
        "This function is deprecated. Use 'normal_predictive' instead.",
        DeprecationWarning,
        stacklevel=1,
    )
    return normal_predictive(distribution=distribution)


@deprecate_prior_parameter("normal_inverse_gamma_prior")
@validate_prior_type
def linear_regression(
    X: NUMERIC,
    y: NUMERIC,
    prior: NormalInverseGamma,
    inv=np.linalg.inv,
) -> NormalInverseGamma:
    """Posterior distribution for a linear regression model with a normal inverse gamma prior.

    Derivation taken from this blog [here](https://gregorygundersen.com/blog/2020/02/04/bayesian-linear-regression/).

    Args:
        X: design matrix
        y: response vector
        prior: NormalInverseGamma prior
        inv: function to invert matrix, defaults to np.linalg.inv

    Returns:
        NormalInverseGamma posterior distribution

    """
    N = X.shape[0]

    delta = inv(prior.delta_inverse)

    delta_post = (X.T @ X) + delta
    delta_post_inverse = inv(delta_post)

    mu_post = (
        # (B, B)
        delta_post_inverse
        # (B, 1)
        # (B, B) * (B, 1) +  (B, N) * (N, 1)
        @ (delta @ prior.mu + X.T @ y)
    )

    alpha_post = prior.alpha + (0.5 * N)
    beta_post = prior.beta + (
        0.5
        * (
            (y.T @ y)
            # (1, B) * (B, B) * (B, 1)
            + (prior.mu.T @ delta @ prior.mu)
            # (1, B) * (B, B) * (B, 1)
            - (mu_post.T @ delta_post @ mu_post)
        )
    )

    return NormalInverseGamma(
        mu=mu_post, delta_inverse=delta_post_inverse, alpha=alpha_post, beta=beta_post
    )


@deprecate_distribution_parameter("normal_inverse_gamma")
@validate_distribution_type
def linear_regression_predictive(
    distribution: NormalInverseGamma,
    X: NUMERIC,
    eye=np.eye,
) -> MultivariateStudentT:
    """Predictive distribution for a linear regression model with a normal inverse gamma prior.

    Args:
        distribution: NormalInverseGamma posterior
        X: design matrix
        eye: function to get identity matrix, defaults to np.eye

    Returns:
        MultivariateStudentT predictive distribution

    """
    mu = X @ distribution.mu
    sigma = (distribution.beta / distribution.alpha) * (
        eye(X.shape[0]) + (X @ distribution.delta_inverse @ X.T)
    )
    nu = 2 * distribution.alpha

    return MultivariateStudentT(
        mu=mu,
        sigma=sigma,
        nu=nu,
    )


@deprecate_prior_parameter("pareto_prior")
@validate_prior_type
def uniform_pareto(
    x_max: NUMERIC,
    n: NUMERIC,
    prior: Pareto,
    max_fn=np.maximum,
) -> Pareto:
    """Posterior distribution for a uniform likelihood with a pareto prior.

    Args:
        x_max: maximum value
        n: number of samples
        prior: Pareto prior
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
            prior=prior
        )
        ```

    """
    alpha_post = prior.alpha + n
    x_m_post = max_fn(prior.x_m, x_max)

    return Pareto(x_m=x_m_post, alpha=alpha_post)


@deprecate_prior_parameter("gamma_prior")
@validate_prior_type
def pareto_gamma(
    n: NUMERIC,
    ln_x_total: NUMERIC,
    x_m: NUMERIC,
    prior: Gamma,
    ln=np.log,
) -> Gamma:
    """Posterior distribution for a pareto likelihood with a gamma prior.

    The parameter x_m is assumed to be known.

    Args:
        n: number of samples
        ln_x_total: sum of the log of all outcomes
        x_m: The known minimum value
        prior: Gamma prior
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
            prior=prior,
        )

        ax = plt.subplot(111)
        posterior.set_bounds(0, 2.5).plot_pdf(ax=ax, label="posterior")
        prior.set_bounds(0, 2.5).plot_pdf(ax=ax, label="prior")
        ax.axvline(x_m_known, color="black", linestyle="--", label="true x_m")
        ax.legend()
        ```
        <!--
        plt.savefig("./docs/images/docstrings/pareto_gamma.png")
        plt.close()
        -->

        ![pareto_gamma](./images/docstrings/pareto_gamma.png)

    """
    alpha_post = prior.alpha + n
    beta_post = prior.beta + ln_x_total - n * ln(x_m)

    return Gamma(alpha=alpha_post, beta=beta_post)


@deprecate_prior_parameter("gamma_proportial_prior")
@validate_prior_type
def gamma(
    x_total: NUMERIC,
    x_prod: NUMERIC,
    n: NUMERIC,
    prior: GammaProportional,
) -> GammaProportional:
    """Posterior distribution for a gamma likelihood.

    Inference on alpha and beta

    Args:
        x_total: sum of all outcomes
        x_prod: product of all outcomes
        n: total number of samples in x_total and x_prod
        prior: GammaProportional prior

    Returns:
        GammaProportional posterior distribution

    """
    p_post = prior.p * x_prod
    q_post = prior.q + x_total
    r_post = prior.r + n
    s_post = prior.s + n

    return GammaProportional(p=p_post, q=q_post, r=r_post, s=s_post)


@deprecate_prior_parameter("gamma_known_rate_proportial_prior")
@validate_prior_type
def gamma_known_rate(
    x_prod: NUMERIC,
    n: NUMERIC,
    beta: NUMERIC,
    prior: GammaKnownRateProportional,
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
    a_post = prior.a * x_prod
    b_post = prior.b + n
    c_post = prior.c + n

    return GammaKnownRateProportional(a=a_post, b=b_post, c=c_post)


@deprecate_prior_parameter("beta_proportial_prior")
@validate_prior_type
def beta(
    x_prod: NUMERIC,
    one_minus_x_prod: NUMERIC,
    n: NUMERIC,
    prior: BetaProportional,
) -> BetaProportional:
    """Posterior distribution for a Beta likelihood.

    Inference on alpha and beta

    Args:
        x_prod: product of all outcomes
        one_minus_x_prod: product of all (1 - outcomes)
        n: total number of samples in x_prod and one_minus_x_prod
        prior: BetaProportional prior

    Returns:
        BetaProportional posterior distribution

    """
    p_post = prior.p * x_prod
    q_post = prior.q * one_minus_x_prod
    k_post = prior.k + n

    return BetaProportional(p=p_post, q=q_post, k=k_post)


@deprecate_prior_parameter("von_mises_known_concentration_prior")
@validate_prior_type
def von_mises_known_concentration(
    cos_total: NUMERIC,
    sin_total: NUMERIC,
    n: NUMERIC,
    kappa: NUMERIC,
    prior: VonMisesKnownConcentration,
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
        prior: VonMisesKnownConcentration prior

    Returns:
        VonMisesKnownConcentration posterior distribution

    """
    sin_total_post = prior.a * sin(prior.b) + sin_total
    a_post = kappa * sin_total_post

    b_post = arctan2(sin_total_post, prior.a * cos(prior.b) + cos_total)

    return VonMisesKnownConcentration(a=a_post, b=b_post)


@deprecate_prior_parameter("von_mises_known_direction_proportial_prior")
@validate_prior_type
def von_mises_known_direction(
    centered_cos_total: NUMERIC,
    n: NUMERIC,
    prior: VonMisesKnownDirectionProportional,
) -> VonMisesKnownDirectionProportional:
    """VonMises likelihood with known direction parameter.

    Taken from <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 2.13.2</a>

    Args:
        centered_cos_total: sum of all centered cosines. sum cos(x - known direction))
        n: total number of samples in centered_cos_total
        prior: VonMisesKnownDirectionProportional prior

    """

    return VonMisesKnownDirectionProportional(
        c=prior.c + n,
        r=prior.r + centered_cos_total,
    )


def _multivariate_normal_known_precision(
    n: NUMERIC,
    x_bar_0: NUMERIC,
    precision_0: NUMERIC,
    x_bar: NUMERIC,
    precision: NUMERIC,
    inv=np.linalg.inv,
) -> tuple[NUMERIC, NUMERIC]:
    mu_post = inv(precision_0 + n * precision) @ (
        (precision_0 @ x_bar_0) + (n * (precision @ x_bar))
    )
    precision_post = precision_0 + (n * precision)

    return mu_post, precision_post


@validate_prior_type
def multivariate_normal_known_covariance(
    n: NUMERIC,
    x_bar: NUMERIC,
    cov: NUMERIC,
    prior: MultivariateNormal,
    inv=np.linalg.inv,
) -> MultivariateNormal:
    """Multivariate normal likelihood with known covariance and multivariate normal prior.

    Args:
        n: number of samples
        x_bar: mean of samples
        cov: known covariance
        prior: MultivariateNormal prior for the mean
        inv: function to invert matrix, defaults to np.linalg.inv

    Returns:
        MultivariateNormal posterior distribution

    """
    mu_bar_0 = prior.mu
    precision_0 = inv(prior.cov)

    precision = inv(cov)

    mu_post, precision_post = _multivariate_normal_known_precision(
        n=n,
        x_bar_0=mu_bar_0,
        precision_0=precision_0,
        x_bar=x_bar,
        precision=precision,
    )

    return MultivariateNormal(mu=mu_post, cov=inv(precision_post))


@validate_distribution_type
def multivariate_normal_known_covariance_predictive(
    distribution: MultivariateNormal,
    cov: NUMERIC,
) -> MultivariateNormal:
    """Predictive distribution for a multivariate normal likelihood with known covariance and a multivariate normal prior.

    Args:
        distribution: MultivariateNormal distribution
        cov: known covariance

    Returns:
        MultivariateNormal predictive distribution

    """
    mu_pred = distribution.mu
    cov_pred = distribution.cov + cov
    return MultivariateNormal(mu=mu_pred, cov=cov_pred)


@validate_prior_type
def multivariate_normal_known_precision(
    n: NUMERIC,
    x_bar: NUMERIC,
    precision: NUMERIC,
    prior: MultivariateNormal,
    inv=np.linalg.inv,
) -> MultivariateNormal:
    """Multivariate normal likelihood with known precision and multivariate normal prior.

    Args:
        n: number of samples
        x_bar: mean of samples
        precision: known precision
        prior: MultivariateNormal prior for the mean
        inv: function to invert matrix, defaults to np.linalg.inv

    Returns:
        MultivariateNormal posterior distribution

    """
    mu_0 = prior.mu
    precision_0 = inv(prior.cov)

    mu_post, precision_post = _multivariate_normal_known_precision(
        n=n,
        x_bar_0=mu_0,
        precision_0=precision_0,
        x_bar=x_bar,
        precision=precision,
    )

    return MultivariateNormal(mu=mu_post, cov=inv(precision_post))


@validate_distribution_type
def multivariate_normal_known_precision_predictive(
    distribution: MultivariateNormal,
    precision: NUMERIC,
    inv: Callable = np.linalg.inv,
) -> MultivariateNormal:
    """Predictive distribution for a multivariate normal likelihood with known precision and a multivariate normal prior.

    Args:
        distribution: MultivariateNormal distribution
        precision: known precision
        inv: function to invert matrix, defaults to np.linalg.inv

    Returns:
        MultivariateNormal predictive distribution

    """
    mu_pred = distribution.mu
    cov_pred = distribution.cov + inv(precision)
    return MultivariateNormal(mu=mu_pred, cov=cov_pred)


@deprecate_prior_parameter("inverse_wishart_prior")
@validate_prior_type
def multivariate_normal_known_mean(
    X: NUMERIC,
    mu: NUMERIC,
    prior: InverseWishart,
) -> InverseWishart:
    """Multivariate normal likelihood with known mean and inverse wishart prior.

    Args:
        X: design matrix
        mu: known mean
        prior: InverseWishart prior

    Returns:
        InverseWishart posterior distribution

    """
    nu_post = prior.nu + X.shape[0]
    psi_post = prior.psi + (X - mu).T @ (X - mu)

    return InverseWishart(
        nu=nu_post,
        psi=psi_post,
    )


@deprecate_prior_parameter("normal_inverse_wishart_prior")
@validate_prior_type
def multivariate_normal(
    X: NUMERIC,
    prior: NormalInverseWishart,
    outer=np.outer,
) -> NormalInverseWishart:
    """Multivariate normal likelihood with normal inverse wishart prior.

    Args:
        X: design matrix
        mu: known mean
        prior: NormalInverseWishart prior
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
            prior=prior,
        )
        ```
    """
    n = X.shape[0]
    X_mean = X.mean(axis=0)
    C = (X - X_mean).T @ (X - X_mean)

    mu_post = (prior.mu * prior.kappa + n * X_mean) / (prior.kappa + n)

    kappa_post = prior.kappa + n
    nu_post = prior.nu + n

    mean_difference = X_mean - prior.mu
    psi_post = (
        prior.psi
        + C
        + outer(mean_difference, mean_difference) * n * prior.kappa / kappa_post
    )

    return NormalInverseWishart(
        mu=mu_post,
        kappa=kappa_post,
        nu=nu_post,
        psi=psi_post,
    )


@deprecate_distribution_parameter("normal_inverse_wishart")
@validate_distribution_type
def multivariate_normal_predictive(
    distribution: NormalInverseWishart,
) -> MultivariateStudentT:
    """Multivariate normal likelihood with normal inverse wishart distribution.

    Args:
        distribution: NormalInverseWishart distribution

    Returns:
        MultivariateStudentT predictive distribution

    Examples:
        Constructed example

        ```python
        import numpy as np

        import matplotlib.pyplot as plt

        from conjugate.distributions import NormalInverseWishart, MultivariateNormal
        from conjugate.models import multivariate_normal, multivariate_normal_predictive

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
            prior=prior,
        )
        prior_predictive = multivariate_normal_predictive(distribution=prior)
        posterior_predictive = multivariate_normal_predictive(distribution=posterior)

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
        ```
        <!--
        plt.savefig("./docs/images/docstrings/multivariate_normal_predictive.png")
        plt.close()
        -->

        ![multivariate_normal_predictive](./images/docstrings/multivariate_normal_predictive.png)
    """

    p = distribution.psi.shape[0]
    mu = distribution.mu
    nu = distribution.nu - p + 1
    sigma = (
        (distribution.kappa + 1)
        * distribution.psi
        / (distribution.kappa * (distribution.nu - p + 1))
    )

    return MultivariateStudentT(mu=mu, sigma=sigma, nu=nu)


@deprecate_prior_parameter("normal_inverse_wishart_prior")
@validate_prior_type
def log_normal(
    ln_x_total: NUMERIC,
    ln_x2_total: NUMERIC,
    n: NUMERIC,
    prior: NormalInverseGamma | NormalGamma,
) -> NormalInverseGamma | NormalGamma:
    """Log normal likelihood.

    By taking the log of the data, we can use the normal inverse gamma posterior.

    Reference: <a href=https://web.archive.org/web/20090529203101/http://www.people.cornell.edu/pages/df36/CONJINTRnew%20TEX.pdf>Section 1.2.1</a>

    Args:
        ln_x_total: sum of the log of all outcomes
        ln_x2_total: sum of the log of all outcomes squared
        n: total number of samples in ln_x_total and ln_x2_total
        prior: NormalInverseGamma or NormalGamma prior

    Returns:
        NormalInverseGamma or NormalGamma posterior distribution

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

        prior = NormalInverseGamma(mu=1, nu=1, alpha=1, beta=1)
        posterior = log_normal_normal_inverse_gamma(
            ln_x_total=ln_data.sum(),
            ln_x2_total=(ln_data**2).sum(),
            n=n_samples,
            prior=prior
        )

        fig, axes = plt.subplots(ncols=2)
        mean, variance = posterior.sample_mean(4000, return_variance=True, random_state=42)

        ax = axes[0]
        ax.hist(mean, bins=20)
        ax.axvline(true_mu, color="black", linestyle="--", label="true mu")

        ax = axes[1]
        ax.hist(variance, bins=20)
        ax.axvline(true_sigma**2, color="black", linestyle="--", label="true sigma^2")
        ```
        <!--
        plt.savefig("./docs/images/docstrings/log_normal_normal_inverse_gamma.png")
        plt.close()
        -->

        ![log_normal_normal_inverse_gamma](./images/docstrings/log_normal_normal_inverse_gamma.png)
    """

    return normal(
        x_total=ln_x_total,
        x2_total=ln_x2_total,
        n=n,
        prior=prior,
    )


@validate_prior_type
def weibull_inverse_gamma_known_shape(
    n: NUMERIC,
    x_beta_total: NUMERIC,
    prior: InverseGamma,
) -> InverseGamma:
    """Posterior distribution for a Weibull likelihood with an inverse gamma prior on shape.

    Args:
        n: total number of samples
        x_beta_total: sum of all x^beta
        prior: InverseGamma prior

    Returns:
        InverseGamma posterior distribution

    """
    alpha_post = prior.alpha + n
    beta_post = prior.beta + x_beta_total

    return InverseGamma(alpha=alpha_post, beta=beta_post)


def _use_predictive_instead(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        name = func.__name__
        warnings.warn(
            f"This function is deprecated and will be removed in future version. Use the {name!r} instead.",
            DeprecationWarning,
            stacklevel=1,
        )
        return func(*args, **kwargs)

    return wrapper


binomial_beta_posterior_predictive = _use_predictive_instead(binomial_beta_predictive)
bernoulli_beta_posterior_predictive = _use_predictive_instead(bernoulli_beta_predictive)
negative_binomial_beta_posterior_predictive = _use_predictive_instead(
    negative_binomial_beta_predictive,
)
categorical_dirichlet_posterior_predictive = _use_predictive_instead(
    categorical_dirichlet_predictive,
)
multinomial_dirichlet_posterior_predictive = _use_predictive_instead(
    multinomial_dirichlet_predictive,
)
poisson_gamma_posterior_predictive = _use_predictive_instead(poisson_gamma_predictive)
exponential_gamma_posterior_predictive = _use_predictive_instead(
    exponential_gamma_predictive
)
gamma_known_shape_posterior_predictive = _use_predictive_instead(
    gamma_known_shape_predictive
)
normal_known_variance_posterior_predictive = _use_predictive_instead(
    normal_known_variance_predictive
)
normal_known_precision_posterior_predictive = _use_predictive_instead(
    normal_known_precision_predictive
)
normal_known_mean_posterior_predictive = _use_predictive_instead(
    normal_known_mean_predictive
)
normal_normal_inverse_gamma_posterior_predictive = _use_predictive_instead(
    normal_normal_inverse_gamma_predictive
)
linear_regression_posterior_predictive = _use_predictive_instead(
    linear_regression_predictive
)
multivariate_normal_posterior_predictive = _use_predictive_instead(
    multivariate_normal_predictive
)
