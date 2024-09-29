import matplotlib.pyplot as plt

import numpy as np
import pandas as pd


def binomial():
    from conjugate.distributions import Beta, Binomial
    from conjugate.models import binomial_beta

    prior = Beta(1, 1)

    n = 10

    true_p = 0.85
    true_distribution = Binomial(n=n, p=true_p)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    def get_posterior(data):
        return binomial_beta(n=len(data) * n, x=data.sum(), prior=prior)

    return sample_data, get_posterior, true_p


def geometric():
    from conjugate.distributions import Beta, Geometric
    from conjugate.models import geometric_beta

    true_p = 0.15
    true_distribution = Geometric(p=true_p)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Beta(1, 1)

    def get_posterior(data):
        return geometric_beta(n=len(data), x_total=data.sum(), prior=prior)

    return sample_data, get_posterior, true_p


def poisson():
    from conjugate.distributions import Gamma, Poisson
    from conjugate.models import poisson_gamma

    prior = Gamma(5, 1)

    true_lam = 5
    true_distribution = Poisson(lam=true_lam)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    def get_posterior(data):
        return poisson_gamma(n=len(data), x_total=data.sum(), prior=prior)

    return sample_data, get_posterior, true_lam


def negative_binomial():
    from conjugate.distributions import Beta, NegativeBinomial
    from conjugate.models import negative_binomial_beta

    r = 10
    true_p = 0.75
    true_distribution = NegativeBinomial(n=r, p=true_p)

    prior = Beta(1, 1)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    def get_posterior(data):
        return negative_binomial_beta(
            n=len(data),
            x=data.sum(),
            r=r,
            prior=prior,
        )

    return sample_data, get_posterior, true_p


def hypergeometric():
    """TODO: Correct this example."""
    from conjugate.distributions import Hypergeometric, BetaBinomial
    from conjugate.models import hypergeometric_beta_binomial

    N = 100
    k = 10
    n = 10
    true_distribution = Hypergeometric(N=N, k=k, n=n)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = BetaBinomial(n=N, alpha=1, beta=1)

    def get_posterior(data):
        return hypergeometric_beta_binomial(
            n=len(data),
            x_total=data.sum(),
            prior=prior,
        )

    return sample_data, get_posterior, k


def normal_known_variance():
    from conjugate.distributions import Normal
    from conjugate.models import normal_known_variance

    true_mu = 5
    known_sigma = 2

    true_distribution = Normal(mu=true_mu, sigma=known_sigma)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Normal(mu=0, sigma=10)

    def get_posterior(data):
        return normal_known_variance(
            n=len(data),
            x_total=data.sum(),
            var=known_sigma**2,
            prior=prior,
        )

    return sample_data, get_posterior, true_mu


def normal_known_precision():
    from conjugate.distributions import Normal
    from conjugate.models import normal_known_precision

    true_mu = 5
    known_tau = 0.5

    true_distribution = Normal(mu=true_mu, sigma=(1 / known_tau) ** 0.5)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Normal(mu=0, sigma=10)

    def get_posterior(data):
        return normal_known_precision(
            n=len(data),
            x_total=data.sum(),
            precision=known_tau,
            prior=prior,
        )

    return sample_data, get_posterior, true_mu


def normal_known_mean():
    from conjugate.distributions import Normal, InverseGamma
    from conjugate.models import normal_known_mean

    known_mu = 5
    true_sigma = 2

    true_distribution = Normal(mu=known_mu, sigma=true_sigma)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = InverseGamma(alpha=2, beta=1)

    def get_posterior(data):
        return normal_known_mean(
            n=len(data),
            x_total=data.sum(),
            x2_total=(data**2).sum(),
            mu=known_mu,
            prior=prior,
        )

    return sample_data, get_posterior, true_sigma**2


def normal_known_mean_alternative():
    from conjugate.distributions import Normal, ScaledInverseChiSquared
    from conjugate.models import normal_known_mean

    known_mu = 5
    true_sigma = 2

    true_distribution = Normal(mu=known_mu, sigma=true_sigma)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = ScaledInverseChiSquared(nu=2, sigma2=1)

    def get_posterior(data):
        return normal_known_mean(
            n=len(data),
            x_total=data.sum(),
            x2_total=(data**2).sum(),
            mu=known_mu,
            prior=prior,
        )

    return sample_data, get_posterior, true_sigma**2


def uniform():
    from conjugate.distributions import Uniform, Pareto
    from conjugate.models import uniform_pareto

    true_theta = 3.5
    true_distribution = Uniform(low=0, high=true_theta)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Pareto(x_m=1, alpha=1)

    def get_posterior(data):
        return uniform_pareto(x_max=data.max(), n=len(data), prior=prior)

    return sample_data, get_posterior, true_theta


def exponential():
    from conjugate.distributions import Exponential, Gamma
    from conjugate.models import exponential_gamma

    true_lam = 5
    true_distribution = Exponential(lam=true_lam)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Gamma(1, 1)

    def get_posterior(data):
        return exponential_gamma(n=len(data), x_total=data.sum(), prior=prior)

    return sample_data, get_posterior, true_lam


def inverse_gamma_known_rate():
    from conjugate.distributions import InverseGamma, Gamma
    from conjugate.models import inverse_gamma_known_rate

    alpha = 1
    beta = 2.5

    true_distribution = InverseGamma(alpha=alpha, beta=beta)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = Gamma(1, 1)

    def get_posterior(data):
        return inverse_gamma_known_rate(
            reciprocal_x_total=(1 / data).sum(),
            n=len(data),
            alpha=alpha,
            prior=prior,
        )

    return sample_data, get_posterior, beta


def weibull_known_shape():
    from conjugate.distributions import Weibull, InverseGamma
    from conjugate.models import weibull_inverse_gamma_known_shape

    beta = 0.5
    theta = 2.5
    true_distribution = Weibull(beta=beta, theta=theta)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = InverseGamma(alpha=1, beta=1)

    def get_posterior(data):
        return weibull_inverse_gamma_known_shape(
            n=len(data),
            x_beta_total=(data**beta).sum(),
            prior=prior,
        )

    return sample_data, get_posterior, theta


def normal_normal_gamma():
    from conjugate.distributions import Normal, NormalGamma
    from conjugate.models import normal

    true_mu = 1.5
    true_precision = 0.25

    true_distribution = Normal(mu=true_mu, sigma=(1 / true_precision) ** 0.5)

    def sample_data(n, rng):
        return true_distribution.dist.rvs(size=n, random_state=rng)

    prior = NormalGamma(mu=0, lam=1, alpha=2, beta=1)

    def get_posterior(data):
        return normal(
            n=len(data),
            x_total=data.sum(),
            x2_total=(data**2).sum(),
            prior=prior,
        ).gamma

    return sample_data, get_posterior, true_precision


def parameter_recovery(
    ns: list[int],
    sample_data,
    get_posterior,
    rng: np.random.Generator,
) -> pd.DataFrame:
    rows = []

    for n in ns:
        data = sample_data(n, rng=rng)
        posterior = get_posterior(data)

        rows.append(
            {
                "n": n,
                "mean": posterior.dist.mean(),
                "lower": posterior.dist.ppf(0.025),
                "upper": posterior.dist.ppf(0.975),
            }
        )

    return pd.DataFrame(rows)


def plot_parameter_recovery(
    df: pd.DataFrame,
    true_value: float,
    ax: plt.Axes | None = None,
) -> plt.Axes:
    ax: plt.Axes = ax or plt.gca()
    ax.plot(df["n"], df["mean"], label="Posterior mean")
    ax.fill_between(df["n"], df["lower"], df["upper"], alpha=0.2, label="95% CI")
    ax.axhline(true_value, color="red", linestyle="--", label="True value")
    ax.set_xlabel("Sample size")
    ax.set_ylabel("Parameter value")
    ax.legend()

    return ax


def main(setup, rng, ns):
    sample_data, get_posterior, true_value = setup()

    df = parameter_recovery(ns, sample_data, get_posterior, rng)

    ax = plot_parameter_recovery(df, true_value=true_value)
    ax.set_xscale("log")
    plt.savefig("parameter_recovery.png")
    plt.close()


if __name__ == "__main__":
    seed = sum(map(ord, "Parameter recovery exercise"))
    rng = np.random.default_rng(seed)

    ns = [5, 10, 25, 50, 100, 250, 500, 1000, 2500]

    setup = normal_known_precision
    setup = normal_known_mean
    setup = uniform
    setup = exponential
    setup = inverse_gamma_known_rate
    setup = normal_known_mean_alternative
    setup = weibull_known_shape
    setup = normal_normal_gamma
    main(setup, ns=ns, rng=rng)
