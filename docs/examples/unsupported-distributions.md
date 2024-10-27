---
comments: true
---
# Unsupported Posterior Predictive Distributions

Suppose we want to use the Pareto model with a Gamma prior which doesn't have a
supported distribution for the posterior predictive.

We can get posterior predictive samples by:

1. Sample from the posterior distribution
2. Sample from the model distribution using posterior samples

## Setup

```python
import numpy as np

from conjugate.distributions import Gamma, Pareto
from conjugate.models import pareto_gamma

seed = sum(map(ord, "Unsupported Posterior Predictive Distributions"))
rng = np.random.default_rng(seed)

n = 10

x_m = 1
alpha = 2.5
true_distribution = Pareto(x_m=x_m, alpha=alpha)

data = true_distribution.dist.rvs(size=n, random_state=rng)

prior = Gamma(1, 1)
posterior: Gamma = pareto_gamma(
    n=n,
    ln_x_total=np.log(data).sum(),
    x_m=x_m,
    prior=prior,
)
```


## 1. Using `conjugate-models`

Since the distributions are vectorized, just:

1. Get the number of samples from the posterior
2. Take a single sample from the model distribution

```python
n_samples = 1_000

alpha_samples = posterior.dist.rvs(size=n_samples, random_state=rng)
posterior_predictive_samples = Pareto(x_m=x_m, alpha=alpha_samples).dist.rvs(random_state=rng)
```

## 2. Using PyMC

Another route would be using PyMC then use the `draw` function.

```python
import pymc as pm

posterior_alpha = pm.Gamma.dist(alpha=posterior.alpha, beta=posterior.beta)
geometric_posterior_predictive = pm.Pareto.dist(m=x_m, alpha=posterior_alpha)

n_samples = 1_000
posterior_predictive_samples = pm.draw(geometric_posterior_predictive, draws=n_samples)
```
