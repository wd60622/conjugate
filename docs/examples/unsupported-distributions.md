---
comments: true 
---
# Unsupported Posterior Predictive Distributions


Suppose we want to use the geometric model with a beta prior which doesn't have a 
supported distribution for the posterior predictive. 

```python
from conjugate.distributions import Beta
from conjugate.models import geometric_beta

prior = Beta(1, 1)
posterior: Beta = geometric_beta(x_total=12, n=10, beta_prior=prior)
```

We can get posterior predictive samples by: 

1. Sample from the posterior distribution
2. Sample from the model distribution using posterior samples

## 1. Using `conjugate-models`

This is easy to do with this package. 

Since the distributions are vectorized, just: 

1. Get the number of samples from the posterior 
2. Take a single sample from the model distribution

```python
from conjugate.distributions import Geometric

n_samples = 1_000
posterior_samples = posterior.dist.rvs(size=n_samples)
posterior_predictive_samples = Geometric(p=posterior_samples).dist.rvs()
```

## 2. Using `pymc`

Another route would be using PyMC then use the `draw` function. 

```python 
import pymc as pm

posterior_dist = pm.Beta.dist(alpha=posterior.alpha, beta=posterior.beta)
geometric_posterior_predictive = pm.Geometric.dist(posterior_dist)

n_samples = 1_000
posterior_predictive_samples = pm.draw(geometric_posterior_predictive, draws=n_samples)
```
