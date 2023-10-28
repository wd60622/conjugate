---
comments: true 
---
# Unsupported Posterior Predictive Distributions with PyMC Sampling

The geometric beta model posterior predictive doesn't have a common dist, but what doesn't mean the posterior predictive can be used. For instance, PyMC can be used to fill in this gap.

```python 
import pymc as pm

from conjugate.distribution import Beta
from conjugate.models import geometric_beta

prior = Beta(1, 1)
posterior: Beta = geometric_beta(x=1, n=10, beta_prior=prior)

posterior_dist = pm.Beta.dist(alpha=posterior.alpha, beta=posterior.beta)
geometric_posterior_predictive = pm.Geometric.dist(posterior_dist)

posterior_predictive_samples = pm.draw(geometric_posterior_predictive, draws=100)
```