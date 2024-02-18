---
comments: true 
---
# Sampling from Distributions

Use the `rvs` method of the scipy distribution stored in `dist` attribute

```python
distribution.dist.rvs(...)
```

## Scalar parameters

If the parameters are scalars, then just pass the number of samples to the `rvs`
method!

```python
from conjugate.distributions import Exponential

lam = 3.5
true_distribution = Exponential(lam=lam)

n_samples = 10
samples = true_distribution.dist.rvs(n_samples)
```

## Vector parameter

If the parameter is a vector, then there will be a broadcast issue from the scipy 
distribution.

```python
import numpy as np

lam = np.array([
    [1, 2], 
    [0.5, 5], 
])

true_distribution = Exponential(lam=lam)

n_samples = 100
try: 
    true_distribution.dist.rvs(n_samples)
except ValueError: 
    print("The number of samples doesn't broadcast with the shape of parameters!")
```

However, this is easy to fix by prepending the number of samples to the shape of 
the model parameter shape

```python
size = (n_samples, *lam.shape)
samples = true_distribution.dist.rvs(size=size, random_state=rng)
```

## Vector parameters

If there are many parameters in your model, then use the `np.broadcast_shapes` 
function in order to get the correct shape before sampling 

```python
from conjugate.distributions import Normal

mu = np.array([1, 2, 3])
sigma = np.array([2.5, 5])[:, None]

true_distribution = Normal(mu=mu, sigma=sigma)

shape = np.broadcast_shapes(mu.shape, sigma.shape)
size = (n_samples, *shape)
samples = true_distribution.dist.rvs(size=size, random_state=rng)
```

