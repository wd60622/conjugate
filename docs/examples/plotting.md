---
comments: true 
---
# Plotting Distributions

All the distributions can be plotted using the `plot_pdf` and `plot_pmf` methods. The `plot_pdf` method is used for continuous distributions and the `plot_pmf` method is used for discrete distributions.

There is limited support for some distributions like the `Dirichlet` or those without a `dist` scipy.


```python 
from conjugate.distributions import Beta, Gamma, Normal

import matplotlib.pyplot as plt

beta = Beta(1, 1)
gamma = Gamma(1, 1)
normal = Normal(0, 1)

bound = 3

dist = [beta, gamma, normal]
labels = ["beta", "gamma", "normal"]
ax = plt.gca()
for label, dist in zip(labels, dist):
    dist.set_bounds(-bound, bound).plot_pdf(label=label)

ax.legend()
plt.show()
```

![Plotting Distributions](../images/plotting-example.png)

The plotting is also supported for [vectorized inputs](vectorized-inputs.md).