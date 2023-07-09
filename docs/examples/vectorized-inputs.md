# Vectorized Inputs

All data and priors will allow for vectorized assuming the shapes work for broadcasting. 

The plotting also supports arrays of results

```python 
import numpy as np

from conjugate.distributions import Beta
from conjugate.models import binomial_beta

import matplotlib.pyplot as plt

# Analytics 
prior = Beta(alpha=1, beta=np.array([1, 5]))
posterior = binomial_beta(n=N, x=x, beta_prior=prior)

# Figure
ax = prior.plot_pdf(label=lambda i: f"prior {i}")
posterior.plot_pdf(ax=ax, label=lambda i: f"posterior {i}")
ax.axvline(x=x / N, ymax=0.05, color="black", linestyle="--", label="MLE")
ax.legend()
plt.show()
```

![Vectorized Priors and Posterior](images/vectorized-plot.png)