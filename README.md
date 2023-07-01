# conjugate priors
Bayesian conjugate models in Python


## Installation

```bash 
pip install conjugate-models
```

## Basic Usage

```python 
from conjugate.distributions import Beta, BetaBinomial
from conjugate.models import binomial_beta, binomial_beta_posterior_predictive

# Observed Data
X = 4
N = 10

# Analytics
prior = Beta(1, 1)
prior_predictive: BetaBinomial = binomial_beta_posterior_predictive(n=N, beta=prior)

posterior: Beta = binomial_beta(n=N, x=X, beta_prior=prior)
posterior_predictive: BetaBinomial = binomial_beta_posterior_predictive(n=N, beta=posterior) 

# Figure
import matplotlib.pyplot as plt

fig, axes = plt.subplots(ncols=2)

ax = axes[0]
ax = posterior.plot_pdf(ax=ax, label="posterior")
prior.plot_pdf(ax=ax, label="prior")
ax.axvline(x=X/N, color="black", ymax=0.05, label="MLE")
ax.set_title("Success Rate")
ax.legend()

ax = axes[1]
posterior_predictive.plot_pmf(ax=ax, label="posterior predictive")
prior_predictive.plot_pmf(ax=ax, label="prior predictive")
ax.axvline(x=X, color="black", ymax=0.05, label="Sample")
ax.set_title("Number of Successes")
ax.legend()
plt.show()
```

<img height=400 src="docs/images/binomial-beta.png" title="Binomial Beta Comparison">

More examples on in the [documentation](https://wd60622.github.io/conjugate/).