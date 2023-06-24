# conjugate priors
Bayesian conjugate models in Python


## Installation

```bash 
pip install conjugate-models
```

## Usage

```python 
from conjugate.distributions import Beta, NegativeBinomial
from conjugate.models import binomial_beta, binomial_beta_posterior_predictive

# Observed Data
X = 4
N = 10

# Analytics
prior = Beta(1, 1)
prior_predictive: NegativeBinomial = binomial_beta_posterior_predictive(n=N, beta=prior)

posterior: Beta = binomial_beta(n=N, x=X, beta_prior=prior)
posterior_predictive: NegativeBinomial = binomial_beta_posterior_predictive(n=N, beta=posterior) 

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

<img height=400 src="images/binomial-beta.png" title="Binomial Beta Comparison">

Though the plotting is meant for numpy and python numbers, the conjugate models work with anything that works like numbers. 

For instance, using SQL Builder

```python
from pypika import Field 

# Columns from table in database
N = Field("N")
X = Field("X")

# Conjugate prior
prior = Beta(alpha=1, beta=1)
posterior = binomial_beta(n=N, x=X, beta_prior=prior)

print("Posterior alpha:", posterior.alpha)
print("Posterior beta:", posterior.beta)
# Posterior alpha: 1+"X"
# Posterior beta: 1+"N"-"X"
```

Example using PyMC 

```python 
import pymc as pm 

alpha = pm.Gamma.dist(alpha=1, beta=20)
beta = pm.Gamma.dist(alpha=1, beta=20)

# Observed Data
N = 10
X = 4

# Conjugate prior 
prior = Beta(alpha=alpha, beta=beta)
posterior = binomial_beta(n=N, x=X, beta_prior=prior)

# Reconstruct the posterior distribution with PyMC
prior_dist = pm.Beta.dist(alpha=prior.alpha, beta=prior.beta)
posterior_dist = pm.Beta.dist(alpha=posterior.alpha, beta=posterior.beta)

samples = pm.draw([alpha, beta, prior_dist, posterior_dist], draws=1000)
```