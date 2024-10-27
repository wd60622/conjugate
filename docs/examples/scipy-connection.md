---
comments: true
---
# Connection to SciPy Distributions

Many distributions have the `dist` attribute which is a <a href=https://docs.scipy.org/doc/scipy/reference/stats.html>scipy.stats distribution</a> object. From there, the methods from scipy.stats to get the pdf, cdf, etc can be leveraged.

```python
from conjugate.distribution import Beta

beta = Beta(1, 1)
scipy_dist = beta.dist

print(scipy_dist.mean())
# 0.5
print(scipy_dist.ppf([0.025, 0.975]))
# [0.025 0.975]

samples = scipy_dist.rvs(100)
```
