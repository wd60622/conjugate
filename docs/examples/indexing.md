---
comments: true 
---
# Indexing Parameters

The distributions can be indexed for subsets. 

```python
beta = np.arange(1, 10)
prior = Beta(alpha=1, beta=beta)

idx = [0, 5, -1]
prior_subset = prior[idx]
prior_subset.plot_pdf(label = lambda i: f"prior {i}")
plt.legend()
plt.show()
```

![Sliced Distribution](./../images/sliced-distribution.png)