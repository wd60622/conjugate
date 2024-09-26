# Bayesian Models with SQL 


Because `conjugate-models` works with [general numerical inputs](generalized-inputs.md), we can use Bayesian models in SQL
with the SQL builder, `PyPika`.

For the example, we will estimate use normal model to estimate the 
total sales amount by group. 

The example table is called `events` and we will assume a normal model for the 
column `sales` for each value of the column `group`.

We can create the sufficient statistics needed for `normal_normal_inverse_gamma`
directly with the SQL builder.


```python
from pypika import Query, Table, functions as fn

event_table = Table("events")

sales = event_table.sales
sales_squared = sales**2

# Sufficient statistics
x_total = fn.Sum(sales)
x2_total = fn.Sum(sales_squared)
n = fn.Count("*")

# Start a query for a groupby
query = (
    Query.from_(event_table)
    .groupby(event_table.group)
    .select(
        event_table.group,
    )
)
```

Perform the Bayesian inference as usual, but using the variables reflecting
the columns. 

```python
from conjugate.distributions import NormalInverseGamma
from conjugate.models import (
    normal_normal_inverse_gamma,
    normal_normal_inverse_gamma_predictive,
)

# Bayesian Inference
prior = NormalInverseGamma(mu=0, nu=1 / 10, alpha=1 / 10, beta=1)
posterior = normal_normal_inverse_gamma(
    x_total=x_total,
    x2_total=x2_total,
    n=n,
    prior=prior,
)
posterior_predictive = normal_normal_inverse_gamma_predictive(distribution=posterior)
```

Then add the columns we want from the inference

```
# Add the posterior predictive estimate
query = query.select(
    posterior_predictive.mu.as_("mu"),
    posterior_predictive.sigma.as_("sigma"),
    posterior_predictive.nu.as_("nu"),
)
```

Which results in this query: 

```sql
SELECT "group",
       (0.0+COUNT(*)*SUM("sales")/COUNT(*))/(0.1+COUNT(*)) "mu",
       POW((1+0.5*(0.0+SUM(POW("sales", 2))-POW((0.0+COUNT(*)*SUM("sales")/COUNT(*))/(0.1+COUNT(*)), 2)*(0.1+COUNT(*))))*(0.1+COUNT(*)+1)/((0.1+COUNT(*))*(0.1+COUNT(*)/2)), 0.5) "sigma",
       2*(0.1+COUNT(*)/2) "nu"
FROM "events"
GROUP BY "group"
```

