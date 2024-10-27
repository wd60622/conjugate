"""Making use of the tool to analyze post engagement on LinkedIn

Incorporate prior knowledge of the average engagement rate on LinkedIn into
the few data points I have from my posts. Then, use the model to make some visuals.

"""

import pandas as pd
import numpy as np

from conjugate.distributions import Beta
from conjugate.models import binomial_beta

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


# Data from my LinkedIn posts
df = pd.DataFrame(
    {
        "impressions": [
            1769,
            1585,
            3803,
            6415,
        ],
        "reactions": [
            43,
            22,
            40,
            81,
        ],
        "label": [
            "1: lc-announcement",
            "2: lc-2-weeks-before",
            "3: lc-after-talk",
            "4: conjugate-models",
        ],
    }
)
IMPRESSIONS = df["impressions"].to_numpy()
REACTIONS = df["reactions"].to_numpy()
LABELS = df["label"].to_numpy()

N_POSTS = len(df)


if __name__ == "__main__":
    # Took the tech engagement rate from this post:
    # https://blog.hootsuite.com/average-engagement-rate/
    # 1.73% engagement rate
    linkedin_average_rate = 0.0173
    # centered around average but with some uncertainty
    prior = Beta.from_mean(mean=linkedin_average_rate, alpha=1)
    # Model for the data
    posterior: Beta = binomial_beta(n=IMPRESSIONS, x=REACTIONS, beta_prior=prior)
    # Inference
    n_posterior_samples = 5_000
    rng = np.random.default_rng(42)
    samples = posterior.dist.rvs(size=(n_posterior_samples, N_POSTS), random_state=rng)

    # Figures with much more code
    fig, axes = plt.subplots(ncols=3)
    fig.suptitle("Post Engagement Analysis with Conjugate Models")

    # Engagement rate from the model
    ax = axes[0]
    ax = posterior.set_bounds(0, 0.05).plot_pdf(label=LABELS, ax=ax)
    ax.axvline(
        linkedin_average_rate, color="black", ymax=0.05, label="LinkedIn Average"
    )
    ax.legend(title="Post")
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.set(
        xlabel="Engagement Rate",
        title="Post Engagement Rate",
    )

    # Difference between two posts
    ax = axes[1]
    diff = samples[:, 0] - samples[:, -1]
    ax.hist(diff, alpha=0.5, edgecolor="black")
    ax.axvline(0, color="black", linestyle="--", label="same engagement rate")
    ax.set(
        xlabel="Difference in Engagement Rate",
        ylabel="# of Posterior Samples",
        title="Engagement between first and most recent post",
    )
    ax.xaxis.set_major_formatter(PercentFormatter(1))
    ax.legend()

    # Most engaged post
    ax = axes[2]
    most_engaged = pd.Series(np.argmax(samples, axis=1))
    most_engaged.value_counts(normalize=True).reindex(df.index).plot.bar(ax=ax)
    ax.yaxis.set_major_formatter(PercentFormatter(1))
    ax.set_xticklabels(df.index + 1, rotation=0)
    ax.set(
        xlabel="Post",
        ylabel="% of Posterior Samples",
        title="Most Engaged Post",
    )
    plt.show()
