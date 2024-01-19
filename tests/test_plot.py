import pytest

import numpy as np

import matplotlib.pyplot as plt

from conjugate.plot import resolve_label

from conjugate.distributions import Beta, Gamma, Normal


@pytest.mark.parametrize(
    "label, yy, expected",
    [
        ("label", np.array([1, 2, 3]), "label"),
        (None, np.array([1, 2, 3]), None),
        (["label1", "label2"], np.array([1, 2]), ["label1", "label2"]),
        ("label", np.ones(shape=(3, 2)), ["label 1", "label 2"]),
        ("label", np.ones(shape=(2, 3)), ["label 1", "label 2", "label 3"]),
        (
            lambda i: f"another {i + 1} label",
            np.ones(shape=(2, 3)),
            ["another 1 label", "another 2 label", "another 3 label"],
        ),
    ],
)
def test_resolve_label(label, yy, expected):
    assert resolve_label(label, yy) == expected


@pytest.mark.mpl_image_compare
def test_plot():
    beta = Beta(1, 1)
    ax = beta.plot_pdf(label="Uniform")
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_plot_multiple():
    beta = Beta(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ax = beta.plot_pdf(label="Beta")
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_plot_multiple_with_labels():
    beta = Beta(np.array([1, 2, 3]), np.array([1, 2, 3]))
    ax = beta.plot_pdf(label=["First Beta", "Second Beta", "Third Beta"])
    ax.legend()
    return plt.gcf()


@pytest.mark.mpl_image_compare
def test_different_distributions() -> None:
    beta = Beta(1, 1)
    gamma = Gamma(1, 1)
    normal = Normal(0, 1)

    ax = beta.plot_pdf(label="Beta")
    gamma.set_bounds(0, upper=5).plot_pdf(ax=ax, label="Gamma")
    normal.set_bounds(-5, 5).plot_pdf(ax=ax, label="Normal")

    ax.legend()
    return plt.gcf()
