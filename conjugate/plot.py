from collections.abc import Callable, Iterable
from dataclasses import asdict
from itertools import zip_longest
from typing import Protocol

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.projections.polar import PolarAxes

import numpy as np
from scipy.stats import gaussian_kde


class Distribution(Protocol):
    def pdf(self, *args, **kwargs) -> np.ndarray: ...  # pragma: no cover

    def pmf(self, *args, **kwargs) -> np.ndarray: ...  # pragma: no cover

    def cdf(self, *args, **kwargs) -> np.ndarray: ...  # pragma: no cover

    def rvs(self, size, *args, **kwargs) -> np.ndarray: ...  # pragma: no cover


LABEL_INPUT = str | Iterable[str] | Callable[[int], str] | None


def label_to_iterable(label: LABEL_INPUT, ncols: int) -> Iterable[str]:
    if label is None:
        return [None] * ncols

    if isinstance(label, str):
        return [f"{label} {i}" for i in range(1, ncols + 1)]

    if callable(label):
        return [label(i) for i in range(ncols)]

    if isinstance(label, Iterable):
        return label

    raise ValueError(
        "Label must be None, a string, iterable, or callable.",
    )  # pragma: no cover


def resolve_label(label: LABEL_INPUT, yy: np.ndarray):
    """

    https://stackoverflow.com/questions/73662931/matplotlib-plot-a-numpy-array-as-many-lines-with-a-single-label
    """
    if yy.ndim == 1:
        return label

    ncols = yy.shape[1]
    if ncols != 1:
        return label_to_iterable(label, ncols)

    return label


class PlotDistMixin:
    """Base mixin in order to support plotting. Requires the dist attribute of the scipy distribution."""

    @property
    def dist(self) -> Distribution:
        raise NotImplementedError(
            "Implement this property in the subclass.",
        )  # pragma: no cover

    @property
    def max_value(self) -> float:
        if not hasattr(self, "_max_value"):
            raise ValueError("Set the max value before plotting.")

        return self._max_value

    @max_value.setter
    def max_value(self, value: float) -> None:
        self._max_value = value

    def set_max_value(self, value: float) -> "PlotDistMixin":
        self.max_value = value

        return self

    @property
    def min_value(self) -> float:
        if not hasattr(self, "_min_value"):
            self._min_value = 0.0

        return self._min_value

    @min_value.setter
    def min_value(self, value: float) -> None:
        self._min_value = value

    def set_min_value(self, value: float) -> "PlotDistMixin":
        """Set the minimum value for plotting."""
        self.min_value = value

        return self

    def set_bounds(self, lower: float, upper: float) -> "PlotDistMixin":
        """Set both the min and max values for plotting."""
        return self.set_min_value(lower).set_max_value(upper)

    def _reshape_x_values(self, x: np.ndarray) -> np.ndarray:
        """Make sure that the values are ready for plotting."""
        for value in asdict(self).values():
            if not isinstance(value, float):
                return x[:, None]

        return x

    def _settle_axis(self, ax: Axes | None = None) -> Axes:
        return ax if ax is not None else plt.gca()


class ContinuousPlotDistMixin(PlotDistMixin):
    """Functionality for plot_pdf method of continuous distributions."""

    def _plot(self, ax: Axes | None = None, cdf: bool = False, **kwargs) -> Axes:
        x = self._create_x_values()
        x = self._reshape_x_values(x)

        ax = self._settle_axis(ax=ax)

        return self._create_plot_on_axis(x=x, cdf=cdf, ax=ax, **kwargs)

    def plot_pdf(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the PDF of distribution

        Args:
            ax: matplotlib Axes, optional
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        Raises:
            ValueError: If the max_value is not set.

        """
        return self._plot(ax=ax, cdf=False, **kwargs)

    def plot_cdf(self, ax: Axes | None = None, **kwargs) -> Axes:
        """Plot the CDF of distribution

        Args:
            ax: matplotlib Axes, optional
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        Raises:
            ValueError: If the max_value is not set.

        """
        return self._plot(ax=ax, cdf=True, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, 100)

    def _setup_labels(self, ax, cdf: bool = False) -> None:
        if isinstance(ax, PolarAxes):
            return

        ylabel = "Density $f(x)$" if not cdf else "Cumulative Density $F(x)$"
        ax.set(xlabel="Domain", ylabel=ylabel)

    def _create_plot_on_axis(self, x, cdf: bool, ax: Axes, **kwargs) -> Axes:
        func = self.dist.cdf if cdf else self.dist.pdf
        yy = func(x)

        if "label" in kwargs:
            label = kwargs.pop("label")
            label = resolve_label(label, yy)
        else:
            label = None

        if "color" in kwargs and isinstance(kwargs["color"], Iterable):
            ax.set_prop_cycle(color=kwargs.pop("color"))

        ax.plot(x, yy, label=label, **kwargs)
        self._setup_labels(ax=ax, cdf=cdf)
        ax.set_ylim(0, None)
        return ax


class DirichletPlotDistMixin(ContinuousPlotDistMixin):
    """Plot the pdf using samples from the dirichlet distribution."""

    def plot_pdf(
        self,
        ax: Axes | None = None,
        samples: int = 1_000,
        random_state=None,
        **kwargs,
    ) -> Axes:
        """Plots the pdf by sampling from the distribution.

        Args:
            ax: matplotlib Axes, optional
            samples: number of samples to take from the distribution
            random_state: random state to use for sampling
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        """
        distribution_samples = self.dist.rvs(size=samples, random_state=random_state)

        ax = self._settle_axis(ax=ax)
        xx = self._create_x_values()

        labels = label_to_iterable(
            kwargs.pop("label", None), distribution_samples.shape[1]
        )

        for x, label in zip_longest(distribution_samples.T, labels):
            kde = gaussian_kde(x)

            yy = kde(xx)
            ax.plot(xx, yy, label=label, **kwargs)

        self._setup_labels(ax=ax)
        return ax


class DiscretePlotMixin(PlotDistMixin):
    """Adding the plot_pmf method to class."""

    def _plot(
        self,
        ax: Axes | None = None,
        cdf: bool = False,
        mark: str = "o-",
        conditional: bool = False,
        **kwargs,
    ) -> Axes:
        x = self._create_x_values()
        x = self._reshape_x_values(x)

        ax = self._settle_axis(ax=ax)
        return self._create_plot_on_axis(
            x,
            ax=ax,
            cdf=cdf,
            mark=mark,
            conditional=conditional,
            **kwargs,
        )

    def plot_pmf(
        self,
        ax: Axes | None = None,
        mark: str = "o-",
        conditional: bool = False,
        **kwargs,
    ) -> Axes:
        """Plot the PMF of distribution

        Args:
            ax: matplotlib Axes, optional
            mark: matplotlib line style
            conditional: If True, plot the conditional probability given the bounds.
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        Raises:
            ValueError: If the max_value is not set.

        """
        return self._plot(
            ax=ax,
            cdf=False,
            mark=mark,
            conditional=conditional,
            **kwargs,
        )

    def plot_cdf(
        self,
        ax: Axes | None = None,
        mark: str = "o-",
        conditional: bool = False,
        **kwargs,
    ) -> Axes:
        """Plot the CDF of distribution

        Args:
            ax: matplotlib Axes, optional
            mark: matplotlib line style
            conditional: If True, plot the conditional probability given the bounds.
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        Raises:
            ValueError: If the max_value is not set.

        """
        return self._plot(ax=ax, cdf=True, mark=mark, conditional=conditional, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.arange(self.min_value, self.max_value + 1, 1)

    def _create_plot_on_axis(
        self,
        x,
        ax,
        cdf: bool,
        mark,
        conditional: bool = False,
        **kwargs,
    ) -> Axes:
        func = self.dist.cdf if cdf else self.dist.pmf
        yy = func(x)

        if conditional:
            yy = yy / np.sum(yy)

            prefix = (
                "Cumulative Probability $F(X \\leq x" if cdf else "Probability $f(x"
            )

            ylabel = f"Conditional {prefix}|{self.min_value} \\leq x \\leq {self.max_value})$"
        else:
            ylabel = (
                "Cumulative Probability $F(X \\leq x)$" if cdf else "Probability $f(x)$"
            )

        if "label" in kwargs:
            label = kwargs.pop("label")
            label = resolve_label(label, yy)
        else:
            label = None

        if "color" in kwargs and isinstance(kwargs["color"], Iterable):
            ax.set_prop_cycle(color=kwargs.pop("color"))

        ax.plot(x, yy, mark, label=label, **kwargs)

        if self.max_value - self.min_value < 15:
            ax.set_xticks(x.ravel())
        else:
            ax.set_xticks(x.ravel(), minor=True)
            ax.set_xticks(x[::5].ravel())

        ax.set_xlabel("Domain")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, None)
        return ax
