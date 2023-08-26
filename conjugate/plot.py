from dataclasses import asdict
from typing import Callable, Iterable, Optional, Protocol, Union

import matplotlib.pyplot as plt

import numpy as np
from scipy.stats import gaussian_kde


class Distribution(Protocol):
    def pdf(self, *args, **kwargs) -> np.ndarray:
        ...

    def pmf(self, *args, **kwargs) -> np.ndarray:
        ...

    def rvs(self, size, *args, **kwargs) -> np.ndarray:
        ...


LABEL_INPUT = Union[str, Iterable[str], Callable[[int], str]]


def resolve_label(label: LABEL_INPUT, yy: np.ndarray):
    """

    https://stackoverflow.com/questions/73662931/matplotlib-plot-a-numpy-array-as-many-lines-with-a-single-label
    """
    if yy.ndim == 1:
        return label

    ncols = yy.shape[1]
    if ncols != 1:
        if isinstance(label, str):
            return [f"{label} {i}" for i in range(1, ncols + 1)]

        if callable(label):
            return [label(i) for i in range(ncols)]

        if isinstance(label, Iterable):
            return label

        raise ValueError("Label must be a string, iterable, or callable.")

    return label


class PlotDistMixin:
    """Base mixin in order to support plotting. Requires the dist attribute of the scipy distribution."""

    @property
    def dist(self) -> Distribution:
        raise NotImplementedError("Implement this property in the subclass.")

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

    def _settle_axis(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        return ax if ax is not None else plt.gca()


class ContinuousPlotDistMixin(PlotDistMixin):
    """Functionality for plot_pdf method of continuous distributions."""

    def plot_pdf(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        """Plot the pdf of distribution

        Args:
            ax: matplotlib Axes, optional
            **kwargs: Additonal kwargs to pass to matplotlib

        Returns:
            new or modified Axes

        """
        ax = self._settle_axis(ax=ax)

        x = self._create_x_values()
        x = self._reshape_x_values(x)

        return self._create_plot_on_axis(x, ax, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.linspace(self.min_value, self.max_value, 100)

    def _setup_labels(self, ax) -> None:
        ax.set_xlabel("Domain")
        ax.set_ylabel("Density $f(x)$")

    def _create_plot_on_axis(self, x, ax, **kwargs) -> plt.Axes:
        yy = self.dist.pdf(x)
        if "label" in kwargs:
            label = kwargs.pop("label")
            label = resolve_label(label, yy)
        else:
            label = None

        ax.plot(x, yy, label=label, **kwargs)
        self._setup_labels(ax=ax)
        ax.set_ylim(0, None)
        return ax


class DirichletPlotDistMixin(ContinuousPlotDistMixin):
    """Plot the pdf using samples from the dirichlet distribution."""

    def plot_pdf(
        self, ax: Optional[plt.Axes] = None, samples: int = 1_000, **kwargs
    ) -> plt.Axes:
        """Plots the pdf"""
        distribution_samples = self.dist.rvs(size=samples)

        ax = self._settle_axis(ax=ax)
        xx = self._create_x_values()

        for x in distribution_samples.T:
            kde = gaussian_kde(x)

            yy = kde(xx)

            ax.plot(xx, yy, **kwargs)

        self._setup_labels(ax=ax)
        return ax


class DiscretePlotMixin(PlotDistMixin):
    """Adding the plot_pmf method to class."""

    def plot_pmf(
        self, ax: Optional[plt.Axes] = None, mark: str = "o-", **kwargs
    ) -> plt.Axes:
        ax = self._settle_axis(ax=ax)

        x = self._create_x_values()
        x = self._reshape_x_values(x)

        return self._create_plot_on_axis(x, ax, mark, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.arange(self.min_value, self.max_value + 1, 1)

    def _create_plot_on_axis(
        self, x, ax, mark, conditional: bool = False, **kwargs
    ) -> plt.Axes:
        yy = self.dist.pmf(x)
        if conditional:
            yy = yy / np.sum(yy)
            ylabel = f"Conditional Probability $f(x|{self.min_value} \\leq x \\leq {self.max_value})$"
        else:
            ylabel = "Probability $f(x)$"

        if "label" in kwargs:
            label = kwargs.pop("label")
            label = resolve_label(label, yy)
        else:
            label = None

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
