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


class PlotDistMixin:
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

    def _reshape_x_values(self, x: np.ndarray) -> np.ndarray:
        """Make sure that the values are ready for plotting."""
        for value in asdict(self).values():
            if not isinstance(value, float):
                return x[:, None]

        return x

    def _resolve_label(self, label: LABEL_INPUT, yy: np.ndarray):
        """

        https://stackoverflow.com/questions/73662931/matplotlib-plot-a-numpy-array-as-many-lines-with-a-single-label
        """
        if yy.ndim == 1:
            return label

        ncols = yy.shape[1]
        if ncols != 1:
            if isinstance(label, Iterable):
                return label

            if isinstance(label, str):
                label = lambda i: f"{label} {i}"

            if callable(label):
                return [label(i) for i in range(1, ncols + 1)]

            raise ValueError("Label must be a string, iterable, or callable.")

        return label

    def _settle_axis(self, ax: Optional[plt.Axes] = None) -> plt.Axes:
        return ax if ax is not None else plt.gca()


class ContinuousPlotDistMixin(PlotDistMixin):
    def plot_pdf(self, ax: Optional[plt.Axes] = None, **kwargs) -> plt.Axes:
        ax = self._settle_axis(ax=ax)

        x = self._create_x_values()
        x = self._reshape_x_values(x)

        return self._create_plot_on_axis(x, ax, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.linspace(0, self.max_value, 100)

    def _setup_labels(self, ax) -> None:
        ax.set_xlabel("Domain")
        ax.set_ylabel("Density $f(x)$")

    def _create_plot_on_axis(self, x, ax, **kwargs) -> plt.Axes:
        yy = self.dist.pdf(x)
        if "label" in kwargs:
            label = kwargs.pop("label")
            label = self._resolve_label(label, yy)
        else:
            label = None

        ax.plot(x, yy, label=label, **kwargs)
        self._setup_labels(ax=ax)
        ax.set_ylim(0, None)
        return ax


class SamplePlotDistMixin(ContinuousPlotDistMixin):
    def plot_pdf(
        self, ax: Optional[plt.Axes] = None, samples: int = 1_000, **kwargs
    ) -> plt.Axes:
        distribution_samples = self.dist.rvs(size=samples)

        ax = self._settle_axis(ax=ax)
        xx = self._create_x_values()

        for x in distribution_samples.T:
            kde = gaussian_kde(x)

            yy = kde(xx)

            ax.plot(xx, yy)

        self._setup_labels(ax=ax)
        return ax


class DiscretePlotMixin(PlotDistMixin):
    def plot_pmf(
        self, ax: Optional[plt.Axes] = None, mark: str = "o-", **kwargs
    ) -> plt.Axes:
        ax = self._settle_axis(ax=ax)

        x = self._create_x_values()
        x = self._reshape_x_values(x)

        return self._create_plot_on_axis(x, ax, mark, **kwargs)

    def _create_x_values(self) -> np.ndarray:
        return np.arange(0, self.max_value + 1, 1)

    def _create_plot_on_axis(self, x, ax, mark, **kwargs) -> plt.Axes:
        yy = self.dist.pmf(x)
        if "label" in kwargs:
            label = kwargs.pop("label")
            label = self._resolve_label(label, yy)
        else:
            label = None

        ax.plot(x, yy, mark, label=label, **kwargs)
        if self.max_value <= 15:
            ax.set_xticks(x.ravel())

        ax.set_xlabel("Domain")
        ax.set_ylabel("Probability $f(x)$")
        ax.set_ylim(0, None)
        return ax
