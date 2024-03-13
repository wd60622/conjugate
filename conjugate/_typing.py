from typing import Protocol


class NUMERIC(Protocol):
    def __add__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __radd__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __sub__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __rsub__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __mul__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __rmul__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __truediv__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover

    def __rtruediv__(self, other: "NUMERIC") -> "NUMERIC":
        ...  # pragma: no cover
