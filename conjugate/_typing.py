from typing import Protocol


class NUMERIC(Protocol):
    def __add__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __radd__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __sub__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __rsub__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __mul__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __rmul__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __truediv__(self, other: "NUMERIC") -> "NUMERIC":
        ...

    def __rtruediv__(self, other: "NUMERIC") -> "NUMERIC":
        ...
