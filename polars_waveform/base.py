"""The contract shared by all waveform kinds.

:class:`~polars_waveform.Waveform` stores numeric data in Polars (lazy, families as group
columns). Other kinds implement the same interface with their own storage, for example a
symbolic waveform holding sympy expressions in numpy object arrays: Polars can store Python
objects but has no kernels to compute with them, so such data needs its own container.

A subclass implements storage, construction and elementwise math (the abstract methods), and
:meth:`WaveformBase.numeric` to turn itself into a numeric :class:`~polars_waveform.Waveform`.
Measurements it does not implement itself (``bandwidth``, ``cross``, ``rise_time``, ...) are
inherited from this class and run on ``numeric()``, so they work on any kind once its values are
numbers.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

__all__ = ["WaveformBase"]

# measurements that need numbers: inherited by other kinds, evaluated on numeric()
_NUMERIC = (
    "value", "ymax", "ymin", "xmax", "xmin", "argmax", "argmin", "average", "mean", "rms", "stddev",
    "cross", "frequency", "period", "rise_time", "fall_time", "slew_rate", "overshoot", "settling_time",
    "bandwidth", "unity_gain_frequency", "phase_margin", "gain_margin", "integ", "iinteg", "deriv",
    "clip", "dft", "to_polars", "plot",
)  # fmt: skip


class WaveformBase(ABC):
    """A value over one or more index columns: groups (one curve per combination) and a sweep.

    The interface every waveform kind provides: names and units, conversion from and to numpy
    arrays (:meth:`from_arrays`, :meth:`to_arrays`), arithmetic, the elementwise functions
    (``db20``, ``phase``, ...), and the measurements, which by default evaluate on
    :meth:`numeric`.
    """

    # --- storage and names (abstract) -------------------------------------------------------
    @classmethod
    @abstractmethod
    def from_arrays(cls, x, y, xlabels=None, ylabel=None, xunits=None, yunit=None) -> WaveformBase:
        """A waveform from numpy arrays: one x array per sweep axis (the last one is the sweep,
        the others become groups) and ``y`` with shape ``(len(x0), len(x1), ...)``, or ragged
        object arrays (see :meth:`polars_waveform.Waveform.from_arrays`)."""

    @abstractmethod
    def to_arrays(self):
        """``(xs, y)`` as numpy arrays in the layout :meth:`from_arrays` accepts."""

    @abstractmethod
    def numeric(self):
        """This waveform as a numeric :class:`~polars_waveform.Waveform`."""

    @property
    @abstractmethod
    def index(self) -> list[str]:
        """Index column names: the groups followed by the sweep."""

    @property
    def groups(self) -> list[str]:
        """Index columns other than the sweep (one curve per combination)."""
        return [c for c in self.index if c != self.xname]

    @property
    @abstractmethod
    def xname(self) -> str: ...

    @property
    @abstractmethod
    def yname(self) -> str: ...

    @property
    @abstractmethod
    def xunit(self): ...

    @property
    @abstractmethod
    def yunit(self): ...

    # --- arithmetic (abstract core, shared operators) ------------------------------------------
    @abstractmethod
    def _binop(self, other, op: str, *, reverse: bool = False) -> WaveformBase:
        """``self <op> other`` (``other <op> self`` if ``reverse``) for op in + - * / < > <= >=."""

    def __add__(self, other):
        return self._binop(other, "+")

    def __radd__(self, other):
        return self._binop(other, "+", reverse=True)

    def __sub__(self, other):
        return self._binop(other, "-")

    def __rsub__(self, other):
        return self._binop(other, "-", reverse=True)

    def __mul__(self, other):
        return self._binop(other, "*")

    def __rmul__(self, other):
        return self._binop(other, "*", reverse=True)

    def __truediv__(self, other):
        return self._binop(other, "/")

    def __rtruediv__(self, other):
        return self._binop(other, "/", reverse=True)

    def __lt__(self, other):
        return self._binop(other, "<")

    def __le__(self, other):
        return self._binop(other, "<=")

    def __gt__(self, other):
        return self._binop(other, ">")

    def __ge__(self, other):
        return self._binop(other, ">=")

    # --- elementwise (abstract) ----------------------------------------------------------------
    @abstractmethod
    def __neg__(self): ...

    @abstractmethod
    def __abs__(self): ...

    @abstractmethod
    def __pow__(self, other): ...

    @abstractmethod
    def real(self): ...

    @abstractmethod
    def imag(self): ...

    @abstractmethod
    def conj(self): ...

    @abstractmethod
    def phase(self, deg: bool = True): ...

    @abstractmethod
    def db10(self): ...

    @abstractmethod
    def db20(self): ...

    def __pos__(self):
        return self

    def abs(self):
        """Magnitude (``abs(w)``)."""
        return abs(self)

    mag = abs

    def conjugate(self):
        return self.conj()


def _numeric_method(name: str):
    def method(self, *args, **kwargs):
        num = self.numeric()
        if type(num) is type(self):  # a numeric kind must implement the measurement itself
            raise NotImplementedError(f"{type(self).__name__}.{name}()")
        return getattr(num, name)(*args, **kwargs)

    method.__name__ = name
    method.__doc__ = f"``{name}`` evaluated on :meth:`numeric` (see ``polars_waveform.Waveform.{name}``)."
    return method


for _name in _NUMERIC:
    setattr(WaveformBase, _name, _numeric_method(_name))
