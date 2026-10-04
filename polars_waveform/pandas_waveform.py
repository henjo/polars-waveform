"""Waveforms of Python objects, for values Polars can store but not compute with.

:class:`PandasWaveform` keeps the same layout as :class:`~polars_waveform.Waveform` (group
columns, a sweep, one value column) in a pandas ``Series`` with a ``MultiIndex``. pandas applies
Python operators element by element, so the values can be any objects that support them, for
example sympy expressions from a symbolic circuit analysis::

    import sympy
    s, R, C = sympy.symbols("s R C")
    h = pw.from_arrays(freqs, [1 / (1 + 2j * sympy.pi * f * R * C) for f in freqs], xlabels=["freq"])
    h.db20()                                  # still symbolic
    h.subs({R: 1e3, C: 1e-9}).bandwidth()     # numbers: measured on numeric()

Measurements that need numbers (``bandwidth``, ``cross``, ``ymax``, plotting, ...) run on
:meth:`PandasWaveform.numeric`, the equivalent :class:`~polars_waveform.Waveform`; it raises,
naming the free symbols, while symbols are left. Needs pandas (``polars-waveform[pandas]``).
"""

from __future__ import annotations

import cmath
import math
import operator
from typing import Any

import polars as pl

from .base import WaveformBase
from .waveform import Waveform

__all__ = ["PandasWaveform", "from_arrays"]

_OPS = {
    "+": operator.add, "-": operator.sub, "*": operator.mul, "/": operator.truediv,
    "<": operator.lt, ">": operator.gt, "<=": operator.le, ">=": operator.ge,
}  # fmt: skip


def _sympy(e) -> Any:
    """sympy if ``e`` is a sympy object (imported only then), else ``None``."""
    mod = type(e).__module__
    if mod.startswith("sympy"):
        import sympy

        return sympy
    return None


def _log10(e):
    sp = _sympy(e)
    if sp is not None:
        return sp.log(e, 10)
    return math.log10(e)


def _phase(e, deg: bool):
    sp = _sympy(e)
    if sp is not None:
        a = sp.arg(e)
        return a * 180 / sp.pi if deg else a
    a = cmath.phase(e)
    return math.degrees(a) if deg else a


def _exp(e):
    sp = _sympy(e)
    return sp.exp(e) if sp is not None else cmath.exp(e) if isinstance(e, complex) else math.exp(e)


def _sqrt(e):
    sp = _sympy(e)
    return sp.sqrt(e) if sp is not None else cmath.sqrt(e) if isinstance(e, complex) or e < 0 else math.sqrt(e)


def _part(e, name: str):
    sp = _sympy(e)
    if sp is not None:
        return {"real": sp.re, "imag": sp.im, "conj": sp.conjugate}[name](e)
    if name == "conj":
        return e.conjugate()
    return getattr(e, name)


class PandasWaveform(WaveformBase):
    """A value over index columns (groups, then the sweep), stored as a pandas ``Series`` with a
    ``MultiIndex``; the values may be any Python objects (sympy expressions, ...).

    Build one with :meth:`from_arrays` or :func:`polars_waveform.from_arrays`, or from a Series
    with named index levels: ``PandasWaveform(series, units={"freq": "Hz"})``; the last level is
    the sweep.
    """

    def __init__(self, series, units: dict | None = None):
        import pandas as pd

        if not isinstance(series, pd.Series):
            raise TypeError("PandasWaveform needs a pandas Series")
        names = list(series.index.names)
        if any(n is None for n in names):
            raise ValueError(f"index levels need names, got {names}")
        if series.name is None or series.name in names:
            raise ValueError("the Series needs a name that is not an index level")
        self._s = series
        self._units = {k: v for k, v in (units or {}).items() if v is not None}

    # --- construction ----------------------------------------------------------------------
    @classmethod
    def from_arrays(cls, x, y, xlabels=None, ylabel=None, xunits=None, yunit=None) -> PandasWaveform:
        """From a grid: ``x`` is one array per axis (a single array for one axis), ``y`` has
        shape ``(len(x0), len(x1), ...)``; the last axis is the sweep, the others are groups.
        Labels default to ``x0, x1, ...`` and ``y``."""
        import numpy as np
        import pandas as pd

        if (isinstance(x, np.ndarray) and x.dtype != object) or all(np.ndim(xi) == 0 for xi in x):
            x = [x]
        xs = [np.asarray(xi) for xi in x]
        names = list(xlabels) if xlabels is not None else [f"x{i}" for i in range(len(xs))]
        if len(names) != len(xs):
            raise ValueError(f"{len(xs)} x arrays but {len(names)} labels")
        ya = np.empty(tuple(len(xi) for xi in xs), dtype=object)
        y = np.asarray(y, dtype=object)
        if y.shape != ya.shape:
            raise ValueError(f"y has shape {y.shape}, the x arrays give {ya.shape}")
        ya[...] = y
        index = pd.MultiIndex.from_product([pd.Index(xi.tolist(), name=n) for xi, n in zip(xs, names, strict=True)])
        units = dict(zip(names, xunits, strict=False)) if xunits is not None else {}
        yname = ylabel if ylabel is not None else "y"
        units[yname] = yunit
        return cls(pd.Series(ya.ravel(), index=index, name=yname, dtype=object), units)

    def to_arrays(self):
        """``(xs, y)`` on the full grid of index values (``y`` an object array)."""
        import numpy as np

        idx = self._s.index
        axes = [np.asarray(idx.unique(level=n)) for n in self.index]
        full = self._s.reindex(_product(axes, self.index))
        return axes, np.asarray(full.to_numpy(dtype=object)).reshape(tuple(len(a) for a in axes))

    def numeric(self) -> Waveform:
        """This waveform as a numeric :class:`~polars_waveform.Waveform` (raises while symbols
        are left; substitute them with :meth:`subs`)."""
        vals = []
        for v in self._s.to_numpy():
            try:
                vals.append(complex(v))
            except (TypeError, ValueError) as e:
                free = set()
                for u in self._s.to_numpy():
                    free |= set(map(str, getattr(u, "free_symbols", ())))
                raise ValueError(f"{self.yname} has free symbols {sorted(free)}: substitute them first (subs)") from e
        frame = self._s.index.to_frame(index=False)
        data = {n: frame[n].to_numpy() for n in self.index}
        if any(v.imag for v in vals):
            yv = pl.DataFrame({"re": [v.real for v in vals], "im": [v.imag for v in vals]}).to_struct(self.yname)
        else:
            yv = pl.Series(self.yname, [v.real for v in vals], dtype=pl.Float64)
        df = pl.DataFrame(data).with_columns(yv.alias(self.yname))
        return Waveform(df, self.yname, index=self.index, units=self._units)

    def to_pandas(self):
        """The underlying ``Series`` (index levels: groups, then the sweep)."""
        return self._s

    # --- names ------------------------------------------------------------------------------
    @property
    def index(self) -> list[str]:
        return list(self._s.index.names)

    @property
    def xname(self) -> str:
        return self.index[-1]

    @property
    def yname(self) -> str:
        return self._s.name

    @property
    def xunit(self):
        return self._units.get(self.xname)

    @property
    def yunit(self):
        return self._units.get(self.yname)

    def __len__(self) -> int:
        return len(self._s)

    def __repr__(self) -> str:
        axes = " x ".join(self.index)
        return f"PandasWaveform({axes} -> {self.yname}, {len(self)} points)"

    # --- elementwise ------------------------------------------------------------------------
    def _new(self, series, name: str | None = None, unit=...) -> PandasWaveform:
        name = name or self.yname
        units = {k: v for k, v in self._units.items() if k != self.yname}
        units[name] = self.yunit if unit is ... else unit
        return PandasWaveform(series.rename(name), units)

    def map(self, func, name: str | None = None) -> PandasWaveform:
        """``func`` applied to every value: ``w.map(sympy.simplify)``, ``w.map(sympy.factor)``."""
        return self._new(self._s.map(func).astype(object), name or f"{getattr(func, '__name__', 'map')}({self.yname})")

    def subs(self, *args, **kwargs) -> PandasWaveform:
        """Substitute symbols (sympy ``subs``) in every value: ``w.subs({R: 1e3})``. Values
        without ``subs`` stay as they are; ``w.subs(...).numeric()`` once no symbols are left."""

        def sub(e):
            return e.subs(*args, **kwargs) if hasattr(e, "subs") else e

        return self._new(self._s.map(sub).astype(object))

    def _binop(self, other, op: str, *, reverse: bool = False):
        if isinstance(other, Waveform):
            return self.numeric()._binop(other, op, reverse=reverse)
        f = _OPS[op]
        if isinstance(other, PandasWaveform):
            if other.index != self.index:
                raise ValueError(f"index columns differ: {self.index} vs {other.index}")
            a, b = self._s.align(other._s, join="inner")
            if len(a) != len(self._s):
                raise ValueError("waveforms have different index values")
            rhs, rname = b, other.yname
            lhs = a
        else:
            lhs, rhs, rname = self._s, other, str(other)
        out = f(rhs, lhs) if reverse else f(lhs, rhs)
        name = f"{rname} {op} {self.yname}" if reverse else f"{self.yname} {op} {rname}"
        unit = self.yunit if op in "+-" else None
        return self._new(out.astype(object), name, unit)

    def __neg__(self):
        return self._new(-self._s, f"-{self.yname}")

    def __abs__(self):
        return self._new(self._s.map(abs).astype(object), f"abs({self.yname})")

    def __pow__(self, other):
        return self._new(self._s.map(lambda e: e**other).astype(object), f"{self.yname}**{other}", None)

    def real(self):
        return self._new(self._s.map(lambda e: _part(e, "real")).astype(object), f"real({self.yname})")

    def imag(self):
        return self._new(self._s.map(lambda e: _part(e, "imag")).astype(object), f"imag({self.yname})")

    def conj(self):
        return self._new(self._s.map(lambda e: _part(e, "conj")).astype(object), f"conj({self.yname})")

    def phase(self, deg: bool = True):
        return self._new(self._s.map(lambda e: _phase(e, deg)).astype(object), f"phase({self.yname})",
                         "deg" if deg else "rad")  # fmt: skip

    def db10(self):
        return self._new(self._s.map(lambda e: 10 * _log10(abs(e))).astype(object), f"db10({self.yname})", "dB")

    def db20(self):
        return self._new(self._s.map(lambda e: 20 * _log10(abs(e))).astype(object), f"db20({self.yname})", "dB")

    def log10(self):
        return self._new(self._s.map(_log10).astype(object), f"log10({self.yname})", None)

    def exp(self):
        return self._new(self._s.map(_exp).astype(object), f"exp({self.yname})", None)

    def sqrt(self):
        return self._new(self._s.map(_sqrt).astype(object), f"sqrt({self.yname})", None)

    # --- selection (no numbers needed) ------------------------------------------------------
    def leaf(self, **values) -> PandasWaveform:
        """One curve (or a smaller family): ``w.leaf(temp=27)``."""
        unknown = set(values) - set(self.groups)
        if unknown:
            raise KeyError(f"not group columns: {sorted(unknown)} (groups: {self.groups})")
        s = self._s.xs(tuple(values.values()), level=list(values), drop_level=True)
        return PandasWaveform(s, self._units)

    def value(self, x):
        """``y`` at sweep value ``x``: exact sweep points work on any values (a number for one
        curve, a waveform over the groups for a family); other points interpolate numerically."""
        levels = self._s.index.get_level_values(self.xname)
        if not (levels == x).any():
            return self.numeric().value(x)
        s = self._s.xs(x, level=self.xname)
        if not self.groups:
            return s.iloc[0] if hasattr(s, "iloc") else s
        return PandasWaveform(s.rename(self.yname), self._units)

    def clip(self, xfrom, xto=None) -> PandasWaveform:
        """Samples with ``xfrom <= x <= xto``."""
        x = self._s.index.get_level_values(self.xname)
        keep = (x >= xfrom) & ((x <= xto) if xto is not None else True)
        return PandasWaveform(self._s[keep], self._units)

    def deriv(self) -> PandasWaveform:
        """Forward difference ``dy/dx`` per curve (one sample shorter), on any values."""
        x = self._s.index.get_level_values(self.xname).to_numpy()
        keys = [self._s.index.get_level_values(g) for g in self.groups] if self.groups else None
        nxt = self._s.groupby(keys).shift(-1) if keys else self._s.shift(-1)
        has_next = nxt.notna()  # before arithmetic: nan - expr is a (non-null) sympy nan
        dx = _shift_per_curve(x, keys) - x
        out = ((nxt - self._s) / dx)[has_next]
        unit = f"{self.yunit}/{self.xunit}" if self.yunit and self.xunit else None
        return self._new(out.astype(object), f"deriv({self.yname})", unit)

    def __getitem__(self, i):
        """The ``i``-th sample of every curve (a value for a single curve), or a slice per curve."""
        if not isinstance(i, (int, slice)):
            raise TypeError("index with an int or a slice")
        if not self.groups:
            return self._s.iloc[i] if isinstance(i, int) else PandasWaveform(self._s.iloc[i], self._units)
        parts = self._s.groupby(level=self.groups, sort=False, group_keys=False)
        if isinstance(i, int):
            return PandasWaveform(parts.nth(i).droplevel(self.xname).rename(self.yname), self._units)
        return PandasWaveform(parts.apply(lambda c: c.iloc[i]), self._units)


def _product(axes, names):
    import pandas as pd

    return pd.MultiIndex.from_product([pd.Index(a.tolist(), name=n) for a, n in zip(axes, names, strict=True)])


def _shift_per_curve(x, keys):
    import pandas as pd

    s = pd.Series(x)
    if keys is None:
        return s.shift(-1).to_numpy()
    return s.groupby([k.to_numpy() for k in keys]).shift(-1).to_numpy()


def _is_number(v) -> bool:
    import numbers

    return isinstance(v, numbers.Number)


def from_arrays(x, y, xlabels=None, ylabel=None, xunits=None, yunit=None) -> WaveformBase:
    """A waveform from arrays (layout of :meth:`Waveform.from_arrays`), of the kind the values
    need: a :class:`Waveform` for numbers, a :class:`PandasWaveform` for other objects (sympy
    expressions, ...)."""
    import numpy as np

    ya = np.asarray(y, dtype=object) if not isinstance(y, np.ndarray) else y
    numeric = ya.dtype != object or all(_is_number(v) or isinstance(v, np.ndarray) for v in ya.ravel())
    if numeric:
        if ya.dtype == object and not any(isinstance(v, np.ndarray) for v in ya.ravel()):
            ya = ya.astype(complex if any(isinstance(v, complex) for v in ya.ravel()) else float)
        return Waveform.from_arrays(x, ya, xlabels, ylabel, xunits, yunit)
    return PandasWaveform.from_arrays(x, ya, xlabels, ylabel, xunits, yunit)
