"""Waveforms on Polars: curves and curve families with circuit-style measurements.

A :class:`Waveform` is a value column over one or more index columns, backed by a lazy
``pl.LazyFrame`` and materialized only when a value is needed. The last index column is the
sweep (time, frequency, offset, ...); any others are groups (corners, Monte Carlo runs, measured
operating points), so one Waveform can hold a whole family of curves::

    import polars as pl
    import polars_waveform as pw

    w = pw.Waveform(pl.scan_parquet("lab.parquet"), "gain", index=["dut", "temp", "freq"])
    w.bandwidth()                  # one row per dut x temp
    w - w.mean()                   # per-curve results broadcast back over the curves
    w.leaf(dut="A1", temp=25.0)    # one curve

Measurements follow OCEAN semantics (``xmax`` is the x at the largest y; ``cross`` counts edges
from 1); the OCEAN and pycircuit spellings (``dB20``, ``unityGainFreq``, ``IIP3``, ...) are aliases.
Complex data is ``Struct{re, im}``; :mod:`.cx` registers ``pl.col(...).cx`` for it.
:func:`from_nested` builds families from nested channel columns, and :class:`ResultSource` is the
shape shared by result objects (``polars_psf.Result`` implements it).

Modules: :mod:`.waveform` (the class), :mod:`.functions` (calculator functions), :mod:`.cx`
(complex helpers), :mod:`.nested`, :mod:`.source`.
"""

from importlib.metadata import PackageNotFoundError
from importlib.metadata import version as _version

from . import cx
from .functions import *  # noqa: F403
from .functions import __all__ as _functions
from .nested import from_nested
from .source import ResultSource
from .waveform import Waveform, either, falling, raising

__all__ = ["ResultSource", "Waveform", "cx", "either", "falling", "from_nested", "raising", *_functions]

try:
    __version__ = _version("polars-waveform")
except PackageNotFoundError:  # running from a source tree without an install
    __version__ = "0+unknown"
