"""The shape shared by result objects that hand out waveforms (simulation results, lab data)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import polars as pl

from .waveform import Waveform

__all__ = ["ResultSource"]


@runtime_checkable
class ResultSource(Protocol):
    """One result with one curve per leaf (corner, Monte Carlo run, measured operating point).

    ``polars_psf.Result`` implements it for Spectre results; a lab-data handler can implement it
    for its own storage, so code written against it works on both::

        def worst_bandwidth(res: ResultSource, signal: str) -> float:
            return res.v(signal).bandwidth()["bandwidth"].min()
    """

    @property
    def leaves(self) -> pl.DataFrame:
        """One row per leaf: its parameter values (the family's group columns)."""
        ...

    @property
    def names(self) -> list[str]:
        """Names of the signals ``v()`` accepts."""
        ...

    def v(self, signal: str, **params) -> Waveform:
        """``signal`` as a Waveform family; ``params`` narrow the leaves."""
        ...

    def scan(self, **kwargs) -> pl.LazyFrame:
        """The result as a lazy table."""
        ...
