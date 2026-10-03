"""Waveform families from nested channel columns.

Measurement tables often keep one row per operating point (the swept columns plus many scalar
results) and store each *channel*, a curve with its own sweep, in a nested column of that row::

    lbw | vdd | idd | ... | pn: {offset: [..], psd: [..]} | spurs: {freq: [..], level: [..]}

:func:`from_nested` turns one such column into a :class:`Waveform` family: one curve per row,
with the given swept columns as groups. It works lazily on ``pl.scan_parquet(...)``: only the
group columns and that channel are read, other channels and scalar columns are not.
"""

from __future__ import annotations

import polars as pl

from .waveform import Waveform

__all__ = ["from_nested"]


def from_nested(
    frame: pl.DataFrame | pl.LazyFrame,
    column: str,
    x: str,
    y: str | None = None,
    groups: list[str] | tuple[str, ...] = (),
    units: dict[str, str | None] | None = None,
) -> Waveform:
    """A Waveform family from the nested channel ``column`` of ``frame``.

    ``column`` holds, per row, either a struct of lists (``{x: [..], y: [..]}``) or a list of
    structs (``[{x, y}, ..]``). ``x`` names the channel's sweep and ``y`` its values (default:
    the only other field). ``groups`` are the row's swept columns; they become the family's
    group columns, so there is one curve per row::

        lf = pl.scan_parquet("pll.parquet")
        pn = from_nested(lf, "pn", x="offset", y="psd", groups=["lbw", "vdd"],
                         units={"offset": "Hz", "psd": "dBc/Hz"})
        pn.value(1e6)                     # phase noise at 1 MHz offset, one row per lbw x vdd

    Rows whose channel is empty contribute no points.
    """
    lf = frame.lazy()
    schema = lf.collect_schema()
    if column not in schema:
        raise KeyError(f"no column {column!r}; have {schema.names()}")
    missing = [g for g in groups if g not in schema]
    if missing:
        raise KeyError(f"group column(s) {missing} not in the frame")
    dtype = schema[column]
    if isinstance(dtype, pl.Struct):  # {x: [..], y: [..]}
        fields = [f.name for f in dtype.fields]
        list_of_structs = False
    elif isinstance(dtype, pl.List) and isinstance(dtype.inner, pl.Struct):  # [{x, y}, ..]
        fields = [f.name for f in dtype.inner.fields]
        list_of_structs = True
    else:
        raise TypeError(f"column {column!r} is {dtype}, not a struct of lists or a list of structs")
    if x not in fields:
        raise KeyError(f"channel {column!r} has no field {x!r}; fields: {fields}")
    if y is None:
        others = [f for f in fields if f != x]
        if len(others) != 1:
            raise ValueError(f"pass y=: channel {column!r} has several value fields {others}")
        y = others[0]
    elif y not in fields:
        raise KeyError(f"channel {column!r} has no field {y!r}; fields: {fields}")
    if y in groups or x in groups:
        raise ValueError("the channel's fields must not share names with the group columns")

    if list_of_structs:
        lf = (
            lf.select(*groups, pl.col(column))
            .explode(column, empty_as_null=False)
            .select(*groups, pl.col(column).struct.field(x), pl.col(column).struct.field(y))
        )
    else:
        lf = lf.select(
            *groups, pl.col(column).struct.field(x), pl.col(column).struct.field(y)
        ).explode(x, y, empty_as_null=False)
    return Waveform(lf, y, index=[*groups, x], units=units)
