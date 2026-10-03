"""Plotting: matplotlib by default, Altair (interactive, through Polars' ``DataFrame.plot``) on request.

Both backends are optional dependencies (``pip install "polars-waveform[matplotlib]"`` or
``[altair]``) and are imported only when you plot. A family is drawn as one line per curve,
labelled with its group values; axis labels carry the units::

    w.plot()                               # matplotlib Axes
    w.semilogx(), w.loglog(), w.stem()
    h.bode()                               # magnitude and phase of a complex response
    w.plot(backend="altair", color="temp") # Altair chart; or pw.set_plot_backend("altair")

Complex data is plotted through :meth:`~polars_waveform.Waveform.bode`, or after reducing it with
``db20()``, ``phase()``, ``real()`` or ``imag()``.
"""

from __future__ import annotations

import math

__all__ = ["bode", "compression_plot", "plot", "set_plot_backend"]

_BACKENDS = ("matplotlib", "altair")
_backend = "matplotlib"


def set_plot_backend(name: str) -> None:
    """Default backend of :func:`plot` and ``Waveform.plot()``: ``"matplotlib"`` or ``"altair"``."""
    global _backend
    if name not in _BACKENDS:
        raise ValueError(f"plot backend must be one of {_BACKENDS}, not {name!r}")
    _backend = name


def _label(name: str, unit) -> str:
    return f"{name} [{unit}]" if unit else name


def _fmt(v) -> str:
    return f"{v:g}" if isinstance(v, (int, float)) and not isinstance(v, bool) else str(v)


def _curve_label(key: dict, w) -> str:
    return ", ".join(f"{k}={_fmt(v)}" for k, v in key.items()) or w.yname


def _no_complex(w) -> None:
    if w.is_complex:
        raise ValueError("cannot plot complex values directly; use bode(), or db20(), phase(), real() or imag()")


def _axes(ax):
    if ax is not None:
        return ax
    import matplotlib.pyplot as plt

    return plt.subplots()[1]


def _matplotlib(w, ax=None, *, logx=False, logy=False, stem=False, **kwargs):
    w = w.numeric()
    _no_complex(w)
    ax = _axes(ax)
    fixed = kwargs.pop("label", None)  # a label for a single curve
    for key, curve in w._curves():
        x, y = curve.x.to_numpy(), curve.y.to_numpy()
        label = fixed if fixed is not None and not key else _curve_label(key, w)
        if stem:
            ax.stem(x, y, label=label, **kwargs)
        else:
            ax.plot(x, y, label=label, **kwargs)
    if logx:
        ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(_label(w.xname, w.xunit))
    ax.set_ylabel(_label(w.yname, w.yunit))
    ax.grid(True, which="both", alpha=0.3)
    if w.groups:
        ax.legend()
    return ax


def _altair(w, mark: str = "line", **encode):
    w = w.numeric()
    _no_complex(w)
    # explicit field specs: names with ":" (e.g. "I1:d") are not misread as Altair shorthand
    encode.setdefault("x", {"field": w.xname, "type": "quantitative", "title": _label(w.xname, w.xunit)})
    encode.setdefault("y", {"field": w.yname, "type": "quantitative", "title": _label(w.yname, w.yunit)})
    if w.groups:
        encode.setdefault("color", {"field": w.groups[0], "type": "nominal"})
        if len(w.groups) > 1:
            encode.setdefault("detail", [{"field": g, "type": "nominal"} for g in w.groups[1:]])
    return getattr(w.to_polars().plot, mark)(**encode)


def plot(w, backend: str | None = None, ax=None, **kwargs):
    """Plot a waveform or family; returns a matplotlib ``Axes`` or an Altair chart.

    matplotlib: ``ax`` draws into existing axes, other keywords go to ``Axes.plot``. Altair:
    ``mark="line"`` (or ``"point"``, ``"area"``, ...) and Altair encode channels as keywords.
    """
    backend = backend or _backend
    if backend == "altair":
        if ax is not None:
            raise ValueError("ax= is for the matplotlib backend")
        return _altair(w, **kwargs)
    if backend != "matplotlib":
        raise ValueError(f"plot backend must be one of {_BACKENDS}, not {backend!r}")
    return _matplotlib(w, ax, **kwargs)


def bode(w, axes=None, deg: bool = True):
    """Magnitude (dB) and phase of a complex response over a log frequency axis, one line per
    curve; ``axes`` is an optional pair of matplotlib axes. Returns ``(fig, (ax_mag, ax_phase))``."""
    import matplotlib.pyplot as plt
    import numpy as np

    w = w.numeric()
    if axes is None:
        fig, axes = plt.subplots(2, 1, sharex=True)
    else:
        fig = axes[0].figure
    ax_mag, ax_ph = axes
    for key, curve in w._curves():
        x = curve.x.to_numpy()
        z = curve.to_numpy() if curve.is_complex else curve.y.to_numpy().astype(complex)
        label = _curve_label(key, w)
        ax_mag.semilogx(x, 20 * np.log10(np.abs(z)), label=label)
        ph = np.unwrap(np.angle(z))
        ax_ph.semilogx(x, np.degrees(ph) if deg else ph, label=label)
    ax_mag.set_ylabel(f"|{w.yname}| [dB]")
    ax_ph.set_ylabel(f"phase({w.yname}) [{'deg' if deg else 'rad'}]")
    ax_ph.set_xlabel(_label(w.xname, w.xunit))
    for ax in axes:
        ax.grid(True, which="both", alpha=0.3)
    if w.groups:
        ax_mag.legend()
    return fig, (ax_mag, ax_ph)


def compression_plot(w_db, slope: float = 1.0, compression: float = 1.0, extrapolation_point=None, ax=None):
    """A gain-compression plot: the response in dB, its extrapolated small-signal line and the
    compression point (pycircuit ``compression_plot``). Returns the matplotlib ``Axes``."""
    from .functions import calc_extrapolation_line, compression_point

    w_db = w_db.numeric()
    if w_db.groups:
        raise ValueError("compression_plot() draws one curve; pick it with leaf()")
    ax = _matplotlib(w_db, ax, label=w_db.yname)
    line = calc_extrapolation_line(w_db, slope, extrapolation_point)
    ax.plot(line.x.to_numpy(), line.y.to_numpy(), "--", label=f"slope {slope:g} line")
    x_c = compression_point(w_db, slope, compression, extrapolation_point)
    if not math.isnan(x_c):
        ax.axvline(x_c, color="gray", linestyle=":", label=f"{compression:g} dB compression at {x_c:.4g}")
    ax.legend()
    return ax
