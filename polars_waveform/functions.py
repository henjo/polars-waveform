"""Calculator functions: ``db20(w)``, ``bandwidth(w, 3, "low")``, ``rise_time(w)``.

Each forwards to the :class:`Waveform` method of the same name, so it works per curve of a
family (corners, Monte Carlo) and returns one row per curve there. The semantics follow OCEAN:
``xmax`` is the x at the largest y, and ``cross`` counts edges from 1 with
``"rising"``/``"falling"``/``"either"``. The OCEAN names (``dB20``, ``unityGainFreq``,
``riseTime``, ``leafValue``, ...) and pycircuit names (``unityGainFrequency``, ``IIP3``, ...) are
aliases.

Arguments follow OCEAN's order where it has positional ones; OCEAN's ``initType``/``finalType``
flags (initial value given as an x instead of a y) are not supported: pass y values.
"""

from __future__ import annotations

import math

from .base import WaveformBase
from .plot import compression_plot, set_plot_backend
from .waveform import Waveform

__all__ = [
    # calculator functions
    "db10", "db20", "mag", "phase", "phase_deg", "phase_rad", "real", "imag", "conjugate",
    "value", "ymax", "ymin", "xmax", "xmin", "average", "rms", "stddev",
    "cross", "deriv", "integ", "iinteg", "clip", "dft", "leaf_value", "log10", "exp", "sqrt",
    "bandwidth", "unity_gain_frequency", "phase_margin", "gain_margin",
    "rise_time", "fall_time", "slew_rate", "overshoot", "settling_time", "delay", "frequency", "period",
    "im2", "im3", "iip2", "iip3", "calc_extrapolation_line", "compression_point", "compression_plot",
    "set_plot_backend",
    # OCEAN aliases
    "dB10", "dB20", "phaseDeg", "phaseRad", "leafValue", "unityGainFreq", "phaseMargin", "gainMargin",
    "riseTime", "fallTime", "slewRate", "settlingTime",
    # pycircuit aliases
    "unityGainFrequency", "IM2", "IM3", "IIP2", "IIP3",
]


# --- elementwise: also on plain numbers and numpy arrays --------------------------------------
def _num(w):
    """True if ``w`` is a plain number or array (not a waveform)."""
    return not isinstance(w, WaveformBase)


def db20(w):
    """``20*log10(|w|)``; works on waveforms, numbers and numpy arrays."""
    if _num(w):
        import numpy as np

        return 20.0 * np.log10(np.abs(w))
    return w.db20()


def db10(w):
    """``10*log10(|w|)``, for power quantities."""
    if _num(w):
        import numpy as np

        return 10.0 * np.log10(np.abs(w))
    return w.db10()


def mag(w):
    if _num(w):
        import numpy as np

        return np.abs(w)
    return w.mag()


def phase(w, deg: bool = True):
    """Phase in degrees (radians with ``deg=False``)."""
    if _num(w):
        import numpy as np

        return np.angle(w, deg=deg)
    return w.phase(deg)


def phase_deg(w):
    return phase(w, deg=True)


def phase_rad(w):
    return phase(w, deg=False)


def real(w):
    if _num(w):
        import numpy as np

        return np.real(w)
    return w.real()


def imag(w):
    if _num(w):
        import numpy as np

        return np.imag(w)
    return w.imag()


def conjugate(w):
    if _num(w):
        import numpy as np

        return np.conjugate(w)
    return w.conj()


def log10(w):
    if _num(w):
        import numpy as np

        return np.log10(w)
    return w.log10()


def exp(w):
    if _num(w):
        import numpy as np

        return np.exp(w)
    return w.exp()


def sqrt(w):
    if _num(w):
        import numpy as np

        return np.sqrt(w)
    return w.sqrt()


def deriv(w: Waveform) -> Waveform:
    return w.deriv()


def iinteg(w: Waveform) -> Waveform:
    return w.iinteg()


def clip(w: Waveform, xfrom: float, xto: float | None = None) -> Waveform:
    return w.clip(xfrom, xto)


def dft(w: Waveform) -> Waveform:
    return w.dft()


def leaf_value(w: Waveform, **values) -> Waveform:
    """One leaf of a family: ``leaf_value(w, temp=27, rval=1000)``."""
    return w.leaf(**values)


# --- measurements (scalar per curve) ----------------------------------------------------------
def value(w: Waveform, x):
    return w.value(x)


def ymax(w: Waveform, axis=-1):
    return w.ymax(axis)


def ymin(w: Waveform, axis=-1):
    return w.ymin(axis)


def xmax(w: Waveform):
    """x where y is largest."""
    return w.xmax()


def xmin(w: Waveform):
    """x where y is smallest."""
    return w.xmin()


def average(w: Waveform, axis=-1):
    return w.average(axis)


def rms(w: Waveform, axis=-1):
    return w.rms(axis)


def stddev(w: Waveform, axis=-1):
    return w.stddev(axis)


def integ(w: Waveform, xfrom: float | None = None, xto: float | None = None):
    return w.integ(xfrom, xto)


def cross(w: Waveform, threshold: float = 0.0, edge: int = 1, type: str = "either"):
    """x of the ``edge``-th crossing of ``threshold`` (from 1; negative counts from the end)."""
    return w.cross(threshold, edge, type)


def bandwidth(w: Waveform, db: float = 3.0, type: str = "low"):
    return w.bandwidth(db, type)


def unity_gain_frequency(w: Waveform):
    return w.unity_gain_frequency()


def phase_margin(w: Waveform):
    return w.phase_margin()


def gain_margin(w: Waveform):
    return w.gain_margin()


def rise_time(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    """Rise time (OCEAN ``riseTime(wave initVal nil finalVal nil theta1 theta2)``)."""
    return w.rise_time(theta1, theta2, initial, final)


def fall_time(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    return w.fall_time(theta1, theta2, initial, final)


def slew_rate(w: Waveform, initial=None, final=None, theta1: float = 10.0, theta2: float = 90.0):
    return w.slew_rate(theta1, theta2, initial, final)


def overshoot(w: Waveform, initial=None, final=None):
    return w.overshoot(initial, final)


def settling_time(w: Waveform, initial=None, final=None, tolerance: float = 1.0):
    return w.settling_time(tolerance, initial, final)


def delay(
    wf1: Waveform,
    wf2: Waveform,
    value1: float = 0.0,
    value2: float | None = None,
    edge1: str = "either",
    nth1: int = 1,
    edge2: str | None = None,
    nth2: int = 1,
):
    """x of ``wf2``'s ``nth2`` crossing of ``value2`` minus that of ``wf1`` (OCEAN ``delay``
    keyword names)."""
    return wf1.delay(wf2, value1, value2, nth1, nth2, edge1, edge2)


def frequency(w: Waveform, threshold: float | None = None):
    return w.frequency(threshold)


def period(w: Waveform, threshold: float | None = None):
    return w.period(threshold)


# --- RF (pycircuit) ---------------------------------------------------------------------------
def im3(w: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Third-order intermodulation tone at ``fund1 + 2*fund2`` of an output spectrum.

    ``fund0`` (the LO of a mixer) is accepted for pycircuit compatibility; output tones are
    measured at IF, so it does not shift them (only :func:`iip3` uses it, for the input)."""
    return abs(w).value(fund1 + 2 * fund2)


def im2(w: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Second-order intermodulation tone at ``fund1 + fund2`` of an output spectrum (``fund0``:
    see :func:`im3`)."""
    return abs(w).value(fund1 + fund2)


def iip3(output: Waveform, input: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Input-referred third-order intercept point."""
    s = abs(output)
    if fund0 is None:
        gain = (s / abs(input)).value(fund1)
    else:
        gain = s.value(abs(fund1)) / abs(input).value(abs(abs(fund1) + fund0))
    return math.sqrt(s.value(abs(fund1)) * s.value(abs(fund2)) ** 2 / s.value(fund1 + 2 * fund2)) / gain


def iip2(output: Waveform, input: Waveform, fund1: float, fund2: float, fund0=None) -> float:
    """Input-referred second-order intercept point."""
    s = abs(output)
    if fund0 is None:
        gain = (s / abs(input)).value(fund1)
    else:
        gain = s.value(abs(fund1)) / abs(input).value(abs(abs(fund1) + fund0))
    return s.value(abs(fund1)) * s.value(abs(fund2)) / s.value(fund1 + fund2) / gain


def calc_extrapolation_line(
    w_db: Waveform, slope: float = 1.0, extrapolation_point: float | None = None, **_
) -> Waveform:
    """Linear (in dB) extrapolation of ``w_db`` through its low end (smallest x)."""
    if extrapolation_point is None:
        extrapolation_point = w_db.x.min()
    m = w_db.value(extrapolation_point) - slope * extrapolation_point
    return m + slope * w_db.xval()


def compression_point(
    w_db: Waveform, slope: float = 1.0, compression: float = 1.0, extrapolation_point=None, **_
) -> float:
    """Input-referred compression point: where the response falls ``compression`` dB below the
    extrapolated small-signal line."""
    line = calc_extrapolation_line(w_db, slope, extrapolation_point)
    return (line - w_db).cross(compression)


# --- aliases ----------------------------------------------------------------------------------
# OCEAN
dB20, dB10 = db20, db10
phaseDeg, phaseRad = phase_deg, phase_rad
leafValue = leaf_value
unityGainFreq = unity_gain_frequency
phaseMargin, gainMargin = phase_margin, gain_margin
riseTime, fallTime, slewRate, settlingTime = rise_time, fall_time, slew_rate, settling_time
# pycircuit
unityGainFrequency = unity_gain_frequency
IM2, IM3, IIP2, IIP3 = im2, im3, iip2, iip3
