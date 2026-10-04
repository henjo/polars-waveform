import numpy as np
import pytest

import polars_waveform as pw
from polars_waveform import WaveformBase


class ArrayWaveform(WaveformBase):
    """A minimal other kind: one curve in numpy arrays."""

    def __init__(self, x, y):
        self.x, self.y = np.asarray(x), np.asarray(y)

    @classmethod
    def from_arrays(cls, x, y, xlabels=None, ylabel=None, xunits=None, yunit=None):
        return cls(x, y)

    def to_arrays(self):
        return [self.x], self.y

    def numeric(self):
        return pw.Waveform.from_arrays(self.x, self.y, xlabels=["t"], ylabel="v")

    index = property(lambda self: ["t"])
    xname = property(lambda self: "t")
    yname = property(lambda self: "v")
    xunit = property(lambda self: None)
    yunit = property(lambda self: None)

    def _binop(self, other, op, *, reverse=False):
        return self.numeric()._binop(other, op, reverse=reverse)

    def __neg__(self):
        return ArrayWaveform(self.x, -self.y)

    def __abs__(self):
        return ArrayWaveform(self.x, np.abs(self.y))

    def __pow__(self, other):
        return ArrayWaveform(self.x, self.y**other)

    def real(self):
        return ArrayWaveform(self.x, self.y.real)

    def imag(self):
        return ArrayWaveform(self.x, self.y.imag)

    def conj(self):
        return ArrayWaveform(self.x, self.y.conj())

    def phase(self, deg=True):
        return ArrayWaveform(self.x, np.angle(self.y, deg=deg))

    def db10(self):
        return ArrayWaveform(self.x, 10 * np.log10(np.abs(self.y)))

    def db20(self):
        return ArrayWaveform(self.x, 20 * np.log10(np.abs(self.y)))


@pytest.fixture
def a():
    return ArrayWaveform([0.0, 1.0, 2.0, 3.0], [-1.0, 1.0, 3.0, 2.0])


def test_measurements_run_on_numeric(a):
    assert a.ymax() == 3.0 and a.xmax() == 2.0 and a.cross(0.0) == pytest.approx(0.5)
    assert pw.functions.ymax(a) == 3.0
    assert pw.functions.db20(a).y.tolist() == pytest.approx([0, 0, 9.5424, 6.0206], abs=1e-4)


def test_arithmetic_with_other_kinds(a):
    w = pw.Waveform.from_arrays(np.array([0.0, 1.0, 2.0, 3.0]), np.array([1.0, 1.0, 1.0, 1.0]), xlabels=["t"],
                                ylabel="v")
    assert isinstance(w - a, pw.Waveform) and (w - a).y.to_list() == [2.0, 0.0, -2.0, -1.0]
    assert (a + w).y.to_list() == [0.0, 2.0, 4.0, 3.0]
