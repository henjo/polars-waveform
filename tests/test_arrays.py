import numpy as np
import polars as pl
import pytest

import polars_waveform as pw


def test_from_arrays_one_axis_and_units():
    w = pw.Waveform.from_arrays(np.array([0.0, 1.0, 2.0]), np.array([0.0, 2.0, 1.0]), xlabels=["t"], ylabel="v",
                                xunits=["s"], yunit="V")
    assert w.index == ["t"] and w.groups == [] and (w.xunit, w.yunit) == ("s", "V")
    assert w.ymax() == 2.0 and w.xmax() == 1.0
    xs, y = w.to_arrays()
    assert len(xs) == 1 and xs[0].tolist() == [0.0, 1.0, 2.0] and y.tolist() == [0.0, 2.0, 1.0]


def test_from_arrays_grid_is_a_family():
    temps, freqs = np.array([-40.0, 85.0]), np.array([1.0, 2.0, 3.0])
    gain = np.array([[1.0, 0.8, 0.5], [0.9, 0.6, 0.3]])
    w = pw.Waveform.from_arrays([temps, freqs], gain, xlabels=["temp", "f"], ylabel="g")
    assert w.groups == ["temp"] and w.xname == "f"
    assert w.ymin().rows() == [(-40.0, 0.5), (85.0, 0.3)]
    xs, y = w.to_arrays()  # round trip: regular grid again
    assert [x.tolist() for x in xs] == [temps.tolist(), freqs.tolist()]
    assert np.array_equal(y, gain)
    with pytest.raises(ValueError):
        pw.Waveform.from_arrays([temps, freqs], gain.T)


def test_from_arrays_complex_default_labels():
    f = np.array([1.0, 10.0, 100.0])
    h = 1 / (1 + 1j * f / 10)
    w = pw.Waveform.from_arrays(f, h)
    assert w.index == ["x0"] and w.yname == "y" and w.is_complex
    assert w.to_numpy() == pytest.approx(h)
    _, y = w.to_arrays()
    assert y.dtype == np.complex128 and y == pytest.approx(h)


def test_from_arrays_ragged_round_trip():
    def obj(*arrays):
        out = np.empty(len(arrays), dtype=object)
        out[:] = [np.asarray(a) for a in arrays]
        return out

    run = obj([1, 1, 1], [2, 2])
    t = obj([0.0, 1.0, 2.0], [0.0, 1.5])
    y = obj([0.0, 1.0, 0.0], [0.0, 2.0])
    w = pw.Waveform.from_arrays([run, t], y, xlabels=["run", "t"], ylabel="v")
    assert w.groups == ["run"] and len(w) == 5
    assert w.ymax().rows() == [(1, 1.0), (2, 2.0)]
    xs, yy = w.to_arrays()  # sweeps differ per curve: ragged again
    assert yy.dtype == object and [a.tolist() for a in yy] == [[0.0, 1.0, 0.0], [0.0, 2.0]]
    assert [a.tolist() for a in xs[1]] == [[0.0, 1.0, 2.0], [0.0, 1.5]]


def test_numpy_ufuncs_stay_waveforms():
    w = pw.Waveform(pl.DataFrame({"t": [1.0, 2.0, 3.0], "v": [-1.0, 10.0, 100.0]}), "v")
    assert isinstance(np.abs(w), pw.Waveform) and np.abs(w).y.to_list() == [1.0, 10.0, 100.0]
    assert np.log10(np.abs(w)).y.to_list() == pytest.approx([0.0, 1.0, 2.0])
    arr = np.array([1.0, 2.0, 3.0])
    assert (arr + w).y.to_list() == [0.0, 12.0, 103.0]  # ndarray on the left
    assert (w * arr).y.to_list() == [-1.0, 20.0, 300.0]
    assert (np.float64(2.0) * w).y.to_list() == [-2.0, 20.0, 200.0]
    assert np.power(np.abs(w), 2).y.to_list() == [1.0, 100.0, 10000.0]
    assert (np.abs(w) > arr).y.to_list() == [False, True, True]
    with pytest.raises(TypeError):
        np.sin(w)  # not supported: numpy raises instead of guessing


class Sampled(pw.WaveformBase):
    """A minimal non-Polars kind: plain numpy arrays, one curve (like a symbolic container)."""

    def __init__(self, x, y):
        self._x, self._y = np.asarray(x, dtype=float), np.asarray(y)

    @classmethod
    def from_arrays(cls, x, y, xlabels=None, ylabel=None, xunits=None, yunit=None):
        return cls(x, y)

    def to_arrays(self):
        return [self._x], self._y

    def numeric(self):
        return pw.Waveform.from_arrays(self._x, self._y.astype(float), xlabels=["t"], ylabel="v")

    index = property(lambda self: ["t"])
    xname = property(lambda self: "t")
    yname = property(lambda self: "v")
    xunit = property(lambda self: None)
    yunit = property(lambda self: None)

    def _binop(self, other, op, *, reverse=False):
        a, b = (other, self._y) if reverse else (self._y, other)
        return Sampled(self._x, {"+": np.add, "-": np.subtract, "*": np.multiply, "/": np.divide}[op](a, b))

    def __neg__(self):
        return Sampled(self._x, -self._y)

    def __abs__(self):
        return Sampled(self._x, np.abs(self._y))

    def __pow__(self, n):
        return Sampled(self._x, self._y**n)

    def real(self):
        return Sampled(self._x, np.real(self._y))

    def imag(self):
        return Sampled(self._x, np.imag(self._y))

    def conj(self):
        return Sampled(self._x, np.conj(self._y))

    def phase(self, deg=True):
        return Sampled(self._x, np.angle(self._y, deg=deg))

    def db10(self):
        return Sampled(self._x, 10 * np.log10(np.abs(self._y)))

    def db20(self):
        return Sampled(self._x, 20 * np.log10(np.abs(self._y)))


def test_other_kinds_share_the_interface():
    s = Sampled([0.0, 1.0, 2.0, 3.0], [0.0, 1.0, 1.0, 0.0])
    assert isinstance(s, pw.WaveformBase) and isinstance(pw.Waveform.from_arrays([1.0], [1.0]), pw.WaveformBase)
    assert isinstance((2 * s + 1), Sampled)  # arithmetic stays in the kind
    assert s.ymax() == 1.0 and s.cross(0.5) == 0.5  # measurements run on numeric()
    assert s.rise_time(final=1.0) == pytest.approx(0.8) and s.groups == []
    assert isinstance(pw.db20(s + 1), Sampled)  # free functions work on any kind
    with pytest.raises(TypeError):
        pw.WaveformBase()  # abstract
