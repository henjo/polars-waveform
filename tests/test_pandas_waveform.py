import math

import numpy as np
import pytest

import polars_waveform as pw

sympy = pytest.importorskip("sympy")
pytest.importorskip("pandas")

R, C = sympy.symbols("R C", positive=True)
FREQS = np.logspace(3, 7, 81)


def rc():
    """H(f) = 1 / (1 + j 2 pi f R C), symbolic in R and C."""
    return pw.from_arrays(FREQS, [1 / (1 + 2 * sympy.pi * sympy.I * f * R * C) for f in FREQS], xlabels=["freq"],
                          ylabel="H", xunits=["Hz"], yunit="V")  # fmt: skip


def test_from_arrays_picks_the_kind():
    assert isinstance(rc(), pw.PandasWaveform)
    assert isinstance(pw.from_arrays(FREQS, np.ones(len(FREQS))), pw.Waveform)
    assert isinstance(pw.from_arrays([1.0, 2.0], [1, 2.5 + 1j]), pw.Waveform)  # Python numbers
    s = pw.from_arrays(np.array([1j, 2j]), [1.0, 2.0], xlabels=["s"])  # complex sweep: Polars cannot index it
    assert isinstance(s, pw.PandasWaveform) and s.value(2j) == 2.0
    w = rc()
    assert w.index == ["freq"] and w.xname == "freq" and w.yname == "H" and w.xunit == "Hz" and w.yunit == "V"
    assert len(w) == 81 and "PandasWaveform(freq -> H" in repr(w)


def test_symbolic_math_then_numbers():
    w = rc()
    g = 2 * w - 1
    assert g.yname == "2 * H - 1" and isinstance(g.to_pandas().iloc[0], sympy.Expr)
    db = w.db20()
    assert db.yunit == "dB" and db.to_pandas().iloc[0].has(R)
    with pytest.raises(ValueError, match="R"):
        w.numeric()
    n = w.subs({R: 1e3, C: 1e-9})
    assert n.bandwidth() == pytest.approx(1 / (2 * math.pi * 1e-6), rel=1e-2)  # measured on numeric()
    num = n.numeric()
    assert isinstance(num, pw.Waveform) and num.is_complex and num.xunit == "Hz"
    assert n.db20().numeric().y[0] == pytest.approx(20 * math.log10(abs(complex(n.to_pandas().iloc[0]))))
    assert n.phase().numeric().y.to_list()[-1] == pytest.approx(-math.degrees(math.atan(2 * math.pi * 1e7 * 1e-6)))
    assert n.map(sympy.simplify).yname == "simplify(H)"
    assert (w - w).map(sympy.simplify).subs({R: 1, C: 1}).numeric().y.to_list() == [0.0] * 81
    with pytest.raises(ValueError):
        w + rc().clip(1e4)  # different sweeps


def test_family_selection_and_numeric_operands():
    t = sympy.Symbol("t")
    temps, xs = np.array([27.0, 85.0]), np.array([0, 1, 2])
    y = np.array([[t * k * x for x in xs] for k in (1, 2)], dtype=object)
    fam = pw.from_arrays([temps, xs], y, xlabels=["temp", "x"], ylabel="v")
    assert fam.groups == ["temp"] and fam.index == ["temp", "x"]
    assert fam.leaf(temp=85.0).to_pandas().tolist() == [0, 2 * t, 4 * t]
    at1 = fam.value(1)  # an exact sweep point: still symbolic, one value per curve
    assert at1.to_pandas().tolist() == [t, 2 * t]
    assert fam[-1].to_pandas().tolist() == [2 * t, 4 * t]
    d = fam.deriv()
    assert d.index == ["temp", "x"] and len(d) == 4  # one sample shorter per curve
    assert all(sympy.simplify(a - b) == 0 for a, b in zip(d.to_pandas(), [t, t, 2 * t, 2 * t], strict=True))
    num = fam.subs(t, 1.5).numeric()
    assert num.ymax().to_dicts() == [{"temp": 27.0, "ymax": 3.0}, {"temp": 85.0, "ymax": 6.0}]
    other = pw.from_arrays([temps, xs], np.ones((2, 3)), xlabels=["temp", "x"], ylabel="one")
    assert (fam.subs(t, 1) + other).to_polars()["v + one"].to_list() == [1, 2, 3, 1, 3, 5]
    xs_, yy = fam.to_arrays()
    assert [a.tolist() for a in xs_] == [[27.0, 85.0], [0, 1, 2]] and yy.shape == (2, 3)
    with pytest.raises(KeyError):
        fam.leaf(x=1)
