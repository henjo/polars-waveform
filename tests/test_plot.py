import numpy as np
import polars as pl
import pytest

import polars_waveform as pw

mpl = pytest.importorskip("matplotlib")
mpl.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


@pytest.fixture(autouse=True)
def _close():
    yield
    plt.close("all")


def family():
    t = np.linspace(0, 1, 11)
    return pw.Waveform.from_arrays([np.array([1.0, 2.0]), t], np.outer([1.0, 2.0], t), xlabels=["k", "t"],
                                   ylabel="v", xunits=[None, "s"], yunit="V")


def test_matplotlib_family_one_line_per_curve_with_units():
    ax = family().plot()
    assert len(ax.lines) == 2 and [ln.get_label() for ln in ax.lines] == ["k=1", "k=2"]
    assert ax.get_xlabel() == "t [s]" and ax.get_ylabel() == "v [V]"
    assert ax.lines[1].get_ydata()[-1] == pytest.approx(2.0)
    _, ax2 = plt.subplots()
    assert family().leaf(k=1.0).semilogy(ax=ax2) is ax2 and ax2.get_yscale() == "log"
    assert family().loglog().get_xscale() == "log"
    assert family().leaf(k=1.0).stem() is not None


def test_bode_complex_and_errors():
    f = np.logspace(3, 7, 50)
    h = pw.Waveform.from_arrays(f, 1 / (1 + 1j * f / 1e5), xlabels=["freq"], ylabel="h", xunits=["Hz"])
    _, (mag, ph) = h.bode()
    assert mag.lines[0].get_ydata()[0] == pytest.approx(0.0, abs=1e-3)
    assert ph.lines[0].get_ydata()[-1] == pytest.approx(np.degrees(np.angle(1 / (1 + 1j * 1e7 / 1e5))), abs=1e-6)
    assert ph.get_xlabel() == "freq [Hz]"
    with pytest.raises(ValueError):
        h.plot()  # complex: use bode() or db20()
    assert len(h.db20().semilogx().lines) == 1
    with pytest.raises(ValueError):
        family().plot(backend="bogus")
    with pytest.raises(ValueError):
        pw.set_plot_backend("bogus")


def test_compression_plot():
    pin = pl.Series([5.0, 10.0, 15.0, 20.0])
    w_db = pw.Waveform(pl.DataFrame({"pin": pin, "gain": [20.0, 20.0, 15.0, 10.0]}), "gain")
    ax = pw.compression_plot(w_db)
    labels = [ln.get_label() for ln in ax.lines]
    assert labels[0] == "gain" and any("compression at 6" in lb for lb in labels)


def test_altair_backend():
    pytest.importorskip("altair")
    spec = family().plot(backend="altair").to_dict()
    assert spec["encoding"]["color"]["field"] == "k" and spec["encoding"]["x"]["title"] == "t [s]"


def test_elementwise_functions_on_plain_numbers_and_arrays():
    assert pw.db20(10.0) == pytest.approx(20.0) and pw.db10(10.0) == pytest.approx(10.0)
    assert pw.dB20(np.array([1.0, 0.1])).tolist() == pytest.approx([0.0, -20.0])
    assert pw.phase(1j) == pytest.approx(90.0) and pw.phase(1j, deg=False) == pytest.approx(np.pi / 2)
    assert pw.mag(3 + 4j) == 5.0 and pw.real(3 + 4j) == 3.0 and pw.imag(3 + 4j) == 4.0
    assert pw.conjugate(1 + 1j) == 1 - 1j and pw.log10(100.0) == 2.0 and pw.sqrt(4.0) == 2.0
    w = pw.Waveform(pl.DataFrame({"t": [1.0, 2.0], "v": [10.0, 100.0]}), "v")
    assert isinstance(pw.db20(w), pw.Waveform) and pw.db20(w).y.to_list() == pytest.approx([20.0, 40.0])
