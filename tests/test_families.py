import numpy as np
import polars as pl
import pytest

import polars_waveform as pw


@pytest.fixture
def fam():
    """gain over temp x rval x freq: gain = rval * 10 + temp + freq."""
    temps, rvals, freqs = np.array([-40.0, 27.0, 125.0]), np.array([1.0, 2.0]), np.array([1.0, 2.0, 3.0, 4.0])
    g = rvals[None, :, None] * 10 + temps[:, None, None] + freqs[None, None, :]
    return pw.Waveform.from_arrays([temps, rvals, freqs], g, xlabels=["temp", "rval", "freq"], ylabel="gain",
                                   yunit="dB")


def test_reduce_over_a_group_keeps_the_rest(fam):
    worst = fam.reduce("min", over="temp")  # index [rval, freq]
    assert worst.index == ["rval", "freq"] and worst.groups == ["rval"] and worst.yunit == "dB"
    df = worst.to_polars()
    assert df.columns == ["rval", "freq", "min(gain)"] and df.height == 8
    assert df.filter((pl.col("rval") == 2.0) & (pl.col("freq") == 3.0))["min(gain)"].item() == 20 - 40 + 3
    assert fam.reduce("mean", over=["temp", "rval"]).to_polars()["mean(gain)"].to_list() == pytest.approx(
        [15 + 112 / 3 + f for f in (1, 2, 3, 4)]  # mean(rval) * 10 + mean(temp) + freq
    )
    med = fam.reduce(lambda e: e.quantile(0.5), over="temp")
    assert med.yname == "gain" and med.to_polars().height == 8
    with pytest.raises(ValueError):
        fam.reduce("max", over="freq")  # the sweep: a per-curve measurement instead
    with pytest.raises(ValueError):
        fam.reduce("bogus", over="temp")


def test_axis_argument_of_reductions(fam):
    per_curve = fam.ymax()  # axis=-1: the sweep
    assert per_curve.columns == ["temp", "rval", "ymax"] and per_curve.height == 6
    assert fam.ymax(axis=-1).equals(per_curve)
    over_temp = fam.ymax(axis="temp")  # pycircuit: reduce that axis
    assert isinstance(over_temp, pw.Waveform) and over_temp.index == ["rval", "freq"]
    assert fam.ymax(axis=0).to_polars().equals(over_temp.to_polars())  # by position
    assert fam.average(axis="rval").index == ["temp", "freq"]
    assert fam.getaxis(-1) == "freq" and fam.getaxis(1) == "rval"
    with pytest.raises(KeyError):
        fam.getaxis("nope")


def test_reorder_swap_along_and_dimension_first(fam):
    t = fam.reorder(["rval", "freq", "temp"])  # curves over temperature
    assert t.xname == "temp" and t.groups == ["rval", "freq"]
    assert t.ymax().height == 8
    assert fam.swapaxes("temp", "freq").index == ["freq", "rval", "temp"]
    seen = [(temp, w.index, w.ymax().height) for temp, w in fam.along("temp")]
    assert seen == [(-40.0, ["rval", "freq"], 2), (27.0, ["rval", "freq"], 2), (125.0, ["rval", "freq"], 2)]
    assert [v for v, _ in fam.axesiterator(0)] == [-40.0, 27.0, 125.0]
    first = fam.dimension_first("temp")
    assert first.index == ["rval", "freq"] and first.to_polars()["gain"].min() == 10 - 40 + 1
    assert fam.reducedimension(["temp", "rval"]).index == ["freq"]
    with pytest.raises(ValueError):
        fam.reorder(["temp", "freq"])
    with pytest.raises(ValueError):
        list(fam.along("freq"))


def test_positional_indexing():
    w = pw.Waveform(pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [5.0, 6.0, 7.0, 8.0]}), "v")
    assert w[0] == 5.0 and w[-1] == 8.0 and w[-2] == 7.0
    with pytest.raises(IndexError):
        w[10]
    assert w[1:3].y.to_list() == [6.0, 7.0] and w[::2].y.to_list() == [5.0, 7.0] and w[-2:].x.to_list() == [2.0, 3.0]
    assert list(w) == [5.0, 6.0, 7.0, 8.0] and np.asarray(w).tolist() == [5.0, 6.0, 7.0, 8.0]
    with pytest.raises(TypeError):
        w["a"]

    df = pl.DataFrame({"run": [0, 0, 0, 1, 1], "t": [0.0, 1.0, 2.0, 0.0, 1.0], "v": [1.0, 2.0, 3.0, 4.0, 5.0]})
    g = pw.Waveform(df, "v", index=["run", "t"])
    assert g[-1].rows() == [(0, 3.0), (1, 5.0)]  # last sample of every curve
    assert g[:2].to_polars()["v"].to_list() == [1.0, 2.0, 4.0, 5.0]
    assert g.filter(pl.col("run") == 1)[-1] == 5.0  # one curve left: a number
    with pytest.raises(TypeError):
        list(g)

    c = pw.Waveform.from_arrays(np.array([1.0, 2.0]), np.array([1 + 2j, 3 - 1j]))
    assert c[-1] == 3 - 1j
