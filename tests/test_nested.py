import numpy as np
import polars as pl
import pytest

import polars_waveform as pw


def pll_rows():
    """Two swept columns, a scalar result and two nested channels with their own sweeps."""
    rows = []
    for lbw in [1e5, 2e5]:
        for vdd in [1.0, 1.2]:
            off = np.logspace(3, 7, 41)
            psd = -100 - 10 * np.log10(1 + (off / lbw) ** 2) - (vdd - 1.0)
            rows.append(
                dict(
                    lbw=lbw,
                    vdd=vdd,
                    idd=0.01 * vdd,
                    pn=dict(offset=off.tolist(), psd=psd.tolist()),
                    spurs=dict(freq=[1e6, 2e6], level=[-70.0, -80.0]),
                )
            )
    return pl.DataFrame(rows)


def test_struct_of_lists_family(tmp_path):
    path = tmp_path / "pll.parquet"
    pll_rows().write_parquet(path)
    lf = pl.scan_parquet(path)
    pn = pw.from_nested(lf, "pn", x="offset", groups=["lbw", "vdd"], units={"offset": "Hz", "psd": "dBc/Hz"})
    assert pn.index == ["lbw", "vdd", "offset"] and pn.yname == "psd" and pn.yunit == "dBc/Hz"
    at_lbw = pn.value(1e5)  # one row per lbw x vdd; -3 dB at the loop bandwidth for lbw = 1e5
    assert at_lbw.height == 4
    row = at_lbw.filter((pl.col("lbw") == 1e5) & (pl.col("vdd") == 1.0))["value"].item()
    assert row == pytest.approx(-103.01, abs=0.05)
    # only the group columns and that channel are read
    plan = pn.lazy.explain()
    assert "spurs" not in plan and "idd" not in plan
    # integrated phase noise: linear power from dBc/Hz, integrated over offset
    integ = (10 ** (pn / 10)).integ(1e4, 1e6)
    assert integ.height == 4 and (integ["integ"] > 0).all()


def test_list_of_structs_and_empty_channel():
    df = pl.DataFrame(
        {
            "dut": ["A", "B", "C"],
            "sweep": [
                [{"f": 1.0, "g": 1.0}, {"f": 2.0, "g": 0.5}, {"f": 3.0, "g": 0.25}],
                [{"f": 1.0, "g": 2.0}, {"f": 2.0, "g": 1.0}],
                [],
            ],
        }
    )
    w = pw.from_nested(df, "sweep", x="f", y="g", groups=["dut"])
    assert w.to_polars()["dut"].to_list() == ["A", "A", "A", "B", "B"]  # C has no points
    assert w.ymax().rows() == [("A", 1.0), ("B", 2.0)]


def test_from_nested_errors():
    df = pll_rows()
    with pytest.raises(KeyError):
        pw.from_nested(df, "nope", x="offset")
    with pytest.raises(KeyError):
        pw.from_nested(df, "pn", x="time")
    with pytest.raises(KeyError):
        pw.from_nested(df, "pn", x="offset", groups=["missing"])
    with pytest.raises(TypeError):
        pw.from_nested(df, "idd", x="offset")


def test_math_and_result_source():
    w = pw.Waveform(pl.DataFrame({"t": [1.0, 2.0, 4.0], "v": [1.0, 10.0, 100.0]}), "v")
    assert w.log10().y.to_list() == pytest.approx([0.0, 1.0, 2.0])
    assert (10 ** w.log10()).y.to_list() == pytest.approx([1.0, 10.0, 100.0])
    assert w.sqrt().y.to_list() == pytest.approx([1.0, 10**0.5, 10.0])
    assert w.log().exp().y.to_list() == pytest.approx([1.0, 10.0, 100.0])
    assert pw.log10 is not None and pw.exp(w).y[0] == pytest.approx(np.e)

    class Lab:  # the ResultSource shape, e.g. a lab-data handler
        leaves = pl.DataFrame({"dut": ["A"]})

        @property
        def names(self):
            return ["v"]

        def v(self, signal, **params):
            return w

        def scan(self, **kwargs):
            return w.lazy

    assert isinstance(Lab(), pw.ResultSource)
