# polars-waveform

Waveforms and curve families on [Polars](https://pola.rs), with circuit-style measurements:
bandwidth, crossings, rise time, phase margin, ... Works on any Polars data: simulation results
(through [polars-psf](https://github.com/henjo/polars-psf) for Cadence® Spectre® PSF files), lab measurements in Parquet,
or a DataFrame you built yourself. Pure Python; depends only on `polars`.

polars-waveform is an independent community project, not affiliated with or endorsed by the Polars
project or by Cadence Design Systems, Inc.

```sh
pip install polars-waveform          # or: uv add polars-waveform
```

## Quick start

```python
import polars as pl
import polars_waveform as pw

df = pl.DataFrame({"t": [0.0, 1.0, 2.0, 3.0], "v": [0.0, 0.4, 0.9, 1.0]})
w = pw.Waveform(df, "v")             # the last index column (t) is the sweep

w.ymax(), w.cross(0.5), w.rise_time(), w.value(1.5)
w.deriv(), w.clip(0.5, 2.5), w * 2 - 1
```

## Families: one curve per operating point

Index columns besides the sweep are groups. A Waveform then holds a family of curves (devices,
temperatures, Monte Carlo runs), and every measurement returns a table with one row per curve:

```python
lf = pl.scan_parquet("lab.parquet")                          # lazy: read on first use
gain = pw.Waveform(lf.select("dut", "temp", "freq", "gain"), "gain", index=["dut", "temp", "freq"])

gain.bandwidth()                     # dut | temp | bandwidth
gain.leaf(dut="A1", temp=25.0)       # one curve (a single curve gives plain numbers)
gain - gain.mean()                   # per-curve results broadcast back over the curves
```

Those tables are ordinary Polars DataFrames: sort, filter, join with other measurements, and
`write_csv` / `write_excel`.

## Nested channels

Measurement tables often keep one row per operating point and store each *channel*, a curve
with its own sweep, in a nested column:

```
lbw | vdd | idd | ... | pn: {offset: [..], psd: [..]} | spurs: {freq: [..], level: [..]}
```

`from_nested` turns a channel into a family, one curve per row, without flattening the table.
Only the group columns and that channel are read:

```python
pn = pw.from_nested(pl.scan_parquet("pll.parquet"), "pn", x="offset", y="psd",
                    groups=["lbw", "vdd"], units={"offset": "Hz", "psd": "dBc/Hz"})

pn.value(1e6)                                    # phase noise at 1 MHz offset, per lbw x vdd
(10 ** (pn / 10)).integ(1e4, 1e7)                # integrated phase noise, per lbw x vdd
```

Both nested shapes work: a struct of lists (`{x: [..], y: [..]}`) and a list of structs.

## Measurements

The semantics follow OCEAN, and the OCEAN and pycircuit names are aliases (`pw.dB20`,
`pw.unityGainFreq`, `pw.riseTime`, `pw.IIP3`, ...):

| | |
|---|---|
| values | `value(x)`, `ymax()`, `ymin()`, `xmax()` (x at the largest y), `xmin()`, `average()`/`mean()`, `rms()`, `stddev()` |
| crossings | `cross(threshold, edge=1, type="either")` (edges from 1, negative from the end), `delay(other, ...)`, `frequency()`, `period()` |
| transient | `rise_time()`, `fall_time()`, `slew_rate()`, `overshoot()`, `settling_time()` |
| frequency | `bandwidth(db, "low"/"high"/"band")`, `unity_gain_frequency()`, `phase_margin()`, `gain_margin()` |
| shape | `deriv()`, `integ(xfrom, xto)`, `iinteg()`, `clip(xfrom, xto)`, `dft()`, `leaf(**groups)` |
| math | `+ - * /`, `**`, `10 ** w`, `abs()`/`mag()`, `db10()`, `db20()`, `phase()`, `real()`, `imag()`, `conj()`, `log10()`, `exp()`, `sqrt()` |
| RF | `pw.im2`, `pw.im3`, `pw.iip2`, `pw.iip3`, `pw.compression_point` |

Complex data is a `Struct{re, im}` column; arithmetic and `db20()`/`phase()` handle it, and
`pl.col("h").cx.db20()` (registered by `polars_waveform.cx`) does the same in plain Polars.

## Result sources

`pw.ResultSource` is the shape shared by result objects that hand out waveforms: `leaves`,
`names`, `v(signal, **params)` and `scan()`. `polars_psf.Result` implements it for simulation
results; a lab-data handler can implement it too, so analysis code runs on both.

## Development

```sh
pip install -e ".[test]"
pytest tests
ruff check .
```

## License

MIT.
