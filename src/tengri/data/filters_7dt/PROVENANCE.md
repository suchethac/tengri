# 7DT Filter Curve Provenance

## What these are

Total system transmission for the 23 7DT bands used by the NGC 1380 program:
3 broad (`g`, `r`, `i`) and 20 medium bands on a 25 nm grid (`m400`-`m875`).
The full 7DT filter set is larger; `u`, `z`, the `*w` filters, and the off-grid
mediums (`m386`, `m438`, ...) are not part of this delivery and are not bundled.

Registered as `7dt_g`, `7dt_r`, `7dt_i`, `7dt_m400` ... `7dt_m875`.

## Source

Delivered 2026-08-27 by Eunjae Herr (Im Lab, Seoul National University) as
`7DT_transmission_23bands.zip`, sha256
`00edf494f825e78e2446301c6c44d6a6c653607c92d7f88f2b8fe4f005f12d06`, one CSV per
band with columns `lam,trans`. Wavelength in **nanometers**, uniform 0.1 nm
grid, 300-1000 nm, 7001 rows per band, zero-padded outside the passband.
Transmission dimensionless.

## Conversion

By `tools/build_7dt_filter_curves.py`, which is committed so these files can be
regenerated from the delivery:

```
python tools/build_7dt_filter_curves.py --source <dir>/7DT_transmission_23bands
```

1. **Wavelength multiplied by 10**, nm to Angstrom. Tengri wavelengths are
   Angstrom throughout. The 0.1 nm delivery grid becomes an exact 1 Angstrom
   integer grid, so the written `%.1f` values are exact.
2. **Transmission untouched.** Not renormalized, per the delivery instruction.
3. **Leading and trailing runs of exact zeros trimmed**, one zero row kept on
   each side as an edge anchor. This drops 25 percent of the rows. It is
   lossless in the only sense that matters: every trimmed row was exactly
   `0.0`, so neither bandpass integral changes. Verified bit-for-bit on
   wavelength and transmission, with the AB zeropoint of each band moving by at
   most 5e-16 mag, which is floating-point noise.

No other transformation. No resampling, no smoothing, no unit change to the
transmission column.

## Why these and not SVO

Peak transmission runs 0.34 at `m400`, up to 0.66 near 500 nm, down to 0.07 at
`m875`. Twenty separate filters do not share one smooth envelope by accident:
detector QE and optics are already folded into these curves. They are the total
system response the program's ADU were actually measured through, which a
filter-glass-only curve would not be.

This is harmless for AB synthetic photometry and in fact preferable. A constant
scale on `T` cancels exactly in the ratio of the two bandpass integrals, so the
overall normalization does not matter; the intra-band QE shape does, and folding
it in makes the curve more correct rather than less. It would matter if these
were used to predict absolute count rates. They should not be.

Whether SVO now serves 7DT was not checked when these were added, and it does
not change the choice: per the rule in `tengri/observation/filters/__init__.py`,
a name must never resolve to an approximation while real data for it exists.

## Known limitation

Transmission is tabulated to 5 decimal places. For the faintest bands that
rounding floor is a non-trivial fraction of the peak: `m875` peaks at 0.0675, so
the 1e-5 quantization is about 1/7000 of peak and the out-of-band rounding
residual contributes roughly 0.5 percent of that band's bandpass weight. It is
rounding, not a measured red leak. The affected bands are `m850` and `m875`;
more decimals from the source would remove it. This is below the band's
zeropoint uncertainty (0.018-0.038 mag) but not negligible against it.

## File digests

sha256, first 16 hex characters:

| file | sha256 |
|---|---|
| `7dt_g.dat` | `102c1819be19a780` |
| `7dt_r.dat` | `c99bd6457db5df6b` |
| `7dt_i.dat` | `c603714acda815ad` |
| `7dt_m400.dat` | `32c03e114a100be4` |
| `7dt_m425.dat` | `c1a430998df705eb` |
| `7dt_m450.dat` | `560dc9b11d1bd3b0` |
| `7dt_m475.dat` | `1492c5da4105d0c5` |
| `7dt_m500.dat` | `f9f8be92d522f615` |
| `7dt_m525.dat` | `6a17553d7a728e48` |
| `7dt_m550.dat` | `473cf1837974bfeb` |
| `7dt_m575.dat` | `c29c7b1c9074dedd` |
| `7dt_m600.dat` | `9c5eeca4cc65146c` |
| `7dt_m625.dat` | `2f403c2acf842c3d` |
| `7dt_m650.dat` | `d6d073f7ba81b71c` |
| `7dt_m675.dat` | `ec84ead1fb20a86d` |
| `7dt_m700.dat` | `93c493eba325aa39` |
| `7dt_m725.dat` | `8d46e051cc012a64` |
| `7dt_m750.dat` | `9a9f52eeaf3a661c` |
| `7dt_m775.dat` | `90743d91ea84d895` |
| `7dt_m800.dat` | `eca6021edcdb015e` |
| `7dt_m825.dat` | `b85969135837643c` |
| `7dt_m850.dat` | `6c4be6947c7d17ec` |
| `7dt_m875.dat` | `22dc0a10e62bb240` |

## Licensing

The numerical data is 7DT instrument characterization, provided by the Im Lab
in response to a question about fitting 7DT photometry with tengri. It is
instrument metadata, not code.

**Redistribution permission is not yet on record.** The curves were sent for
use, which is not the same as consent to ship them in a public package. Confirm
with the depositor (Eunjae Herr, Im Lab, SNU) and record the answer here before
these files go out in a release. Cite the 7DT instrument papers when publishing
photometry synthesized through them.
