# tengri TODO log

Single-line / single-paragraph TODOs lifted out of source files so they don't rot
in place. Larger architectural-context blocks may stay in code where a contributor
will read them (see `forward/_kernels/hybrid.py` line ~2252 for an example).

## forward/_kernels/hybrid.py

- **Preintegrated non-stellar photometry paths**: implementations exist for
  CLOUDY, DL07 and SKIRTOR in `_nebular_phot_preintegrated()` but are gated
  behind calibration verification. All non-stellar components currently take
  the full-wavelength path. Re-enable once cross-validation against the
  full-wavelength integrators is in.
  (Was: comment at hybrid.py ~line 1548, lifted 2026-05-17.)

## components/agn/disc.py

- **ADAF disc model rewrite**: The current implementation contains documented
  physics discrepancies vs Mahadevan (1997) ApJ 477 585. Steps: (1) rewrite
  equations to match paper exactly, (2) add regression test vs original paper,
  (3) wire precompute adapter via `components/agn/adaf_precompute.py`,
  (4) register in `forward/precompute/registry.py`.
  (Was: comment block at disc.py ~line 1549, lifted 2026-05-19.)

## forward/component_factory.py

- **Port legacy `compute_uv_slope_beta` etc.** into the new
  `compute_sed_quantities` path. Currently UV-slope, Dn4000, Balmer-break,
  M_UV, and luminosity-weighted quantities are returned as `NaN`. The
  legacy machinery lives in older `sed_model.py` paths.
  (Was: TODO in component_factory.py:416, lifted 2026-05-18.)
