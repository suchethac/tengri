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
