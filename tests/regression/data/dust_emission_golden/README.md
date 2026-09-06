# Dust Emission Golden Baseline

Captured dust emission SEDs at commit ccb1b6eda, before migration to
SEDModelComponent subclasses.

## Captured Templates (11)

- `astrodust`
- `bosa`
- `casey2012`
- `dale2014`
- `dale2014_cigale`
- `draine_li2007`
- `draine_li2014`
- `modified_blackbody`
- `pah_drude`
- `schreiber2016`
- `themis`

## Skipped Templates (1)

- `schreiber2018`: 'schreiber2018' lazy loader is in an inconsistent state — the first resolution did not replace the registry entry. Check schreiber2018_templates.h5 or use the modern SEDComponent path.

## Input Parameters

See `params.json` for exact wavelength grid, L_ir, and per-template parameters.

## Generation

```bash
python scripts/baseline_dust_emission_golden.py
```

All outputs use 64-bit JAX arrays (jax_enable_x64=True).

## Re-frozen 2026-09-05 (CMB contrast fix)

`casey2012.npy`, `modified_blackbody.npy` and `schreiber2016.npy` changed at
one node only, the grid's first (1000 A): the old contrast factor clipped
the blue end to an exact zero there, and the fix returns the physical value
(at most 7.6e-7 of the peak). Every other node agrees to 1.4e-14.
