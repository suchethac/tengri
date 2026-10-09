# Dust Emission Golden Baseline

Captured dust emission SEDs at commit ccb1b6eda, before migration to
SEDModelComponent subclasses.

## Captured Templates (11)

- `bosa`
- `casey2012`
- `dale2014`
- `dale2014_cigale`
- `draine_li2007`
- `draine_li2014`
- `graybody`
- `modified_blackbody`
- `pah_drude`
- `schreiber2018`
- `themis`

## Skipped Templates (0)



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

## Regenerated 2026-09-05 (evaluation-grid normalization)

Template models now resample to the evaluation grid first and normalize
there, with `integrate_lnu_over_nu` as the shared trapezoid in frequency,
so `integral(sed_dust_ir) / L_ir` is 1 to eight digits on every grid. On
this 512-point grid the previous files were off by: bosa 4.0%, dale2014
3.1%, dale2014_cigale 2.9%, draine_li2014 3.1%, themis 12.9%. The frozen
component goldens `astrodust.npy` (0.99%) and `draine2021_pah_ir.npy`
(0.74%) were re-frozen with the same call path as
`tests/regression/test_dust_goldens_852.py`. `graybody.npy` and
`schreiber2018.npy` are new.

## Regenerated 2026-10-01 (#2596)

`graybody.npy` — the closure no longer carries the optically-thin `(nu/nu_ref)^beta` factor on top of the general-opacity term; every other node of every other template unchanged.

## Regenerated 2026-10-04 (#2708)

`casey2012.npy` — emission is zero below 1 um and the shape is normalized on the evaluation grid after that mask. Nodes at 1 um and longer are scaled by one constant, 1.0002297314, relative to the previous file; the node below 1 um (1000 A) is zero. Every other template unchanged.

## Removed 2026-10-06 (schreiber2016 is the tabulated library)

`schreiber2016.npy` captured the analytic modified-blackbody plus Drude stand-in, which no longer exists: `schreiber2016` is the tabulated Schreiber et al. (2018) library, whose expected values are mixed in the tests from the template arrays instead of frozen from the model's own output.

## Regenerated 2026-10-09 (schreiber2018 native-grid normalization, guarded)

`schreiber2018.npy` is re-frozen from the current component and is now read by
`TestSchreiber2018GridPort` in `test_dust_emission_grid_components.py`. The change
has two parts, both measured on this 512-point grid:

- The normalization constant. The closure divides by the native-grid integral
  instead of the caller-grid integral, so each node is scaled by one constant
  `C = I_caller / I_native = 1.0466432589673538` on the templates the golden was
  frozen with. Applied to the old file, `old * C` reproduces the current closure
  on those templates to 1e-16 relative.
- The template data. `data/schreiber2018_templates.h5` changed after this golden
  was frozen (#2291, 2026-09-12). That shifts the shape by at most 6e-4 of the
  peak, and by up to 1.3 % at a few nodes between about 6e4 and 1.8e5 A. This
  predates the native-grid change, and it is the only part of the file change
  that is not the constant.
