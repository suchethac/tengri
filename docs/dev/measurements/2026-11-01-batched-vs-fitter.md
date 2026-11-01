# Batched spectroscopic stack against the per-galaxy Fitter (12 mocks)

Measured 2026-11-01 for issue #3118.

- Code: tengri at `a608d682f` (the #3117 head, merged into the lane). The lane head adds only documentation.
- Machine: Apple M4 Pro, 14 CPU cores (`hw.ncpu` 14), 48 GiB memory (`hw.memsize` 51539607552), Darwin 25.6.0. CPU only, `JAX_PLATFORMS=cpu`, float64, jax 0.11.1 from `<repo>/.venv`.
- Machine load: other sessions shared the machine. No step exited with code 137.

## Mock set

- 12 galaxies, `i = 0..11`. Pixel count `560, 600, 640` for `i mod 3 = 0, 1, 2` (four each), on `linspace(4000, 7500, n)`.
- Redshift `z = 0.05 + 0.10 i / 11`. Truth `log10 M = 9.0 + 0.25 (i mod 5)`.
- Template: DPL SFH, mass free on `Uniform(8, 12)`; `neb={"type": "none"}`; calibration tilt 5 percent (`1 + 0.05 x`); `Spectroscopy` banded resolution R = 300 (`gaussian_resolution_bands(wave, 300, 21)`), `eline_mode="marginalized"`, flat prior, `eline_prior_sigma = 1e-14`, catalog Hbeta, OIII_4959, OIII_5007, Halpha, `eline_sigma_kms = 120`.
- Noise: S/N 20 per pixel on the median continuum; each line injected at matched-filter S/N 5.
- Fits: MAP with 300 Adam steps at learning rate 0.05 on both paths; `calibration_marginalize=True`, `cal_n_poly=3`, `cal_prior_sigma=1.0`.
- Batched template is free-redshift on `Uniform(0, 0.5)` with the 600-pixel reference grid. The per-galaxy path uses `Fixed(z)` builds on each galaxy's own grid, with `Fitter(..., approx=None)`.
- Source: `<job-tmp>/out/3118/common.py`, `<job-tmp>/out/3118/step_a.py`, `<job-tmp>/out/3118/step_b.py`, `<job-tmp>/out/3118/step_c.py`, `<job-tmp>/out/3118/step_d.py`. Logs: `<job-tmp>/out/3118/a_p560.log`, `a_p600.log`, `a_p640.log`, `b.log`, `c_*_g*.log`, `d_free.log`, `d_fixed.log`.

## Commands

Run from the lane worktree, one python process per command, with
`PYTHONPATH=<worktree>/src` and `JAX_PLATFORMS=cpu`.

```
PY=<repo>/.venv/bin/python
J=<job-tmp>/out/3118
$PY <job-tmp>/out/3118/step_a.py 0 > <job-tmp>/out/3118/a_p560.log 2>&1      # 560 pixels, galaxies 0,3,6,9
$PY <job-tmp>/out/3118/step_a.py 1 > <job-tmp>/out/3118/a_p600.log 2>&1      # 600 pixels, galaxies 1,4,7,10
$PY <job-tmp>/out/3118/step_a.py 2 > <job-tmp>/out/3118/a_p640.log 2>&1      # 640 pixels, galaxies 2,5,8,11
$PY <job-tmp>/out/3118/step_b.py > <job-tmp>/out/3118/b.log 2>&1             # batched, quantum=64, two calls in one process
$PY <job-tmp>/out/3118/step_c.py batched 0 > <job-tmp>/out/3118/c_batched_g0.log 2>&1
$PY <job-tmp>/out/3118/step_c.py fitter 0 > <job-tmp>/out/3118/c_fitter_g0.log 2>&1
$PY <job-tmp>/out/3118/step_c.py batched 4 > <job-tmp>/out/3118/c_batched_g4.log 2>&1
$PY <job-tmp>/out/3118/step_c.py fitter 4 > <job-tmp>/out/3118/c_fitter_g4.log 2>&1
$PY <job-tmp>/out/3118/step_c.py batched 11 > <job-tmp>/out/3118/c_batched_g11.log 2>&1
$PY <job-tmp>/out/3118/step_c.py fitter 11 > <job-tmp>/out/3118/c_fitter_g11.log 2>&1
$PY <job-tmp>/out/3118/step_d.py free > <job-tmp>/out/3118/d_free.log 2>&1
$PY <job-tmp>/out/3118/step_d.py fixed > <job-tmp>/out/3118/d_fixed.log 2>&1
```

A probe run `step_a.py 0 0` (galaxy 0 alone, `a_g0.log`) preceded the table and is not in it. Each command finished in under 25 seconds.

## (a) Per-galaxy path

Each galaxy is a fresh `Fitter`, so each first evaluation and each Adam step function compile again per galaxy. Times are summed over the four galaxies of each pixel count.

| Stage | 560 pixels | 600 pixels | 640 pixels | All 12 |
|---|---|---|---|---|
| `SEDModel.build`, `Fixed(z)` | 0.267 s | 0.358 s | 0.509 s | 1.134 s |
| `Fitter` construction and initial point | 2.488 s | 2.231 s | 2.362 s | 7.081 s |
| First `neg_log_posterior` evaluation (compile) | 1.636 s | 2.468 s | 2.429 s | 6.533 s |
| 300 Adam MAP steps (includes the step function compile) | 3.097 s | 4.360 s | 3.490 s | 10.947 s |
| Sum of the four stages | 7.488 s | 9.417 s | 8.790 s | 25.695 s |

MAP `log10 M` minus truth, per galaxy, in catalog order: 0.0493, -0.0014, -0.0025, 0.0005, -0.0004, 0.0477, -0.0014, 0.0002, 0.0002, -0.0002, 0.0482, 0.0006.

## (b) Batched path

`build_spectro_batches(records, quantum=64, model=template)`: 0.053 s, two buckets. `n_max = 576` (4 galaxies) and `n_max = 640` (8 galaxies).

`fit_spectra_map_vmap(template, batches, n_steps=300, learning_rate=0.05, calibration_marginalize=True, cal_n_poly=3, cal_prior_sigma=1.0)`, two calls in one process. The counter is `jax._src.test_util.count_jit_compilation_cache_miss`.

| Call | Wall | JIT cache misses |
|---|---|---|
| 1 (includes all compiles) | 22.717 s | 573 |
| 2 (same step count) | 0.816 s | 2 |

The counter includes build-time compiles. The two misses on call 2 were not traced.

MAP `log10 M` minus truth, catalog order: 0.0021, -0.0003, -0.0025, 0.0005, -0.0004, -0.0004, -0.0004, 0.0002, 0.0002, -0.0002, 0.0005, 0.0018.

## (c) Posterior check, three galaxies

Galaxies 0 (560 pixels, z = 0.050, truth 9.00), 4 (600 pixels, z = 0.0864, truth 10.00) and 11 (640 pixels, z = 0.150, truth 9.25). `n_warmup = 100`, `n_samples = 200`, `seed = 0`, one galaxy per process.

- Batched: `fit_spectra_catalog_mcmc(template, batches, sampler="nuts", n_warmup=100, n_burnin=0, n_samples=200, seed=0, calibration_marginalize=True, cal_n_poly=3, cal_prior_sigma=1.0)` on the free-redshift template, `build_spectro_batches([record], quantum=64, model=template)`.
- Fitter: `Fitter(model_i, data=flux, noise=sigma, data_type="spectroscopy", approx=None, calibration_marginalize=True, cal_n_poly=3, cal_prior_sigma=1.0).run("mcmc_nuts", key=PRNGKey(0), n_warmup=100, n_burnin=0, n_samples=200, verbose=False)` on the `Fixed(z)` build.

| Galaxy | Pixels | z | Truth | Batched median | Batched 16-84 width | Fitter median | Fitter 16-84 width | (batched median - Fitter median) / Fitter width | Wall batched | Wall Fitter |
|---|---|---|---|---|---|---|---|---|---|---|
| 0 | 560 | 0.0500 | 9.00 | 9.002415 | 0.002011 | 9.002262 | 0.002119 | 0.072 | 11.11 s | 11.15 s |
| 4 | 600 | 0.0864 | 10.00 | 9.999599 | 0.002031 | 9.999638 | 0.001697 | -0.023 | 12.14 s | 11.22 s |
| 11 | 640 | 0.1500 | 9.25 | 9.251749 | 0.001427 | 9.251776 | 0.001500 | -0.018 | 12.44 s | 9.06 s |

Monte Carlo error: each median has about 0.04 widths with 200 draws, so the difference of two medians has about 0.06 widths. The galaxy 0 difference of 0.072 is within that error. The contract test uses 0.05 at 6000 draws; this check did not use that many draws.

## (d) Peak RSS, fresh process per build

`resource.getrusage(RUSAGE_SELF).ru_maxrss`, reported in bytes on darwin.

| Build | Peak RSS after import | Peak RSS after build | Build wall |
|---|---|---|---|
| Free-redshift template | 494.08 MB | 504.84 MB | 0.189 s |
| `Fixed(z)`, galaxy 0 | 497.35 MB | 508.31 MB | 0.138 s |

The synthetic grid (100 wavelengths, 3 metallicities, 20 ages) makes these builds small. The memory is dominated by the process baseline. The builds are not comparable to the real-grid build numbers in the page.

## Summary

- Build plus compile plus MAP, all 12 galaxies: per-galaxy path 25.7 s (sum of four stages, each galaxy compiled separately); batched path 22.7 s on the first call and 0.82 s on the second call.
- Posterior medians agree within 0.072 Fitter widths on the three galaxies checked.
- Per-galaxy MAP for the three truth-9.00 galaxies (0, 5, 10) ends 0.048 to 0.049 dex high. Batched MAP for the same galaxies is within 0.002 dex. Cause not investigated; the per-galaxy build is `Fixed(z)`.
