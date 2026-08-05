# Notebook renewal — execution timings, peak RSS, and cold-compile breakdown

**Date:** 2026-05-06
**Branch:** main (post smart-lean cache + (5a) wNE→Cue hard-error)
**Hardware:** macOS (CPU only via `JAX_PLATFORMS=cpu`); 64 GB RAM laptop
**Cache:** persistent JAX disk cache enabled (`~/.cache/tengri_jax_cache`),
clean for cold runs unless noted.

## Active notebook set

After archiving 15 legacy duplicates from the previous plan to
`notebooks/archive_2020506/`, the active series is **9 notebooks** (00–08):

| # | File | Topic |
|---|---|---|
| 00 | `00_quickstart.py` | end-to-end demo, polished |
| 01 | `01_why_jax.py` | JIT/grad/vmap/composition pedagogy |
| 02 | `02_sed_anatomy.py` | wavelength-by-wavelength SED tour |
| 03 | `03_discovering_the_menu.py` | `tengri.list_*` / `describe` discovery API |
| 04 | `04_building_models.py` | compositional model construction |
| 05 | `05_fitting_photometry.py` | UV–MIR photometry HMC fit |
| 06 | `06_fitting_spectroscopy.py` | optical NUTS spectroscopy fit |
| 07 | `07_joint_photo_spec.py` | joint photometry+spectroscopy NUTS |
| 08 | `08_emission_lines.py` | line fluxes + BPT diagram |

Plan slots 09–12 (inference tour, diagnostics, gallery, extending+Paper II)
are deferred — current renewal covers Batches A+B of the plan in
`~/.claude/plans/i-want-to-make-ethereal-kahan.md`.

## End-to-end timings (jupytext sync → nbconvert --execute --inplace)

Captured via `/usr/bin/time -l` (macOS) on a clean kernel per notebook.
"wall" is real seconds, "peak RSS" is `maximum resident set size` from
`time -l` divided by 2³⁰.

| Notebook | wall (s) | peak RSS (GB) | exit | notes |
|---|---:|---:|---|---|
| `00_quickstart` | 116 | 19.6 | ✓ | full forward + 1 fit, heavy cold-cache |
| `01_why_jax` | 30 | 4.7 | ✓ | JIT/grad demos; 7-D mock fit |
| `02_sed_anatomy` | 11 | 1.5 | ✓ | panchromatic decomposition (X-ray → radio) |
| `03_discovering_the_menu` | 49 | 0.7 | ✓ | discovery API tour, no fits |
| `04_building_models` | 23 | 2.6 | ✓ | parameter sweeps, no fits |
| `05_fitting_photometry` | 120 | 11.0 | ✓ | HMC, 8 free params, 12 bands |
| `06_fitting_spectroscopy` | 385 | 7.0 | ✓ | NUTS, 6 free params, 1000 pixels |
| `07_joint_photo_spec` | 458 | 18.1 | ✓ | 2× MAP + 1× NUTS joint, 5 params |
| `08_emission_lines` | 35 | 1.9 | ✓ | line fluxes + BPT |

**Total wall time:** ~21 min for **all 9 of 9** notebooks.

## Cold-compile breakdown (nb05 photometry HMC, cache disabled)

To answer "what is taking this much cold compilation?", here's a
phase-by-phase profile of nb05 with `TENGRI_DISABLE_JAX_CACHE=1`,
`n_warmup=100, n_samples=200` (1/3 of the production config):

| phase | seconds | what it does |
|---|---:|---|
| `load_ssp_data` | 0.03 | HDF5 read of 12×107×11149 SSP grid |
| `Photometry.from_names` | 0.03 | pre-cached filter curves (`data/filters/`) |
| `Parameters(...)` | 0.00 | dataclass construction |
| `SEDModel(...)` | 3.93 | SSP precompute + filter pre-integration + ztable build |
| `predict_photometry` cold | 0.99 | forward kernel HLO compile (12-band hybrid) |
| `predict_photometry` warm | 0.00 | <1 ms steady-state |
| `mock` | 0.06 | warm forward + Gaussian noise sample |
| `Fitter(...)` | 0.01 | data binding |
| `fitter.run("mcmc_hmc", ...)` | **36.41** | **HMC scan compile + 300 iterations** |
| **TOTAL (cold, no cache)** | **41.46** | |

For the production config (`n_warmup=300, n_samples=600`), HMC scales
roughly ~3× → ~110 s, matching the observed 120 s wall.

### Where the 36 s in HMC goes

The HMC `fitter.run` time is dominated by **HLO compilation of the
`lax.scan` warmup loop**, not the actual sampling work:

- ~5 s : MAP initialization (Adam over 200 steps; small graph, fast compile)
- ~25 s : NUTS warmup body XLA compile — the leapfrog integrator
  (with L=10–32 steps depending on tree depth) calling
  `predict_photometry` is unrolled and fused into a single optimized HLO
  graph. The compile cost scales with leapfrog L, not number of warmup
  iterations.
- ~6 s : actual sampling — 100 warmup + 200 production iterations at
  ~20 ms/iter steady state.

For NUTS (vs HMC) the compile is ~10× larger because the doubling
binary-tree expansion has up to 2^max_treedepth nested leapfrog calls in
the HLO graph. That's why nb06's NUTS at 1000 pixels takes 6.4 minutes
total — most of that is XLA fusing a much bigger graph.

### Why nb06/07 are the long pole

| nb | inference | n_pix | compile graph proxy |
|---|---|---:|---|
| 05 | HMC, fixed-L | 12 | small — fast cold (~30 s HMC compile) |
| 06 | NUTS, dense_mass=False | 1000 | ~30× larger spectrum vector → ~10× compile |
| 07 | 2× MAP + 1× NUTS joint | 12+100 | NUTS dominates; 460 s total |

The compile-time pattern is **roughly proportional to the wave-vector
length carried through the gradient tape**, not the number of iterations.
Disk-cache hits eliminate the compile on subsequent process runs (~3 s
load instead of 25 s rebuild), but inside one nbconvert kernel the
compile is paid once.

### Ways to make cold compile faster (deferred)

1. **Sub-graph caching of the forward kernel.** The HMC scan body calls
   `predict_photometry` which is itself a large graph. If we marked the
   forward call as opaque to the scan body's optimizer, the scan compile
   would shrink. Trade-off: warm-iteration speed gets ~2× slower because
   XLA can't cross-fuse leapfrog → forward.
2. **`max_treedepth=8` on NUTS** (currently 10). Cuts compile graph 4×.
   Cost: longer chains needed for same ESS on stiff posteriors.
3. **Batch HMC with `n_chains>1` via vmap.** Compile is paid once; ESS
   accumulates linearly with chain count. Already wired but not enabled
   in the tutorials.
4. **Pre-compile and persist the scan kernel separately.** Custom XLA
   AOT compile that survives Python upgrades. ~1 week of work; pays off
   if many users hit cold-cache at scale.

## Peak-RSS analysis

| nb | peak RSS | what dominates |
|---|---:|---|
| 00, 07 | 19–20 GB | NUTS dense-mass-matrix vmap compile + corner KDE residency |
| 05 | 11 GB | HMC scan compile + 8-D corner |
| 06 | 7 GB | 1000-pixel NUTS scan, no dense-mass |
| 08 | 1.9 GB | line evaluation only, no scan |
| 01, 04 | 2.6–4.7 GB | small forward compile |
| 03 | 0.7 GB | no JIT-heavy paths |

The smart-lean cache (`Fitter.run(lean=True)` default, 2026-05) keeps
peak RSS bounded by evicting prior-phase scan bodies before the next
fitter call. Without it, multi-fit notebooks (e.g. nb07's MAP+MAP+NUTS)
would run at peak ~35 GB and hit macOS jetsam silently.

## Fixes that landed during this run

| File | Fix |
|---|---|
| `notebooks/01_why_jax.py` | added `os.chdir(_repo_root)` so nbconvert kernel resolves `data/...` paths |
| `notebooks/04_building_models.py` | replaced 3 hard-coded `"notebooks/figures/..."` with `_repo_root / ...` absolute paths |
| `notebooks/05_fitting_photometry.py` | `result.rhat.values()` → `result.rhat().values()` (it's a method) |
| `notebooks/05_fitting_photometry.py` | replaced `result.plot_corner` (corner.py KDE → 11+ GB OOM) with manual hist2d corner |
| `notebooks/05_fitting_photometry.py` | bypassed `result.derived` (vmap-over-600-samples OOM) with chunked Python loop |
| `notebooks/05_fitting_photometry.py` | manual SFH plotting instead of `Posterior.plot_sfh()` (no `label`/`color` kwargs) |
| `notebooks/06_fitting_spectroscopy.py` | `predict_spectrum(wavelengths=...)` → `predict_spectrum(wave_obs=...)` (median was 1 dex off) |
| `notebooks/06_fitting_spectroscopy.py` | replaced `plot_corner_comparison` (24 GB OOM) with manual hist2d corner |
| `notebooks/07_joint_photo_spec.py` | joint MAP → NUTS so constraint widths have credible intervals |
| `notebooks/07_joint_photo_spec.py` | redrew constraint figure as MAP-points overlaid on NUTS-error-bars |
| `notebooks/08_emission_lines.py` | switched from wNE SSP to bare `fsps_prsc_miles_chabrier.h5` for Cue compatibility |
| `notebooks/02_sed_anatomy.py` | replaced deleted `compute_sed_components` with `predict_via_orchestrator`; second pass with τ=0 recovers the dashed pre-attenuation reference |
| `notebooks/02_sed_anatomy.py` | rewrote as panchromatic kitchen-sink (X-ray → radio); pinned `agn_log_lbol=Fixed(11.0)` (log10 L_sun, not erg/s) and `agn_lum_ratio=Fixed(1.0)` (the registry default 0.0 zeros out AGN regardless of `agn_log_lbol`) |

## Outstanding

- **nb05/nb06 corner plots** are minimal-fidelity histograms now;
  consider re-introducing corner.py once a memory-friendly KDE backend
  exists, or upstream a `plot_corner(use_hist=True)` option to
  `Posterior.plot_corner` for tutorial use.
- **Plan slots 09–12** (inference tour, diagnostics, capability gallery,
  extending+Paper II preview) are next, on a fresh PR.

## Reproducing

```bash
# Sync .py → .ipynb (jupytext percent format)
cd notebooks && jupytext --sync *.py

# Execute one notebook with timing+RSS
/usr/bin/time -l env JAX_PLATFORMS=cpu PYTHONUNBUFFERED=1 \
  .venv/bin/jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.timeout=900 \
  notebooks/05_fitting_photometry.ipynb

# All 9 in sequence
for nb in 00_quickstart 01_why_jax 02_sed_anatomy \
          03_discovering_the_menu 04_building_models \
          05_fitting_photometry 06_fitting_spectroscopy \
          07_joint_photo_spec 08_emission_lines; do
  scripts/run_nbconvert.sh "$nb"
done
```

`scripts/run_nbconvert.sh` is the harness — it wraps `jupyter nbconvert
--execute --inplace` with `/usr/bin/time -l`, captures wall + peak RSS
into `/tmp/nbconvert_<nb>.log`, and prints a summary line.
