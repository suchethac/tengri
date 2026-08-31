# Canonical recipes performance audit — 2026-05-23

**TL;DR.** The performance targets the user remembers (tens of microseconds
per forward eval, <1 s MAP, ~5 s HMC, <20 s NUTS) are still achievable on
this codebase — but only by passing `approx=WavePrecomp()` at build time.
None of the six recipes in `tengri.recipes` wires this up, so users
landing on the recipe path silently pay the slow exact-wave-grid integral
(~5 ms forward, ~6 s MAP). Beyond the missing wiring, the audit surfaced
five reproducible bugs that block or distort canonical use cases.

## Reproducibility

```
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_canonical_recipes.py
JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_canonical_recipes.py --mcmc --recipe mock_recovery_minimal
```

Platform: macOS, CPU backend, JAX 0.x, persistent JIT cache warm
(`~/.cache/tengri_jax_cache` at 25 GB). Filters: 8 broadbands
(SDSS ugriz + 2MASS JHKs). SSP: `ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5`.

## Headline numbers

`mock_recovery_minimal` (D=7, photo only, 8 filters):

|                      | exact (default)     | `approx=WavePrecomp()` | speedup |
| -------------------- | ------------------- | ---------------------- | ------- |
| forward steady       | **5160 µs**         | **96 µs** (warm cache) | 54×     |
| forward steady (cold)| 5160 µs             | 240 µs                 | 21×     |
| gradient steady      | 3949 µs             | 131 µs                 | 30×     |
| MAP (200 ADAM steps) | **5856 ms**         | **820 ms**             | 7.1×    |
| HMC (200 warm + 200) | n/a                 | **2966 ms**            | —       |
| NUTS (200 warm + 200)| n/a                 | **2764 ms**            | —       |

`quiescent_z0` (D=5, photo only, 8 filters, neb=none stub):

|                      | exact (default)     | `approx=WavePrecomp()` |
| -------------------- | ------------------- | ---------------------- |
| forward steady       | 5165 µs             | 253 µs                 |
| gradient steady      | 4388 µs             | 323 µs                 |

Filter-count probe (5 SDSS vs 8 SDSS+2MASS, PRECOMP): 254 µs vs 234 µs.
**Filter count is not the bottleneck.**

## Speedup is real, the wiring is missing

Every public recipe in `src/tengri/recipes/__init__.py` returns a
dict that is splatted into `SEDModel.build(**recipes.X())`. `approx=` is
accepted by `SEDModel.build` (passes through `**model_kwargs` to
`__init__`), so adding it to the recipe dict is a one-line fix per
recipe. Suggested patch (per recipe):

```python
from tengri import WavePrecomp

def star_forming_photometry() -> dict:
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust=...,
        neb=builders.neb.cue(defaults=FIXED),
        redshift=Uniform(0.01, 6.0),
        apply_igm=True,
        approx=WavePrecomp(),            # ← add this line
    )
```

Recommended scope: all five photometry recipes
(`mock_recovery_minimal`, `quiescent_z0`, `star_forming_photometry`,
`stochastic_sfh_jwst`, `agn_panchromatic`). `dust_demo` is forward-only
gallery code and can stay exact for clarity.

For users that need to fit a catalog at a fixed redshift grid, the
`WavePrecomp(catalog_z_range=(z_min, z_max), n_z=200)` form gives further
amortization across galaxies — surfacing this in the recipes' docstrings
would be high-leverage.

## Bugs that block canonical use

### B1. `stochastic_sfh_jwst` recipe raises ConcretizationTypeError

`src/tengri/components/stellar/component.py:507`
```python
d_log_age = float(log_age_grid[1] - log_age_grid[0])
```
`log_age_grid` is `jnp.linspace(...)` from `make_log_age_grid`, so it is a
traced array under `jit`. Calling `float()` on the difference fails.

**Fix.** `n_grid` is static configuration and the bounds (`6.0`, `10.14`)
are constants. Replace the line with:
```python
LOG_AGE_MIN, LOG_AGE_MAX = 6.0, 10.14   # mirror make_log_age_grid defaults
d_log_age = (LOG_AGE_MAX - LOG_AGE_MIN) / (n_grid - 1)
```
Or expose a sibling `log_age_grid_step(n_grid)` from `gp_sfh.py` returning
a Python float; the existing `make_log_age_grid` keeps returning the
jnp array for the rest of `apply()`.

A regression test for this should fit one synthetic mock with
`stochastic_sfh_jwst()` through `predict_photometry` — the bug only fires
on the `field=True` branch.

### B2. `WavePrecomp + neb=none` triggers UnexpectedTracerError on star_forming_photometry

`star_forming_photometry [PRECOMP]` with `neb` swapped to `none()` (to
work around B5 below) raises `UnexpectedTracerError`. The exact-path
build of the same configuration works. Needs investigation — likely a
component that closes over a precompute-time tracer when the nebular
slot is empty.

### B3. `n_chains` removed from HMC / NUTS backends but Fitter docs still imply multi-chain

```python
fitter.run("mcmc_hmc", n_warmup=200, n_samples=200, n_chains=1)
# TypeError: run_hmc() got an unexpected keyword argument 'n_chains'
```
`tengri.inference.fitter.Fitter.run` docstring lists multi-chain MCMC as
a feature; the actual backends in `inference/backends/` don't accept the
kwarg. Either re-thread `n_chains` through `run_hmc` / `run_nuts` (via
`vmap` over keys) or update docs to say "single-chain only; for chains
call `run()` N times with different keys".

### B4. `benchmark_forward_model.py` times the same path 3×

`bench/scripts/benchmark_forward_model.py:64-71` comments admit that
PR #135 collapsed the `mode="exact|compositional|hybrid"` axis to a
single path, and the benchmark loop now times an identical path three
times. The headline numbers in
`bench/reports/2026-05-06_forward_model_speedup.md` (30–400× hybrid
speedup) are therefore historical, not currently reproducible from the
canonical bench script.

**Status (2026-08-31):** `2026-05-06_forward_model_speedup.md` marked superseded (#2092);
the bench script now compares `approx=None` vs `approx=WavePrecomp()` with honest gradient harness.

**Fix.** Rewrite the bench to compare `approx=None` vs
`approx=WavePrecomp()` across the same configs, dropping the `mode=`
loop. The benchmark added in this audit
(`bench/scripts/benchmark_canonical_recipes.py`) is a starting point
but currently only covers four recipes — extend per the matrix.

### B5. Four of six recipes fail on the default local SSP files

`star_forming_photometry`, `quiescent_z0`, `stochastic_sfh_jwst`, and
`agn_panchromatic` all use Cue nebular and require a bare-stellar SSP
(`data/ssp_prsc_miles_chabrier.h5`). A fresh checkout typically only
ships the wNE files (`ssp_prsc_miles_chabrier_wNE_*.h5`), so four
recipes raise `CueWNESSPError` immediately on `SEDModel.build(...)`.

The error message points to the SSP type mismatch but does not tell the
user how to obtain a bare-stellar SSP. Options:

* Document the required SSP file and a download URL in each affected
  recipe's docstring + a top-level `data/README.md`.
* Add a `tengri.data.download_bare_stellar()` helper.
* Provide a Cue-free fallback nebular block as a recipe variant.

## Memory / tracer behavior

* First recipe build: ~1.2 GB RSS jump (JAX warm + SSP load). Subsequent
  recipe builds add 30–60 MiB.
* No tracer-OOM signature observed on D ≤ 8. The B1 tracer error is a
  bug, not memory pressure.
* JIT cache is on by default and warm; cold compile of a recipe forward
  is ~500 ms, warm is ~70–250 ms.

## Suggested follow-on PRs (ordered)

1. **Wire `approx=WavePrecomp()` into the five photometry recipes** —
   one-line additions, gates 7–50× speedup for default users.
2. **Fix B1** (stellar/component.py:507) — unblocks
   `stochastic_sfh_jwst`. Add regression test.
3. **Fix B4** — rewrite `benchmark_forward_model.py` to compare
   `approx=None` vs `WavePrecomp()`. Republish a current speedup
   report.
4. **Investigate B2** — the WavePrecomp+neb=none interaction.
5. **Resolve B3** — restore `n_chains` or update docs.
6. **B5** — SSP fetch story.
