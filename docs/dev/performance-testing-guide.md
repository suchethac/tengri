# Performance Testing Guide

Guide for running, interpreting, and updating forward model benchmarks.
Written for future AI agents and developers.

## Quick benchmark command

```bash
source .venv/bin/activate
JAX_PLATFORMS=cpu python analysis/bench_speed_all_modes.py  # if it exists
# Or inline:
JAX_PLATFORMS=cpu python -c "
from tengri import SEDModel
import jax, jax.numpy as jnp, time
m = SEDModel.from_config(ssp='data/pgny_mist_c3k_chabrier.h5', sfh='dpl',
    filters=['sdss_u','sdss_g','sdss_r','sdss_i','sdss_z'], redshift=0.1)
p = m.spec.sample(jax.random.PRNGKey(42))
for mode in ['exact','compositional','hybrid']:
    _ = m.predict_photometry(p, mode=mode)  # warmup
    _ = m.predict_photometry(p, mode=mode)
    n = 10 if mode == 'exact' else 200
    t0 = time.perf_counter()
    for _ in range(n): m.predict_photometry(p, mode=mode)
    print(f'{mode}: {(time.perf_counter()-t0)/n*1e6:.0f} μs')
"
```

## Key rules

1. **Pin `JAX_PLATFORMS=cpu` unless the benchmark is about the device.** CPU is
   the reference platform, and Apple Metal in particular gives unreliable
   timings. Every benchmark here is a CPU number except
   `benchmark_device_matrix.py` and `benchmark_catalog_throughput.py`, which are
   deliberately device-agnostic and select the platform from the environment.
   CUDA numbers live in `bench/reports/2026-08-20_cuda_device_matrix.md`.

2. **Always warm up** — first call triggers XLA compilation (~30-60s).
   Run the function twice before timing. The persistent XLA cache at
   `~/.cache/tengri_jax_cache` avoids recompilation across sessions.

3. **Clear XLA cache when measuring compilation** — `rm -rf ~/.cache/tengri_jax_cache`
   to get cold-start numbers.

4. **Use pgny SSP** (`data/pgny_mist_c3k_chabrier.h5`) as the default
   benchmark SSP. It's a pure-continuum progenitor SSP without baked-in
   nebular emission.

5. **Report errors on meaningful bands only** — at high z, some optical
   bands have flux < 1e-45. Relative errors on near-zero quantities are
   misleading. Mask with `jnp.abs(exact) > 1e-45`.

## What to benchmark

### Standard configurations

```python
configs = {
    "stellar":      dict(),
    "cue":          dict(nebular='cue'),
    "cloudy":       dict(nebular='cloudy', cloudy_grid_path='data/cloudy_grid_mist.h5'),
    "dl07":         dict(dust_emission='draine_li2007'),
    "themis":       dict(dust_emission='themis'),
    "agn_simple":   dict(agn='simple'),
    "agn_kd_full":  dict(agn='kubota_done_full'),
    "kitchen_sink": dict(nebular='cue', dust_emission='themis',
                         agn='kubota_done_full', radio=True, xray=True),
}
```

### What to measure

For each config, report:

| Metric | How |
|--------|-----|
| **Latency** (μs/ms) | `time.perf_counter()` over N iterations, post-warmup |
| **Speedup** vs exact | `t_exact / t_mode` |
| **Max error** vs exact | `max(abs(mode - exact) / abs(exact))` on bands with flux > 1e-45 |
| **Gradient NaN** | `jax.grad(lambda p: sum(predict(p)))` — check all params |

### Redshift sweep

Run the kitchen-sink config at z = {0.01, 0.1, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0}
to verify IGM handling and high-z stability.

## Profiling the forward model

### Python-level profiling

```python
import time

m = SEDModel.from_config(...)
p = m.spec.sample(key)

# Profile each stage
ip = m._get_internal_params(p)           # param translation
sfr = m._compute_sfr(ip)                 # SFH
rest = m.predict_rest_sed(p)             # rest-frame SED (the bottleneck)
obs = m.predict_obs_sed(p)               # observed SED = rest + redshift
phot = m.predict_photometry(p)           # photometry = obs + filters
```

### XLA-level profiling

```python
# Dump XLA HLO for the compositional kernel
from jax import make_jaxpr
jaxpr = make_jaxpr(m._compositional.photometry)(sfr_on_ssp, p)
print(jaxpr)

# Or use JAX's built-in profiler
with jax.profiler.trace("/tmp/jax-trace"):
    for _ in range(100):
        m.predict_photometry(p, mode='compositional')
# View with: tensorboard --logdir /tmp/jax-trace
```

### Identifying bottlenecks

The compositional kernel fuses everything into one XLA graph, so
Python-level profiling won't show internal breakdowns. Instead:

1. **Benchmark incrementally** — add one component at a time:
   - Stellar only → + nebular → + dust → + AGN → + radio → + xray
   - The delta between configs shows each component's marginal cost

2. **Check XLA compilation log** — set `XLA_FLAGS="--xla_dump_to=/tmp/xla_dump"`
   to inspect the generated HLO graph

3. **Watch for recompilation** — if the JIT cache misses (different shapes,
   different Python control flow), compilation adds 30-60s. Use
   `JAX_LOG_COMPILES=1` to detect this.

## Updating benchmarks after code changes

When you modify the forward model, fused kernels, or precomputation:

1. **Run the standard benchmark suite** (all configs × 3 modes)
2. **Compare against the table in `docs/dev/optimization-architecture.md`**
3. **Update the table** if numbers changed significantly (>20% latency change
   or >0.1% error change)
4. **Include the benchmark in the commit message** — e.g., "hybrid 35 μs → 28 μs
   for stellar-only" or "compositional error 0.001% → 0.000%"

### Regression thresholds

| Metric | Acceptable | Investigate |
|--------|-----------|-------------|
| Latency increase | <10% | >10% |
| Error increase | <0.01% absolute | >0.01% |
| New NaN gradient | Never acceptable | Always investigate |

## Known performance characteristics

### Why compositional is 0.000% error

The compositional kernel computes the SAME physics as exact, just fused
into one `@jax.jit` graph. The only difference is floating-point operation
order (XLA may reorder commutative ops), giving ~1e-12 relative difference
— which rounds to 0.0000% at 4 decimal places.

### Why hybrid degrades with complex non-stellar

Hybrid's advantage is the stellar CSP: `einsum("i,if->f")` over
`(n_age, n_filters)` instead of `(n_age, n_wave)`. But non-stellar
components are always computed at full wavelength. As non-stellar
components grow (Cue: 4 neural net passes; K&D: 3-zone disc + nthcomp),
they dominate and the stellar savings matter less.

### Why exact mode is slow

The exact pipeline dispatches each physics function through Python,
allocating intermediate arrays. Each function call has ~1 μs Python
overhead × thousands of operations. The rest-frame SED computation
accounts for 99.8% of total time.

### The SFR bug pattern

A recurring bug: using `weights[-1]` (CSP mass in youngest age bin, ~10^8
Msun) instead of `sfr_on_ssp[-1]` (SFR in Msun/yr, ~0.2) for nebular
Q_H scaling and X-ray luminosity. This inflates nebular emission by ~10^9.

The bug existed independently in three code locations:
- `nonstell.py` — compositional kernel
- `fused_kernels.py` `_hybrid_phot_body` — hybrid kernel
- `fused_kernels.py` z-table hybrid kernel

When adding new non-stellar components that depend on SFR, always use
`sfr_on_ssp[-1]` or `p.get("_sfr_current")`, never `weights[-1]`.

## Files involved

| File | Role |
|------|------|
| `core/fused_kernels.py` | All JIT kernel builders (hybrid, compositional, z-table) |
| `core/nonstell.py` | Non-stellar SED builder for compositional rest_sed_kernel |
| `core/preintegrate.py` | Generic template preintegration |
| `core/precompute_templates.py` | DL07/SKIRTOR/Dale preintegration wrappers |
| `core/model.py` | Mode dispatch, PrecomputedData, kernel wiring |
| `models/sps/precompute.py` | SSP photometry/spectroscopy/z-table precomputation |
| `docs/dev/optimization-architecture.md` | Benchmark tables (update after changes) |
| `docs/dev/dust-preintegration.md` | Dust IR preintegration details |
| `analysis/bench_speed_all_modes.py` | Benchmark script (if exists) |
