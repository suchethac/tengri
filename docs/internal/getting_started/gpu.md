# Running on a GPU

Tengri's forward model is pure JAX, so the same code that runs on CPU
runs on GPU and TPU without modification. This page shows how to verify
your install picks up the GPU and what to expect once it does.

```{warning}
Benchmark numbers in [Performance](../performance/index.md) are CPU only.
GPU performance is functionally tested but not characterized with
published wall-clocks. To contribute benchmarks on A100 / H100 / 4090,
see [`bench/RERUN.md`](https://github.com/suchethac/tengri/blob/main/bench/RERUN.md).
```

## Install JAX with CUDA

Tengri installs the CPU-only JAX wheel by default. For GPU, replace it
with the CUDA wheel — [JAX's install docs](https://docs.jax.dev/en/latest/installation.html)
are authoritative for versions:

```bash
pip install -U "jax[cuda12]"
```

## Verify the install

```python
import jax
print(jax.devices())             # should list CUDA / GPU devices
print(jax.default_backend())     # should print 'gpu'
```

Or via the consolidated CLI:

```bash
python -m tengri.bench
```

prints the JAX backend and default device at the top of the report —
look for `default device: cuda` or `gpu` rather than `cpu`.

## Running a fit on the GPU

Most code paths just work:

```python
import jax
import tengri

# Pin to GPU explicitly (optional — JAX picks GPU by default if available).
jax.config.update("jax_platform_name", "gpu")

ssp = tengri.load_ssp()
# ... build Parameters / SEDModel / Fitter as in the spine notebooks ...
result = fitter.run("mcmc_nuts")
```

Things to be aware of:

- **First call compiles for the GPU device.** Cold compile is cached by
  device; the first run after switching CPU→GPU pays the cost once, then
  subsequent runs hit the cache.
- **VRAM is tighter than RAM.** Small photometric fits sit comfortably
  in 8 GB; large stochastic-SFH models can peak at about twice that
  during compile. If you hit OOM, set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` to leave headroom for cuDNN.
- **`predict_*_batch` is where the GPU wins.** Single-galaxy calls are
  CPU-bound on Python overhead; the batch path `Fitter` uses for
  catalog fits is what fills the GPU.

## Is it worth it? (measured)

Benchmarked 2026-08-20 on an RTX 3060 against a Ryzen 9 5900X — see
[`bench/reports/2026-08-20_cuda_device_matrix.md`](https://github.com/suchethac/tengri/blob/main/bench/reports/2026-08-20_cuda_device_matrix.md)
and `notebooks/nvidia_cuda.py`. The GPU is a **width** instrument:

| shape | faster |
|---|---|
| one galaxy, forward | CPU by 32.7x |
| one galaxy, gradient | CPU by 13.2x |
| one MAP fit, 300 steps | CPU by 8.8x |
| batch 2048, gradient | **GPU by 8.7x** (14.7x in float32) |
| catalog NUTS, 256 galaxies | **GPU by 1.2x** |

The crossover is 128–512 galaxies for a batched forward or gradient. The forward
model runs at ~0.12 FLOP/byte, so a single galaxy leaves the card waiting on memory
and dispatch; GPU wall clock is nearly flat in batch size (2048x the work for 1.24x
the time). Note that consumer GeForce cards run float64 at 1/64 rate, which puts a
3060 *below* a 5900X on dense float64 arithmetic; datacenter cards are ~1/2.

## float32 on CUDA: set the matmul precision

If you run with `JAX_ENABLE_X64=0`, also set:

```bash
export JAX_DEFAULT_MATMUL_PRECISION=highest
```

On Ampere and later, XLA lowers float32 matmuls to TF32 (10-bit mantissa) by
default. tengri's own float32 Fisher-matrix test fails on CUDA without this — a
4.5% error on parameter error bars — and passes with it. `NVIDIA_TF32_OVERRIDE=0`
alone does **not** fix it: XLA chooses its own algorithm, so the JAX-level knob is
the one that binds. It costs no measurable speed, since float32's advantage here is
halved memory traffic rather than tensor cores.

Two float32 caveats on CUDA beyond that:

- `jax.grad` of a raw observable (e.g. `sum(predict_photometry)`) returns
  **identically zero** in float32, on any device. Fits are unaffected — the
  likelihood standardizes by sigma before squaring — but do not differentiate raw
  fluxes in float32.
- float32 geoVI with marginalized emission lines does not run: cuBLASLt refuses the
  GEMM and JAX 0.11 removed the legacy fallback.

## Apple Metal (experimental — not supported for benchmarks)

JAX-Metal is incomplete: several primitives silently fall back to CPU,
and test failures on Metal do not reproduce on CPU. Use **CPU** instead:

```bash
JAX_PLATFORMS=cpu python -m tengri.bench
```

If `python -m tengri.bench` reports `default device: METAL` and
1-galaxy timing is much slower than the [Performance table](../performance/index.md),
the silent CPU fallback is the cause. Force CPU as above.

## TPU

Untested but expected to work: tengri uses no TPU-incompatible
operations (no `jax.device_put` to CPU, no Python loops in JIT'd code,
no NumPy in the forward graph). If you try it, reports welcome.

## Multi-device

`predict_photometry_batch` is `vmap`-based and stays on one device.

Catalog fits do shard across devices. `CatalogFitter.run(..., devices="all")`
maps the galaxy axis over a `Mesh` via GSPMD — `_resolve_devices` and
`_sharded_vmap` in `src/tengri/inference/catalog_fitter.py` — for the two
backends the catalog path vectorizes, `mcmc_nuts` and `mcmc_hmc`. The galaxy
axis must be a multiple of `lcm(forward_chunk_size, n_devices)`. Every other
method still runs one galaxy at a time, where a second device has nothing to
do; for those, use one process per device over catalog slices.

`shard_map` specifically is not used: BlackJAX's NUTS carries a `lax.cond`
that trips manual varying-axis tracking, so the seam is `jax.jit(jax.vmap(...))`
over a sharded axis instead.

## See also

- [Performance](../performance/index.md) — CPU benchmarks.
- [Memory expectations](../performance/memory.md) — peak RSS table,
  measured on CPU.
- [JAX install docs](https://docs.jax.dev/en/latest/installation.html)
  — authoritative for CUDA versions and wheels.
