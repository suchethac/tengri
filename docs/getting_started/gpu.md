# Running on a GPU

Tengri's forward model is pure JAX, so the same code that runs on CPU
runs on GPU and TPU without modification. This page shows how to verify
your install picks up the GPU and what to expect once it does.

```{warning}
Benchmark numbers in [Performance](../performance/index.md) are CPU
only. GPU performance is functionally tested but **not yet
characterised** with published wall-clocks. If you can run the bench
suite on an A100 / H100 / 4090, please contribute results — see
[`bench/RERUN.md`](https://github.com/suchethac/tengri/blob/main/bench/RERUN.md).
```

## Install JAX with CUDA

Tengri's pyproject installs the CPU-only JAX wheel by default. For GPU,
follow [JAX's official CUDA install](https://docs.jax.dev/en/latest/installation.html)
and replace it. As of JAX 0.4.30+:

```bash
pip install -U "jax[cuda12]"
```

(For older JAX or specific CUDA toolkits, use
`pip install -U "jax[cuda12_pip]"` or the version-pinned variant from
the JAX docs.)

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

ssp = tengri.load_ssp_data("data/ssp_fsps_v3.2.h5")
# ... build Parameters / SEDModel / Fitter as in the spine notebooks ...
result = fitter.run("mcmc_nuts")
```

Things to be aware of:

- **First call compiles for the GPU device.** That cold compile is
  cached at `~/.cache/tengri_jax_cache` (via the persistent compile
  cache), keyed by device — so the first run after switching CPU →
  GPU pays the compile cost once, but subsequent runs hit the cache.
- **VRAM is tighter than RAM.** A smooth D = 7 NUTS fit fits comfortably
  in 8 GB. Stochastic D = 137 fits or `dust_emission="dale2014"` may
  push toward 16 GB during compile. If you hit OOM, set
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.85` (or similar) so JAX leaves
  headroom for cuDNN.
- **`predict_*_batch` is where the GPU wins.** Single-galaxy
  evaluations are usually CPU-bound on Python overhead. The
  `predict_photometry_batch` path (used by `Fitter` for catalogue fits
  and by `nb09` for parameter sweeps) is what fills the GPU.

## Apple Metal (experimental — not supported for benchmarks)

JAX-Metal exists but is incomplete: several primitives tengri uses
fall back to CPU silently, and we have observed test failures on
Metal that don't reproduce on CPU. The supported reference platform
on Apple silicon is **CPU** — be explicit:

```bash
JAX_PLATFORMS=cpu python -m tengri.bench
```

If `python -m tengri.bench` reports `default device: METAL` and your
1-galaxy timing is much slower than the [Performance page's table](../performance/index.md),
the silent CPU fallback is probably the cause; force CPU as above.

## TPU

Untested but expected to work: tengri uses no TPU-incompatible
operations (no `jax.device_put` to CPU, no Python loops in JIT'd code,
no NumPy in the forward graph). If you try it, reports welcome.

## Multi-device

`predict_photometry_batch` is `vmap`-based and stays on one device.
Multi-GPU sharding via `jax.pmap` or the newer `shard_map` is **not
yet wired up** — large catalogue fits currently run sequentially on
one GPU per Python process. If you need to fan out across devices,
the cheapest pattern today is one Python process per device, each
fitting a slice of the catalogue.

## See also

- [Performance](../performance/index.md) — CPU benchmarks; will be
  extended with GPU columns when those exist.
- [Memory expectations](../performance/memory.md) — peak RSS table
  applies to CPU; subtract ~30 % as a rough guess for VRAM.
- [JAX install docs](https://docs.jax.dev/en/latest/installation.html)
  — authoritative for CUDA versions and wheels.
