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
Multi-GPU sharding via `jax.pmap` or `shard_map` is not yet wired.
Large catalog fits run sequentially on one GPU per Python process. To fan
out across devices, use one Python process per device, each fitting a
catalog slice.

## See also

- [Performance](../performance/index.md) — CPU benchmarks.
- [Memory expectations](../performance/memory.md) — peak RSS table,
  measured on CPU.
- [JAX install docs](https://docs.jax.dev/en/latest/installation.html)
  — authoritative for CUDA versions and wheels.
