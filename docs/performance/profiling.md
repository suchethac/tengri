# Profiling

How to measure performance in your own setup and identify bottlenecks.

## Quick timing with `%timeit`

In a Jupyter notebook or IPython session, use `%timeit` to measure individual
operations after JIT warmup:

```python
import jax
import jax.numpy as jnp
from tengri import Model, ParamSpec, Uniform, load_ssp_data, load_filter_set

ssp = load_ssp_data("data/ssp.h5")
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
spec = ParamSpec(
    sfh_dpl_alpha=Uniform(0.5, 3.0),
    sfh_dpl_beta=Uniform(0.5, 3.0),
    sfh_dpl_tau_gyr=Uniform(0.5, 13.0),
    sfh_dpl_log_peak_sfr=Uniform(-1.0, 2.5),
    met_logzsol=Uniform(-2.0, 0.5),
    dust_tau_bc=Uniform(0.0, 2.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_slope=-0.7,
    redshift=0.1,
    mean_sfh_type="dpl",
)
model = Model(spec, ssp, filters=filters)

# Sample parameters and warm up the JIT
key = jax.random.PRNGKey(0)
params = spec.sample(key)
_ = model.predict_photometry(params)  # trigger compilation

# Time the forward model
%timeit model.predict_photometry(params)

# Time the gradient
grad_fn = jax.grad(lambda p: model.predict_photometry(p).sum())
_ = grad_fn(params)  # warm up
%timeit grad_fn(params)
```

:::{important}
Always call the function once before timing to ensure JIT compilation has completed.
JAX operations are asynchronous --- use `jax.block_until_ready()` if you are timing
outside of `%timeit` (which handles this automatically).
:::

## Manual timing with `block_until_ready`

For scripts where `%timeit` is not available:

```python
import time
import jax

def bench(fn, *args, n=200, warmup=3):
    """Benchmark a JAX function with proper synchronization."""
    for _ in range(warmup):
        jax.block_until_ready(fn(*args))

    t0 = time.perf_counter()
    for _ in range(n):
        jax.block_until_ready(fn(*args))
    elapsed = time.perf_counter() - t0

    return elapsed / n * 1e6  # microseconds

us = bench(model.predict_photometry, params)
print(f"Forward model: {us:.0f} us")
```

:::{warning}
Without `jax.block_until_ready()`, JAX returns a future immediately and your timing
will measure dispatch overhead, not actual computation time.
:::

## Component-level profiling

The `analysis/profile_all_components.py` script profiles every model component
individually --- SFH, dust, metallicity interpolation, SPS, filters --- and reports
both wall-clock time and array memory:

```bash
cd ~/Projects/tengri
python analysis/profile_all_components.py
```

This produces a table like:

```
Component                    Time (us)   Memory (MB)
─────────────────────────────────────────────────────
SFH computation                   73        0.01
SFR interpolation                 49        0.01
CSP weights (trapezoid)            3        0.01
Metallicity interpolation        209        0.44
Dust attenuation               1700       41.00
CSP SED einsum                   506        0.04
Photometric integration          197        0.34
```

## Inference benchmarks

To compare inference methods (MAP, Ray Tracing, NUTS, geoVI), run the paper
benchmark script:

```bash
python analysis/fig07_speed_benchmarks.py --n-repeats 3
```

This measures wall-clock time for each method on both smooth (D=7) and stochastic
(D=137) models and produces a grouped bar chart.

## JAX profiling tools

### XLA compilation log

To see what XLA is compiling and how long it takes:

```python
import jax
jax.config.update("jax_log_compiles", True)

# Now any JIT compilation will print to stderr:
# Compiling predict_photometry for args (ShapedArray(float64[7]),)
model.predict_photometry(params)
```

### TensorBoard profiling

For detailed traces of device computation:

```python
import jax

# Start the profiler
jax.profiler.start_trace("/tmp/tengri_profile")

# Run your computation
for _ in range(100):
    jax.block_until_ready(model.predict_photometry(params))

# Stop and save
jax.profiler.stop_trace()
```

Then view in TensorBoard:

```bash
pip install tensorboard-plugin-profile
tensorboard --logdir /tmp/tengri_profile
```

### XLA HLO dump

To inspect the optimized computation graph:

```bash
XLA_FLAGS="--xla_dump_to=/tmp/xla_dump" python -c "
import jax
from tengri import Model, ParamSpec, load_ssp_data, load_filter_set
# ... set up model and run predict_photometry
"
```

The dump directory will contain `.txt` files with the HLO intermediate representation
before and after optimization passes.

## Memory considerations

### Large models

If you are fitting models with many components (AGN + nebular + dust IR), memory
usage can grow. Key strategies:

- Use `forward_dtype="float32"` to halve SSP memory (66.9 MB to 33.5 MB)
- Photometry precomputation reduces the working set from `(15, 93, 5994)` to
  `(15, 93, n_filters)` --- typically a 1000x reduction
- For spectroscopy, precomputation reduces the wavelength axis to only the observed
  pixels

### Batch fitting

When fitting catalogs with `fitter.fit_batch(galaxies)`, memory scales linearly with
the number of galaxies being fit in parallel. For very large catalogs, the batch
fitter processes galaxies sequentially by default.

### Monitoring memory

```python
import jax

# Check current device memory usage
for device in jax.devices():
    stats = device.memory_stats()
    if stats:
        used = stats["bytes_in_use"] / 1e6
        print(f"{device}: {used:.1f} MB in use")
```

## Checklist: diagnosing slow fits

1. **Is JIT compilation the bottleneck?** Check if the first call is slow but
   subsequent calls are fast. If so, the XLA cache should help across sessions.

2. **Is the redshift fixed?** A free redshift disables photometry precomputation and
   fused kernels, which can make the forward model 20x slower.

3. **Are you using the exact path unnecessarily?** If you only need photometry,
   make sure filters are provided at Model init so precomputation activates.

4. **Is float64 needed?** Switch to `forward_dtype="float32"` for a ~1.5x speedup
   with negligible accuracy loss.

5. **Is the compilation cache working?** Check that `/tmp/tengri_jax_cache` exists
   and is growing. Clear it if you have upgraded JAX.

6. **Are you running on CPU?** tengri is tested on CPU. GPU support via JAX is
   available but experimental (JAX Metal on Apple Silicon has known issues ---
   use `JAX_PLATFORMS=cpu` for reliable results).
