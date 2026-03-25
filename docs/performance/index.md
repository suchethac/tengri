# Performance

tengri is designed for speed. The forward model runs in hundreds of microseconds,
gradients are cheaper than function evaluations, and the entire inference pipeline
is JIT-compiled. This section covers what to expect, how to make things faster,
and how to measure performance in your own setup.

```{toctree}
:maxdepth: 1

benchmarks
inference_comparison
optimization
profiling
```
