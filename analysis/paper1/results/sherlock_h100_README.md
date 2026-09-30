# Sherlock H100 device-matrix benchmark — data provenance

Measured for `fig:gpu` under issue #2444. Committed as data rather than
transcribed into a figure script: `fig04_gpu_crossover.py` currently carries
hand-copied arrays from `bench/reports/2026-08-20_cuda_device_matrix.md`, and a
caption that names hardware the arrays did not come from is exactly the drift a
committed file prevents.

## Files

| file | contents |
|---|---|
| `sherlock_h100_batch.json` | **the one to read.** Shape C (batch sweep) shaped as a drop-in for the `FORWARD_DATA` / `GRADIENT_DATA` / `FINDING_1` / `FINDING_3` dicts, plus provenance, caveats and the per-call view |
| `sherlock_h100_device_matrix_gpu.json` | full unmodified GPU rows as the benchmark emitted them (shapes A, B, C, D, G, I) |
| `sherlock_h100_device_matrix_cpu.json` | the same for the same-node CPU arm |
| `figures/fig_*.pdf` | rendered figure set |
| `figures/device_matrix_table.csv` | every measurement as a flat table |

`sherlock_h100_batch.json` keys `forward` and `gradient` each carry `batch`,
`cpu_f64`, `cpu_f32`, `gpu_f64`, `gpu_f32` — the same names and the same shape
as the hardcoded dicts, so adopting it replaces a literal with a load.

## Hardware and stack

```
cluster   Sherlock (Stanford Research Computing)
node      sh04-01n01.int      partition gpu      slurm job 44335875
GPU       NVIDIA H100 80GB HBM3, compute capability 9.0, driver 550.163.01
          UUID GPU-1e5e0726-a3b7-ab23-aba6-5382183e69ab
CPU       Intel Xeon Platinum 8462Y+ (Sapphire Rapids), 8 cores allocated
commit    0e2bae3d4c91403c9891473a3ab7a0f9dd6d36af  (paper1/mock-joint-d44, tree clean)
JAX       0.7.0 / jaxlib 0.7.0
CUDA      12.8.0, cuDNN 9.14.0.64   (both from Sherlock's module stack)
python    3.12.1
```

**The CPU arm is the same node and the same allocation**, selected per cell with
`JAX_PLATFORMS=cpu`. A crossover measured against a different machine's CPU is
not a crossover.

JAX is pinned at 0.7.0 by the cluster, not by preference: Sherlock runs CentOS 7
(glibc 2.17), so only `manylinux2014` wheels install, and 0.7.0 is the last
jaxlib/CUDA-plugin release built that way.

## Read the caveats before quoting a number

They are in the JSON under `caveats`, and they are not boilerplate:

1. **The whole GPU sweep is launch-latency bound.** Per-*call* wall time on the
   H100 is constant to within 3.4% (f64) and 1.0% (f32) from batch 1 to batch
   2048 — 13.52 to 13.98 ms. The per-galaxy curve is therefore exactly `1/n`
   and **has not flattened** at the largest batch measured. Shape A agrees
   independently: 600,960 FLOPs in 15.4 ms is ~39 MFLOP/s on a card capable of
   tens of TFLOP/s.
2. **Do not quote an asymptotic per-galaxy cost.** 6.83 µs/galaxy (f64, batch
   2048) is a lower bound that is still falling. "Still falling at batch 2048,
   at 6.8 µs/galaxy" is supportable; "reaches X µs/galaxy" is not.
3. **f64 ≈ f32 on the H100 (ratio 1.02) is a latency result, not an FP64
   throughput result.** The same ratio on the Xeon is 1.92. The supportable
   claim is that at these batch sizes the forward model is launch-bound, so
   precision is nearly free on a datacentre GPU — *not* that the H100's FP64
   units are what make float64 affordable. That would need batches large enough
   to be compute-bound, which are not in this file.
4. **`throughput_chunked` understates the GPU.** Shape I's default
   `--chunk-size 1000` pays the 13.6 ms latency floor every 1000 galaxies
   (14.7 µs/gal) where shape C at batch 2048 already reaches 6.8. The chunk
   size is the limit there, not the card.

`aa_control` holds the spread of repeated identical measurements (1.001–1.012
on GPU, 1.007–1.11 on CPU). Ratios below that are not resolvable.

## What is measured, and what the crossover is

Reading the tables: the CPU/GPU crossover sits **between batch 128 and 512 in
float64**, and **between 512 and 2048 in float32**. Float64 crosses over
*earlier* than float32, because f64 costs the Xeon 2–3x and costs the H100
essentially nothing.

## Shapes not measured

`E` (catalog NUTS) and `H` (converged catalog fit) are absent. Both need
`blackjax>=1.6`, which requires `jax>=0.9.0`; glibc 2.17 caps jax at 0.7.0.
Unresolvable without a container, and not worked around by pinning an older
blackjax. See #2444.

## Reproducing

The harness is at `bench/slurm/` (allocation scripts) and
`bench/scripts/plot_device_matrix.py` (rendering, which reads these JSON files
and needs neither JAX nor a GPU). The measurement itself is the repository's own
`bench/scripts/benchmark_device_matrix.py`, unmodified, at its defaults.
