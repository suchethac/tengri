# SLURM scripts — MCMC catalog fitting on a GPU

Fit a catalog of galaxies with NUTS **in parallel on a GPU**. `CatalogFitter`
vmaps `forward_chunk_size` galaxies per step, so they sample simultaneously on
the card. Each galaxy is fit independently; you get a posterior per galaxy.

These are general templates — set three environment variables and submit.

## What you provide

- **`CATALOG`** — an `.npz` with `flux_obs` shape `(N, n_band)` and `noise`
  shape `(N, n_band)` (per-band 1σ errors).
- **`MODEL_BUILDER`** — `"module:function"`, an importable zero-arg callable
  returning a built `SEDModel` whose observation matches the catalog's bands:

  ```python
  # myfit.py  (on your PYTHONPATH / in the repo)
  def build_model():
      from tengri import SEDModel, Observation, Photometry, Parameters, Uniform, Fixed
      from tengri.sps.dsps_wrapper import load_ssp_data
      ssp = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
      obs = Observation(photometry=Photometry.from_names(["sdss_u","sdss_g","sdss_r","sdss_i","sdss_z"]))
      spec = Parameters(sfh_dpl_log_total_mass=Uniform(-1, 3), met_logzsol=Uniform(-2, 0.2),
                        dust_tau_diff=Uniform(0, 2), redshift=Fixed(0.1), mean_sfh_type="dpl")
      return SEDModel(spec, ssp, observation=obs)
  ```

- **`OUTDIR`** — where shards are written.

Optional (with defaults): `METHOD=mcmc_nuts`, `CHUNK=64` (galaxies in parallel),
`N_WARMUP=300`, `N_SAMPLES=1000`, `TENGRI_REPO`, `TENGRI_VENV`.

## One GPU (the simple case)

```bash
CATALOG=cat.npz MODEL_BUILDER=myfit:build_model OUTDIR=out CHUNK=256 \
    sbatch fit_catalog.sbatch
```

One GPU fits the whole catalog. Raise `CHUNK` until you fill the card
(bigger CHUNK = more galaxies in flight = higher throughput, until VRAM caps).
Result: `out/shard_00000.npz` — per-galaxy posterior summaries
(`stellar_mass_mean`, `..._p16/p50/p84`, etc.), keyed by `global_index`.

## Many GPUs (large catalogs)

Split the catalog across GPUs — one array task per GPU, spread over whatever
nodes are free:

```bash
CATALOG=cat.npz MODEL_BUILDER=myfit:build_model OUTDIR=out \
    sbatch --array=0-15%8 fit_catalog_array.sbatch      # 16 slices, <=8 at once
python merge_shards.py --shards out --out out/catalog_posteriors.npz
```

The array size is the number of slices; `%8` caps concurrent tasks. Each task
reads `SLURM_ARRAY_TASK_ID` / `SLURM_ARRAY_TASK_COUNT` automatically.

## Sherlock notes

- **Install the CUDA wheel** in your venv: `pip install -U "jax[cuda12]"`.
- **Pick a GPU** with the `#SBATCH --constraint` line: `GPU_SKU:H100_SXM5`,
  `GPU_SKU:A100_SXM4`, `GPU_SKU:V100_PCIE`, or by memory `GPU_MEM:80GB`. List
  what's available with `node_feat -p gpu | grep GPU_`. Partitions: `gpu`
  (shared), `owners` (your PI's), `dev` (`sh_dev -g 1` for a quick interactive test).
- **Caches on `$SCRATCH`** (`env.sh` sets `TENGRI_JAX_CACHE_DIR` /
  `TENGRI_PRECOMP_CACHE_DIR`): the first task compiles the kernels, the rest
  load them in milliseconds. For a large array, run one task to completion first
  so the shared cache is warm before the swarm starts.

## Files

| File | What it does |
|------|--------------|
| `fit_catalog.sbatch` | One GPU, whole catalog, vectorized NUTS. Start here. |
| `fit_catalog_array.sbatch` | Array job: one slice per GPU, scales across nodes. |
| `run_catalog_slice.py` | The worker both sbatch scripts call. Runnable standalone. |
| `merge_shards.py` | Concatenate array shards into one ordered table. |
| `env.sh` | Sourced setup: venv, `$SCRATCH` caches, GPU memory knobs. |
