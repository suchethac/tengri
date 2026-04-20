"""Verify vmap MCMC batch fitting works end-to-end."""

import time
from pathlib import Path

import jax
import jax.numpy as jnp

jax.config.update("jax_enable_x64", True)

from tengri import Fitter, Parameters, SEDModel, generate_mock, load_filter_set, load_ssp_data

DATA = Path(__file__).resolve().parents[1] / "data"
ssp_file = DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not ssp_file.is_file():
    print("SSP file not found, skipping")
    raise SystemExit(0)

ssp = load_ssp_data(str(ssp_file))
filters = load_filter_set(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
spec = Parameters(mean_sfh_type="tsnorm")
model = SEDModel(spec, ssp, filters=filters)

# Create batch of 5 galaxies with different parameters
n_gal = 5
batch = []
for i in range(n_gal):
    params = spec.sample(jax.random.PRNGKey(i))
    mock = generate_mock(model, params, snr=20.0, key=jax.random.PRNGKey(100 + i))
    batch.append({"flux_obs": mock["flux_obs"], "noise": mock["noise"]})

fitter = Fitter(model, batch[0]["flux_obs"], batch[0]["noise"])

# Test vmap NUTS
print("=" * 60)
print("Testing vmap MCMC batch fitting")
print("=" * 60)

for method in ["mcmc_nuts", "mcmc_hmc", "mcmc_ghmc", "mcmc_dynamic_hmc"]:
    print(f"\n--- {method} ---")
    t0 = time.time()
    results = fitter.fit_batch(
        batch, method=method, key=jax.random.PRNGKey(42),
        n_warmup=50, n_burnin=10, n_samples=50, verbose=False,
    )
    t_total = time.time() - t0
    print(f"  {n_gal} galaxies in {t_total:.1f}s ({t_total / n_gal:.2f}s/galaxy)")

    # Verify results
    assert len(results) == n_gal, f"Expected {n_gal} results"
    for i, r in enumerate(results):
        assert "n_samples" in r.diagnostics, "Missing diagnostics"
        assert r.diagnostics["n_samples"] == 50
        assert r.diagnostics["batch_size"] == n_gal
        n_params = len(r.params)
        assert n_params > 0, f"Galaxy {i}: no params"
    print(f"  All {n_gal} posteriors valid")

# Compare: vmap vs sequential
print("\n--- Comparison: vmap NUTS vs sequential ---")

t0 = time.time()
vmap_results = fitter.fit_batch(
    batch, method="mcmc_nuts", key=jax.random.PRNGKey(42),
    n_warmup=50, n_burnin=10, n_samples=100, verbose=False,
)
t_vmap = time.time() - t0

t0 = time.time()
seq_results = []
for i, gal in enumerate(batch):
    f = Fitter(model, gal["flux_obs"], gal["noise"])
    r = f.run(
        "mcmc_nuts", key=jax.random.fold_in(jax.random.PRNGKey(42), i),
        n_warmup=50, n_burnin=10, n_samples=100, verbose=False,
    )
    seq_results.append(r)
t_seq = time.time() - t0

print(f"  vmap:       {t_vmap:.1f}s ({t_vmap / n_gal:.2f}s/galaxy)")
print(f"  sequential: {t_seq:.1f}s ({t_seq / n_gal:.2f}s/galaxy)")
speedup = t_seq / t_vmap if t_vmap > 0 else float("inf")
print(f"  Speedup: {speedup:.1f}x")

print("\nvmap MCMC batch fitting: PASS")
