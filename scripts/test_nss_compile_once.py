"""Verify NSS compile-once: second galaxy should NOT retrace."""

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

# Galaxy A: young, dusty
params_a = spec.sample(jax.random.PRNGKey(0))
mock_a = generate_mock(model, params_a, snr=20.0, key=jax.random.PRNGKey(1))

# Galaxy B: different
params_b = spec.sample(jax.random.PRNGKey(42))
mock_b = generate_mock(model, params_b, snr=15.0, key=jax.random.PRNGKey(43))

# NSS Galaxy A (cold)
fitter_a = Fitter(model, mock_a["flux_obs"], mock_a["noise"])
t0 = time.time()
result_a = fitter_a.run(
    "nss", key=jax.random.PRNGKey(10),
    n_live=50, num_delete=5, max_iterations=20, verbose=False,
)
t_cold = time.time() - t0
print(f"NSS Galaxy A (cold):  {t_cold:.1f}s  logZ={result_a.log_evidence:.2f}")

# NSS Galaxy B (should be cached)
fitter_b = Fitter(model, mock_b["flux_obs"], mock_b["noise"])
t0 = time.time()
result_b = fitter_b.run(
    "nss", key=jax.random.PRNGKey(20),
    n_live=50, num_delete=5, max_iterations=20, verbose=False,
)
t_cached = time.time() - t0
print(f"NSS Galaxy B (cached): {t_cached:.1f}s  logZ={result_b.log_evidence:.2f}")

speedup = t_cold / t_cached if t_cached > 0 else float("inf")
print(f"Speedup: {speedup:.1f}x")

# Verify algo was cached on SEDModel
cache = getattr(model, "_nss_algo_cache", {})
print(f"NSS algo cache entries: {len(cache)}")

assert len(cache) == 1, f"Expected 1 cached algo, got {len(cache)}"
assert result_a.log_evidence != 0.0, "logZ should be non-zero"
assert result_b.log_evidence != 0.0, "logZ should be non-zero"
print("\nNSS compile-once: PASS")
