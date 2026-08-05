"""Diagnose MCMC performance bottleneck.

Quick test to measure:
1. Single model evaluation time
2. Single gradient evaluation time
3. NUTS iteration overhead
"""

import time

import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree

from tengri import (
    Fitter,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    load_ssp_data,
)
from tengri.observation.filters import load_filter_set
from tengri.parameters.priors import Fixed, Uniform

jax.config.update("jax_enable_x64", True)

# Load data
filter_names = [
    "hst_f435w",
    "hst_f606w",
    "hst_f775w",
    "hst_f814w",
    "hst_f850lp",
    "hst_f125w",
    "hst_f140w",
    "hst_f160w",
    "vista_ks",
    "irac_36",
    "irac_45",
]
filter_set = load_filter_set(filter_names)
obs = Observation(photometry=Photometry.from_filter_set(filter_set))

# Mock data
flux = jnp.ones(11) * 1e-26
flux_unc = flux / 10.0

# D=11 model (same as D1 test)
params = Parameters(
    mean_sfh_type=["dense_basis", "field"],
    sfh_dbp_log_total_mass=Uniform(9.0, 12.0),
    sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
    sfh_field_psd_sigma=Uniform(0.1, 3.0),
    sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
    met_logzsol=Uniform(-2.0, 0.2),
    dust_law_bc="salim_sbl18",
    dust_tau_bc=Uniform(0.0, 3.0),
    dust_tau_diff=Uniform(0.0, 2.0),
    dust_model="two_component",
    dust_emission="draine_li2007",
    dust_umin=Fixed(1.0),
    dust_gamma_dl=Uniform(0.0, 0.1),
    dust_qpah=Uniform(0.5, 4.5),
    nebular_ssp=True,
    apply_igm=True,
    redshift=Fixed(1.0),
    n_grid=64,
)

ssp_data = load_ssp_data("data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5")
model = SEDModel(params, ssp_data, observation=obs)
fitter = Fitter(model, data=flux, noise=flux_unc)

print(f"Free parameters: {len(params.free_params)}")
print(f"Parameter names: {params.free_params}")

# Get loss function (same as MCMC uses)
loss_fn = fitter._get_or_build_loss_fn()
data_args = fitter._data_args

# Initialize parameters
rng_key = jax.random.PRNGKey(42)
init_params = fitter._initialize_unbounded(rng_key)
init_flat, unravel_fn = ravel_pytree(init_params)


# Create log prob function (negative loss)
def log_prob_flat(position):
    params = unravel_fn(position)
    return -loss_fn(params, data_args)


print(f"\nModel configuration:")
print(f"  Number of free params (D): {len(init_flat)}")

print("\n1. Measuring single log_prob evaluation time...")
jax.clear_caches()
t0 = time.perf_counter()
log_p = log_prob_flat(init_flat)
t1 = time.perf_counter()
print(f"   First call (with JIT): {(t1 - t0) * 1000:.1f} ms")
print(f"   Log prob value: {log_p:.2f}")

# Warmup
for _ in range(3):
    _ = log_prob_flat(init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01)

t0 = time.perf_counter()
log_p = log_prob_flat(init_flat)
t1 = time.perf_counter()
print(f"   After warmup: {(t1 - t0) * 1000:.3f} ms")

print("\n2. Measuring gradient evaluation time...")
grad_fn = jax.grad(log_prob_flat)

jax.clear_caches()
t0 = time.perf_counter()
grad = grad_fn(init_flat)
t1 = time.perf_counter()
print(f"   First call (with JIT): {(t1 - t0) * 1000:.1f} ms")

# Warmup
for _ in range(3):
    _ = grad_fn(init_flat + jax.random.normal(rng_key, (len(init_flat),)) * 0.01)

t0 = time.perf_counter()
grad = grad_fn(init_flat)
t1 = time.perf_counter()
print(f"   After warmup: {(t1 - t0) * 1000:.3f} ms")
print(f"   Gradient norm: {jnp.linalg.norm(grad):.2e}")

print("\n3. Measuring 100 sequential evaluations...")
t0 = time.perf_counter()
for i in range(100):
    theta_i = (
        init_flat + jax.random.normal(jax.random.fold_in(rng_key, i), (len(init_flat),)) * 0.01
    )
    _ = log_prob_flat(theta_i)
t1 = time.perf_counter()
print(f"   Total time: {(t1 - t0) * 1000:.1f} ms")
print(f"   Per evaluation: {(t1 - t0) * 1000 / 100:.3f} ms")

print("\n4. Estimating NUTS time for 20 warmup + 20 samples...")
# NUTS typically does ~10-15 gradient evaluations per iteration
n_iterations = 40
grads_per_iter = 12
total_grad_evals = n_iterations * grads_per_iter
estimated_time = total_grad_evals * (t1 - t0) / 100.0
print(f"   Estimated gradient evaluations: {total_grad_evals}")
print(f"   Estimated total time: {estimated_time:.1f} seconds")

print("\n5. Checking model prediction overhead...")
# Direct model call vs loss function
theta_dict = unravel_fn(init_flat)
t0 = time.perf_counter()
for _ in range(10):
    _ = model.predict_rest_sed(theta_dict)
t1 = time.perf_counter()
print(f"   10 model.predict_rest_sed() calls: {(t1 - t0) * 1000:.1f} ms")
print(f"   Per call: {(t1 - t0) * 1000 / 10:.3f} ms")
