#!/usr/bin/env python
"""Minimal reproducer for kitchen-sink gradient ConcretizationTypeError.

Reproduces the error from benchmark_forward_model.py:
Full traceback lands at a concrete-value check in JAX tracing.
"""

import os
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
jax.config.update("jax_enable_x64", True)

from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.sps.dsps_wrapper import load_ssp_data

# Load SSP data
SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
if not os.path.exists(SSP_PATH):
    SSP_PATH = "data/ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ssp_data = load_ssp_data(SSP_PATH)

# Minimal kitchen-sink config: every component enabled
kitchen_sink_kwargs = dict(
    nebular_ssp=True,
    dust_emission="themis",
    dust_qpah=Fixed(2.5),
    dust_umin=Fixed(1.0),
    agn_model="kubota_done_full",
    agn_log_lbol=Fixed(10.0),
    radio=True,
    xray=True,
    radio_q_ir=Fixed(2.64),
)

# Build minimal model  on stochastic field SFH
obs = Observation(
    photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
)
base_kwargs = dict(
    met_logzsol=Uniform(-2, 0.2),
    dust_tau_bc=Uniform(0, 2),
    dust_tau_diff=Uniform(0, 1.5),
    dust_slope=Fixed(-0.7),
    redshift=Fixed(0.1),
    mean_sfh_type=["dense_basis", "field"],
    sfh_dbp_log_total_mass=Uniform(8, 12),
    sfh_dbp_tx_frac_0=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_1=Uniform(0.05, 0.95),
    sfh_dbp_tx_frac_2=Uniform(0.05, 0.95),
    sfh_field_psd_sigma=Uniform(0.1, 2.0),
    sfh_field_psd_tau_myr=Uniform(10, 1000),
)
base_kwargs.update(kitchen_sink_kwargs)
spec = Parameters(**base_kwargs)
model = SEDModel(spec, ssp_data, observation=obs, approx=None)
params = spec.sample(jax.random.PRNGKey(42))

print("Built kitchen-sink model. Attempting gradient over free-parameter vector...")
print(f"Model: {type(model).__name__}")
print(f"Params keys: {list(params.keys())}")
print(f"Free params: {list(spec.free_params)}")

# Split params into free and fixed
free_param_names = set(spec.free_params)
free_params = {k: v for k, v in params.items() if k in free_param_names}
fixed_params = {k: v for k, v in params.items() if k not in free_param_names}

print(f"Free params subdict: {list(free_params.keys())}")
print(f"Fixed params subdict: {list(fixed_params.keys())}")

# Attempt the gradient: differentiate loss over free params only
def loss_fn(p_free):
    # Merge free params with fixed params for full prediction
    p_full = {**fixed_params, **p_free}
    return jnp.sum(model.predict_photometry(p_full))

try:
    grad_fn = jax.jit(jax.grad(loss_fn))
    result = grad_fn(free_params)
    print("Gradient succeeded!")
    print(f"Result type: {type(result)}")
except Exception as exc:
    print(f"\nGradient FAILED with {type(exc).__name__}:")
    print(f"{exc}")
    import traceback
    traceback.print_exc()
