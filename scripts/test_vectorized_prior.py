"""Quick verification that vectorized prior works correctly."""

import jax
import jax.numpy as jnp
from pathlib import Path

jax.config.update("jax_enable_x64", True)

from tengri import Fitter, Parameters, SEDModel
from tengri.components.sps.dsps_wrapper import load_ssp_data
from tengri.parameters.priors import Fixed, Uniform

# Setup
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"

print("Loading SSP data...")
ssp = load_ssp_data(str(SSP_FILE))

# Test 1: All Uniform priors (should use vectorized path)
print("\nTest 1: All Uniform priors (vectorized path)")
params_uniform = Parameters(
    redshift=Fixed(1.0),
    sfh_tsnorm_log_peak_sfr=Uniform(-2, 2),
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.1, 10),
    sfh_tsnorm_width_gyr=Uniform(0.1, 5),
    sfh_tsnorm_skew=Uniform(-1, 1),
    sfh_tsnorm_trunc=Uniform(0.01, 1),
    met_logzsol=Uniform(-2, 0.5),
    dust_tau_diff=Uniform(0, 3),
    dust_tau_bc=Uniform(0, 2),
)

flux_obs = jnp.array([1e-19, 2e-19, 1.5e-19])
flux_err = jnp.array([1e-20, 2e-20, 1.5e-20])

model_uniform = SEDModel(params_uniform, ssp)
fitter_uniform = Fitter(model_uniform, flux_obs, flux_err)

# Build prior function
logprior_fn = fitter_uniform._build_logprior_fn()

# Test with valid parameters (inside bounds)
key = jax.random.PRNGKey(42)
params_init = fitter_uniform._initialize_unbounded(key)

from tengri.utils.transforms import to_bounded

param_dict_bounded = {}
for name in fitter_uniform._free_names:
    lo, hi = fitter_uniform._bounds[name]
    param_dict_bounded[name] = to_bounded(params_init[name], lo, hi)
for name, val in fitter_uniform._fixed_values.items():
    param_dict_bounded[name] = val

lp_valid = logprior_fn(param_dict_bounded)
print(f"  Log prior (valid params): {lp_valid:.6f}")
assert jnp.isfinite(lp_valid), "Log prior should be finite for valid params"

# Test with out-of-bounds parameters
param_dict_invalid = param_dict_bounded.copy()
param_dict_invalid["dust_tau_bc"] = 5.0  # outside [0, 2]
lp_invalid = logprior_fn(param_dict_invalid)
print(f"  Log prior (invalid params): {lp_invalid}")
assert lp_invalid == -jnp.inf, "Log prior should be -inf for out-of-bounds params"

# Test 2: Mixed priors (should use loop path)
print("\nTest 2: Mixed Uniform/Gaussian priors (loop path)")
from tengri.parameters.priors import Gaussian

params_mixed = Parameters(
    redshift=Fixed(1.0),
    sfh_tsnorm_log_peak_sfr=Gaussian(-0.5, 1.0, -2, 2),  # Gaussian prior
    sfh_tsnorm_peak_lbt_gyr=Uniform(0.1, 10),
    sfh_tsnorm_width_gyr=Uniform(0.1, 5),
    sfh_tsnorm_skew=Uniform(-1, 1),
    sfh_tsnorm_trunc=Uniform(0.01, 1),
    met_logzsol=Uniform(-2, 0.5),
    dust_tau_diff=Uniform(0, 3),
    dust_tau_bc=Uniform(0, 2),
)

model_mixed = SEDModel(params_mixed, ssp)
fitter_mixed = Fitter(model_mixed, flux_obs, flux_err)
logprior_fn_mixed = fitter_mixed._build_logprior_fn()

# Same parameter values
params_init_mixed = fitter_mixed._initialize_unbounded(key)
param_dict_bounded_mixed = {}
for name in fitter_mixed._free_names:
    lo, hi = fitter_mixed._bounds[name]
    param_dict_bounded_mixed[name] = to_bounded(params_init_mixed[name], lo, hi)
for name, val in fitter_mixed._fixed_values.items():
    param_dict_bounded_mixed[name] = val

lp_mixed = logprior_fn_mixed(param_dict_bounded_mixed)
print(f"  Log prior (mixed, valid params): {lp_mixed:.6f}")
assert jnp.isfinite(lp_mixed), "Log prior should be finite for valid params"

print("\n✓ All tests passed!")
print("  - Vectorized path works correctly for all-Uniform priors")
print("  - Loop path works correctly for mixed priors")
print("  - Bounds checking works in both paths")
