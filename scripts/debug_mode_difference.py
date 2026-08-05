"""Debug why mode='auto' and mode='_traceable' give different results."""

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

# D=11 model
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

# Get loss function
loss_fn = fitter._get_or_build_loss_fn()
data_args = fitter._data_args

# Initialize parameters
rng_key = jax.random.PRNGKey(42)
init_params = fitter._initialize_unbounded(rng_key)

print("=" * 70)
print("DEBUGGING MODE DIFFERENCES")
print("=" * 70)

# Compare predictions using the same physical parameters
print("\nTest 1: Compare forward model predictions at same physical params")
print("-" * 70)

# Get a physical parameter dict from the init
free_names = fitter._free_names
bounds = fitter._bounds


def to_bounded(xi, lo, hi):
    z = jax.nn.sigmoid(xi)
    return lo + (hi - lo) * z


physical_params = {}
for name in free_names:
    lo, hi = bounds[name]
    physical_params[name] = to_bounded(init_params[name], lo, hi)

print(f"Testing with physical parameters:")
for name, val in physical_params.items():
    if jnp.ndim(val) == 0:
        print(f"  {name}: {float(val):.4f}")
    else:
        print(f"  {name}: array shape {val.shape}, mean={jnp.mean(val):.4f}")

# Predict with different modes
pred_traceable = model.predict_photometry(physical_params, mode="_traceable")
pred_auto = model.predict_photometry(physical_params, mode="auto")
pred_exact = model.predict_photometry(physical_params, mode="exact")
pred_compositional = model.predict_photometry(physical_params, mode="compositional")

print(f"\nForward model predictions (photometry):")
print(f"  _traceable:    {pred_traceable}")
print(f"  auto:          {pred_auto}")
print(f"  exact:         {pred_exact}")
print(f"  compositional: {pred_compositional}")

print(f"\nDifferences:")
print(f"  _traceable vs auto:          {jnp.max(jnp.abs(pred_traceable - pred_auto)):.2e}")
print(f"  _traceable vs exact:         {jnp.max(jnp.abs(pred_traceable - pred_exact)):.2e}")
print(
    f"  _traceable vs compositional: {jnp.max(jnp.abs(pred_traceable - pred_compositional)):.2e}"
)
print(f"  auto vs compositional:       {jnp.max(jnp.abs(pred_auto - pred_compositional)):.2e}")

# Check chi-squared
chi2_traceable = jnp.sum(((pred_traceable - flux) / flux_unc) ** 2)
chi2_auto = jnp.sum(((pred_auto - flux) / flux_unc) ** 2)
chi2_exact = jnp.sum(((pred_exact - flux) / flux_unc) ** 2)

print(f"\nChi-squared:")
print(f"  _traceable:    {chi2_traceable:.4f}")
print(f"  auto:          {chi2_auto:.4f}")
print(f"  exact:         {chi2_exact:.4f}")
print(f"  Difference (auto - _traceable): {chi2_auto - chi2_traceable:.4f}")

print("\n" + "=" * 70)
print("Test 2: Check which kernel path is being used")
print("-" * 70)

# Check if hybrid or compositional kernels are built
print(f"\nModel kernel status:")
print(f"  Has _SEDModel__hybrid: {hasattr(model, '_SEDModel__hybrid')}")
print(f"  Has _SEDModel__compositional: {hasattr(model, '_SEDModel__compositional')}")

if hasattr(model, "_SEDModel__hybrid"):
    hybrid = model._SEDModel__hybrid
    print(f"  Hybrid is None: {hybrid is None}")
    if hybrid is not None:
        print(f"  Hybrid has _photometry_raw: {hasattr(hybrid, '_photometry_raw')}")

if hasattr(model, "_SEDModel__compositional"):
    comp = model._SEDModel__compositional
    print(f"  Compositional is None: {comp is None}")
    if comp is not None:
        print(f"  Compositional has _photometry_raw: {hasattr(comp, '_photometry_raw')}")

print("\n" + "=" * 70)
print("CONCLUSION")
print("=" * 70)

if jnp.max(jnp.abs(pred_traceable - pred_auto)) < 1e-6:
    print("✓ Forward model predictions are IDENTICAL")
    print("  The numerical difference in the experiment is NOT from the forward model")
    print("  It must be from the parameter transformation or prior computation")
else:
    print("⚠ Forward model predictions DIFFER")
    print("  This explains the numerical differences in the experiment")
    print(f"  Max difference: {jnp.max(jnp.abs(pred_traceable - pred_auto)):.2e}")

print("=" * 70)
