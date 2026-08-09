"""
Dust energy balance: L_IR = L_UV_absorbed across opacity variations
===================================================================

A cornerstone of dust modeling is energy conservation: the UV light
absorbed by dust must be re-radiated in the infrared. This example
constructs 15 tengri SEDModels with optical depth τ_V ∈ {0, 0.1, ..., 4}
and validates that integrated infrared luminosity (8–1000 μm) matches
the absorbed UV (912–3000 Å rest-frame).

The test is performed at z = 0.05 with star-forming galaxy kinematics
(tsnorm SFH peaking at 0.3 Gyr lookback time) and a two-component dust
model (Calzetti attenuation + Dale14 IR emission).

**Key Physics:**
L_UV_absorbed = L_UV_intrinsic − L_UV_attenuated

For any self-consistent dust model:
L_IR ≈ L_UV_absorbed (within ~5% when using tabulated Dale14 templates)

Non-conservation flags calibration issues in the dust emission routing.

Reference: da Cunha et al. 2008, MNRAS, 388, 1595 (energy-balance principle).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    load_ssp,
)
from tengri.analysis.plotting import setup_style

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

setup_style()

# ============================================================================
# 1. Setup: bare-stellar SSP and wavelength grid for integration
# ============================================================================

# Load a bare-stellar SSP (required for Cue nebular). `load_ssp` walks parent
# directories for `data/`, so it finds the committed grid whatever the working
# directory is — the gallery runner executes each example from its own folder.
ssp = load_ssp("fsps_prsc_miles_chabrier")

# Redshift
z = 0.05

# Rest-frame wavelength grid for integration (high resolution)
# Extend to 1000 μm = 1e7 Å for full IR coverage
wave_rest = np.logspace(2.0, 7.0, 3000)  # 100 Å – 1000 μm rest-frame

# Integration bounds: UV (912–3000 Å) and IR (8–1000 μm)
wave_uv_min, wave_uv_max = 912.0, 3000.0
wave_ir_min, wave_ir_max = 8e4, 1e7  # 8 μm – 1000 μm in Angstroms

# ============================================================================
# 2. Observation setup: minimal photometry (broadband for SED anchoring)
# ============================================================================

filters = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "irac_36",
    "irac_45",
    "irac_58",
    "irac_80",
    "mips_24",
    "mips_70",
    "herschel_100",
    "herschel_160",
]

obs = Observation(photometry=Photometry.from_names(filters))

# ============================================================================
# 3. Optical depth grid and model construction
# ============================================================================

tau_v_grid = np.array(
    [0.0, 0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 0.15, 0.75, 2.25, 3.75]
)  # 15 values total

lir_grid = []  # Integrated IR luminosity (8–1000 μm)
luv_absorbed_grid = []  # Absorbed UV luminosity

# Dummy parameters for reproducibility
key = jax.random.PRNGKey(42)

for tau_v in tau_v_grid:
    # ========================================================================
    # Build the intrinsic (dust-free) SED
    # ========================================================================
    model_intrinsic = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(z),
        sfh={"type": "tsnorm", "all_params": FIXED, "peak_lbt_gyr": 0.3},
        dust={"type": "single_component", "law_bc": "calzetti", "all_params": FIXED, "tau_v": 0.0},
        neb={"type": "cue", "all_params": FIXED},
    )

    # Generate intrinsic SED on rest-frame wavelength grid
    # Sample params and use all fixed values
    params_intrinsic = model_intrinsic.spec.sample(key)
    result_intrinsic = model_intrinsic.predict(params_intrinsic)
    sed_intrinsic_np = np.asarray(result_intrinsic.rest_sed(wave_rest))

    # Integrate UV: 912–3000 Å
    # Convert L_nu to luminosity: ∫ L_nu dν = ∫ L_nu * (c/λ²) dλ
    # where c = 3e10 cm/s and λ is in Angstroms
    mask_uv = (wave_rest >= wave_uv_min) & (wave_rest <= wave_uv_max)
    c_cgs = 2.99792458e10  # cm/s
    integrand_uv_intrinsic = sed_intrinsic_np[mask_uv] * c_cgs / (wave_rest[mask_uv] ** 2)
    luv_intrinsic = float(np.trapezoid(integrand_uv_intrinsic, wave_rest[mask_uv]))

    # ========================================================================
    # Build the dust model with current tau_v
    # ========================================================================
    model_dust = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(z),
        sfh={"type": "tsnorm", "all_params": FIXED, "peak_lbt_gyr": 0.3},
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "all_params": FIXED,
            "tau_bc": tau_v,
            "emission": {"type": "dale2014", "all_params": FIXED},
        },
        neb={"type": "cue", "all_params": FIXED},
    )

    # Generate attenuated SED
    params_dust = model_dust.spec.sample(key)
    result_attenuated = model_dust.predict(params_dust)
    sed_attenuated_np = np.asarray(result_attenuated.rest_sed(wave_rest))

    # Integrate UV from attenuated SED
    integrand_uv_attenuated = sed_attenuated_np[mask_uv] * c_cgs / (wave_rest[mask_uv] ** 2)
    luv_attenuated = float(np.trapezoid(integrand_uv_attenuated, wave_rest[mask_uv]))

    # Absorbed UV = intrinsic − attenuated
    luv_absorbed = luv_intrinsic - luv_attenuated

    # ========================================================================
    # Extract dust IR emission from the attenuated model
    # ========================================================================
    # The SED returned by predict_panchromatic() is the sum of all components.
    # Dust IR was added by the two_component model.
    # The tail (λ > 3000 Å) is dominated by dust emission + nebular continuum.
    # For a clean IR-only measurement, subtract the intrinsic at long λ.

    # Dust IR SED ≈ (attenuated SED) − (intrinsic SED at long λ)
    # At λ > 3000 Å, intrinsic is negligible (stellar + nebular tail),
    # so sed_attenuated ≈ dust_ir + small_continuum.
    # For energy balance, we integrate the IR part of the attenuated SED.

    mask_ir = (wave_rest >= wave_ir_min) & (wave_rest <= wave_ir_max)
    sed_ir_total_np = sed_attenuated_np[mask_ir]

    # Subtract intrinsic (which should be tiny in the IR, but do it for rigor)
    sed_ir_dust = sed_ir_total_np - sed_intrinsic_np[mask_ir]

    # Make sure we don't have numerical negatives (cap at 0)
    sed_ir_dust = np.maximum(sed_ir_dust, 0.0)

    # Integrate IR: 8–1000 μm
    integrand_ir = sed_ir_dust * c_cgs / (wave_rest[mask_ir] ** 2)
    lir = float(np.trapezoid(integrand_ir, wave_rest[mask_ir]))

    lir_grid.append(lir)
    luv_absorbed_grid.append(luv_absorbed)

    print(
        f"τ_V = {tau_v:5.2f}: "
        f"L_UV_absorbed = {luv_absorbed:.4e} erg/s, "
        f"L_IR = {lir:.4e} erg/s, "
        f"ratio = {lir / (luv_absorbed + 1e-20):.3f}"
    )

lir_grid = np.array(lir_grid)
luv_absorbed_grid = np.array(luv_absorbed_grid)
tau_v_grid = np.array(tau_v_grid)

# ============================================================================
# 4. Plotting
# ============================================================================

fig, ax = plt.subplots(figsize=(6.5, 5.5))

# Plot data points
ax.scatter(
    luv_absorbed_grid,
    lir_grid,
    s=80,
    alpha=0.7,
    color="C0",
    edgecolors="black",
    linewidth=0.5,
)

# Overlay the y = x line (perfect energy conservation)
lum_range = np.logspace(
    np.log10(min(luv_absorbed_grid.min(), lir_grid.min())),
    np.log10(max(luv_absorbed_grid.max(), lir_grid.max())),
    100,
)
ax.plot(lum_range, lum_range, "r--", lw=1.5, label="Perfect balance (y=x)", alpha=0.8)

# Shaded regions for ±10% tolerance
ax.fill_between(
    lum_range, lum_range * 0.9, lum_range * 1.1, alpha=0.15, color="green", label="±10% tolerance"
)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$L_{\mathrm{UV,absorbed}}$ [erg/s]", fontsize=12)
ax.set_ylabel(r"$L_{\mathrm{IR}}$ [8–1000 μm, erg/s]", fontsize=12)
ax.legend(loc="lower right", fontsize=10)
ax.grid(True, alpha=0.3, which="both")

fig.tight_layout()
plt.savefig("plot_usecase_uv_to_ir_bolometric_balance.png", dpi=150, bbox_inches="tight")

# ============================================================================
# 5. Energy balance validation
# ============================================================================

# Check for significant non-conservation
ratio = lir_grid / (luv_absorbed_grid + 1e-20)
mean_ratio = np.mean(ratio)
std_ratio = np.std(ratio)
max_deviation = np.max(np.abs(1.0 - ratio))

print("\n" + "=" * 70)
print("ENERGY BALANCE SUMMARY")
print("=" * 70)
print(f"Mean L_IR / L_UV_absorbed ratio: {mean_ratio:.4f} ± {std_ratio:.4f}")
print(f"Max deviation from y=x: {max_deviation * 100:.2f}%")
print(f"Models within ±10%: {np.sum(np.abs(1.0 - ratio) < 0.1)} / {len(ratio)}")
print(f"Models within ±20%: {np.sum(np.abs(1.0 - ratio) < 0.2)} / {len(ratio)}")

if max_deviation > 0.10:
    print("\n⚠ WARNING: Energy non-conservation detected (>10% deviation).")
    print("File an issue with the above table and commit hash.")
else:
    print("\n✓ Energy balance validated: L_IR ≈ L_UV_absorbed (±10%)")
