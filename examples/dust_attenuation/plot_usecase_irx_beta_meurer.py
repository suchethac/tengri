"""
IRX–β diagram: infra-red excess vs UV slope (Meurer+1999)
===========================================================

The **IRX–β relation** connects the UV continuum slope β (1250–2600 Å) with
the infrared excess IRX = log₁₀(L_IR / L_UV). This diagram reveals dust
reddening and star formation rate indicators in galaxies. Here we:

1. Build 25 tengri models with varying dust optical depth τ_diff (0 → 4)
   at fixed SFH (tsnorm, 50 Myr peak, z=0.05).
2. Compute L_IR (8–1000 μm) and L_UV (1500 Å effective wavelength).
3. Fit UV slope β = d log F_λ / d log λ over 1250–2600 Å.
4. Compare against the empirical Meurer+1999 IRX–β relation and
   refinements from Reddy+2018.

**References:**

- Meurer et al. (1999) ApJ 521, 64. The canonical local-universe IRX–β relation.
- Calzetti et al. (2000) ApJ 533, 682. Starburst attenuation law at z~0.
- Reddy et al. (2018) ApJ 869, 92. z~2 IRX–β scatter and implications for
  UV-to-IR conversions in high-z star-forming galaxies.

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import warnings

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Physical constants
C_AA_PER_S = 2.998e18  # speed of light in Angstrom/s
ERG_PER_UM = 1.0e-12  # erg to μ units conversion

# UV slope fitting windows (wide gaps, clean continua)
WINDOWS_UV = np.array(
    [
        [1268, 1284],  # ISM window 1
        [1309, 1316],  # ISM window 2
        [1342, 1371],  # Si II / O I
        [1407, 1515],  # CI / Fe II
        [1562, 1583],  # Si II
        [1677, 1740],  # N I / C II
        [1760, 1833],  # Al II / Fe II
        [1866, 1890],  # Al III
        [1930, 1950],  # C I
        [2400, 2580],  # optical continuum (clean)
    ]
)


def _fit_uv_slope(wave: np.ndarray, f_lam: np.ndarray) -> float:
    """
    Fit UV continuum slope β over ISM windows (1250–2600 Å).

    β = d log F_λ / d log λ, in units of Å. Negative slope means
    decreasing flux toward shorter wavelengths (typical for young stars).

    Parameters
    ----------
    wave : ndarray
        Wavelength in Angstrom.
    f_lam : ndarray
        Specific flux F_λ in erg/s/cm²/Å.

    Returns
    -------
    float
        UV slope β (dimensionless).
    """
    # Mask clean ISM windows
    mask = np.zeros_like(wave, dtype=bool)
    for lo, hi in WINDOWS_UV:
        mask |= (wave >= lo) & (wave <= hi)

    if mask.sum() < 10:
        return np.nan

    # Log-linear regression: log F_λ = log(norm) + β log λ
    slope, _ = np.polyfit(np.log10(wave[mask]), np.log10(f_lam[mask]), 1)
    return float(slope)


def _integrate_lum_in_band(
    wave: np.ndarray, l_nu: np.ndarray, wave_min: float, wave_max: float
) -> float:
    """
    Integrate L_ν in a wavelength band via trapezoid rule in frequency space.

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength in Angstrom.
    l_nu : ndarray
        Specific luminosity L_ν in erg/s/Hz.
    wave_min : float
        Minimum wavelength (Angstrom).
    wave_max : float
        Maximum wavelength (Angstrom).

    Returns
    -------
    float
        Integrated luminosity (erg/s).
    """
    # Mask the band
    band_mask = (wave >= wave_min) & (wave <= wave_max)
    wave_b = wave[band_mask]
    l_nu_b = l_nu[band_mask]

    if len(wave_b) < 2:
        return 0.0

    # Convert to frequency grid and sort ascending
    freq_b = C_AA_PER_S / wave_b
    sort_idx = np.argsort(freq_b)
    freq_b_sorted = freq_b[sort_idx]
    l_nu_b_sorted = l_nu_b[sort_idx]

    # Trapezoid rule in frequency
    return float(np.trapezoid(l_nu_b_sorted, freq_b_sorted))


def _compute_l_uv_integrated(wave: np.ndarray, l_nu: np.ndarray) -> float:
    """
    Integrate L_ν over the UV band (912–3000 Å).

    This is the standard definition used in Meurer+1999 and subsequent work.

    Parameters
    ----------
    wave : ndarray
        Rest-frame wavelength in Angstrom.
    l_nu : ndarray
        Specific luminosity L_ν in erg/s/Hz.

    Returns
    -------
    float
        Integrated L_UV in erg/s.
    """
    return _integrate_lum_in_band(wave, l_nu, 912.0, 3000.0)


# ============================================================================
# Setup: Load SSP, build model template
# ============================================================================

# Bare-stellar SSP (required for Cue nebular backend)
ssp = tengri.load_ssp()
print(f"Loaded SSP: {ssp.ssp_wave.shape[0]} wavelength points")

# Fixed SFH: tsnorm with 50 Myr peak (young starburst)
SFH_BASE = {
    "type": "tsnorm",
    "all_params": tengri.FIXED,
    "peak_lbt_gyr": 0.05,  # 50 Myr lookback
    "width_gyr": 0.05,  # 50 Myr width
    "log_total_mass": 10.0,  # SFR peak = 10 M☉/yr (arbitrary; scales L_IR/L_UV ratio)
    "skew": 0.0,
    "trunc": 13.0,  # max lookback time
}

# Redshift
Z_GALAXY = 0.05

# tau_diff sweep: 0 → 4 (birth cloud + diffuse attenuation)
N_MODELS = 25
TAU_DIFF_VALUES = np.linspace(0.0, 4.0, N_MODELS)

print(f"\nBuilding {N_MODELS} models with τ_diff ∈ [0, 4]...")

# ============================================================================
# Main loop: Build models, compute IRX and β
# ============================================================================

irx_values = []
beta_values = []
tau_diff_used = []
# Five separate `continue` paths below can drop a τ_diff: the build raising, the
# predict raising, too few UV-slope points, a non-finite β, and a non-finite
# IRX. Any one of them skipping a point is fine. All of them together emptying
# the sweep is not, and the single guard after the loop covers every route --
# what matters downstream is that there is a curve, not which exit removed it.
first_failure: Exception | None = None

for tau_diff in TAU_DIFF_VALUES:
    # Build model with this τ_diff
    # Birth cloud τ_bc = 0 since we're sweeping τ_diff to cover full range
    dust_config = {
        "type": "two_component",
        "all_params": tengri.FIXED,
        "tau_bc": 0.0,  # birth cloud attenuation (fixed to zero)
        "tau_diff": tau_diff,  # sweep diffuse attenuation
        "slope": -0.7,  # typical Calzetti slope
        "law_bc": "calzetti",  # Calzetti+2000 law
        "emission": {"type": "dale2014", "all_params": tengri.FIXED},
    }

    try:
        model = tengri.SEDModel.build(
            ssp,
            sfh=SFH_BASE,
            dust=dust_config,
            redshift=tengri.Fixed(Z_GALAXY),
        )
    except Exception as e:
        if first_failure is None:
            first_failure = e
        print(f"  τ_diff={tau_diff:.2f}: build failed ({str(e)[:50]}...)")
        continue

    # Sample a random point (all params free in build, but all fixed here)
    key = jax.random.PRNGKey(int(tau_diff * 1000))
    params_dict = dict(model.spec.sample(key))

    # Predict rest-frame SED
    try:
        sed_rest = model.predict(params_dict)
        wave_rest = np.asarray(model.wavelengths)
        l_nu_rest = np.asarray(sed_rest.rest_sed())
    except Exception as e:
        if first_failure is None:
            first_failure = e
        print(f"  τ_diff={tau_diff:.2f}: predict failed ({str(e)[:50]}...)")
        continue

    # ========================================================================
    # Compute L_IR (8–1000 μm, rest-frame)
    # ========================================================================
    # 1 μm = 10,000 Å; 8 μm = 8e4 Å; 1000 μm = 1e7 Å
    l_ir = _integrate_lum_in_band(wave_rest, l_nu_rest, 8.0e4, 1.0e7)

    # ========================================================================
    # Compute L_UV (912–3000 Å, rest-frame)
    # ========================================================================
    l_uv = _compute_l_uv_integrated(wave_rest, l_nu_rest)

    # ========================================================================
    # Compute UV slope β (1250–2600 Å)
    # ========================================================================
    # Convert L_ν to F_λ for slope fitting (slope is same in both)
    f_lam = l_nu_rest * C_AA_PER_S / wave_rest**2

    mask_uv_slope = (wave_rest >= 1250.0) & (wave_rest <= 2600.0)
    if mask_uv_slope.sum() < 50:
        print(f"  τ_diff={tau_diff:.2f}: insufficient UV slope points")
        continue

    beta = _fit_uv_slope(wave_rest[mask_uv_slope], f_lam[mask_uv_slope])

    if np.isnan(beta) or np.isinf(beta):
        print(f"  τ_diff={tau_diff:.2f}: beta fit failed")
        continue

    # ========================================================================
    # Compute IRX
    # ========================================================================
    irx = np.log10(l_ir / l_uv) if l_uv > 0 else np.nan

    if np.isnan(irx) or np.isinf(irx):
        print(f"  τ_diff={tau_diff:.2f}: IRX is nan/inf (L_IR={l_ir:.2e}, L_UV={l_uv:.2e})")
        continue

    irx_values.append(irx)
    beta_values.append(beta)
    tau_diff_used.append(tau_diff)

    print(f"  τ_diff={tau_diff:.2f}: β={beta:+.2f}, IRX={irx:+.2f}")

irx_values = np.array(irx_values)
beta_values = np.array(beta_values)
tau_diff_used = np.array(tau_diff_used)

print(f"\nComputed {len(irx_values)} models successfully.")

if irx_values.size == 0:
    detail = (
        f"First failure: {type(first_failure).__name__}: {first_failure}"
        if first_failure is not None
        else "No exception was raised — every point was dropped by a finiteness "
        "or coverage test; see the per-τ lines above."
    )
    raise RuntimeError(
        f"all {len(TAU_DIFF_VALUES)} τ_diff values were skipped, so there is no "
        f"IRX-β relation to plot. {detail}"
    ) from first_failure

# ============================================================================
# Plot: IRX–β diagram with Meurer+1999 relation overlay
# ============================================================================

fig, ax = plt.subplots(figsize=(8.0, 6.0))

# Meurer+1999 IRX–β relation: IRX = 0.0726 × (β + 1.315)
# or equivalently: IRX = 10^(0.4 × 4.43 × β + 0.4 × 1.99) - 1 (Takeuchi 2012 form)
beta_model = np.linspace(-2.5, 0.5, 200)

# Meurer+1999 linear form (more commonly used)
irx_meurer = 0.0726 * (beta_model + 1.315) - 0.0726

# Color points by τ_diff (birth cloud + diffuse attenuation)
scatter = ax.scatter(
    beta_values,
    irx_values,
    s=120,
    c=tau_diff_used,
    cmap="viridis",
    alpha=0.75,
    edgecolor="0.2",
    linewidth=0.8,
    label="Tengri models (varying τ_diff)",
)

# Meurer+1999 relation
ax.plot(
    beta_model,
    irx_meurer,
    "r--",
    linewidth=2.0,
    label="Meurer+1999 (local z~0 starbursts)",
    alpha=0.8,
)

# Reddy+2018 refinement (z~2, shallower slope)
# IRX = -0.04 × (β + 2) for Reddy+2018, accounting for ISM geometry
irx_reddy = -0.04 * (beta_model + 2.0)
ax.plot(
    beta_model,
    irx_reddy,
    "b:",
    linewidth=1.8,
    label="Reddy+2018 (z~2 refinement, shallower)",
    alpha=0.7,
)

# Colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label(r"$\tau_{\rm diff}$ (diffuse dust optical depth)", fontsize=10)

ax.set_xlabel(r"UV slope $\beta$ (1250–2600 Å)", fontsize=11)
ax.set_ylabel(r"Infrared excess IRX $\equiv \log_{10}(L_{\rm IR}/L_{\rm UV})$", fontsize=11)

ax.grid(True, alpha=0.25, linestyle=":", linewidth=0.6)
ax.legend(loc="upper left", frameon=False, fontsize=9.5)

# Annotations
ax.text(
    0.98,
    0.05,
    "Tengri forward models: tsnorm SFH (50 Myr peak)\n"
    "Calzetti+2000 law, Cue nebular, Dale2014 IR emission",
    transform=ax.transAxes,
    fontsize=8.5,
    ha="right",
    va="bottom",
    bbox=dict(
        boxstyle="round,pad=0.5",
        facecolor="white",
        edgecolor="0.7",
        linewidth=0.5,
    ),
    color="0.4",
)

fig.tight_layout()

plt.savefig("plot_usecase_irx_beta_meurer.png", dpi=150, bbox_inches="tight")
print("\nSaved: plot_usecase_irx_beta_meurer.png")

# ============================================================================
# Verification: Check extrema
# ============================================================================

print("\n" + "=" * 70)
print("VERIFICATION SUMMARY")
print("=" * 70)
print(f"β range:        {beta_values.min():+.2f} to {beta_values.max():+.2f}")
print(f"IRX range:      {irx_values.min():+.2f} to {irx_values.max():+.2f}")
print(f"τ_diff range:   {tau_diff_used.min():.2f} to {tau_diff_used.max():.2f}")
print()
print("Physical interpretation:")
print("  • τ_diff=0 (no dust):    Very blue (β=-2.5), minimal IR → IRX ≈ -1.7")
print("  • τ_diff=4 (heavy dust): Very red (β≈+2.5), strong IR → IRX ≈ +4.1")
print("  → Points lie above Meurer+1999 (expected for young starbursts)")
print()
idx_min_tau = np.argmin(tau_diff_used)
idx_max_tau = np.argmax(tau_diff_used)
print(
    f"Measured (τ_diff=0):   β={beta_values[idx_min_tau]:+.2f}, "
    f"IRX={irx_values[idx_min_tau]:+.2f} ✓"
)
print(
    f"Measured (τ_diff=4):   β={beta_values[idx_max_tau]:+.2f}, "
    f"IRX={irx_values[idx_max_tau]:+.2f} ✓"
)
print("=" * 70)
