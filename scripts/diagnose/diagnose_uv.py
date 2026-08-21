"""
Diagnostic script to understand the UV absorption calculation issue.
"""
import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import warnings
from pathlib import Path

import jax
import numpy as np

from tengri import (
    FIXED,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    load_ssp_data,
)

warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Setup
ssp_path = Path("data/fsps_prsc_miles_chabrier.h5")
if not ssp_path.exists():
    import tengri
    ssp_path = Path(tengri.download_ssp("fsps_prsc_miles_chabrier"))
ssp = load_ssp_data(str(ssp_path))

z = 0.05
wave_rest = np.logspace(2.0, 7.0, 3000)
wave_uv_min, wave_uv_max = 912.0, 3000.0

filters = [
    "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z",
    "irac_36", "irac_45", "irac_58", "irac_80",
    "mips_24", "mips_70", "herschel_100", "herschel_160",
]
obs = Observation(photometry=Photometry.from_names(filters))

key = jax.random.PRNGKey(42)

# Build intrinsic model (no dust)
print("=" * 70)
print("Building INTRINSIC model (no dust)")
print("=" * 70)
model_intrinsic = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    redshift=Fixed(z),
    sfh={"type": "tsnorm", "all_params": FIXED, "peak_lbt_gyr": 0.3},
    neb={"type": "cue", "all_params": FIXED},
)
print(f"Intrinsic model free params: {model_intrinsic.spec.free_params}")

params_intrinsic = model_intrinsic.spec.sample(key)
print(f"Sampled params keys: {list(params_intrinsic.keys())}")

result_intrinsic = model_intrinsic.predict_rest_sed(params_intrinsic, wave=wave_rest)
sed_intrinsic_np = np.array(result_intrinsic.sed)

print(f"Intrinsic SED shape: {sed_intrinsic_np.shape}")
print(f"Intrinsic SED min/max: {sed_intrinsic_np.min():.4e} / {sed_intrinsic_np.max():.4e}")

mask_uv = (wave_rest >= wave_uv_min) & (wave_rest <= wave_uv_max)
sed_uv_intrinsic = sed_intrinsic_np[mask_uv]
print(f"Intrinsic UV SED (912-3000 Å) min/max: {sed_uv_intrinsic.min():.4e} / {sed_uv_intrinsic.max():.4e}")
print(f"Intrinsic UV SED has NaN: {np.isnan(sed_uv_intrinsic).any()}")
print(f"Intrinsic UV SED has negative: {(sed_uv_intrinsic < 0).any()}")

luv_intrinsic = float(np.trapz(sed_uv_intrinsic, wave_rest[mask_uv]))
print(f"Integrated L_UV intrinsic: {luv_intrinsic:.4e}\n")

# Build dust model
print("=" * 70)
print("Building DUST model (tau_v=0.5)")
print("=" * 70)
model_dust = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    redshift=Fixed(z),
    sfh={"type": "tsnorm", "all_params": FIXED, "peak_lbt_gyr": 0.3},
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": FIXED,
        "tau_bc": 0.5,
    }, dust_emission={"type": "dale2014", "all_params": FIXED},
    neb={"type": "cue", "all_params": FIXED},
)
print(f"Dust model free params: {model_dust.spec.free_params}")

params_dust = model_dust.spec.sample(key)
print(f"Sampled params keys: {list(params_dust.keys())}")

result_dust = model_dust.predict_rest_sed(params_dust, wave=wave_rest)
sed_dust_np = np.array(result_dust.sed)

print(f"Dust SED shape: {sed_dust_np.shape}")
print(f"Dust SED min/max: {sed_dust_np.min():.4e} / {sed_dust_np.max():.4e}")

sed_uv_dust = sed_dust_np[mask_uv]
print(f"Dust UV SED (912-3000 Å) min/max: {sed_uv_dust.min():.4e} / {sed_uv_dust.max():.4e}")
print(f"Dust UV SED has NaN: {np.isnan(sed_uv_dust).any()}")
print(f"Dust UV SED has negative: {(sed_uv_dust < 0).any()}")

luv_dust = float(np.trapz(sed_uv_dust, wave_rest[mask_uv]))
print(f"Integrated L_UV dust: {luv_dust:.4e}")

# The key comparison
print("\n" + "=" * 70)
print("COMPARISON")
print("=" * 70)
print(f"L_UV intrinsic: {luv_intrinsic:.4e}")
print(f"L_UV dust:      {luv_dust:.4e}")
print(f"Ratio (dust/intrinsic): {luv_dust / luv_intrinsic:.4f}")
print(f"L_UV absorbed (intrinsic - dust): {luv_intrinsic - luv_dust:.4e}")

# Check which one should be higher
print("\nInterpretation:")
if luv_intrinsic > luv_dust:
    print("✓ CORRECT: Intrinsic > Dust (UV is attenuated by dust)")
else:
    print("✗ WRONG: Dust > Intrinsic (this causes negative absorption)")

# Sample a few wavelength points
print("\n" + "=" * 70)
print("Sample SED values at specific wavelengths")
print("=" * 70)
sample_waves = [1000, 1500, 2000, 2500, 3000]
for w in sample_waves:
    idx = np.argmin(np.abs(wave_rest - w))
    actual_w = wave_rest[idx]
    print(f"λ={actual_w:.0f} Å: intrinsic={sed_intrinsic_np[idx]:.4e}, dust={sed_dust_np[idx]:.4e}, ratio={sed_dust_np[idx]/sed_intrinsic_np[idx]:.4f}")
