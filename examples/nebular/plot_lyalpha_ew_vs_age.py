"""
Lyman-alpha equivalent width peaks during O-star dominance
===========================================================

Lyman-alpha (Lyα) equivalent width (EW) traces stellar population age
through the presence and strength of massive O stars. We construct a
sequence of constant star-formation-rate (CSF) models with ages ranging
from 1 Myr to 30 Myr at fixed metallicity (Z = Zsun; logZ = 0), compute
the rest-frame Lyα emission line luminosity and the underlying continuum
at 1216 Å, then derive EW(Lyα) = L(Lyα) / L_continuum.

Key result: EW peaks at ~3–5 Myr when spectral type O dominates ionization,
then decays past 10 Myr as stars age past the main sequence.

References
----------
Charlot & Fall 1993 (ApJ 405, 538) — Empirical population synthesis and
spectral evolution across age and metallicity.
Schaerer 2003 (A&A 397, 527) — Ionizing photon production in massive
starburst populations.
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style
from tengri.utils.physics_constants import L_SUN

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")
warnings.filterwarnings("ignore", message=".*deprecated.*")

# ── Load bare stellar SSP (Cue requires bare-stellar, not wNE)
ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")

# ── Rest-frame Lyα at 1216 Å
wave_lya = 1216.0
# ── Conversion constants for EW
C_AA_PER_S = 2.998e18  # speed of light [Å/s]

# ── Age range: 1–30 Myr (young starbursts)
# Scan ages logarithmically to capture the O-star to WR transition
ages_myr = np.logspace(0.0, np.log10(30), 18)  # 1 to 30 Myr

# ── Nebular model: Cue emulator, fixed logU and metallicity
neb_config = {
    "type": "cue",
    "*": tengri.FIXED,
    # Short-form keys inside the `neb` group; full `neb_*` keys are silently ignored.
    "logZ_gas": 0.0,
    "logU": -2.0,
    "fesc": 0.0,
    "fesc_lya": 0.0,  # 0 = no resonant destruction → full intrinsic Lyα emission
}

# ── Dust: no attenuation for clean nebular emission view
dust_config = {
    "type": "two_component",
    "*": tengri.FIXED,
    "tau_diff": 0.0,
    "tau_bc": 0.0,
}

# ── Collect EW measurements
ew_lya = []

for age_myr in ages_myr:
    # Build base model with constant SFH: vary the age via start_gyr/end_gyr window
    # population spans [age_myr / 1000, 0] Gyr (older age at start → younger, present day)
    sfh_config_age = {
        "type": "const",
        "*": tengri.FIXED,
        "log_sfr": 0.0,
        "start_gyr": age_myr / 1e3,
        "end_gyr": 0.0,
    }

    model = tengri.SEDModel.build(
        ssp,
        sfh=sfh_config_age,
        dust=dust_config,
        neb=neb_config,
        redshift=tengri.Fixed(0.0),
    )

    # Sample parameters
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    # Predict emission lines and rest-frame SED
    lines = model.predict_emission_lines(params)
    sed_result = model.predict_rest_sed(params)

    # Extract Lyman-alpha luminosity [Lsun]
    lya_lum = float(lines.lya)

    # Extract continuum density at 1216 Å via linear interpolation
    wave = np.asarray(sed_result.wavelength)
    sed = np.asarray(sed_result.sed)

    # Find closest wavelength to 1216 Å
    idx_lya = np.argmin(np.abs(wave - wave_lya))
    continu_at_lya = sed[idx_lya]

    # Compute equivalent width: EW [Å] = L_line [erg/s] / L_lambda_continuum [erg/s/Å]
    # Convert L_line from Lsun to erg/s, then L_nu to L_lambda via L_lambda = L_nu * c / λ²
    if continu_at_lya > 0:
        # EW [Å] = L_line [erg/s] / L_lambda_continuum [erg/s/Å]
        # `lines.lya` is already in erg/s (verified empirically).
        continu_lambda = continu_at_lya * C_AA_PER_S / (wave_lya ** 2)  # L_nu [erg/s/Hz] → L_lambda [erg/s/Å]
        ew = lya_lum / continu_lambda
    else:
        ew = np.nan

    ew_lya.append(ew)

ew_lya = np.asarray(ew_lya)

# ── Plotting
fig, ax = plt.subplots(figsize=(7, 5))

ax.plot(
    ages_myr,
    ew_lya,
    "o-",
    color="C0",
    markersize=6,
    lw=2,
    label=r"Lyα EW (Cue, Z = Z$_\odot$)",
)

ax.axvline(3.0, color="0.5", ls="--", lw=0.8, alpha=0.5, label="O-star peak (~3 Myr)")
ax.axvline(10.0, color="0.6", ls=":", lw=0.8, alpha=0.5, label="WR phase (~10 Myr)")

ax.set_xlabel(r"Population age [Myr]")
ax.set_ylabel(r"Lyα equivalent width [Å]")
ax.set_xscale("log")
ax.set_xlim(0.7, 35)
ax.set_ylim(0, max(ew_lya) * 1.15)

ax.legend(frameon=False, fontsize=10, loc="upper right")
ax.grid(True, alpha=0.2, which="both")

fig.tight_layout()
plt.savefig("plot_lyalpha_ew_vs_age.png", dpi=150, bbox_inches="tight")

print(f"Peak EW(Lyα): {np.nanmax(ew_lya):.1f} Å at age {ages_myr[np.nanargmax(ew_lya)]:.1f} Myr")
