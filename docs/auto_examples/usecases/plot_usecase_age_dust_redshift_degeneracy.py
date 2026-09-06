"""
The age-dust-redshift degeneracy in photometry
==============================================

Optical photometry alone cannot uniquely break the degeneracy between stellar
age, dust attenuation, and redshift — a fundamental limitation in photo-z and
SED fitting. Three physically distinct galaxy populations can produce nearly
identical SDSS ugriz photometry:

A. Young + dusty + low-z:      200 Myr, τ_V ≈ 2.0, z = 0.5
B. Old + clean + mid-z:        5 Gyr,  τ_V ≈ 0.1, z = 1.0
C. Post-starburst + dust + hi-z: 1 Gyr, τ_V ≈ 0.7, z = 1.5

We build three `tengri.SEDModel` instances via `SEDModel.build()` and tune the
total stellar mass of each to match a reference r-band magnitude. This demonstrates
why multiwavelength observations (spectroscopy, X-ray, FIR) are critical for
constraining both age and dust.

See Papovich et al. 2001 (AJ, 122, 1) for the seminal discussion of this
degeneracy in the context of Hubble Deep Field galaxies.

References:

- Papovich et al. 2001, AJ, 122, 1
- Poggianti & Barbaro 1997, A&A, 325, 1025

"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # suppress XLA/PjRt C++ INFO+WARNING logs

import jax
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri import Observation, Photometry, SEDModel

# Setup
tengri.analysis.plotting.setup_style()

ssp = tengri.load_ssp("fsps_prsc_miles_chabrier")
sdss_filters = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
observation = Observation(photometry=sdss_filters)

# Reference magnitude (r-band) to which all three models will be tuned
m_r_target = 20.0

# Scenario A: Young + dusty + low-z
# Use a declining exponential SFH peaking early (young light-weighted age)

model_a = SEDModel.build(
    ssp,
    observation=observation,
    sfh={
        "type": "dexp",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_gyr": 0.3,
        "log_total_mass": 10.0,  # Will be tuned for magnitude match
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_bc": 2.0,
        "tau_diff": 0.8,
    },
    neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
    redshift=tengri.Fixed(0.5),
    igm={"type": "inoue"},
)

baseline_a = dict(model_a.spec.sample(jax.random.PRNGKey(0)))
baseline_a["met_logzsol"] = -0.1
baseline_a["dust_tau_diff"] = 0.8

# Scenario B: Old + clean + mid-z
# Use a very declining SFH with long timescale (old light-weighted age)

model_b = SEDModel.build(
    ssp,
    observation=observation,
    sfh={
        "type": "dexp",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_gyr": 8.0,
        "log_total_mass": 10.0,  # Will be tuned
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_bc": 0.1,
        "tau_diff": 0.05,
    },
    neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
    redshift=tengri.Fixed(1.0),
    igm={"type": "inoue"},
)

baseline_b = dict(model_b.spec.sample(jax.random.PRNGKey(1)))
baseline_b["met_logzsol"] = -0.1
baseline_b["dust_tau_diff"] = 0.05

# Scenario C: Post-starburst + dust + high-z
# Use log-normal peak with intermediate age at peak

model_c = SEDModel.build(
    ssp,
    observation=observation,
    sfh={
        "type": "lnorm",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "peak_gyr": 1.0,
        "width_gyr": 0.5,
        "log_total_mass": 10.0,  # Will be tuned
    },
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "all_params": tengri.Fixed(tengri.DEFAULT),
        "tau_bc": 0.7,
        "tau_diff": 0.3,
    },
    neb={"type": "cue", "all_params": tengri.Fixed(tengri.DEFAULT)},
    redshift=tengri.Fixed(1.5),
    igm={"type": "inoue"},
)

baseline_c = dict(model_c.spec.sample(jax.random.PRNGKey(2)))
baseline_c["met_logzsol"] = -0.1
baseline_c["dust_tau_diff"] = 0.3


# Bisection helper to find log_total_mass that produces target r-band magnitude
def _bisect_log_total_mass(model, sfh_param_name, baseline, m_r_target, lo=-1.0, hi=3.0):
    """Binary search for log_total_mass that produces m_r = m_r_target.

    Raises
    ------
    RuntimeError
        If no iteration produced a usable magnitude. Without this the search
        would treat every failure as "flux too high", drive ``hi`` down thirty
        times, and return a finite number close to ``lo`` -- a normalization
        derived from zero successful evaluations, indistinguishable in the
        figure from a converged one. A plausible wrong number is worse than a
        crash: the plot still renders and the degeneracy it claims to show is
        an artifact of the failure.
    """
    evaluated = 0
    first_failure: Exception | None = None
    for _iteration in range(30):
        mid = 0.5 * (lo + hi)
        params = {**baseline, sfh_param_name: mid}
        try:
            photo = model.predict_photometry(params)
            flux_array = np.asarray(photo)
            evaluated += 1
            if np.any(flux_array <= 0) or np.any(np.isnan(flux_array)):
                hi = mid  # Too high
                continue
            m_r = -2.5 * np.log10(flux_array[2]) - 48.6
            if abs(m_r - m_r_target) < 0.01:
                return mid
            if m_r > m_r_target:
                lo = mid  # Flux too low, need higher SFR
            else:
                hi = mid  # Flux too high, need lower SFR
        except Exception as e:
            if first_failure is None:
                first_failure = e
            hi = mid
            continue
    if evaluated == 0:
        raise RuntimeError(
            f"bisection for {sfh_param_name} never evaluated the model, so its "
            f"returned normalization would be meaningless. First failure: "
            f"{type(first_failure).__name__}: {first_failure}"
        ) from first_failure
    return 0.5 * (lo + hi)


# Tune each scenario to match target r-band magnitude
log_total_mass_a = _bisect_log_total_mass(
    model_a, "sfh_dexp_log_total_mass", baseline_a, m_r_target
)
baseline_a["sfh_dexp_log_total_mass"] = log_total_mass_a
params_a = baseline_a

log_total_mass_b = _bisect_log_total_mass(
    model_b, "sfh_dexp_log_total_mass", baseline_b, m_r_target
)
baseline_b["sfh_dexp_log_total_mass"] = log_total_mass_b
params_b = baseline_b

log_total_mass_c = _bisect_log_total_mass(
    model_c, "sfh_lnorm_log_total_mass", baseline_c, m_r_target
)
baseline_c["sfh_lnorm_log_total_mass"] = log_total_mass_c
params_c = baseline_c

# Predict photometry for all three scenarios
photo_a = model_a.predict_photometry(params_a)
photo_b = model_b.predict_photometry(params_b)
photo_c = model_c.predict_photometry(params_c)

flux_a = np.asarray(photo_a)
flux_b = np.asarray(photo_b)
flux_c = np.asarray(photo_c)

mag_a = -2.5 * np.log10(flux_a) - 48.6
mag_b = -2.5 * np.log10(flux_b) - 48.6
mag_c = -2.5 * np.log10(flux_c) - 48.6

# Predict rest-frame SEDs for display
sed_a = model_a.predict(params_a)
sed_b = model_b.predict(params_b)
sed_c = model_c.predict(params_c)

wave_a = np.asarray(model_a.wavelengths)
wave_b = np.asarray(model_b.wavelengths)
wave_c = np.asarray(model_c.wavelengths)

sed_a_lnu = np.asarray(sed_a.rest_sed())
sed_b_lnu = np.asarray(sed_b.rest_sed())
sed_c_lnu = np.asarray(sed_c.rest_sed())

# Create figure
fig, axes = plt.subplots(2, 1, figsize=(10, 8))

# TOP PANEL: Observed photometry (should cluster within degeneracy)
ax_phot = axes[0]
filter_names = ["u", "g", "r", "i", "z"]
band_positions = np.arange(len(filter_names))
ax_phot.scatter(
    band_positions - 0.12,
    mag_a,
    s=100,
    marker="o",
    label="A: 200 Myr + tau=2.0 + z=0.5",
    color="tab:blue",
    alpha=0.8,
    edgecolors="black",
    linewidth=1.2,
)
ax_phot.scatter(
    band_positions,
    mag_b,
    s=100,
    marker="s",
    label="B: 5 Gyr + tau=0.1 + z=1.0",
    color="tab:orange",
    alpha=0.8,
    edgecolors="black",
    linewidth=1.2,
)
ax_phot.scatter(
    band_positions + 0.12,
    mag_c,
    s=100,
    marker="^",
    label="C: 1 Gyr + tau=0.7 + z=1.5",
    color="tab:green",
    alpha=0.8,
    edgecolors="black",
    linewidth=1.2,
)

ax_phot.set_xticks(band_positions)
ax_phot.set_xticklabels([f"SDSS {name}" for name in filter_names])
ax_phot.set_ylabel("Magnitude (AB)", fontsize=11)
ax_phot.invert_yaxis()
ax_phot.legend(loc="upper left", frameon=False, fontsize=10)
ax_phot.grid(True, alpha=0.3, linestyle=":")
y_spread = np.max(mag_a) - np.min(mag_c)
ax_phot.set_ylim(np.max(mag_a) + 0.5, np.min(mag_c) - 0.5)
ax_phot.set_title(
    "Observed-frame SDSS photometry: three scenarios overlap within degeneracy",
    fontsize=12,
    weight="bold",
)

# BOTTOM PANEL: Rest-frame SEDs (should look different)
ax_sed = axes[1]

# Wave range for visualization
wave_rest_min = 2000.0
wave_rest_max = 8000.0

mask_a = (wave_a >= wave_rest_min) & (wave_a <= wave_rest_max)
mask_b = (wave_b >= wave_rest_min) & (wave_b <= wave_rest_max)
mask_c = (wave_c >= wave_rest_min) & (wave_c <= wave_rest_max)

ax_sed.loglog(
    wave_a[mask_a],
    sed_a_lnu[mask_a],
    lw=2.5,
    label="A: young + dusty + low-z",
    color="tab:blue",
    alpha=0.85,
)
ax_sed.loglog(
    wave_b[mask_b],
    sed_b_lnu[mask_b],
    lw=2.5,
    label="B: old + clean + mid-z",
    color="tab:orange",
    alpha=0.85,
)
ax_sed.loglog(
    wave_c[mask_c],
    sed_c_lnu[mask_c],
    lw=2.5,
    label="C: post-starburst + dust + high-z",
    color="tab:green",
    alpha=0.85,
)

ax_sed.set_xlabel(r"Rest-frame wavelength [Angstrom]", fontsize=11)
ax_sed.set_ylabel(r"$L_\nu$ [erg s$^{-1}$ Hz$^{-1}$]", fontsize=11)
ax_sed.legend(loc="upper right", frameon=False, fontsize=10)
ax_sed.grid(True, alpha=0.3, linestyle=":", which="both")
ax_sed.set_xlim(wave_rest_min, wave_rest_max)

plt.tight_layout()
plt.savefig("plot_usecase_age_dust_redshift_degeneracy.png", dpi=150, bbox_inches="tight")
