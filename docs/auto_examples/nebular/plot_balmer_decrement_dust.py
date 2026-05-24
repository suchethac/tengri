"""
Balmer decrement Hα/Hβ as a dust meter
========================================

In Case B recombination at T = 10⁴ K, n_e ≈ 10² cm⁻³, intrinsic
Hα/Hβ = 2.86. Any larger ratio in the observed flux is interpreted
as a dust reddening signal — the workhorse measurement for E(B−V)
in star-forming galaxies.

We sweep birth-cloud and diffuse dust opacities over the same young
SF galaxy with Cue, read out the Hα and Hβ fluxes via
``predict_emission_lines``, and overlay the observed Hα/Hβ ratio
against the standard Calzetti+2000 conversion:

    E(B − V) = 1.97 × log₁₀ [ (Hα/Hβ)_obs / 2.86 ]
"""

import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

import tengri
from tengri.analysis.plotting import setup_style

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

INTRINSIC = 2.86
SSP = tengri.load_ssp("fsps_prsc_miles_chabrier")


def _line_flux(wave, l_nu, lam0, win=30.0):
    """Integrated line flux above local continuum (rough rectangle)."""
    f_lam = l_nu * 2.998e18 / wave**2
    line = (wave >= lam0 - win) & (wave <= lam0 + win)
    cont_lo = (wave >= lam0 - 80) & (wave <= lam0 - 60)
    cont_hi = (wave >= lam0 + 60) & (wave <= lam0 + 80)
    cont = 0.5 * (np.median(f_lam[cont_lo]) + np.median(f_lam[cont_hi]))
    delta = wave[line][1] - wave[line][0] if line.sum() > 1 else 1.0
    return float(np.sum(np.maximum(f_lam[line] - cont, 0.0)) * delta)


def _ratio(tau_diff):
    model = tengri.SEDModel.build(
        SSP,
        sfh={"type": "dpl", "*": tengri.FIXED,
             "tau_gyr": 0.05, "log_peak_sfr": 1.5,
             "alpha": 4.0, "beta": 2.0},
        dust={"type": "two_component", "*": tengri.FIXED,
              "tau_diff": tau_diff, "tau_bc": 0.3, "slope": -0.7,
              "law_diff": "calzetti"},
        neb={"type": "cue", "*": tengri.FIXED},
        redshift=tengri.Fixed(0.05),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed(p)
    wave = np.asarray(out.wavelength)
    l_nu = np.asarray(out.sed)
    return _line_flux(wave, l_nu, 6562.8) / _line_flux(wave, l_nu, 4861.3)


tau_grid = np.linspace(0.0, 1.8, 16)
ratio = np.array([_ratio(float(t)) for t in tau_grid])

# Calzetti+2000 inversion: E(B-V) from observed Hα/Hβ
ebv_meas = 1.97 * np.log10(np.maximum(ratio / INTRINSIC, 1.0))
# Convert input tau_diff (V-band optical depth) to E(B-V) via R_V = 4.05 (Calzetti)
ebv_input = tau_grid / 4.05 / np.log(10)

fig, (ax_r, ax_e) = plt.subplots(1, 2, figsize=(11.5, 4.3),
                                  gridspec_kw={"wspace": 0.28})

ax_r.plot(ebv_input, ratio, color="C3", lw=1.6)
ax_r.axhline(INTRINSIC, color="0.55", lw=0.6, ls="--")
ax_r.text(0.02, INTRINSIC + 0.03, r"Case B: $H\alpha/H\beta = 2.86$",
          fontsize=8, color="0.4")
ax_r.set(xlabel=r"input nebular reddening $E(B-V)$  [mag]",
         ylabel=r"observed $H\alpha\,/\,H\beta$",
         xlim=(0.0, ebv_input.max()))

ax_e.plot(ebv_input, ebv_meas, color="C0", lw=1.6, label="recovered")
diag = np.linspace(0, ebv_input.max(), 50)
ax_e.plot(diag, diag, color="0.55", lw=0.5, ls="--", label="1:1")
ax_e.set(xlabel=r"input $E(B-V)$  [mag]",
         ylabel=r"$E(B-V)$ from Calzetti $H\alpha/H\beta$ inversion  [mag]",
         xlim=(0.0, ebv_input.max()), ylim=(0.0, ebv_input.max() * 1.1))
ax_e.legend(frameon=False, fontsize=9, loc="upper left")

fig.savefig("plot_balmer_decrement_dust.png", dpi=150, bbox_inches="tight")
