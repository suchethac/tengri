"""
Dust attenuation law validation: tengri vs dust_extinction PyPI package
======================================================================

Diagnostic figure comparing tengri's Calzetti and Cardelli/CCM89 attenuation
laws against the reference implementations in the ``dust_extinction`` package
(Barbary et al., widely used by astropy workflows). Residuals reveal systematic
offsets and validity ranges. If k(λ) residuals exceed 5% outside known
singularities, the implementation may need verification against the original papers.

Reference: Calzetti et al. 2000, ApJ 533 682; Cardelli et al. 1989, ApJ 345 245.
"""

import warnings
from pathlib import Path

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from tengri.analysis.plotting import setup_style
from tengri.dust import resolve_dust_law

setup_style()
warnings.filterwarnings("ignore", message=".*BakedInBackend.*")

# Wavelength grid: 1000 – 22000 Angstrom
wave_aa = np.logspace(3, 4.34, 300)  # 1000–22000 Å
wave_micron = wave_aa / 1e4

# Import dust_extinction reference models
try:
    from dust_extinction.parameter_averages import CCM89, F19, G23

    has_dust_extinction = True
except ImportError:
    has_dust_extinction = False
    print("Warning: dust_extinction not installed. Using tengri only.")

fig, (ax_main, ax_res) = plt.subplots(
    2, 1, figsize=(7.0, 5.2), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
)

# ──── Calzetti ────
calzetti_fn = resolve_dust_law("calzetti")
k_calzetti = np.array(calzetti_fn(jnp.asarray(wave_aa)))
ax_main.loglog(wave_aa, k_calzetti, "C0-", lw=1.5, label="Calzetti+2000 (tengri)")

# ──── Cardelli CCM89 (Rv=3.1) ────
cardelli_fn = resolve_dust_law("cardelli")
k_cardelli = np.array(cardelli_fn(jnp.asarray(wave_aa), dust_Rv=3.1))
ax_main.loglog(wave_aa, k_cardelli, "C1-", lw=1.5, label="Cardelli+1989 Rv=3.1 (tengri)")

if has_dust_extinction:
    # dust_extinction uses inverse wavelength in microns
    x = 1.0 / wave_micron

    # CCM89 from dust_extinction
    ccm89_model = CCM89()
    k_ccm89_ref = ccm89_model.evaluate(x, Rv=3.1)
    ax_main.loglog(
        wave_aa,
        k_ccm89_ref,
        "C1--",
        lw=1.2,
        alpha=0.7,
        label="Cardelli+1989 Rv=3.1 (dust_extinction)",
    )

    # F19 from dust_extinction (newer parametrization)
    f19_model = F19()
    k_f19_ref = f19_model.evaluate(x, Rv=3.1)
    ax_main.loglog(wave_aa, k_f19_ref, "C2-", lw=1.5, label="Fitzpatrick+2019 Rv=3.1")

    # G23 from dust_extinction (Gordon+2023)
    g23_model = G23()
    k_g23_ref = g23_model.evaluate(x, Rv=3.1)
    ax_main.loglog(wave_aa, k_g23_ref, "C3-", lw=1.5, label="Gordon+2023 Rv=3.1")

ax_main.set_ylabel(r"$k(\lambda)$ [mag]")
ax_main.legend(frameon=False, fontsize=8, loc="lower right")
ax_main.set_ylim(0.1, 10)

# ──── Residual panel ────
if has_dust_extinction:
    # Residual: (tengri - reference) / reference
    res_cardelli = (k_cardelli - k_ccm89_ref) / np.abs(k_ccm89_ref + 1e-10)
    ax_res.semilogx(wave_aa, res_cardelli * 100, "C1-", lw=1.2, label="Cardelli residual")

    res_f19 = (k_calzetti - k_f19_ref) / np.abs(k_f19_ref + 1e-10)
    ax_res.semilogx(wave_aa, res_f19 * 100, "C2-", lw=1.2, label="Calzetti vs F19")

    ax_res.axhline(0, color="k", lw=0.5, alpha=0.3)
    ax_res.axhline(5, color="r", lw=0.5, alpha=0.3, ls="--")
    ax_res.axhline(-5, color="r", lw=0.5, alpha=0.3, ls="--")
    ax_res.set_ylabel("Residual [%]")
    ax_res.set_ylim(-20, 20)
    ax_res.legend(frameon=False, fontsize=8)
else:
    # Show tengri's two laws relative to each other
    ratio = k_calzetti / np.abs(k_cardelli + 1e-10)
    ax_res.semilogx(wave_aa, ratio, "C0-", lw=1.2)
    ax_res.set_ylabel(r"$k_{\rm Calzetti} / k_{\rm Cardelli}$")
    ax_res.set_ylim(0.5, 2.0)

ax_res.set_xlabel(r"Wavelength [$\mathrm{\AA}$]")
fig.tight_layout()

outfile = Path(__file__).with_suffix(".png")
plt.savefig(outfile, dpi=150, bbox_inches="tight")
