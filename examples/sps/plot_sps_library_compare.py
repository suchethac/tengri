"""
SSP library comparison at a fixed age and metallicity
======================================================

Different stellar population synthesis codes use different stellar
spectral libraries, isochrone families, and binary treatments. The
SED of a ~1 Gyr-old, solar-metallicity simple stellar population
already shows visible differences in the UV (BPASS binaries add a
hot continuum) and in the NIR (treatment of TP-AGB).

At a fixed lookback age of 1 Gyr (single 50 Myr-wide burst) we
overlay the rest-frame ``νL_ν`` for each bundled SSP grid, all
normalized at 5500 Å so the chromatic differences read cleanly.

The bundled SSPs include both ``wNE`` ("with nebular emission" baked
into the grid at fixed ``log U``) and bare-stellar variants; we use
bare-stellar wherever available so the differences trace stellar
physics, not nebular treatment.
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
warnings.filterwarnings("ignore", message=".*deprecated.*")

# (ssp_short_name, label)
LIBRARIES = [
    ("fsps_prsc_miles_chabrier", "FSPS-Padova / MILES (Chabrier)"),
    ("fsps_mist_c3k_a_chabrier", "FSPS-MIST / C3K (Chabrier)"),
    ("bpss_stars_c3k_a_chabrier", "BPASS stars-only / C3K (Chabrier)"),
    ("bc03_pdva_stelib_chabrier", "BC03-Padova / STELIB (Chabrier)"),
    ("cb19_templates", "Charlot & Bruzual 2019"),
]
COLORS = plt.cm.viridis(np.linspace(0.05, 0.92, len(LIBRARIES)))

C_AA_PER_S = 2.998e18
SFH = {
    "type": "tsnorm",
    "all_params": tengri.FIXED,
    "peak_lbt_gyr": 1.0,
    "width_gyr": 0.05,
    "log_total_mass": 10.0,
    "skew": 0.0,
    "trunc": 13.0,
}
DUST = {"type": "two_component", "all_params": tengri.FIXED, "tau_diff": 0.0, "tau_bc": 0.0}

fig, ax = plt.subplots(figsize=(7.2, 4.6))

plotted = 0
first_failure: Exception | None = None

for (ssp_name, label), color in zip(LIBRARIES, COLORS):
    try:
        ssp = tengri.load_ssp(ssp_name)
    except Exception as e:
        # Was `except (FileNotFoundError, Exception)` — the second member makes
        # the first redundant, so it caught everything while reading as a
        # missing-file check. Absent grids are expected (this example is skipped
        # in CI); a loader bug was being hidden alongside them.
        if first_failure is None:
            first_failure = e
        continue
    model = tengri.SEDModel.build(
        ssp,
        sfh=SFH,
        dust=DUST,
        redshift=tengri.Fixed(0.01),
    )
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict(p)
    wave = np.asarray(model.wavelengths)
    nu_l_nu = C_AA_PER_S / wave * np.asarray(out.rest_sed())
    norm = nu_l_nu[np.argmin(np.abs(wave - 5500.0))]
    if norm > 0:
        nu_l_nu = nu_l_nu / norm
    ax.loglog(wave, nu_l_nu, color=color, lw=1.4, label=label)
    plotted += 1

# The Lyman-limit and Lyalpha reference lines below are drawn unconditionally, so
# an empty comparison still yields a plausible-looking annotated axes.
if plotted == 0:
    raise RuntimeError(
        f"none of the {len(LIBRARIES)} SSP libraries loaded, so there is "
        f"nothing to compare. First failure: "
        f"{type(first_failure).__name__}: {first_failure}"
    ) from first_failure

ax.axvline(1216, color="0.55", lw=0.6, ls=":")
ax.text(1216, 0.012, r"Ly$\alpha$", color="0.4", fontsize=8, rotation=90, va="bottom", ha="right")
ax.axvline(912, color="0.55", lw=0.6, ls=":")
ax.text(912, 0.012, "Lyman limit", color="0.4", fontsize=8, rotation=90, va="bottom", ha="right")
ax.set(
    xlim=(700, 5e4),
    ylim=(1e-2, 5.0),
    xlabel=r"Rest-frame wavelength $\lambda$ [$\mathrm{\AA}$]",
    ylabel=r"$\nu L_\nu / \nu L_\nu(5500\,\mathrm{\AA})$",
)
ax.legend(frameon=False, fontsize=8, loc="lower right")

fig.tight_layout()
plt.savefig("plot_sps_library_compare.png", dpi=150, bbox_inches="tight")
