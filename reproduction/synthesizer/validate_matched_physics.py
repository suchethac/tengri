"""Matched-input pixel validation: is the physics correct?

Companion to ``01_synthesizer.py`` (which compares tengri-native vs
Synthesizer-native *configurations*, where offsets are input-driven). This
script removes every input difference — same stellar grid, same metallicity,
same delayed-tau SFH, same Calzetti screen — and checks the stellar SED
**pixel by pixel**. At matched inputs the two implementations must agree; a
surviving residual is a physics/mapping problem.

Run:

    PYTHONHASHSEED=0 JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/synthesizer/validate_matched_physics.py

Needs ``synthesizer`` with its test grids downloaded
(``synthesizer-download --stellar-test-grids --agn-test-grids --dust-grid``)
and the re-shaped stellar grid cached under ``_drivers/data/``.

``PYTHONHASHSEED=0`` is pinned deliberately: ``UnifiedAGN.get_spectra``
assembles its emission-model tree in dict/set order, so the AGN line-region
spectrum genuinely changes with the seed (``reproduction/CONTRACT.md`` §7,
01_synthesizer.py §9c). This script touches only the stellar channel, where
the seed does not matter, but the pin keeps every reproduction entry point
consistent.

FINDING -- the dust mapping matters
-----------------------------------
Synthesizer's ``attenuate(..., name="calzetti", av=A_V)`` applies a **single**
screen of optical depth ``tau_V = A_V / 1.086`` to the *entire* stellar
continuum. tengri's ``two_component`` is **Charlot & Fall**: it adds a
birth-cloud optical depth ``tau_bc`` to the *young-star* continuum on top of
the diffuse ``tau_diff``. The Synthesizer-equivalent tengri build is therefore
a single diffuse screen -- ``tau_bc = 0``, ``tau_diff = A_V / 1.086``. The
CIGALE, Prospector and BAGPIPES validators reach the same conclusion
independently (issue #747).

Nebular is OFF on both sides: Synthesizer bakes Cloudy grids into its SSP
grids while tengri uses the Cue emulator, which are different models by design
(01_synthesizer.py §8).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")
os.environ.setdefault("PYTHONHASHSEED", "0")

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from reproduction.synthesizer._drivers import synthesizer_driver as S, units as U

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

L_SUN = 3.828e33
C_AA = 2.998e18
LOG10_ZSUN = -1.848  # Asplund 2009, Zsun = 0.0142 (CLAUDE.md metallicity note)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched parameters (01_synthesizer.py setup).
Z_ABS = 0.02
MET_LOGZSOL = float(np.log10(Z_ABS) - LOG10_ZSUN)
LOG_MASS = 10.0
TAU_GYR, AGE_GYR = 1.0, 5.0
A_V = 1.0
TAU_DIFF = A_V / 1.086  # single Calzetti screen


def synthesizer_stellar_dust():
    """Synthesizer stellar SED behind a single Calzetti screen.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    w, L = S.stellar_sed(
        tau_gyr=TAU_GYR,
        max_age_gyr=AGE_GYR,
        metallicity=Z_ABS,
        log_mass=LOG_MASS,
        nebular=False,
    )
    return w, S.attenuate(w, L, name="calzetti", av=A_V)


def tengri_stellar_dust(ssp, tau_bc):
    """tengri stellar SED at the matched parameters.

    Parameters
    ----------
    ssp : SSPData
        The re-shaped Synthesizer stellar grid.
    tau_bc : float
        Birth-cloud optical depth. ``0.0`` is the Synthesizer-equivalent
        mapping.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(TAU_GYR),
            "age_gyr": Fixed(AGE_GYR),
            "log_total_mass": Fixed(LOG_MASS),
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(TAU_DIFF),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.sed_intrinsic)


BANDS = {
    "FUV 1216-1900 Å": (1216, 1900),
    "NUV 2000-3000 Å": (2000, 3000),
    "optical 0.3-0.8 µm": (3000, 8000),
    "NIR 1-3 µm": (1e4, 3e4),
}


def report(w_s, L_t, L_s, title):
    """Print a band-by-band median-ratio table.

    Parameters
    ----------
    w_s : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom].
    L_t, L_s : array_like, shape (n_wave,)
        tengri and Synthesizer L_nu on that grid [erg/s/Hz].
    title : str
        Heading printed above the table.
    """
    print(f"\n  {title}")
    print(f"  {'band':<20} {'tengri/Synth':>15} {'med|resid|':>11}")
    print("  " + "-" * 51)
    for name, (a, b) in BANDS.items():
        m = (w_s >= a) & (w_s <= b) & (L_s > 0) & (L_t > 0)
        if not m.any():
            continue
        r = L_t[m] / L_s[m]
        med = float(np.median(r))
        flag = " OK" if 0.95 <= med <= 1.05 else "  <-- check"
        print(f"  {name:<20} {med:>14.3f}× {float(np.median(np.abs(r - 1))):>10.1%}{flag}")


def main():
    U.verify_unit_conversion(rtol=1e-3)
    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "synthesizer_test_grid.h5"))

    w_s, L_s = synthesizer_stellar_dust()

    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=TAU_DIFF)
    report(w_s, U.regrid(w_t, L_t, w_s), L_s, "tau_bc + tau_diff split (NOT Synthesizer-equivalent)")

    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=0.0)
    L_t_on_s = U.regrid(w_t, L_t, w_s)
    report(w_s, L_t_on_s, L_s, "tau_bc = 0, single diffuse screen (Synthesizer-equivalent)")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    um = w_s / 1e4
    nu = C_AA / w_s
    ax.loglog(um, nu * L_s / L_SUN, "C0-", lw=1.5, label="Synthesizer (matched)")
    ax.loglog(um, nu * L_t_on_s / L_SUN, "C1--", lw=1.5, label="tengri (matched)")
    ax.set_xlim(0.09, 3.0)
    ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
    ax.set_title("Matched-input stellar+attenuation — tengri vs Synthesizer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    resid = np.full(w_s.shape, np.nan)
    mm = (L_s > 0) & (L_t_on_s > 0)
    resid[mm] = L_t_on_s[mm] / L_s[mm] - 1.0
    axr.axhspan(-0.05, 0.05, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.set_xscale("log")
    axr.set_xlim(0.09, 3.0)
    axr.set_ylim(-0.3, 0.3)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
    axr.set_ylabel("tengri/Synthesizer − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGS / "synthesizer_validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
