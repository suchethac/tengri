"""Matched-input pixel validation: is the physics correct?

Companion to ``01_prospect_r.py`` (which compares tengri-native vs
ProSpect-native *configurations*, where offsets are input-driven). This script
removes every input difference — same BC03 library, same metallicity, same
skew-normal SFH, same Charlot & Fall attenuation — and checks the stellar SED
**pixel by pixel**. At matched inputs the two implementations must agree; a
surviving residual is a physics/mapping problem.

Run:

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/prospect_r/validate_matched_physics.py

Needs R with the ``ProSpect`` and ``ProSpectData`` packages, reachable through
``rpy2``.

FINDING -- here the birth-cloud/diffuse split is the CORRECT mapping
-------------------------------------------------------------------
The other four matched-input validators in this repo (CIGALE, Prospector,
BAGPIPES, Synthesizer) all conclude that tengri's ``two_component`` dust must
be collapsed to a **single diffuse screen** (``tau_bc = 0``) to match the
reference code, because those codes apply one screen to the whole continuum.

ProSpect is the exception, and the contrast is the point: it is natively
**Charlot & Fall**, with ``tau_birth`` on young stars and ``tau_screen`` on all
stars, each a power law. tengri's ``two_component`` is the same model, so the
mapping is direct — ``tau_bc = tau_birth``, ``tau_diff = tau_screen``, both
laws ``power_law`` at the shared slope. Collapsing to a single screen would be
*wrong* here.

The lesson generalizes: the correct dust mapping is a property of the reference
code's attenuation model, not a house convention. See issue #747 for the
CIGALE case where the split was applied wrongly.

Nebular is OFF on both sides: ProSpect ties Halpha to the SFR and distributes
other lines via Levesque et al. (2010), while tengri uses the Cue emulator on
Cloudy 17 — different models by design (01_prospect_r.py §8).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from reproduction._validation import (
    UV_TO_NIR,
    convention_sensitivity,
    filter_rows,
    print_filter_table,
)
from reproduction.prospect_r._drivers import prospect_driver as P, units as U

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

L_SUN = 3.828e33
C_AA = 2.998e18
LOG10_ZSUN = -1.848  # Asplund 2009, Zsun = 0.0142 (CLAUDE.md metallicity note)

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched parameters (01_prospect_r.py setup).
Z_ABS = 0.02
MET_LOGZSOL = float(np.log10(Z_ABS) - LOG10_ZSUN)
TAU_BIRTH, TAU_SCREEN = 1.0, 0.3
POW_SLOPE = -0.7
SFH_PARS = {"mSFR": 10.0, "mpeak": 10.0, "mperiod": 0.3, "mskew": 0.0}


def prospect_stellar_dust():
    """ProSpect attenuated stellar SED at the matched parameters.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    out = P.prospect_sed(
        massfunc="snorm",
        sfh_pars=SFH_PARS,
        Z=Z_ABS,
        tau_birth=TAU_BIRTH,
        tau_screen=TAU_SCREEN,
        pow_birth=POW_SLOPE,
        pow_screen=POW_SLOPE,
    )
    return out["StarsAtten"]


def tengri_stellar_dust(ssp, log_mass, *, collapse_screen=False):
    """tengri stellar SED at the matched parameters.

    Parameters
    ----------
    ssp : SSPData
        The BC03 grid tengri reads.
    log_mass : float
        ``log10`` of the total mass formed [Msun], matched to ProSpect's
        skew-normal normalization.
    collapse_screen : bool, optional
        If True, put all optical depth on the diffuse component -- the mapping
        the other four validators need, and which is *wrong* for ProSpect.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    tau_bc = 0.0 if collapse_screen else TAU_BIRTH
    tau_diff = (TAU_BIRTH + TAU_SCREEN) if collapse_screen else TAU_SCREEN
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "*": FIXED},
        sfh={
            "type": "snorm",
            "peak_lbt_gyr": Fixed(SFH_PARS["mpeak"]),
            "width_gyr": Fixed(SFH_PARS["mperiod"]),
            "skew": Fixed(SFH_PARS["mskew"]),
            "log_total_mass": Fixed(log_mass),
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "law_bc": "power_law",
            "law_diff": "power_law",
            "slope_bc": POW_SLOPE,
            "slope_diff": POW_SLOPE,
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(tau_diff),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.sed_intrinsic)


def report(w_p, L_t, L_p, title, *, compact=False):
    """Print the bandpass ratio table for one configuration.

    Parameters
    ----------
    w_p : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom]; both SEDs are already on it.
    L_t, L_p : array_like, shape (n_wave,)
        tengri and ProSpect L_nu on that grid [erg/s/Hz].
    title : str
        Heading printed above the table.
    compact : bool, optional
        One summary line instead of the full ladder.

    Notes
    -----
    Scoped to :data:`~reproduction._validation.UV_TO_NIR`: this comparison is
    stellar + attenuation, with no matched dust-emission block on either side.
    """
    print_filter_table(
        filter_rows(w_p, L_t, L_p, filters=UV_TO_NIR),
        ref_name="ProSpect",
        title=title,
        compact=compact,
    )


def main():
    U.verify_unit_conversion(rtol=1e-3)
    import tengri

    ssp_path = tengri.download_ssp(
        "bc03_pdva_stelib_chabrier", dest=str(HERE / "_drivers" / "data")
    )
    ssp = load_ssp_data(str(ssp_path))

    w_p, L_p = prospect_stellar_dust()

    # Normalize tengri to ProSpect's formed mass by matching the NIR, where
    # attenuation is negligible, so the comparison is of shape not bookkeeping.
    w_t, L_t = tengri_stellar_dust(ssp, log_mass=10.0)
    ref = (w_p > 1.5e4) & (w_p < 2.5e4)
    scale = float(np.median(L_p[ref] / U.regrid(w_t, L_t, w_p)[ref]))
    log_mass = 10.0 + float(np.log10(scale))

    w_t, L_t = tengri_stellar_dust(ssp, log_mass=log_mass)
    L_t_on_p = U.regrid(w_t, L_t, w_p)
    report(w_p, L_t_on_p, L_p, "tau_bc = tau_birth, tau_diff = tau_screen (ProSpect-equivalent)")

    w_t2, L_t2 = tengri_stellar_dust(ssp, log_mass=log_mass, collapse_screen=True)
    print("\n  Control (wrong in a known way):")
    report(
        w_p,
        U.regrid(w_t2, L_t2, w_p),
        L_p,
        "collapsed to one screen (the OTHER codes' mapping — wrong here)",
        compact=True,
    )

    print(
        f"\n  bandpass-convention sensitivity (photon vs energy weight): "
        f"{convention_sensitivity(w_p, L_t_on_p, L_p, filters=UV_TO_NIR):.2e}"
    )

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    um = w_p / 1e4
    nu = C_AA / w_p
    ax.loglog(um, nu * L_p / L_SUN, "C0-", lw=1.5, label="ProSpect (matched)")
    ax.loglog(um, nu * L_t_on_p / L_SUN, "C1--", lw=1.5, label="tengri (matched)")
    ax.set_xlim(0.09, 3.0)
    ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
    ax.set_title("Matched-input stellar+Charlot&Fall — tengri vs ProSpect")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    resid = np.full(w_p.shape, np.nan)
    mm = (L_p > 0) & (L_t_on_p > 0)
    resid[mm] = L_t_on_p[mm] / L_p[mm] - 1.0
    axr.axhspan(-0.05, 0.05, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.set_xscale("log")
    axr.set_xlim(0.09, 3.0)
    axr.set_ylim(-0.3, 0.3)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
    axr.set_ylabel("tengri/ProSpect − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGS / "prospect_r_validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
