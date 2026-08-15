"""Matched-input pixel validation: is the physics correct?

Companion to ``validate_moneyshot.py`` (which compares tengri-native vs
CIGALE-native *configurations*, where offsets are input-driven). This script
removes every input difference — same BC03 SSP, same metallicity, same
delayed-tau SFH, same Calzetti screen, same Dale-2014 templates+alpha — and
checks the stellar+dust SED **pixel by pixel**. At matched inputs the two
implementations must agree; a surviving residual is a physics/mapping problem.

Run:

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/cigale/validate_matched_physics.py

Needs ``_drivers/data/bc03_from_cigale.h5`` (regen:
``python reproduction/cigale/_drivers/cigale_ssp_to_dsps.py``) and the
CIGALE-sourced Dale templates ``data/dale2014_templates_cigale.h5``.

FINDING — the dust mapping matters
----------------------------------
CIGALE's ``dustatt_modified_starburst(E_BV_lines)`` applies a **single**
Calzetti screen ``E_BV_cont = 0.44 * E_BV_lines`` to the *entire* stellar
continuum. tengri's ``two_component`` is **Charlot & Fall**: it adds a
birth-cloud optical depth ``tau_bc`` to the *young-star* continuum on top of
the diffuse ``tau_diff``. So the CIGALE-equivalent tengri build is a single
diffuse screen — ``tau_bc = 0``, ``tau_diff = R_V * 0.44 * E_BV / 1.086`` —
NOT the ``tau_bc + tau_diff`` split used in 01_cigale.py §setup. With that
split both terms hit the continuum, over-attenuating the (young-star-dominated)
FUV by ~2x and inflating the re-radiated dust IR / radio by ~12-20%:

    mapping                FUV     opt     FIR
    tau_bc + tau_diff      0.50x   0.96x   1.12x   (01_cigale.py §setup — wrong)
    tau_bc = 0 (this file) 0.97x   1.00x   0.98x   (CIGALE-equivalent)

The two attenuation *models* are genuinely different (Charlot&Fall vs a single
modified-starburst screen) — tengri is not buggy — but they must not be mapped
onto each other via a BC+diffuse split. See issue #747.

Nebular is OFF on both sides (Cue emulator vs CIGALE BC03 nebular grid are
different models, §8). AGN disc parity is bit-exact in §9 (``schartmann2005``
== ``disk_type=1``); X-ray is issue #746; radio is §11 (1.5% at matched q_IR).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from reproduction._validation import (
    convention_sensitivity,
    filter_rows,
    print_filter_table,
)
from reproduction.cigale._drivers import cigale_driver as C, units as U

from tengri import FIXED, Fixed, SEDModel
from tengri.components.dust.emission_templates import register_dale2014_tabulated
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

L_SUN = 3.828e33
C_AA = 2.998e18

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched parameters (01_cigale.py setup).
MET_LOGZSOL = float(np.log10(0.02) + 1.848)  # CIGALE bc03 Z=0.02 -> +0.149
R_V, F_CONT, E_BV = 4.05, 0.44, 0.3
TAU_DIFF = R_V * F_CONT * E_BV / 1.086  # single Calzetti screen = E_BV_cont
TAU_GYR, AGE_GYR = 6.0, 12.0


def cigale_stellar_dust():
    sed = C.run_chain(
        [
            (
                "sfhdelayed",
                dict(
                    tau_main=TAU_GYR * 1e3,
                    age_main=AGE_GYR * 1e3,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            ("dustatt_modified_starburst", dict(E_BV_lines=E_BV)),
            ("dale2014", dict(alpha=2.0, fracAGN=0.0)),
        ]
    )
    return C.to_lnu(sed)


def tengri_stellar_dust(ssp, tau_bc):
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "*": FIXED},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(TAU_GYR),
            "age_gyr": Fixed(AGE_GYR),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(TAU_DIFF),
            "lyman_cutoff": True,
            "*": FIXED,
            "emission": {"type": "dale2014", "alpha_dale": Fixed(2.0), "*": FIXED},
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.sed_intrinsic)


def report(w_c, L_t, L_c, title, *, compact=False):
    """Print the bandpass ratio table for one configuration.

    Parameters
    ----------
    w_c : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom]; both SEDs are already on it.
    L_t, L_c : array_like, shape (n_wave,)
        tengri and CIGALE L_nu on that grid [erg/s/Hz].
    title : str
        Heading printed above the table.
    compact : bool, optional
        One summary line instead of the full ladder.
    """
    print_filter_table(
        filter_rows(w_c, L_t, L_c),
        ref_name="CIGALE",
        title=title,
        compact=compact,
    )


def main():
    register_dale2014_tabulated(
        str(HERE.parent.parent / "data" / "dale2014_templates_cigale.h5"), name="dale2014"
    )
    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "bc03_from_cigale.h5"))
    w_c, L_c = cigale_stellar_dust()

    # Wrong mapping (BC+diffuse split, as in 01_cigale.py §setup).
    tau_bc_split = R_V * (1.0 - F_CONT) * E_BV / 1.086
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=tau_bc_split)
    print("\n  Control (wrong in a known way):")
    report(
        w_c,
        U.regrid(w_t, L_t, w_c),
        L_c,
        "BC+diffuse split (01_cigale.py §setup mapping) — over-attenuates FUV",
        compact=True,
    )

    # Correct CIGALE-equivalent mapping (single diffuse screen).
    w_t2, L_t2 = tengri_stellar_dust(ssp, tau_bc=0.0)
    L_t2_c = U.regrid(w_t2, L_t2, w_c)
    report(w_c, L_t2_c, L_c, "single diffuse Calzetti screen (tau_bc=0) — CIGALE-equivalent")

    print(
        f"\n  bandpass-convention sensitivity (photon vs energy weight): "
        f"{convention_sensitivity(w_c, L_t2_c, L_c):.2e}"
    )

    # Parameter map.
    print("\n  parameter mapping (tengri <-> CIGALE):")
    rows = [
        ("SSP", "bc03_from_cigale.h5 (identical)", "ok"),
        (
            "metallicity",
            f"met_logzsol={MET_LOGZSOL:+.3f} <-> Z=0.02; Zsun=0.0142 vs 0.02",
            "convention",
        ),
        ("SFH norm", "log_total_mass=0 (1 Msun) <-> normalise=True", "ok"),
        (
            "dust amount",
            f"tau_diff={TAU_DIFF:.3f}=R_V·0.44·E_BV/1.086 <-> E_BV_lines={E_BV}",
            "ok (single screen)",
        ),
        (
            "dust model",
            "two_component (Charlot&Fall, tau_bc on young) != modified_starburst",
            "ISSUE: not 1:1",
        ),
        ("dust IR", "dale2014 alpha_dale=2.0 + cigale templates <-> dale2014 alpha=2.0", "ok"),
        ("AGN disc", "disc.schartmann2005 == skirtor2016 disk_type=1 (bit-exact, §9)", "ok"),
        ("AGN norm", "agn_log_lbol (abs) vs fracAGN (rel to galaxy)", "convention"),
        ("radio q_IR", "radio_q_ir default 2.64 (Bell03) vs qir_sf 2.5", "convention"),
        ("X-ray", "corona dropped in build (l_2500 not wired); N_H not exposed", "ISSUE #746"),
    ]
    for a, b, c in rows:
        print(f"    {a:<13} {b:<58} [{c}]")

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    nu = C_AA / w_c
    um = w_c / 1e4
    ax.loglog(um, nu * L_c / L_SUN, "C0-", lw=1.5, label="CIGALE (matched inputs)")
    ax.loglog(um, nu * L_t2_c / L_SUN, "C1--", lw=1.4, label="tengri (single screen, matched)")
    ax.loglog(
        um,
        nu * U.regrid(w_t, L_t, w_c) / L_SUN,
        "C3:",
        lw=1.0,
        label="tengri (BC+diffuse split — wrong mapping)",
    )
    ax.set_xlim(1e-2, 1e4)
    ymax = float(np.nanmax(nu * L_c / L_SUN))
    ax.set_ylim(ymax * 1e-4, ymax * 3)
    ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$, 1 $M_\odot$]")
    ax.set_title("Matched-input pixel validation — tengri vs CIGALE (stellar + dust)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    resid = np.full(w_c.shape, np.nan)
    mm = (L_c > 0) & (L_t2_c > 0)
    resid[mm] = L_t2_c[mm] / L_c[mm] - 1.0
    axr.axhspan(-0.05, 0.05, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.set_xscale("log")
    axr.set_ylim(-0.3, 0.3)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
    axr.set_ylabel("tengri/CIGALE − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS / "validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
