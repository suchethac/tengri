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
    print_radio_table,
    print_xray_table,
    radio_rows,
    xray_rows,
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


# Radio section (01_cigale.py §11). CIGALE's qir_sf = 2.5, alpha_sf = 0.8;
# tengri's bucket default q_IR is 2.64, so it must be pinned to match.
Q_IR, ALPHA_SF = 2.5, 0.8

# X-ray section (01_cigale.py §10). alpha_ox = -1.4 is CIGALE's default; the
# Just+2007 relation L_2keV = L_2500 * 10**(alpha_ox/0.3838) then fixes the
# L_2500 both codes must be solved onto.
ALPHA_OX = -1.4
LOG_L2500_TARGET = (2.638 - ALPHA_OX) / 0.137
INCL_DEG = 30.0


def xray_seds(ssp):
    """AGN corona from both codes, solved onto one intrinsic ``L_2500``.

    Parameters
    ----------
    ssp : SSPData
        The repackaged CIGALE BC03 grid.

    Returns
    -------
    tuple
        ``(w_cigale, L_cigale, w_tengri, L_tengri, log_l2500_c, log_l2500_t)``;
        L_nu in [erg/s/Hz], corona only on both sides.

    Notes
    -----
    The anchor is the whole comparison. :math:`\\alpha_{ox}` is *defined*
    between :math:`L_{2500}` and :math:`L_{2\\,\\mathrm{keV}}`, so unless both
    codes sit on the same intrinsic disc :math:`L_{2500}` the X-ray ratio is
    reporting the disc normalization rather than the corona. Each side is
    therefore solved from one trial run: CIGALE's disc scales linearly with
    ``sfr_A`` at fixed ``fracAGN``, tengri's ``schartmann2005`` has an
    L_bol-independent shape, so one evaluation gives the scale factor exactly.

    Inclination is pinned to 30 deg on both sides (Yang+2020, #980) and
    ``N_H`` to zero. tengri's constant 1% scattered floor (Ricci et al. 2017)
    has no CIGALE counterpart and shows up as roughly +1% in the ratio.
    """
    sfh = dict(
        tau_main=TAU_GYR * 1e3,
        age_main=AGE_GYR * 1e3,
        tau_burst=50,
        age_burst=20,
        f_burst=0.0,
        sfr_A=1.0,
        normalise=False,
    )

    def _cigale(sfr_a):
        return C.run_chain(
            [
                ("sfhdelayed", {**sfh, "sfr_A": sfr_a}),
                ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
                ("dustatt_modified_starburst", dict(E_BV_lines=E_BV)),
                ("dale2014", dict(alpha=2.0)),
                (
                    "skirtor2016",
                    dict(
                        t=7, pl=1.0, q=1.0, oa=40, R=20, Mcl=0.97, i=INCL_DEG,
                        disk_type=1, delta=-0.36, fracAGN=0.3, law=0, EBV=0.0,
                        temperature=100, emissivity=1.6,
                    ),
                ),
                (
                    "yang20",
                    dict(
                        gam=1.8, E_cut=300.0, alpha_ox=ALPHA_OX,
                        max_dev_alpha_ox=0.2, angle_coef="0.5 & 0",
                        det_lmxb=0.0, det_hmxb=0.0,
                    ),
                ),
            ]
        )

    def _l2500(sed):
        return float(sed.info["agn.intrin_Lnu_2500A_30deg"]) * 1e7  # W/Hz -> erg/s/Hz

    trial = _cigale(1e8)
    sed_c = _cigale(1e8 * 10.0**LOG_L2500_TARGET / _l2500(trial))
    w_c, L_c = U.wnm_to_erg_per_hz_per_aa(
        np.asarray(sed_c.wavelength_grid), np.asarray(sed_c.luminosities["xray.agn"])
    )
    l2500_c = _l2500(sed_c)

    cos_inc = float(np.cos(np.radians(INCL_DEG)))

    def _tengri(log_lbol):
        return SEDModel.build(
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
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(TAU_DIFF),
                "*": FIXED,
            },
            agn={
                "type": "composable",
                "disc": {"type": "schartmann2005", "*": FIXED},
                "torus": {"type": "skirtor", "*": FIXED},
                "agn_log_lbol": Fixed(log_lbol),
                "agn_cos_inc": Fixed(cos_inc),
                "*": FIXED,
            },
            xray={"type": "yang20", "log_nh": Fixed(0.0), "*": FIXED},
            redshift=Fixed(0.0),
        ).predict_state({})

    st = _tengri(11.5)
    log_lbol = 11.5 + float(
        np.log10(l2500_c / float(np.asarray(st.derived["L_2500_intrinsic"])))
    )
    st = _tengri(log_lbol)
    l2500_t = float(np.asarray(st.derived["L_2500_intrinsic"]))
    return (
        w_c,
        L_c,
        np.asarray(st.wave),
        np.asarray(st.derived["sed_xray"]),
        np.log10(l2500_c),
        np.log10(l2500_t),
    )


def radio_seds(ssp):
    """Star-forming radio from both codes at matched q_IR and alpha_sf.

    Parameters
    ----------
    ssp : SSPData
        The repackaged CIGALE BC03 grid.

    Returns
    -------
    tuple of ndarray
        ``(w_cigale, L_cigale, w_tengri, L_tengri)``; L_nu in [erg/s/Hz].
        CIGALE's is ``radio.sf_nonthermal``, synchrotron only -- its module
        carries no free-free term (#863). tengri's ``sed_radio`` is
        synchrotron + Murphy 2011 free-free, so the two are deliberately not
        the same quantity; that is the finding, not a defect.

    Notes
    -----
    The dust setup must match the rest of this file exactly. Both codes anchor
    the synchrotron on the absorbed luminosity through
    ``L_ref = L_dust / (3.75e12 * 10**q_IR)``, so any mismatch in the dust
    budget lands in the radio 1:1 and would read as a radio disagreement.
    """
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
            ("dale2014", dict(alpha=2.0)),
            ("radio", dict(qir_sf=Q_IR, alpha_sf=ALPHA_SF, R_agn=0.0, alpha_agn=0.7)),
        ]
    )
    w_c, L_c = U.wnm_to_erg_per_hz_per_aa(
        np.asarray(sed.wavelength_grid), np.asarray(sed.luminosities["radio.sf_nonthermal"])
    )

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
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(TAU_DIFF),
            "lyman_cutoff": True,
            "*": FIXED,
            "emission": {"type": "dale2014", "alpha_dale": Fixed(2.0), "*": FIXED},
        },
        radio={
            "type": "condon92",
            "radio_q_ir": Fixed(Q_IR),
            "radio_alpha_sf": Fixed(ALPHA_SF),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return w_c, L_c, np.asarray(s.wave), np.asarray(s.derived["sed_radio"])


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

    # X-ray, both codes on one intrinsic L_2500.
    w_xc, L_xc, w_xt, L_xt, lg_c, lg_t = xray_seds(ssp)
    print(
        f"\n  matched disc log10 L_2500: CIGALE {lg_c:.4f}, tengri {lg_t:.4f} "
        f"(target {LOG_L2500_TARGET:.4f})"
    )
    print_xray_table(
        xray_rows(w_xc, U.regrid(w_xt, L_xt, w_xc), L_xc),
        ref_name="CIGALE",
        title="AGN corona — tengri xray.yang20 vs CIGALE xray.agn (Yang+2020)",
    )
    print(
        "  N_H = 0 and i = 30 deg on both sides. tengri adds a constant 1%\n"
        "  scattered floor (Ricci et al. 2017) that CIGALE has no counterpart\n"
        "  for, which is the expected sign and size of a flat offset here."
    )

    # Radio, at matched q_IR and alpha_sf.
    w_rc, L_rc, w_rt, L_rt = radio_seds(ssp)
    print_radio_table(
        radio_rows(w_rc, U.regrid(w_rt, L_rt, w_rc), L_rc),
        ref_name="CIGALE",
        title="Star-forming radio — tengri condon92 vs CIGALE radio.sf_nonthermal",
    )
    print(
        "  CIGALE's module is synchrotron only (#863); tengri's condon92 adds\n"
        "  Murphy 2011 free-free. Free-free is flat (alpha ~ 0.1) where\n"
        "  synchrotron is steep (0.8), so its share climbs with frequency —\n"
        "  which is what the flatter tengri spectral index above is measuring."
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
        ("X-ray", "xray.yang20 corona, log_nh exposed; matched on L_2500 (#746 closed)", "ok"),
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
