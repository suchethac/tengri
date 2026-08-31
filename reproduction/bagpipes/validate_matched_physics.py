"""Matched-input pixel validation: is the physics correct?

Companion to ``01_bagpipes.py`` (which compares tengri-native vs
BAGPIPES-native *configurations*, where offsets are input-driven). This script
removes every input difference — same BC03+MILES Kroupa SSP, same metallicity,
same delayed-tau SFH, same Calzetti screen — and checks the stellar SED
**pixel by pixel**. At matched inputs the two implementations must agree; a
surviving residual is a physics/mapping problem.

Run:

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/bagpipes/validate_matched_physics.py

Needs ``bagpipes`` and the repackaged grid
``_drivers/data/bc03_miles_from_bagpipes.h5`` (regen:
``python -m reproduction.bagpipes._drivers.bagpipes_ssp_to_dsps``).

FINDING -- the dust mapping matters
-----------------------------------
BAGPIPES' ``{"type": "Calzetti", "Av": A_V}`` applies a **single** screen of
optical depth ``tau_V = A_V / 1.086`` to the *entire* stellar continuum.
tengri's ``two_component`` is **Charlot & Fall**: it adds a birth-cloud optical
depth ``tau_bc`` to the *young-star* continuum on top of the diffuse
``tau_diff``. The BAGPIPES-equivalent tengri build is therefore a single
diffuse screen -- ``tau_bc = 0``, ``tau_diff = A_V / 1.086`` -- not a
birth-cloud/diffuse split. The two attenuation *models* are genuinely
different and must not be mapped onto each other via a split. The CIGALE and
Prospector validators reach the same conclusion independently (issue #747).

Nebular is OFF on both sides: BAGPIPES uses Cloudy v25 grids and tengri uses
the Cue emulator trained on Cloudy v17, which are different models by design
(01_bagpipes.py §9 quantifies the Halpha ratio).
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
    line_rows,
    print_filter_table,
    print_line_table,
)
from reproduction.bagpipes._drivers import bagpipes_driver as B, units as U

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

L_SUN = 3.828e33
C_AA = 2.998e18

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched parameters (01_bagpipes.py setup).
MET_LOGZSOL = 0.0  # BAGPIPES metallicity = 1.0 Z/Zsun
LOG_MASS = 10.0
TAU_GYR, AGE_GYR = 1.0, 5.0
A_V = 1.0
TAU_DIFF = A_V / 1.086  # single Calzetti screen


def bagpipes_stellar_dust():
    """BAGPIPES stellar SED behind a single Calzetti screen.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    return B.attenuated_lnu(
        dust_block={"type": "Calzetti", "Av": A_V},
        sfh_type="delayed",
        massformed=LOG_MASS,
        metallicity=1.0,
        age=AGE_GYR,
        tau=TAU_GYR,
    )


def tengri_stellar_dust(ssp, tau_bc):
    """tengri stellar SED at the matched parameters.

    Parameters
    ----------
    ssp : SSPData
        The repackaged BC03+MILES Kroupa grid.
    tau_bc : float
        Birth-cloud optical depth. ``0.0`` is the BAGPIPES-equivalent mapping.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(TAU_GYR),
            "age_gyr": Fixed(AGE_GYR),
            "log_total_mass": Fixed(LOG_MASS),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(tau_bc),
            "tau_diff": Fixed(TAU_DIFF),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.sed_intrinsic)


# Nebular section: a young constant-SFR population, where the lines dominate
# (01_bagpipes.py §9 fiducial).
NEB_AGE = 0.01  # Gyr
NEB_LOGU, NEB_LOGZ, NEB_LOGMASS = -2.0, 0.0, 9.0

# BAGPIPES' default 747-point grid spans 1 A to 1e8 A and smears Cloudy v25's
# lines into bumps; §9 hands it a dense optical grid instead.
_NEB_SPEC_WAVS = np.arange(900.0, 7000.0, 1.0)


def report(w_b, L_t, L_b, title, *, compact=False):
    """Print the bandpass ratio table for one configuration.

    Parameters
    ----------
    w_b : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom]; both SEDs are already on it.
    L_t, L_b : array_like, shape (n_wave,)
        tengri and BAGPIPES L_nu on that grid [erg/s/Hz].
    title : str
        Heading printed above the table.
    compact : bool, optional
        One summary line instead of the full ladder.
    """
    print_filter_table(
        filter_rows(w_b, L_t, L_b, filters=UV_TO_NIR),
        ref_name="BAGPIPES",
        title=title,
        compact=compact,
    )


def nebular_only(ssp):
    """Nebular-only L_nu from both codes at matched gas parameters.

    BAGPIPES is isolated as (neb on) - (neb off) at the same SFH; tengri
    publishes ``sed_nebular`` directly.

    Parameters
    ----------
    ssp : SSPData
        The repackaged BC03/MILES grid.

    Returns
    -------
    tuple of ndarray
        ``(w_bagpipes, L_bagpipes, w_tengri, L_tengri)``; L_nu in [erg/s/Hz].
    """
    base = {"metallicity": 1.0, "age_min": 0.0, "age_max": NEB_AGE, "massformed": NEB_LOGMASS}
    on = B._build_model(
        {"redshift": 0.0, "constant": base, "nebular": {"logU": NEB_LOGU}},
        spec_wavs=_NEB_SPEC_WAVS,
    )
    off = B._build_model({"redshift": 0.0, "constant": base}, spec_wavs=_NEB_SPEC_WAVS)
    w_b = on.spectrum[:, 0]
    to_lnu = w_b**2 / U.C_ANGSTROM_PER_S
    L_b = np.clip((on.spectrum[:, 1] - off.spectrum[:, 1]) * to_lnu, 0.0, None)

    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
        sfh={
            "type": "const",
            "start_gyr": Fixed(NEB_AGE),
            "end_gyr": Fixed(0.0),
            "log_total_mass": Fixed(NEB_LOGMASS),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_logU": Fixed(NEB_LOGU),
            "neb_logZ_gas": Fixed(NEB_LOGZ),
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return w_b, L_b, np.asarray(s.wave), np.asarray(s.derived["sed_nebular"])


def main():
    U.verify_unit_conversion(rtol=1e-3)
    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "bc03_miles_from_bagpipes.h5"))

    w_b, L_b = bagpipes_stellar_dust()

    # The wrong mapping: a birth-cloud/diffuse split of the same A_V.
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=TAU_DIFF)
    print("\n  Control (wrong in a known way):")
    report(
        w_b,
        U.regrid(w_t, L_t, w_b),
        L_b,
        "tau_bc + tau_diff split (NOT BAGPIPES-equivalent)",
        compact=True,
    )

    # The BAGPIPES-equivalent mapping: one diffuse screen.
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=0.0)
    L_t_on_b = U.regrid(w_t, L_t, w_b)
    report(w_b, L_t_on_b, L_b, "tau_bc = 0, single diffuse screen (BAGPIPES-equivalent)")

    print(
        f"\n  bandpass-convention sensitivity (photon vs energy weight): "
        f"{convention_sensitivity(w_b, L_t_on_b, L_b, filters=UV_TO_NIR):.2e}"
    )
    print(
        "  ladder stops at 2MASS Ks by design: BAGPIPES applies energy balance and\n"
        "  re-emits in the IR, the matched tengri build carries no dust.emission\n"
        "  block, so redder bands would compare two different models (see UV_TO_NIR)."
    )

    # Emission lines, at matched gas parameters. Not a parity check -- see
    # print_line_table's Notes.
    w_bn, L_bn, w_tn, L_tn = nebular_only(ssp)
    print_line_table(
        line_rows(w_tn, L_tn, w_bn, L_bn, line_lum=U.line_lum),
        ref_name="BAGPIPES",
        title="Emission lines — tengri Cue vs BAGPIPES Cloudy v25 (matched logU, logZ_gas)",
    )

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    um = w_b / 1e4
    nu = C_AA / w_b
    ax.loglog(um, nu * L_b / L_SUN, "C0-", lw=1.5, label="BAGPIPES (matched)")
    ax.loglog(um, nu * L_t_on_b / L_SUN, "C1--", lw=1.5, label="tengri (matched)")
    ax.set_xlim(0.09, 3.0)
    ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
    ax.set_title("Matched-input stellar+attenuation — tengri vs BAGPIPES")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    resid = np.full(w_b.shape, np.nan)
    mm = (L_b > 0) & (L_t_on_b > 0)
    resid[mm] = L_t_on_b[mm] / L_b[mm] - 1.0
    axr.axhspan(-0.05, 0.05, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.set_xscale("log")
    axr.set_xlim(0.09, 3.0)
    axr.set_ylim(-0.3, 0.3)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
    axr.set_ylabel("tengri/BAGPIPES − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGS / "bagpipes_validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
