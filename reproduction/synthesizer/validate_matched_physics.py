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

Nebular is OFF for the stellar+attenuation comparison: Synthesizer bakes Cloudy
grids into its SSP grids while tengri uses the Cue emulator, which are
different models by design (01_synthesizer.py §8).

FINDING -- the dust IR agrees in energy and disagrees in shape, by one knob
--------------------------------------------------------------------------
The stellar+attenuation ladder stops at 2MASS Ks, so the IR is compared
separately on the **isolated Draine & Li 2007 component** -- Synthesizer's
``TotalEmission`` tree label, tengri's build differenced with and without the
``emission`` block. That reaches SCUBA-2 850 um, 13 bands from 3.4 to 863 um.

Integrated energy agrees: **L_IR (8-1000 um) ratio 1.052x**. The shape does
not. At tengri's defaults the MIR runs low and the sub-mm high:

    WISE W3 0.598x   MIPS 24 0.383x   PACS 160 1.503x   SPIRE 500 1.655x

That is a colder SED at fixed total energy, and it is **one unmatched
parameter**, not a modelling disagreement. Only two of DL07's three knobs map
through the two grammars: both expose ``qpah`` (Synthesizer's is a mass
*fraction*, 0.025 <-> tengri's percent, 2.5) and ``umin``. Synthesizer then
exposes ``alpha``, the :math:`dU \\propto U^{-\\alpha}` index; tengri exposes
``gamma_dl``, the PDR mass fraction. Different parameters, not two names for
one.

Sweeping the one tengri exposes moves the shape monotonically and leaves the
energy alone -- which is what a warm/cold redistribution at fixed L_IR must do:

    gamma_dl    W3      MIPS24   PACS160  SPIRE500   L_IR
    0.00        0.520x  0.206x   1.667x   1.858x     1.055x
    0.01 (def)  0.598x  0.383x   1.503x   1.655x     1.052x
    0.05        0.788x  0.815x   1.102x   1.159x     1.045x
    0.10        0.905x  1.082x   0.854x   0.853x     1.040x

Synthesizer's DL07 at ``alpha = 2.0`` therefore behaves like ``gamma_dl ~ 0.1``
in tengri's parameterization -- an order of magnitude above tengri's default of
0.01. The table below is printed at the default, i.e. matched on two knobs of
three, because tuning the third until the residual vanished would be fitting
the answer. The sweep is the evidence that one knob explains it.
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

from reproduction._validation import (
    IR_BANDS,
    UV_TO_NIR,
    convention_sensitivity,
    filter_rows,
    line_rows,
    print_filter_table,
    print_line_table,
)
from reproduction.synthesizer._drivers import synthesizer_driver as S, units as U

from tengri import DEFAULT, Fixed, SEDModel
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


# Dust IR section. Synthesizer's qpah is a mass *fraction*, tengri's is in
# percent: 0.025 <-> 2.5. umin matches directly. The third DL07 knob does not
# map -- see dust_emission_only.
QPAH_FRAC, QPAH_PCT = 0.025, 2.5
UMIN = 1.0
ALPHA_SYNTH = 2.0

# Nebular section: a young constant-SFR population, where the lines dominate
# (01_synthesizer.py fiducial).
NEB_AGE = 0.01  # Gyr
NEB_LOGU, NEB_LOGZ, NEB_LOGMASS = -2.0, 0.0, 9.0


def dust_emission_only(ssp):
    """The Draine & Li 2007 IR component alone, from both codes.

    Isolating the dust-emission component is what makes the IR comparable at
    all here. Synthesizer's ``TotalEmission`` attenuates the *reprocessed*
    (stellar + nebular) spectrum, so its total carries a Cloudy c23.01 nebular
    contribution that the matched tengri build does not; taking the tree's
    ``dust_emission`` label keeps that out of the IR question. On the tengri
    side the component is recovered by differencing the same build with and
    without the ``emission`` block, so the attenuated stellar continuum
    cancels exactly.

    Parameters
    ----------
    ssp : SSPData
        The re-shaped Synthesizer stellar grid.

    Returns
    -------
    tuple of ndarray
        ``(w_synth, L_synth, w_tengri, L_tengri)``; L_nu in [erg/s/Hz].

    Notes
    -----
    Only two of the three DL07 knobs can be matched through the two grammars.
    Both expose ``qpah`` and ``umin``; Synthesizer then exposes ``alpha`` (the
    :math:`dU \\propto U^{-\\alpha}` index) while tengri exposes ``gamma_dl``
    (the PDR mass fraction). They are different parameters, not two names for
    one, so each side keeps its own default for the third and the comparison is
    that much weaker. Reported rather than hidden: a validator that quietly
    matched two of three and presented the result as fully matched would be
    making the mistake this whole file exists to catch.
    """
    w_s, L_s = S.total_emission(
        tau_gyr=TAU_GYR,
        max_age_gyr=AGE_GYR,
        metallicity=Z_ABS,
        log_mass=LOG_MASS,
        av=A_V,
        qpah=QPAH_FRAC,
        umin=UMIN,
        alpha=ALPHA_SYNTH,
        components=("dust_emission",),
    )["dust_emission"]

    def _tengri(with_emission):
        dust = {
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(TAU_DIFF),
            "all_params": Fixed(DEFAULT),
        }
        if with_emission:
            dust["emission"] = {
                "type": "draine_li2007",
                "qpah": Fixed(QPAH_PCT),
                "umin": Fixed(UMIN),
                "all_params": Fixed(DEFAULT),
            }
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
            dust_attenuation=dust,
            redshift=Fixed(0.0),
        )
        s = m.predict_state({})
        return np.asarray(s.wave), np.asarray(s.sed_intrinsic)

    w_on, L_on = _tengri(True)
    w_off, L_off = _tengri(False)
    # Attaching DL07 extends the master grid past the stellar templates, so the
    # two builds do not share one. Regrid the no-emission baseline onto the
    # with-emission grid before differencing; beyond the stellar grid's reach
    # regrid returns 0, which is the right baseline there.
    L_off_on = U.regrid(w_off, L_off, w_on)
    return w_s, np.clip(L_s, 0.0, None), w_on, np.clip(L_on - L_off_on, 0.0, None)


def nebular_only(ssp):
    """Nebular-only L_nu from both codes at matched metallicity.

    Synthesizer returns its ``nebular`` spectrum directly from
    ``NebularEmission``; tengri publishes ``sed_nebular``.

    Parameters
    ----------
    ssp : SSPData
        The repackaged Synthesizer test grid.

    Returns
    -------
    tuple of ndarray
        ``(w_synth, L_synth, w_tengri, L_tengri)``; L_nu in [erg/s/Hz].
    """
    w_s, L_s = S.nebular_sed(age_gyr=NEB_AGE, metallicity=Z_ABS, log_mass=NEB_LOGMASS)
    L_s = np.clip(L_s, 0.0, None)

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
    return w_s, L_s, np.asarray(s.wave), np.asarray(s.derived["sed_nebular"])


def report(w_s, L_t, L_s, title, *, compact=False):
    """Print the bandpass ratio table for one configuration.

    Parameters
    ----------
    w_s : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom]; both SEDs are already on it.
    L_t, L_s : array_like, shape (n_wave,)
        tengri and Synthesizer L_nu on that grid [erg/s/Hz].
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
        filter_rows(w_s, L_t, L_s, filters=UV_TO_NIR),
        ref_name="Synth",
        title=title,
        compact=compact,
    )


def main():
    U.verify_unit_conversion(rtol=1e-3)
    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "synthesizer_test_grid.h5"))

    w_s, L_s = synthesizer_stellar_dust()

    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=TAU_DIFF)
    print("\n  Control (wrong in a known way):")
    report(
        w_s,
        U.regrid(w_t, L_t, w_s),
        L_s,
        "tau_bc + tau_diff split (NOT Synthesizer-equivalent)",
        compact=True,
    )

    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=0.0)
    L_t_on_s = U.regrid(w_t, L_t, w_s)
    report(w_s, L_t_on_s, L_s, "tau_bc = 0, single diffuse screen (Synthesizer-equivalent)")

    print(
        f"\n  bandpass-convention sensitivity (photon vs energy weight): "
        f"{convention_sensitivity(w_s, L_t_on_s, L_s, filters=UV_TO_NIR):.2e}"
    )

    # The IR half, on the isolated Draine & Li 2007 component so the ladder
    # reaches SCUBA-2 without the stellar+attenuation comparison's scope limit.
    w_ds, L_ds, w_dt, L_dt = dust_emission_only(ssp)
    L_dt_on_s = U.regrid(w_dt, L_dt, w_ds)
    print_filter_table(
        filter_rows(w_ds, L_dt_on_s, L_ds, filters=IR_BANDS),
        ref_name="Synth",
        title="Dust IR alone — Draine & Li 2007, matched qpah and umin",
    )
    # L_IR = int L_lambda dlambda over 8-1000 um, the standard definition.
    # Integrate on an ascending grid: reversing both arrays flips the sign.
    order = np.argsort(w_ds)
    w_o = w_ds[order]
    band = (w_o >= 8e4) & (w_o <= 1e7)
    to_llam = C_AA / w_o**2
    lir_t = float(np.trapezoid((L_dt_on_s[order] * to_llam)[band], w_o[band]))
    lir_s = float(np.trapezoid((L_ds[order] * to_llam)[band], w_o[band]))
    if lir_s > 0:
        print(
            f"  integrated L_IR (8-1000 um) ratio {lir_t / lir_s:.3f}x — the "
            "energy-balance budgets.\n  Synthesizer re-emits a nebular "
            "contribution the matched tengri build has not."
        )

    # Emission lines, at matched gas parameters. Not a parity check -- see
    # print_line_table's Notes.
    w_sn, L_sn, w_tn, L_tn = nebular_only(ssp)
    print_line_table(
        line_rows(w_tn, L_tn, w_sn, L_sn, line_lum=U.line_lum),
        ref_name="Synth",
        title="Emission lines — tengri Cue vs Synthesizer Cloudy c23.01 (matched Z)",
    )

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
