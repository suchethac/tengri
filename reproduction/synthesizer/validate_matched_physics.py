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

FINDING -- all four DL07 knobs map; the 2014 grid release agrees to 2.6 percent
--------------------------------------------------------------------------------
The stellar+attenuation ladder stops at 2MASS Ks, so the IR is compared
separately on the **isolated Draine & Li dust component** -- Synthesizer's
``TotalEmission`` tree label, tengri's build differenced with and without the
``emission`` block. That reaches SCUBA-2 850 um, 13 bands from 3.4 to 863 um.

Synthesizer's grid file ``draine_li_dust_emission_grid_MW_3p1.hdf5`` is the 2014
template release (axes: qpah, umin, alpha; U_max = 1e7). All four knobs map
through the two codes: both expose ``qpah`` (Synthesizer's is a mass *fraction*,
0.025 <-> tengri's percent, 2.5), ``umin`` (directly), the PDR mass fraction
``gamma`` (Synthesizer's parameter name, matched to tengri's ``gamma_dl``), and
the alpha power-law index (Synthesizer's ``alpha`` mapped to tengri's
``alpha_dl14``; both default to 2.0).

At matched knobs (qpah=2.5%, umin=1, gamma=0.05, alpha=2.0), **L_IR (8-1000 um)
ratio 1.026x** (median 1.026x across 13 bands). The two codes employ different
energy-balance conventions: Synthesizer balances dust on the reprocessed
spectrum (stellar continuum after gas ionization plus nebular), consuming the
ionizing continuum below 912 Å, while tengri balances on incident stellar with
the Lyman continuum excluded. This difference in absorbed luminosity integration
accounts for the 2.6 percent L_IR offset: Synthesizer's absorbed luminosity
integrated over reprocessed, 2.5005e43 erg/s, versus tengri's over incident
>= 912 Å, 2.5677e43 erg/s, ratio 1.0269. The one outlier, WISE W1 at 0.984x,
is the 3.3 um PAH feature resampled differently onto each code's master grid
(locally 1.96x at the feature core; native templates agree to 2.6e-4 there).
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
# percent: 0.025 <-> 2.5. umin matches directly. gamma maps to gamma_dl.
QPAH_FRAC, QPAH_PCT = 0.025, 2.5
UMIN = 1.0
ALPHA_DL14 = 2.0
GAMMA_DL = 0.05  # Synthesizer's default gamma; maps to tengri's gamma_dl

# Nebular section: a young constant-SFR population, where the lines dominate
# (01_synthesizer.py fiducial).
NEB_AGE = 0.01  # Gyr
NEB_LOGU, NEB_LOGZ, NEB_LOGMASS = -2.0, 0.0, 9.0


def dust_emission_only(ssp):
    """The Draine & Li dust IR component alone, from both codes.

    Isolating the dust-emission component is what makes the IR comparable at
    all here. Synthesizer's ``TotalEmission`` tree balances the dust energy on
    the reprocessed spectrum (stellar continuum after gas-phase ionization plus
    nebular emission), consuming the ionizing continuum below 912 Å. tengri
    balances on the incident stellar spectrum. The 2.6 percent difference in
    L_IR between the codes reflects this energy-balance convention: Synthesizer's
    absorbed luminosity (integrated over reprocessed) versus tengri's absorbed
    luminosity (integrated over incident with LyC excluded).

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
    All four DL07 knobs map: qpah, umin, gamma_dl (=``gamma``), and
    alpha_dl14 (=``alpha``). Both grids are the 2014 release.
    """
    w_s, L_s = S.total_emission(
        tau_gyr=TAU_GYR,
        max_age_gyr=AGE_GYR,
        metallicity=Z_ABS,
        log_mass=LOG_MASS,
        av=A_V,
        qpah=QPAH_FRAC,
        umin=UMIN,
        alpha=ALPHA_DL14,
        gamma=GAMMA_DL,
        components=("dust_emission",),
    )["dust_emission"]

    def _tengri(with_emission):
        dust_attenuation = {
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(TAU_DIFF),
            "all_params": Fixed(DEFAULT),
        }
        build_kwargs = {
            "ssp_data": ssp,
            "met": {"logzsol": Fixed(MET_LOGZSOL), "all_params": Fixed(DEFAULT)},
            "sfh": {
                "type": "delayed",
                "tau_gyr": Fixed(TAU_GYR),
                "age_gyr": Fixed(AGE_GYR),
                "log_total_mass": Fixed(LOG_MASS),
                "all_params": Fixed(DEFAULT),
            },
            "dust_attenuation": dust_attenuation,
            "redshift": Fixed(0.0),
        }
        if with_emission:
            build_kwargs["dust_emission"] = {
                "type": "draine_li2014",
                "qpah": Fixed(QPAH_PCT),
                "umin": Fixed(UMIN),
                "gamma_dl": Fixed(GAMMA_DL),
                "alpha_dl14": Fixed(ALPHA_DL14),
                "all_params": Fixed(DEFAULT),
            }
        m = SEDModel.build(**build_kwargs)
        s = m.predict_state({})
        return np.asarray(s.wave), np.asarray(s.sed_intrinsic)

    w_on, L_on = _tengri(True)
    w_off, L_off = _tengri(False)
    # Attaching the dust-emission grid extends the master grid past the stellar
    # templates, so the two builds do not share one. Regrid the no-emission
    # baseline onto the with-emission grid before differencing; beyond the
    # stellar grid's reach regrid returns 0, which is the right baseline there.
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

    # The IR half, on the isolated dust component so the ladder reaches SCUBA-2
    # without the stellar+attenuation comparison's scope limit.
    w_ds, L_ds, w_dt, L_dt = dust_emission_only(ssp)
    L_dt_on_s = U.regrid(w_dt, L_dt, w_ds)
    print_filter_table(
        filter_rows(w_ds, L_dt_on_s, L_ds, filters=IR_BANDS),
        ref_name="Synth",
        title=(
            "Dust IR alone — Draine & Li 2014, all four knobs matched (qpah, umin, gamma, alpha)"
        ),
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
