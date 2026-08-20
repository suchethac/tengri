"""Matched-input pixel validation: is the physics correct?

Companion to ``01_prospector.py`` (which compares tengri-native vs
Prospector-native *configurations*, where offsets are input-driven). This
script removes every input difference — same FSPS MIST+MILES Chabrier SSP,
same metallicity, same delayed-tau SFH, same Calzetti screen, same
Draine & Li 2007 templates — and checks the stellar+dust SED **pixel by
pixel**. At matched inputs the two implementations must agree; a surviving
residual is a physics/mapping problem.

Run:

    SPS_HOME=/path/to/fsps JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/prospector/validate_matched_physics.py

Needs ``python-fsps`` with ``SPS_HOME`` set, and the repackaged grid
``_drivers/data/fsps_mist_miles_chabrier.h5``.

FINDING -- the dust mapping matters, exactly as it does for CIGALE
------------------------------------------------------------------
FSPS ``dust_type=2`` applies a **single** Calzetti screen of optical depth
``tau_V = A_V / 1.086`` to the *entire* stellar continuum. tengri's
``two_component`` is **Charlot & Fall**: it adds a birth-cloud optical depth
``tau_bc`` to the *young-star* continuum on top of the diffuse ``tau_diff``.
The FSPS-equivalent tengri build is therefore a single diffuse screen --
``tau_bc = 0``, ``tau_diff = A_V / 1.086`` -- not a birth-cloud/diffuse split.
With a split, both terms hit the continuum and the young-star-dominated FUV is
over-attenuated, which then inflates the re-radiated dust IR through energy
balance. The two attenuation *models* are genuinely different; they must not be
mapped onto each other via a split. This is the same trap ``validate_matched_
physics.py`` documents on the CIGALE side (issue #747), reached independently.

FINDING -- and the dust IR needs a SECOND convention matched
------------------------------------------------------------
Getting the screen right fixes the stellar continuum but leaves the dust IR
~10% low, because the two codes draw the energy-balance budget differently:
FSPS re-emits all absorbed luminosity including LyC, while tengri's canonical
balance excludes ``lambda < 912`` A -- those photons re-emerge as nebular, the
CIGALE convention. 01_prospector.py §6 documents this and puts it at ~11% of
absorbed energy at this fiducial (#961, #922).

Measured here, both screens correct, MIR / FIR median ratio:

    eb_include_lyc = False    0.900x / 0.894x
    eb_include_lyc = True     0.999x / 0.999x

So the deficit is a convention, not a defect, and the opt-in closes it to
floating point. Worth stating plainly because the first version of this script
reported the 10% as an unexplained discrepancy: a matched-input validator is
only as good as the conventions it remembers to match.

Nebular is off for the *continuum* comparison -- FSPS uses Byler+2017 Cloudy
grids and tengri uses the Cue emulator, which are different models by design
(01_prospector.py §8 quantifies the Halpha ratio). AGN is off; Prospector has
no X-ray or radio.

FINDING -- the bandpass ladder reaches where the wavelength windows could not
-----------------------------------------------------------------------------
The band table is 23 real transmission curves from GALEX FUV (0.15 um) to
SCUBA-2 850 um -- 3.75 decades -- replacing six hand-drawn wavelength windows
that stopped at 300 um. At the fully-matched configuration all 23 agree to a
median 0.999x.

One band does not: **WISE W1 at 1.046x**, the only entry outside 5%. It was
invisible before because 3.39 um fell in the gap between the old ``NIR 1-3 um``
and ``MIR 8-30 um`` windows. W1 sits where the stellar Rayleigh-Jeans tail hands
over to the hottest dust, so it is the band most sensitive to where each code
draws that boundary -- worth a look, and exactly the kind of thing a window
table cannot report.

The ratios are safe to read without knowing which bandpass weight FSPS uses
internally: the photon (1/lambda) and energy (1/lambda^2) conventions move
them by 2.6e-4, measured in the run rather than assumed, because both spectra
pass through the identical average.

FINDING -- Lyman alpha, and only Lyman alpha, disagrees
------------------------------------------------------
With nebular on and the gas parameters matched (logU = -2, logZ_gas = 0), the
optical lines agree well: [O II] 0.98x, Hbeta 1.03x, [O III] 0.87-0.88x,
Halpha 0.95x, [S II] 1.16-1.18x. Both Balmer decrements are sound --
Halpha/Hbeta = 3.01 (FSPS) and 2.79 (tengri) against Case B ~ 2.86 -- so the
recombination physics matches on both sides.

Lyman alpha does not, by a factor of ~120:

    FSPS      Lya/Hbeta =  0.60
    tengri    Lya/Hbeta = 71.19

Lya is a resonant line: its escape depends on scattering and on dust
destruction inside the H II region, which every code treats by its own
convention, so a large divergence here is a modelling choice rather than a
defect. What the table adds is that the divergence is *confined* to it. Two
caveats on the number: both grids carry only three points at 10 A spacing
across the line, so it is resolution-limited (the ratio survives because both
sides are measured identically), and no claim is made here about which value is
right -- that needs the Case B Lya/Hbeta checked against a source, not recalled.
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
    convention_sensitivity,
    filter_rows,
    line_rows,
    print_filter_table,
    print_line_table,
)
from reproduction.prospector._drivers import prospector_driver as P, units as U

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

L_SUN = 3.828e33
C_AA = 2.998e18

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched parameters (01_prospector.py setup).
MET_LOGZSOL = 0.0
TAU_GYR, AGE_GYR = 1.0, 5.0
A_V = 1.0
TAU_DIFF = A_V / 1.086  # single Calzetti screen, FSPS dust_type=2
Q_PAH, U_MIN, GAMMA = 2.5, 1.0, 0.05


def fsps_stellar_dust():
    """FSPS stellar+dust SED at the matched parameters.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    return P.csp_lnu(
        logzsol=MET_LOGZSOL,
        tau=TAU_GYR,
        tage=AGE_GYR,
        sfh=4,
        av=A_V,
        dust_type=2,
        add_dust_emission=True,
        duste_qpah=Q_PAH,
        duste_umin=U_MIN,
        duste_gamma=GAMMA,
        add_neb_emission=False,
    )


def tengri_stellar_dust(ssp, tau_bc, *, include_lyc=False):
    """tengri stellar+dust SED at the matched parameters.

    Parameters
    ----------
    ssp : SSPData
        The repackaged FSPS MIST+MILES Chabrier grid.
    tau_bc : float
        Birth-cloud optical depth. ``0.0`` is the FSPS-equivalent mapping;
        a non-zero value reproduces the over-attenuation described above.
    include_lyc : bool, optional
        Put LyC photons into the energy-balance budget. tengri's canonical
        balance excludes ``lambda < 912`` A (those photons re-emerge as
        nebular, the CIGALE convention); FSPS re-emits everything. Matching
        FSPS therefore needs this on -- see the FINDING above.

    Returns
    -------
    tuple of ndarray
        ``(wave_aa, L_nu)`` with shapes ``(n_wave,)``; L_nu in [erg/s/Hz].
    """
    dust = {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(tau_bc),
        "tau_diff": Fixed(TAU_DIFF),
        "*": FIXED,
    }
    # A peer group now, not a sub-block of the attenuation dict.
    dust_emission = {
        "type": "draine_li2007",
        "qpah": Fixed(Q_PAH),
        "umin": Fixed(U_MIN),
        "gamma_dl": Fixed(GAMMA),
        "*": FIXED,
    }
    if include_lyc:
        dust["eb_include_lyc"] = True
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
        dust_attenuation=dust,
        dust_emission=dust_emission,
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return np.asarray(s.wave), np.asarray(s.sed_intrinsic)


# Nebular section: a young constant-SFR population, where the lines dominate
# (01_prospector.py §8 fiducial).
NEB_AGE = 0.01  # Gyr
NEB_LOGU, NEB_LOGZ, NEB_LOGMASS = -2.0, 0.0, 9.0


def report(w_f, L_t, L_f, title, *, compact=False):
    """Print the bandpass ratio table for one configuration.

    Parameters
    ----------
    w_f : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom]; both SEDs are already on it.
    L_t, L_f : array_like, shape (n_wave,)
        tengri and FSPS L_nu on that grid [erg/s/Hz].
    title : str
        Heading printed above the table.
    compact : bool, optional
        One summary line instead of the full ladder.
    """
    print_filter_table(
        filter_rows(w_f, L_t, L_f),
        ref_name="FSPS",
        title=title,
        compact=compact,
    )


def nebular_only():
    """Nebular-only L_nu from both codes at matched gas parameters.

    FSPS is isolated as (neb on) - (neb off) at the same SFH; tengri publishes
    ``sed_nebular`` directly. FSPS's curve is per 1 Msun formed, so it is
    rescaled to the tengri build's formed mass.

    Returns
    -------
    tuple of ndarray
        ``(w_fsps, L_fsps, w_tengri, L_tengri)``; L_nu in [erg/s/Hz].
    """
    w_p, L_p = P.isolate(
        dict(
            sfh=1,
            const=1.0,
            tage=NEB_AGE,
            add_neb_emission=True,
            gas_logu=NEB_LOGU,
            gas_logz=NEB_LOGZ,
        ),
        dict(sfh=1, const=1.0, tage=NEB_AGE),
    )
    L_p = np.clip(L_p, 0.0, None) * 10.0**NEB_LOGMASS

    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "fsps_mist_miles_chabrier.h5"))
    m = SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(MET_LOGZSOL), "*": FIXED},
        sfh={
            "type": "const",
            "start_gyr": Fixed(NEB_AGE),
            "end_gyr": Fixed(0.0),
            "log_total_mass": Fixed(NEB_LOGMASS),
            "*": FIXED,
        },
        dust_attenuation={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        neb={
            "type": "cue",
            "neb_logU": Fixed(NEB_LOGU),
            "neb_logZ_gas": Fixed(NEB_LOGZ),
            "*": FIXED,
        },
        redshift=Fixed(0.0),
    )
    s = m.predict_state({})
    return w_p, L_p, np.asarray(s.wave), np.asarray(s.derived["sed_nebular"])


def main():
    U.verify_unit_conversion(rtol=1e-3)
    ssp = load_ssp_data(str(HERE / "_drivers" / "data" / "fsps_mist_miles_chabrier.h5"))

    w_f, L_f = fsps_stellar_dust()

    print("\n  Controls (each wrong in a known way):")

    # The wrong mapping: a birth-cloud/diffuse split of the same A_V.
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=TAU_DIFF)
    report(
        w_f,
        U.regrid(w_t, L_t, w_f),
        L_f,
        "tau_bc + tau_diff split (NOT FSPS-equivalent)",
        compact=True,
    )

    # Right screen, but tengri's canonical energy balance: the dust IR runs
    # ~10% low because LyC is excluded from the budget.
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=0.0)
    report(
        w_f,
        U.regrid(w_t, L_t, w_f),
        L_f,
        "tau_bc = 0, LyC excluded (tengri default)",
        compact=True,
    )

    # Both conventions matched: single screen AND LyC in the budget.
    w_t, L_t = tengri_stellar_dust(ssp, tau_bc=0.0, include_lyc=True)
    L_t_on_f = U.regrid(w_t, L_t, w_f)
    report(w_f, L_t_on_f, L_f, "tau_bc = 0 + eb_include_lyc (fully FSPS-equivalent)")

    # The ratios above are read without knowing which bandpass weight FSPS
    # uses internally; this is the evidence that that is safe.
    print(
        f"\n  bandpass-convention sensitivity (photon vs energy weight): "
        f"{convention_sensitivity(w_f, L_t_on_f, L_f):.2e}"
    )

    # Emission lines, at matched gas parameters. Not a parity check -- see
    # print_line_table's Notes.
    w_pn, L_pn, w_tn, L_tn = nebular_only()
    print_line_table(
        line_rows(w_tn, L_tn, w_pn, L_pn, line_lum=U.line_lum),
        ref_name="FSPS",
        title="Emission lines — tengri Cue vs FSPS Byler+2017 (matched logU, logZ_gas)",
    )

    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(12, 7), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    um = w_f / 1e4
    nu = C_AA / w_f
    ax.loglog(um, nu * L_f / L_SUN, "C0-", lw=1.5, label="FSPS (matched)")
    ax.loglog(um, nu * L_t_on_f / L_SUN, "C1--", lw=1.5, label="tengri (matched)")
    ax.set_ylabel(r"$\nu L_\nu$ [$L_\odot$]")
    ax.set_title("Matched-input stellar+dust — tengri vs FSPS")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    resid = np.full(w_f.shape, np.nan)
    mm = (L_f > 0) & (L_t_on_f > 0)
    resid[mm] = L_t_on_f[mm] / L_f[mm] - 1.0
    axr.axhspan(-0.05, 0.05, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.set_xscale("log")
    axr.set_ylim(-0.3, 0.3)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
    axr.set_ylabel("tengri/FSPS − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGS / "prospector_validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
