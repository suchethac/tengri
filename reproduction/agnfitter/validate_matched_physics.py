"""Matched-input pixel validation: is the physics correct?

Companion to ``01_agnfitter.py`` (which compares tengri-native vs
AGNFITTER-RX-native *configurations*, where offsets are input-driven). This
script removes every input difference — same template library, same node
parameters — and checks the cold-dust and accretion-disk SEDs **pixel by
pixel**. At matched inputs the two implementations must agree; a surviving
residual is a physics/mapping problem.

Run:

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/agnfitter/validate_matched_physics.py

Needs only the committed reference grids under ``data/``
(``agnfitter_bbb_reference.h5``, ``agnfitter_torus_reference.h5``,
``agnfitter_cold_dust_reference.h5``). It never reads an upstream
AGNFITTER-RX checkout -- ``tests/contract/test_reproduction_driver_no_clone.py``
pins that, after #1035 and #792 both regressed it.

Why this comparison is different from the other four
----------------------------------------------------
The CIGALE, Prospector, BAGPIPES and Synthesizer validators run the reference
code live and compare a *computed* SED. AGNFITTER-RX's model libraries are
tabulated, so the reference side here is the repackaged table itself. That
makes this the strictest of the five: tengri's block and the reference are
evaluating the same published nodes, so a node-exact library should agree to
interpolation error, not to a few percent.

Both cold-dust libraries below are node-exact matches by construction --
``schreiber2018`` reads the S17 tables and ``dh02_ce01`` the legacy
Dale & Helou 2002 + Chary & Elbaz 2001 grid -- so they are the sharpest test
of the repackaging. Shapes are peak-normalized, because AGNFITTER-RX and
tengri carry different luminosity bookkeeping for the same template.
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

import jax.numpy as jnp
from reproduction._validation import (
    IR_BANDS,
    filter_rows,
    print_filter_table,
    print_radio_table,
    radio_rows,
)
from reproduction.agnfitter._drivers import agnfitter_driver as A, units as U

from tengri.dust import DUST_EMISSION_MODELS

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)

# Matched nodes (01_agnfitter.py §6).
T_DUST, F_PAH = 35.0, 0.02
LOG_LIR = 12.0
IR_BAND = (3e4, 3e6)  # 3-300 um, where the cold-dust shape is defined


def _peak_norm(wave, L, band=IR_BAND):
    """Normalize an SED to its peak inside ``band``.

    Parameters
    ----------
    wave : array_like, shape (n_wave,)
        Wavelength grid [Angstrom].
    L : array_like, shape (n_wave,)
        Luminosity on that grid; units cancel.
    band : tuple of float, optional
        ``(lo, hi)`` wavelength window [Angstrom] defining the peak.

    Returns
    -------
    ndarray, shape (n_wave,)
        ``L`` divided by its maximum inside ``band``.
    """
    m = (wave > band[0]) & (wave < band[1])
    return np.asarray(L) / float(np.asarray(L)[m].max())


def compare(name, w_ref, L_ref, L_tengri, results):
    """Report the peak-normalized shape residual for one library.

    Parameters
    ----------
    name : str
        Library label for the printed table.
    w_ref : array_like, shape (n_wave,)
        Reference wavelength grid [Angstrom].
    L_ref, L_tengri : array_like, shape (n_wave,)
        Reference and tengri SEDs on that grid.
    results : dict
        Accumulator, keyed by ``name``, holding ``(w, ref, tengri)`` for the
        figure.
    """
    ref_n = _peak_norm(w_ref, L_ref)
    ten_n = _peak_norm(w_ref, L_tengri)
    m = (w_ref > IR_BAND[0]) & (w_ref < IR_BAND[1]) & (ref_n > 1e-3)
    resid = np.abs(ten_n[m] - ref_n[m]) / ref_n[m]
    med = float(np.median(resid))

    lam_ref = float(w_ref[m][np.argmax(ref_n[m])]) / 1e4
    lam_ten = float(w_ref[m][np.argmax(ten_n[m])]) / 1e4
    flag = " OK" if med < 0.05 else "  <-- check"
    print(
        f"  {name:<26} {med:>9.3%}   {lam_ref:>7.1f}   {lam_ten:>7.1f}"
        f"   {abs(lam_ten - lam_ref) / lam_ref:>7.1%}{flag}"
    )
    results[name] = (w_ref, ref_n, ten_n)


# AGN radio (01_agnfitter.py §11). Both codes are compared as *shapes*,
# normalized at 5 GHz, because the amplitude is set by radio loudness — a free
# parameter on both sides and not a physics claim.
L_AGN_BOL = 1e45  # erg/s
DPL_PARS = dict(alpha1=-0.75, alpha2=-0.1, log_nu_t=10.0, log_nu_cut=13.0)
NU_NORM = 5.0e9  # Hz


def radio_report():
    """Compare both AGN radio laws against the AGNFITTER-RX branches they mirror.

    Notes
    -----
    Two laws, two upstream branches: ``radio_agn`` is the single power law
    (``nRADdata == 1``) and ``radio_agn_dpl`` the broken one (``nRADdata > 3``
    / DPL-4) of Martinez-Ramirez et al. 2024, Eqs. 9-10. Comparing the SPL
    against the DPL branch, or vice versa, would measure the difference between
    two deliberately different laws and read as a defect.

    Shapes are normalized at 5 GHz so the table tests the *slope*, which is the
    physics, rather than radio loudness, which is a fitted amplitude.
    """
    import numpy as _np

    from tengri.radio import radio_agn, radio_agn_dpl

    freq = _np.geomspace(1e8, 1e12, 400)  # 0.1-1000 GHz
    wave = U.C_ANGSTROM_PER_S / freq

    def _norm(w, L):
        ref = float(_np.interp(U.C_ANGSTROM_PER_S / NU_NORM, w[::-1], _np.asarray(L)[::-1]))
        return _np.asarray(L) / ref if ref > 0 else _np.asarray(L)

    for label, ten, ref in (
        (
            "SPL — radio_agn vs AGNFITTER nRADdata==1",
            radio_agn(jnp.asarray(wave), L_AGN_BOL, radio_loudness=1.0, alpha_agn=0.75),
            A.agn_radio_spl(freq)[1],
        ),
        (
            "DPL — radio_agn_dpl vs AGNFITTER DPL-4",
            radio_agn_dpl(jnp.asarray(wave), L_AGN_BOL, radio_loudness=1.0, **DPL_PARS),
            A.agn_radio_dpl(freq, **DPL_PARS)[1],
        ),
    ):
        print_radio_table(
            radio_rows(wave, _norm(wave, ten), _norm(wave, ref)),
            ref_name="AGNFITTER",
            title=f"AGN radio, normalized at 5 GHz — {label}",
        )


def main():
    A.require_available()
    U.verify_unit_conversion(rtol=1e-3)

    print("\n  Cold-dust libraries, peak-normalized over 3-300 um")
    print(f"  {'library':<26} {'med|res|':>9}   {'ref um':>7}   {'ten um':>7}   {'dpeak':>7}")
    print("  " + "-" * 72)

    results = {}

    w_s17, L_s17 = A.cold_dust_template("S17", tdust=T_DUST, fpah=F_PAH)
    L_s18 = np.asarray(
        DUST_EMISSION_MODELS["schreiber2018"](
            jnp.asarray(w_s17), 1.0, dust_T=T_DUST, dust_f_pah=F_PAH
        )
    )
    compare("S17 / schreiber2018", w_s17, L_s17, L_s18, results)

    w_dh, L_dh = A.cold_dust_template("DH02_CE01", log_irlum=LOG_LIR)
    L_dh02 = np.asarray(
        DUST_EMISSION_MODELS["dh02_ce01"](jnp.asarray(w_dh), 1.0, dust_log_lir=LOG_LIR)
    )
    compare("DH02_CE01 / dh02_ce01", w_dh, L_dh, L_dh02, results)

    # The same templates through real IR bandpasses. A pixel residual says the
    # curves agree; a band ratio says what a MIPS or SPIRE measurement of them
    # would agree to, which is the number an observer actually compares.
    for name, (w, ref_n, ten_n) in results.items():
        print_filter_table(
            filter_rows(w, ten_n, ref_n, filters=IR_BANDS),
            ref_name="AGNFITTER",
            title=f"{name} — peak-normalized templates through IR bandpasses",
        )

    radio_report()

    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(6 * n, 7), squeeze=False,
                             gridspec_kw={"height_ratios": [3, 1]})
    for i, (name, (w, ref_n, ten_n)) in enumerate(results.items()):
        ax, axr = axes[0][i], axes[1][i]
        um = w / 1e4
        ax.loglog(um, ref_n, "C0-", lw=2.5, alpha=0.5, label="AGNFITTER-RX table")
        ax.loglog(um, ten_n, "C1--", lw=1.2, label="tengri")
        ax.set_xlim(1, 1e3)
        ax.set_ylim(1e-3, 2)
        ax.set_title(name)
        ax.set_ylabel("peak-normalized $L$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

        resid = np.full(w.shape, np.nan)
        mm = ref_n > 1e-3
        resid[mm] = ten_n[mm] / ref_n[mm] - 1.0
        axr.axhspan(-0.05, 0.05, color="0.85")
        axr.axhline(0.0, color="0.5", lw=0.8)
        axr.plot(um, resid, "C1-", lw=1.0)
        axr.set_xscale("log")
        axr.set_xlim(1, 1e3)
        axr.set_ylim(-0.3, 0.3)
        axr.set_xlabel(r"Rest-frame wavelength $\lambda$ [$\mu$m]")
        axr.set_ylabel("tengri/ref − 1")
        axr.grid(True, alpha=0.3)
    fig.tight_layout()

    out = FIGS / "agnfitter_validate_matched_physics.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
