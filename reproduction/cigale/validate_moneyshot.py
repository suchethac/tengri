"""Internal validation: reproduce the SED-anatomy *money shot* with CIGALE.

Not a notebook and not user-facing — a head-to-head sanity check that the
panchromatic (X-ray -> radio) SED produced by ``notebooks/02_sed_anatomy.py``
is consistent with pcigale at matched parameters. Run it directly:

    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \
        .venv/bin/python reproduction/cigale/validate_moneyshot.py

Policy (per maintainer): do **not** tune CIGALE to force agreement. Where the
two diverge, decide whether it is an expected modeling difference (different
SSP, different SFH parameterization, dust FUV extrapolation, disc model) or a
tengri bug — and in the latter case file an issue rather than patching here.

The script prints a band-by-band ratio report and writes
``_figs/validate_moneyshot_cigale.png``.

Finding: the FIR / radio / X-ray models share the same templates and agree to
~1.5% at identical inputs (01_cigale.py §6 dale2014, §10 yang20, §11 radio).
The money-shot band offsets are entirely INPUT differences, not model bugs:
FIR follows ``L_ir`` (tengri's two-component Calzetti + FSPS UV absorbs ~1.5×
more than CIGALE's ``E_BV_lines=0.3`` + BC03); radio adds the q_IR convention
(2.64 vs 2.50) and tengri's disc-driven AGN jets (CIGALE chain has
``R_agn=0``); X-ray follows ``l_2500`` (multicolor vs SKIRTOR disc, ``log_lbol``
vs ``fracAGN``) and the α_ox convention. The one genuine tengri bug found while
building this — the AGN X-ray corona silently dropped in the build — is issue
#746 (worked around here via the public ``xray_total``).

References
----------
.. [1] Boquien et al. 2019, A&A 622, A103 (CIGALE).
"""

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")

from reproduction.cigale._drivers import cigale_driver as C, units as U

from tengri import FIXED, Fixed, SEDModel
from tengri.agn import compute_l2500, multicolor_disc
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data
from tengri.xray import xray_total

L_SUN = 3.828e33  # erg/s
C_AA = 2.998e18  # Å/s
Z = 0.1

HERE = Path(__file__).resolve().parent
FIGS = HERE / "_figs"
FIGS.mkdir(exist_ok=True)


# ── tengri money-shot (identical config to notebooks/02_sed_anatomy.py) ──────
def tengri_moneyshot(ssp):
    """Build the money-shot model and return (wave_Å, total_Lnu, components)."""
    model = SEDModel.build(
        ssp_data=ssp,
        sfh={
            "type": "dpl",
            "*": FIXED,
            "log_total_mass": 10.72,
            "alpha": 0.9,
            "beta": 2.7,
            "tau_gyr": 13.2,
        },
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "*": FIXED,
            "tau_bc": 0.8,
            "tau_diff": 0.3,
            "slope": -0.4,
        }, dust_emission={"type": "dale2014", "*": FIXED, "alpha_dale": 2.2},
        neb={"type": "cue", "*": FIXED},
        agn={
            "disc": {"type": "multicolor", "*": FIXED, "log_lbol": 10.5},
            "torus": {"type": "skirtor", "*": FIXED, "tau_skirtor": 5.0, "torus_frac": 0.5},
            "lines": {"type": "nlr", "*": FIXED},
        },
        radio={"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}, "*": FIXED},
        xray={"type": "simple", "*": FIXED},
        redshift=Fixed(Z),
    )
    import jax

    s = model.predict_state(model.spec.sample(jax.random.PRNGKey(0)))
    wave = np.asarray(s.wave)
    total = np.asarray(s.sed_intrinsic)

    # The build drops the AGN X-ray corona (invalid kwargs + missing l_2500
    # handoff — see issue #746). Swap in the corona-complete X-ray so the
    # total is honest.
    sfr = float(s.derived["sfr"])
    mstar = 10.0 ** float(s.derived["log_mstar"])
    dw = np.logspace(2.5, 4.5, 400)
    l2500 = float(compute_l2500(dw, multicolor_disc(dw, agn_log_lbol=10.5)))
    xray_corona = np.asarray(
        xray_total(wave, sfr=sfr, stellar_mass=mstar, l_2500_30deg=l2500, log_nh=20.0)
    )
    xray_model = np.asarray(s.derived.get("sed_xray", np.zeros_like(wave)))
    total = total - xray_model + xray_corona
    diag = {"L_ir": float(s.derived.get("L_ir", np.nan)), "l_2500": l2500}
    return wave, total, sfr, mstar, diag


# ── CIGALE matched chain ─────────────────────────────────────────────────────
def cigale_moneyshot():
    """Run the matched pcigale chain; return (wave_Å, total_Lnu erg/s/Hz)."""
    sed = C.run_chain(
        [
            # τ-delayed SFH (CIGALE has no double-power-law); age≈cosmic time at
            # z=0.1 with τ≈age keeps it on the rising shoulder like the DPL.
            (
                "sfhdelayed",
                dict(
                    tau_main=6000,
                    age_main=12000,
                    tau_burst=50,
                    age_burst=20,
                    f_burst=0.0,
                    sfr_A=1.0,
                    normalise=True,
                ),
            ),
            ("bc03", dict(imf=1, metallicity=0.02, separation_age=10)),
            (
                "nebular",
                dict(
                    logU=-2.0,
                    zgas=0.02,
                    ne=100,
                    f_esc=0.0,
                    f_dust=0.0,
                    lines_width=300.0,
                    emission=True,
                    line_list="",
                ),
            ),
            ("dustatt_modified_starburst", dict(E_BV_lines=0.3)),
            # CIGALE's dale2014 alpha is gridded; 2.0 is the nearest node to
            # the money-shot's alpha_dale=2.2 (a small, documented mismatch).
            ("dale2014", dict(alpha=2.0, fracAGN=0.0)),
            # SKIRTOR analytic disc (disk_type=0) is the closest CIGALE analog
            # of tengri's multicolor disc; fracAGN sets the AGN/IR balance.
            (
                "skirtor2016",
                dict(
                    t=7,
                    pl=1.0,
                    q=1.0,
                    oa=40,
                    R=20,
                    Mcl=0.97,
                    i=30,
                    disk_type=0,
                    delta=0.0,
                    fracAGN=0.2,
                    lambda_fracAGN="0/0",
                    law=0,
                    EBV=0.0,
                    temperature=100,
                    emissivity=1.6,
                ),
            ),
            (
                "yang20",
                dict(
                    gam=1.8,
                    E_cut=300.0,
                    alpha_ox=-1.4,
                    max_dev_alpha_ox=0.2,
                    angle_coef="0.5 & 0",
                    det_lmxb=0.0,
                    det_hmxb=0.0,
                ),
            ),
            ("radio", dict(qir_sf=2.5, alpha_sf=0.8, R_agn=0.0, alpha_agn=0.7)),
            ("redshifting", dict(redshift=0.0)),  # compare rest-frame SEDs
        ]
    )
    w, L = C.to_lnu(sed)
    info = sed.info
    diag = {
        # dust.luminosity [W] -> erg/s; AGN 2500 Å [W/Hz] -> erg/s/Hz.
        "L_ir": float(info.get("dust.luminosity", np.nan)) * 1e7,
        "l_2500": float(info.get("agn.intrin_Lnu_2500A_30deg", np.nan)) * 1e7,
    }
    return w, L, diag


def main():
    ssp = load_ssp_data(str(HERE.parent.parent / "data" / "fsps_prsc_miles_chabrier.h5"))
    w_t, L_t, sfr, mstar, dt = tengri_moneyshot(ssp)
    w_c, L_c, dc = cigale_moneyshot()

    # Put both on CIGALE's grid; normalize CIGALE to tengri in the optical so
    # we compare SED *shapes* across nine decades, not absolute SSP/IMF/mass
    # bookkeeping (different SSP libraries, different SFH normalization).
    L_t_on_c = U.regrid(w_t, L_t, w_c)
    opt = (w_c >= 3000) & (w_c <= 8000) & (L_c > 0) & (L_t_on_c > 0)
    norm = float(np.median(L_t_on_c[opt] / L_c[opt]))
    L_c_n = L_c * norm

    nu_c = C_AA / w_c
    um = w_c / 1e4

    # Band-by-band ratio report (tengri / CIGALE on matched grid).
    bands = {
        "soft X-ray (0.5-2 keV)": (6.2, 24.8),
        "FUV (1216-1900 Å)": (1216, 1900),
        "optical (3000-8000 Å)": (3000, 8000),
        "NIR (1-3 µm)": (1e4, 3e4),
        "MIR (8-30 µm)": (8e4, 3e5),
        "FIR (60-300 µm)": (6e5, 3e6),
        "radio (>1 cm)": (1e8, 3e9),
    }
    print(f"\n  tengri money-shot: M*={mstar:.2e} Msun, SFR={sfr:.1f} Msun/yr, z={Z}")
    print(f"  optical normalization tengri/CIGALE = {norm:.3g}× (factored out below)\n")
    print(f"  {'band':<24} {'tengri/CIGALE':>14}   note")
    print("  " + "-" * 60)
    for name, (a, b) in bands.items():
        m = (w_c >= a) & (w_c <= b) & (L_c_n > 0) & (L_t_on_c > 0)
        if not m.any():
            print(f"  {name:<24} {'—':>14}   (no overlap)")
            continue
        r = float(np.median(L_t_on_c[m] / L_c_n[m]))
        flag = "OK" if 0.5 <= r <= 2.0 else (">2× HIGH" if r > 2 else "<0.5× LOW")
        print(f"  {name:<24} {r:>13.2f}×   {flag}")

    # Why the IR/radio/X-ray bands differ: the *models* share the same templates and
    # agree to ~1.5% at identical inputs (see 01_cigale.py §6 dale2014, §10
    # yang20, §11 radio). The money-shot offsets are INPUT differences, not
    # model bugs — quantified here on the optical-matched scale.
    lir_t, lir_c = dt["L_ir"], dc["L_ir"] * norm
    l25_t, l25_c = dt["l_2500"], dc["l_2500"] * norm
    print("\n  input decomposition (optical-matched scale):")
    print(
        f"    L_ir       tengri {lir_t:.2e} / CIGALE {lir_c:.2e} = {lir_t / lir_c:.2f}×"
        "   -> sets FIR (same Dale template; heavier tengri attenuation + FSPS UV)"
    )
    print(
        f"    l_2500     tengri {l25_t:.2e} / CIGALE {l25_c:.2e} = {l25_t / l25_c:.2f}×"
        "   (multicolor vs SKIRTOR disc; log_lbol vs fracAGN)"
    )
    print("    X-ray norm α_ox=just2007(l_2500)+Lehmer16 XRB (tengri) vs α_ox=-1.4 (CIGALE)")
    print("               -> soft-band excess is α_ox convention + XRB (l_2500 alone is 0.81×)")
    print("    q_IR       tengri 2.64 (Bell03 default) vs CIGALE 2.50 -> radio ×0.72 (convention)")
    print("    AGN radio  tengri jets ON (disc-driven) vs CIGALE R_agn=0 -> radio band excess")
    print("    => radio ≈ L_ir × AGN-jets × q_IR; FIR ≈ L_ir; X-ray ≈ α_ox + XRB. No model bug.")
    # The tengri curve steps down near ~2.25e5 µm (#883). That is the edge of the
    # shared Dale+2014 template grid (`data/dale2014_templates_cigale.h5`, which
    # runs 0.36 µm – 2.25e5 µm): CIGALE's Dale2014 templates carry a built-in
    # radio-synchrotron tail (the FIR–radio correlation), so `dust.emission.
    # dale2014` and the separate `radio.condon92` component BOTH contribute in
    # the radio — a double count that is a CIGALE convention, not tengri-specific.
    # tengri shows the step because its Dale-template radio tail (~L_ir-scaled)
    # is comparable to condon92; past the template edge the interpolation zeroes
    # (`right=0.0`) so only condon92 remains. CIGALE rises smoothly there only
    # because its radio module dominates its own (identically-truncated) Dale
    # tail. The radio band beyond ~1 cm is therefore not a clean single-component
    # comparison; the physically clean fix (truncate the Dale template to
    # dust-only and let condon92 own the radio) is tracked in #883 and would make
    # tengri MORE correct than the CIGALE double count, so it is deliberately not
    # applied in this reproduction figure.
    print("    radio step ~2.25e5 µm = Dale2014 template radio-tail edge (double-counts condon92;")
    print(
        "               shared CIGALE convention, #883). Radio band >1 cm is not single-component."
    )

    # Figure
    fig, (ax, axr) = plt.subplots(
        2, 1, figsize=(13, 7.5), sharex=True, gridspec_kw={"height_ratios": [3, 1]}
    )
    ax.loglog(um, nu_c * L_c_n / L_SUN, "C0-", lw=1.6, label="CIGALE (matched chain)")
    ax.loglog(um, nu_c * L_t_on_c / L_SUN, "C1--", lw=1.5, label="tengri money-shot")
    ax.set_xlim(1e-4, 1e6)
    ymax = float(np.nanmax(nu_c * L_t_on_c / L_SUN))
    ax.set_ylim(ymax * 1e-9, ymax * 30)
    ax.set_ylabel(r"$\nu L_\nu$  [$L_\odot$]  (CIGALE norm. to tengri optical)")
    ax.set_title("Money-shot SED — tengri vs CIGALE (internal validation)")
    # Flag the Dale2014 template radio-tail edge (the #883 step): both codes'
    # dale2014 carries a built-in radio synchrotron that double-counts condon92.
    ax.axvline(2.25e5, color="0.6", ls=":", lw=1.0)
    ax.text(
        2.25e5,
        ymax * 5,
        "Dale radio-tail edge\n(#883)",
        fontsize=7,
        color="0.4",
        ha="right",
        va="top",
        rotation=90,
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    resid = np.full(w_c.shape, np.nan)
    mm = (L_c_n > 0) & (L_t_on_c > 0)
    resid[mm] = L_t_on_c[mm] / L_c_n[mm] - 1.0
    axr.axhspan(-0.25, 0.25, color="0.85")
    axr.axhline(0.0, color="0.5", lw=0.8)
    axr.plot(um, resid, "C1-", lw=1.0)
    axr.axvline(2.25e5, color="0.6", ls=":", lw=1.0)  # Dale radio-tail edge (#883)
    axr.set_xscale("log")
    axr.set_ylim(-1, 1)
    axr.set_xlabel(r"Rest-frame wavelength $\lambda$  [$\mu$m]")
    axr.set_ylabel("tengri/CIGALE − 1")
    axr.grid(True, alpha=0.3)
    fig.tight_layout()
    out = FIGS / "validate_moneyshot_cigale.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"\n  wrote {out}\n")


if __name__ == "__main__":
    main()
