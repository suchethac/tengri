# SPDX-License-Identifier: BSD-3-Clause
"""Wavelength-resolved dust-attenuation curve differential: tengri vs external codes.

Investigates the report that tengri's two-component dust attenuation comes out
*lower* (spectrum too bright) than FSPS/Prospector, CIGALE and bagpipes
(documented as CROSSVAL-01 in ``docs/known_bugs.md``: tengri/FSPS NUV flux
ratio = 1.291 at tau_BC=1, tau_diff=0.5).

Why a curve differential, not an SED differential
-------------------------------------------------
The attenuation curve ``A(lambda) = -2.5 log10(T(lambda))`` **cancels the SSP**
entirely (it is the ratio of attenuated to intrinsic light). That removes the
stellar-population differences which led ``analysis/crossval_external_seds.py``
to *deliberately skip* tengri on dust-law crossval, and isolates the dust
physics alone. The comparison is therefore fully deterministic and needs **no
external code installed** — each external code's curve is a closed-form
expression we reimplement here.

Findings (see ``__main__`` summary)
-----------------------------------
* Single-screen laws (Calzetti) are a control: tengri == analytic, residual ~0.
* Two-component birth-cloud slope is the discrepancy. tengri ties the
  birth-cloud and diffuse power-law indices to a single ``dust_slope`` (-0.7,
  faithful to the *original* Charlot & Fall 2000). FSPS defaults the birth
  cloud to a steeper ``dust1_index = -1.0``, so FSPS attenuates young-star UV
  ~24% more. tengri has no knob to match it -> structural under-attenuation in
  the UV for young populations.

Run
---
    JAX_PLATFORMS=cpu PYTHONPATH=$PWD/src:$PWD \\
        /path/to/.venv/bin/python analysis/dust_attenuation_curve_differential.py

Writes ``analysis/figures/dust_attenuation_curve_differential.png`` and prints a
pass/fail table.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np

from tengri.components.dust.attenuation import calzetti, two_component_dust

V_BAND = 5500.0
LN10_OVER_2P5 = np.log(10.0) / 2.5  # A_mag = 1.0857 * tau ; tau = A / 1.0857
MAG_PER_TAU = 2.5 / np.log(10.0)  # = 1.0857

# Diagnostic wavelengths.
FUV, NUV, VBAND = 1500.0, 2700.0, 5500.0

# Common rest-frame grid (Angstrom), log-spaced over the SED-relevant range.
WAVE = jnp.asarray(np.logspace(np.log10(1200.0), np.log10(25000.0), 400))


# ── tengri attenuation curves (A(lambda)/A_V, dimensionless) ──────────────


def tengri_single_screen_calzetti() -> np.ndarray:
    """A(lambda)/A_V for a single Calzetti screen = k(lambda) (k(V)=1)."""
    return np.asarray(calzetti(WAVE))


def tengri_two_component_curve(
    tau_bc: float, tau_diff: float, *, young: bool, slope: float = -0.7
) -> np.ndarray:
    """A(lambda) [mag] for the young (weight->1) or old (weight->0) limit.

    Uses the public ``two_component_dust`` transmission at a single age far
    inside the birth-cloud (1e6 yr) or far outside it (1e10 yr), so the sigmoid
    weight collapses to 1 or 0 and the curve is the pure component combination.
    Both birth-cloud and diffuse laws are ``power_law`` and currently share the
    one ``n_slope`` argument (the crux of the discrepancy).
    """
    age = jnp.asarray([1.0e6 if young else 1.0e10])
    trans = two_component_dust(
        wavelength=WAVE,
        age_grid=age,
        tau_v1=tau_bc,
        tau_v2=tau_diff,
        law_bc="power_law",
        law_diff="power_law",
        # Per-component birth-cloud slope via the new bc_params overlay (the
        # diffuse ISM keeps the shared -0.7). For tau_diff=0 this isolates the
        # birth cloud; for the default slope it reproduces the old behaviour.
        bc_params={"n_slope": slope},
    )[0]
    return -2.5 * np.log10(np.asarray(trans))


# ── External codes' closed-form curves ────────────────────────────────────


def analytic_calzetti_k() -> np.ndarray:
    """Calzetti et al. (2000) k(lambda)/k(V), R_V=4.05 — independent reimpl.

    Used as a control: this must equal tengri's ``calzetti`` to machine level,
    proving the harness and the single-screen path agree. Bagpipes' and
    Prospector's single-screen Calzetti reduce to exactly this curve.
    """
    w_um = np.asarray(WAVE) / 1e4
    x = 1.0 / w_um
    rv = 4.05
    k_ir = 2.659 * (-1.857 + 1.040 * x)
    k_uv = 2.659 * (-2.156 + 1.509 * x - 0.198 * x**2 + 0.011 * x**3)
    k_prime = np.where(w_um >= 0.63, k_ir, k_uv)
    return np.clip((k_prime + rv) / rv, 0.0, None)


def fsps_cf00_curve(
    tau_bc: float,
    tau_diff: float,
    *,
    young: bool,
    dust1_index: float = -1.0,
    dust_index: float = -0.7,
) -> np.ndarray:
    """FSPS Charlot & Fall (dust_type=0) A(lambda) [mag].

    FSPS applies independent power-law indices to the birth cloud (``dust1`` /
    ``dust1_index``, default -1.0) and the diffuse ISM (``dust2`` /
    ``dust_index``, default -0.7). ``tau_bc``/``tau_diff`` here are the V-band
    optical depths (FSPS ``dust1``/``dust2``). Young stars see both screens;
    old stars see only the diffuse screen.
    """
    w = np.asarray(WAVE)
    tau_diff_lambda = tau_diff * (w / V_BAND) ** dust_index
    if young:
        tau_bc_lambda = tau_bc * (w / V_BAND) ** dust1_index
        tau = tau_bc_lambda + tau_diff_lambda
    else:
        tau = tau_diff_lambda
    return MAG_PER_TAU * tau


def _ratio(curve_mag: np.ndarray, w0: float, w1: float) -> float:
    """A(w0)/A(w1) from a magnitude curve, interpolated onto the grid."""
    wg = np.asarray(WAVE)
    a0 = float(np.interp(w0, wg, curve_mag))
    a1 = float(np.interp(w1, wg, curve_mag))
    return a0 / a1 if a1 != 0 else float("nan")


def _at(curve_mag: np.ndarray, w0: float) -> float:
    return float(np.interp(w0, np.asarray(WAVE), curve_mag))


# ── Report ────────────────────────────────────────────────────────────────


def _make_figure(out_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wg = np.asarray(WAVE)
    av = 1.0
    tau_v = av / MAG_PER_TAU  # match A_V=1 mag between codes

    # Control: single-screen Calzetti.
    k_teng = tengri_single_screen_calzetti()
    k_anal = analytic_calzetti_k()

    # Two-component, young population: A_V split tau_bc=tau_diff for illustration.
    tau_bc = tau_v
    tau_diff = 0.0
    teng_young = tengri_two_component_curve(tau_bc, tau_diff, young=True)
    fsps_young = fsps_cf00_curve(tau_bc, tau_diff, young=True)
    # Target: tengri *would* match FSPS if the BC slope could be set to -1.0.
    teng_young_fixed = tengri_two_component_curve(tau_bc, tau_diff, young=True, slope=-1.0)

    fig, axes = plt.subplots(2, 2, figsize=(12, 9))

    ax = axes[0, 0]
    ax.plot(wg, av * k_teng, label="tengri calzetti", lw=2)
    ax.plot(wg, av * k_anal, "--", label="analytic Calzetti (bagpipes/Prospector)", lw=1.5)
    ax.set_title("Control: single-screen Calzetti (must overlap)")
    ax.set_xscale("log")
    ax.set_xlabel("rest wavelength [A]")
    ax.set_ylabel("A(lambda) [mag], A_V=1")
    ax.legend()

    ax = axes[0, 1]
    ax.plot(wg, teng_young, label="tengri BC (n=-0.7)", lw=2)
    ax.plot(wg, fsps_young, "--", label="FSPS BC (dust1_index=-1.0)", lw=1.5)
    ax.plot(wg, teng_young_fixed, ":", label="tengri BC n=-1.0 (target)", lw=2)
    ax.set_title("Young population, birth-cloud term (tau_bc=A_V/1.0857)")
    ax.set_xscale("log")
    ax.set_xlabel("rest wavelength [A]")
    ax.set_ylabel("A(lambda) [mag]")
    ax.legend()

    ax = axes[1, 0]
    resid_control = av * (k_teng - k_anal)
    resid_young = teng_young - fsps_young
    ax.axhline(0, color="k", lw=0.5)
    ax.plot(wg, resid_control, label="control (Calzetti) residual", lw=1.5)
    ax.plot(wg, resid_young, label="young BC residual (tengri - FSPS)", lw=2)
    ax.set_title("Residual A(lambda) [mag] — negative = tengri under-attenuates")
    ax.set_xscale("log")
    ax.set_xlabel("rest wavelength [A]")
    ax.set_ylabel("delta A [mag]")
    ax.legend()

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ("metric", "tengri", "FSPS", "tengri/FSPS"),
        (
            "A(NUV 2700A) young [mag]",
            f"{_at(teng_young, NUV):.3f}",
            f"{_at(fsps_young, NUV):.3f}",
            f"{_at(teng_young, NUV) / _at(fsps_young, NUV):.3f}",
        ),
        (
            "A(FUV 1500A) young [mag]",
            f"{_at(teng_young, FUV):.3f}",
            f"{_at(fsps_young, FUV):.3f}",
            f"{_at(teng_young, FUV) / _at(fsps_young, FUV):.3f}",
        ),
        (
            "NUV/V color ratio",
            f"{_ratio(teng_young, NUV, VBAND):.3f}",
            f"{_ratio(fsps_young, NUV, VBAND):.3f}",
            "—",
        ),
    ]
    tbl = ax.table(cellText=rows, loc="center", cellLoc="left")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1.0, 1.6)
    ax.set_title("Birth-cloud UV under-attenuation")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def main() -> None:
    av = 1.0
    tau_v = av / MAG_PER_TAU

    # Control check.
    k_teng = tengri_single_screen_calzetti()
    k_anal = analytic_calzetti_k()
    control_max_resid = float(np.max(np.abs(k_teng - k_anal)))

    # Young birth-cloud discrepancy.
    teng_young = tengri_two_component_curve(tau_v, 0.0, young=True)
    fsps_young = fsps_cf00_curve(tau_v, 0.0, young=True)
    teng_fixed = tengri_two_component_curve(tau_v, 0.0, young=True, slope=-1.0)

    a_nuv_teng = _at(teng_young, NUV)
    a_nuv_fsps = _at(fsps_young, NUV)
    a_nuv_fixed = _at(teng_fixed, NUV)
    # Flux ratio tengri/FSPS = 10^(0.4*(A_fsps - A_teng)) (less attenuation -> brighter).
    flux_ratio_nuv = 10.0 ** (0.4 * (a_nuv_fsps - a_nuv_teng))
    flux_ratio_nuv_fixed = 10.0 ** (0.4 * (a_nuv_fsps - a_nuv_fixed))

    print("=" * 70)
    print("DUST ATTENUATION CURVE DIFFERENTIAL — tengri vs external codes")
    print("=" * 70)
    print(
        f"\n[control] single-screen Calzetti, max |k_tengri - k_analytic| = "
        f"{control_max_resid:.2e}  "
        f"({'PASS' if control_max_resid < 1e-3 else 'FAIL'} < 1e-3)"
    )
    print("  -> single-screen path is correct; bagpipes/Prospector Calzetti match.\n")

    print("[two-component] young population, birth-cloud only (tau_bc=A_V/1.0857):")
    print(
        f"  A(NUV 2700A): tengri={a_nuv_teng:.3f}  FSPS={a_nuv_fsps:.3f} mag  "
        f"(tengri {a_nuv_fsps - a_nuv_teng:.3f} mag low)"
    )
    print(
        f"  => NUV flux ratio tengri/FSPS = {flux_ratio_nuv:.3f} (cf. CROSSVAL-01 measured 1.291)"
    )
    print("  With BC slope set to -1.0 (the proposed fix knob):")
    print(
        f"  A(NUV)={a_nuv_fixed:.3f} mag -> flux ratio = {flux_ratio_nuv_fixed:.3f} "
        f"({'RECOVERED' if abs(flux_ratio_nuv_fixed - 1.0) < 0.01 else 'residual'})"
    )

    out = Path(__file__).resolve().parent / "figures" / "dust_attenuation_curve_differential.png"
    if os.environ.get("TENGRI_NO_FIG") != "1":
        _make_figure(out)
        print(f"\nfigure: {out}")


if __name__ == "__main__":
    main()
