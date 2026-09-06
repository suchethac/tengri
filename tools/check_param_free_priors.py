#!/usr/bin/env python3
"""CI guard: every parameter is either freeable or refused on the record.

``all_params: FREE`` resolves each parameter to its declared ``free_prior``. A
parameter that has none stays pinned, and before #887 that was indistinguishable
from a parameter someone had decided should stay pinned -- ``free_prior=None``
meant both "we chose this" and "nobody looked". 86 of 358 parameters were in that
state, only 3 with a recorded reason, so the backlog was invisible.

This guard removes the ambiguity. A parameter with a ``Fixed`` default and no
``free_prior`` must appear in :data:`REFUSED` with a reason. Adding a new one
fails the build until someone writes down which it is -- a range, or why there
isn't one.

The reasons below are not decoration. They record four genuinely different
grounds for refusing, and the distinction matters when revisiting them:

``inert``
    Freeing it would add a dimension the selected code path never reads. Either
    the group's wildcard is not scoped to the structural variant that owns the
    parameter, or the data needed to make it live is not shipped
    (``dust_frac_agn`` needs the QSO template). Fixing the scoping makes these
    declarable, and that is not hypothetical: the four ``dust`` attenuation-law
    shape modifiers left this list once ``parse_groups`` began narrowing the
    group wildcard to the laws a build actually selects.
``fixed-by-physics``
    A perfectly good range exists but the quantity is not a per-object degree of
    freedom -- an analytic constant, or a population-level calibration whose
    per-object freedom is already exposed as a separate parameter.
``not-continuous``
    The value is discrete or a sentinel, so a continuous prior is not merely
    wide but wrong: it would spend its mass on values that name nothing, or give
    the sentinel measure zero.
``no-evidence``
    A range would be defensible but could not be sourced. These are the ones to
    revisit first; each names what is missing.
``target-dependent``
    The admissible range is set by the source being fitted rather than by
    physics or a grid, so no static interval is correct for every target. An
    absolute luminosity or SFR has no galaxy-independent scale; an SF-onset
    lookback is capped by the age of the universe at the source redshift (8.6
    Gyr at z=0.5, 0.9 at z=6), so a bound generous enough for z~0 admits
    zero-star-formation draws at z=2. These must be freed against the caller's
    own target.
``explicit-only``
    A genuine per-object freedom with a genuine range, withheld from the
    wildcard because the data a default fit has cannot constrain it. Freeing it
    by default would add a dimension whose posterior is its prior to every model
    that opts the block free -- real cost, no information. The caller can and
    should free it explicitly when their data supports it.

Usage
-----
    python tools/check_param_free_priors.py

Exit 0 if every pinned parameter is accounted for; 1 with the unlisted ones
otherwise.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))

from tengri.parameters.priors import Fixed
from tengri.parameters.registry import registry

#: name -> (ground, reason). See the module docstring for the four grounds.
REFUSED: dict[str, tuple[str, str]] = {
    # ── inert: the wildcard is not scoped to the variant that reads them ──
    # The four attenuation-law shape modifiers that used to sit here --
    # dust_Rv, dust_delta, dust_bump_strength, dust_slope -- are declared now.
    # Scoping the dust wildcard to the laws the build selects is what made them
    # declarable, which is the general remedy this ground was pointing at.
    "dust_frac_agn": ("inert", "needs templates_qso; the default Dale file ships only SF"),
    "neb_logZ_gas": ("inert", "range is the selected nebular backend's grid; neb is unscoped"),
    "neb_xid": ("inert", "Feltre NLR only, and a 3-node grid inside a wider validator"),
    # ── fixed-by-physics: real range, but not a per-object freedom ──
    "radio_alpha_ff": ("fixed-by-physics", "optically-thin bremsstrahlung is -0.1 analytically"),
    "radio_delv_mass_slope": ("fixed-by-physics", "FIRRC slope; degenerate at fixed (M*, z)"),
    "radio_delv_z_slope": ("fixed-by-physics", "as radio_delv_mass_slope"),
    "radio_mcch_mass_slope": ("fixed-by-physics", "as radio_delv_mass_slope"),
    "radio_mcch_z_slope": ("fixed-by-physics", "as radio_delv_mass_slope"),
    "xray_gamma_hmxb": ("fixed-by-physics", "Lehmer+2016 population constant; see xray_det_hmxb"),
    "xray_gamma_lmxb": ("fixed-by-physics", "Lehmer+2016 population constant; see xray_det_lmxb"),
    "met_alpha_fe": ("fixed-by-physics", "pre-existing decision; only constrained by spectra"),
    "met_alpha_fe_young": ("fixed-by-physics", "as met_alpha_fe"),
    "dust_f_obscuration": ("fixed-by-physics", "pre-existing decision; degenerate with tau_diff"),
    # ── target-dependent: the bound is set by the source, not by physics ──
    "dust_L_agn_ir": ("target-dependent", "absolute luminosity; no galaxy-independent scale"),
    "sfh_exp_start_gyr": ("target-dependent", "onset capped by the age of the universe at z"),
    "sfh_dexp_start_gyr": ("target-dependent", "as sfh_exp_start_gyr; caught by test_bug_1031"),
    "sfh_const_start_gyr": ("target-dependent", "as sfh_exp_start_gyr"),
    # ── not-continuous: discrete values, sentinels, or ordering constraints ──
    "sfh_periodic_burst_type": (
        "not-continuous",
        "validator requires int; selects one of 3 shapes",
    ),
    "neb_hbfrac": ("not-continuous", "snapped to the nearest of 2 grid values at load time"),
    "shock_log_density": ("not-continuous", "snapped to nearest grid point -- zero gradient"),
    "shock_b_over_sqrt_n": ("not-continuous", "snapped to nearest grid point -- zero gradient"),
    "noise_dof": ("not-continuous", "0 is a sentinel selecting the Gaussian likelihood"),
    "dla_z": ("not-continuous", "0 is a sentinel meaning 'use the source redshift'"),
    "redshift": ("not-continuous", "top-level grammar argument; survey-dependent range"),
    "sfh_const_end_gyr": ("not-continuous", "ordering constraint with sfh_const_start_gyr"),
    "sfh_dpl_lookback_end_gyr": (
        "not-continuous",
        "ordering constraint with sfh_dpl_lookback_age_gyr; overlapping supports are refused",
    ),
    "sfh_trunc_exp_end_gyr": (
        "not-continuous",
        "ordering constraint with sfh_trunc_exp_age_gyr; overlapping supports are refused",
    ),
    # ── explicit-only: real freedom, but not one a default fit can constrain ──
    "met_logzsol_scatter": (
        "explicit-only",
        "MDF second moment; declaring one added it to 6 of 10 shipped recipes",
    ),
    # ── no-evidence: revisit these first ──
    "radio_alpha_thin": ("no-evidence", "needs Table 1 of Martinez-Ramirez+2024"),
    "radio_alpha_thick": (
        "no-evidence",
        "needs Table 1 of Martinez-Ramirez+2024; sign convention",
    ),
    "agn_grahsp_a_bc": ("no-evidence", "needs the GRAHSP prior table (arXiv:2405.19297)"),
    "xray_delta_alpha_ox": ("no-evidence", "needs the Just+2007 alpha_ox intrinsic scatter"),
    "agn_xray_delta_alpha_ox": ("no-evidence", "as xray_delta_alpha_ox; kept in step with it"),
}

VALID_GROUNDS = {
    "inert",
    "fixed-by-physics",
    "not-continuous",
    "no-evidence",
    "explicit-only",
    "target-dependent",
}


def main() -> int:
    reg = registry()
    pinned = {
        name
        for name in reg
        if getattr(reg.get(name), "free_prior", None) is None
        and isinstance(getattr(reg.get(name), "prior", None), Fixed)
    }

    bad_ground = {n: g for n, (g, _) in REFUSED.items() if g not in VALID_GROUNDS}
    unlisted = sorted(pinned - set(REFUSED))
    stale = sorted(set(REFUSED) - pinned)

    if bad_ground:
        print("Unknown refusal ground (must be one of " + ", ".join(sorted(VALID_GROUNDS)) + "):")
        for name, ground in sorted(bad_ground.items()):
            print(f"  {name}: {ground!r}")

    if unlisted:
        print(f"\n{len(unlisted)} parameter(s) resolve to a Fixed default with no free_prior")
        print("and no recorded reason. 'all_params: FREE' silently leaves each one pinned:\n")
        for name in unlisted:
            print(f"  {name}")
        print(
            "\nGive each a free_prior (its admissible range -- measured from the grid it\n"
            "indexes where there is one, never transcribed from a description), or add it\n"
            "to REFUSED in this file with one of the four grounds and a reason."
        )

    if stale:
        print(f"\n{len(stale)} REFUSED entr(y/ies) no longer describe a pinned parameter.")
        print("They now carry a free_prior, or were renamed or removed. Drop them:\n")
        for name in stale:
            print(f"  {name}")

    if unlisted or stale or bad_ground:
        return 1

    by_ground: dict[str, int] = {}
    for ground, _ in REFUSED.values():
        by_ground[ground] = by_ground.get(ground, 0) + 1
    freeable = sum(1 for n in reg if getattr(reg.get(n), "free_prior", None) is not None)
    print(f"OK: {len(reg)} parameters, {freeable} freeable by 'all_params: FREE'.")
    print(
        f"{len(REFUSED)} deliberately pinned: "
        + ", ".join(f"{g} {c}" for g, c in sorted(by_ground.items()))
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
