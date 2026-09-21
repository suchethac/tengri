#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
"""Which SFH forms have a staircase in their age parameter, and which do not.

``mean_sfh.lognormal`` evaluated a density in ``T = age - t_lookback`` and
masked it where ``T <= 0``. At a wide width that density is nearly flat across
the grid, so every age node switched on at full value and the trapezoid jumped
each time the moving boundary crossed one. Configuration V's fits collapsed on
it: an energy error at every step, divergences, and an adapted step size
driven toward zero.

That was one form. The question this answers is which others do it, and the
answer has to be enumerated from the registry rather than listed by hand --
there are thirty-eight forms and the paper uses five of them.

The probe is a fine sweep of one SFH's age parameter, with everything else at
its registry default, reporting the largest single step in the summed
photometry as a multiple of the median step. A smooth curve sits near 1. The
log-normal read 7.6 before its fix on this measure, with a dedicated sweep at
width 2.14 dex.

Two rules it follows, both learned from getting them wrong:

* a form whose age parameter does not move the photometry is reported as
  inert, not as smooth;
* a median step of zero makes the ratio undefined, so the verdict is
  "undetermined" rather than a pass. An absent measurement is not a good one.

CLI:
    python analysis/paper1/sfh_staircase_sweep.py [--ssp NAME] [--verbose]
"""

from __future__ import annotations

import argparse
import sys
import warnings

import numpy as np

#: Above this multiple of the median step, the curve is not smooth.
STAIRCASE = 4.0
#: Between this and STAIRCASE, worth a look but not a verdict.
WATCH = 2.0
#: Teeth fall roughly 20 Myr apart near this age on a log-spaced grid.
AGES = np.linspace(1.30, 1.50, 201)
PROBE_BANDS = ["sdss_g", "sdss_r", "sdss_i", "2mass_j"]
DEFAULT_SSP = "fsps_mist_c3k_a_chabrier"


def step_excess(values: np.ndarray) -> float | None:
    """Largest step over the median step, or None when that is undefined."""
    steps = np.abs(np.diff(values)) / np.abs(values[:-1])
    median = float(np.median(steps))
    if not np.isfinite(median) or median == 0.0:
        return None
    return float(steps.max() / median)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ssp", default=DEFAULT_SSP)
    parser.add_argument("--verbose", action="store_true", help="show build failures")
    args = parser.parse_args(argv)

    warnings.filterwarnings("ignore")
    import jax

    import tengri
    from tengri import DEFAULT, Fixed, SEDModel, Uniform
    from tengri.components.stellar.sfh import registry as reg

    try:
        ssp = tengri.load_ssp(args.ssp)
    except FileNotFoundError as exc:
        print(f"{exc}\n\nPass --ssp with a library this machine has.", file=sys.stderr)
        return 2
    observation = tengri.Observation(photometry=tengri.Photometry.from_names(PROBE_BANDS))
    specs = getattr(reg, "SFH_REGISTRY", {})
    names = sorted(specs)
    print(f"{len(names)} registry forms, swept in their own age parameter\n")
    print(f"{'form':<26}{'max step / median':>18}  verdict")

    findings = []
    for name in names:
        params = list(getattr(specs[name], "params", {}) or {})
        age_param = next((p for p in params if p.endswith("_age_gyr")), None)
        if age_param is None:
            print(f"{name:<26}{'-':>18}  no age parameter: no moving onset")
            continue
        short = age_param.split(f"sfh_{name}_")[-1]
        try:
            model = SEDModel.build(
                ssp_data=ssp,
                observation=observation,
                neb={"type": "none"},
                sfh={"type": name, "all_params": Fixed(DEFAULT), short: Uniform(0.5, 3.0)},
                redshift=Fixed(1.0),
                approx=None,
            )
            # Bound as defaults, not captured: a closure over the loop
            # variables would evaluate whichever form the loop had reached by
            # the time it ran, which is a real trap even where it happens to
            # work because the call is immediate.
            predict = jax.jit(
                lambda a, _m=model, _p=age_param: sum(_m.predict_photometry({_p: a}))
            )
            values = np.array([float(predict(float(a))) for a in AGES])
        except Exception as exc:
            detail = f": {type(exc).__name__}" + (f" {exc}"[:60] if args.verbose else "")
            print(f"{name:<26}{'-':>18}  could not build{detail}")
            continue

        if values.std() == 0.0:
            print(f"{name:<26}{'-':>18}  age is inert: the sweep cannot judge it")
            continue
        excess = step_excess(values)
        if excess is None:
            print(f"{name:<26}{'undetermined':>18}  median step is zero; no verdict")
            continue
        verdict = "STAIRCASE" if excess > STAIRCASE else ("watch" if excess > WATCH else "smooth")
        print(f"{name:<26}{excess:>18.2f}  {verdict}")
        if excess > STAIRCASE:
            findings.append((name, excess))

    if findings:
        print("\nforms with a staircase in age:")
        for name, excess in findings:
            print(f"  {name}: {excess:.1f}x the median step")
        return 1
    print("\nno form showed a staircase in its age parameter")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
