"""Measure the IGM node fold and the exact fold against the exact integrator.

The reference is not another precomputed path: it is ``approx=None``, the exact
wavelength-grid integrator, which applies IGM transmission on the full grid and
never forms a sub-band average at all. Both folds are approximations of it, and
the question this script answers is how much each one costs, band by band, as
the Lyman break sweeps through the filter set.

Why a separate reference matters. The node fold and the exact fold share a code
path up to the point where transmission enters, so comparing them to each other
measures their difference and not their error -- they could agree closely and
both be wrong. The exact integrator is the only arm here that does not use the
sub-band quadrature.

What to expect if the exact fold is correct:

* At low redshift, where no break lies inside any band, all three agree to the
  quadrature floor and the two folds are indistinguishable. A large difference
  here means the exact fold has broken something unrelated to the IGM.
* As the break enters a band, the node fold's error grows sharply -- it forms
  <S><T> where the flux needs <S.T> -- while the exact fold's error stays at
  the quadrature floor, because the step is resolved by the integrand rather
  than sampled at a point.
* On the BARE-STELLAR control the exact fold must reproduce the exact
  integrator to the quadrature floor. One that does not has almost certainly
  evaluated the transmission in the wrong frame: T_IGM takes observed-frame
  wavelength, so the rest-frame SSP grid needs a (1+z) before the lookup, and
  omitting it produces a plausible but wrong answer that still varies with
  redshift.

What this script must NOT conclude, and once did. In a model carrying dust and
nebular emission, BOTH folds inherit the sub-band quadrature's error on those
spectral shapes -- a term that is not about the IGM at all and that can exceed
the IGM error itself. The node fold's own IGM error oscillates in sign as the
break sweeps a band, so wherever it crosses zero the two errors partially
cancel and the node fold's *absolute* error can be the smaller one. Measured at
z=1.0 in galex_nuv: node -0.398%, exact -0.693% in the full model, which reads
as the exact fold losing; strip dust and nebular and it is node +0.365% against
exact +0.0000%. Ranking the folds by absolute error in a rich model therefore
measures a sign coincidence. The rich-model table is kept because it reports
the magnitudes a real fit sees, but the pass/fail gate is the control.

Usage::

    python -m paper1.validate_igm_fold                    # default flag spelling
    python -m paper1.validate_igm_fold --flag igm_fold --exact-value exact
"""

from __future__ import annotations

import argparse

import jax
import numpy as np

import tengri
from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp

jax.config.update("jax_enable_x64", True)

#: Bands chosen so the break sweeps through them at different redshifts: GALEX
#: FUV is where the node fold is known to fail, the optical bands are where the
#: science lives and where any regression would matter.
PROBE_FILTERS = [
    "galex_fuv",
    "galex_nuv",
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
]

#: Redshifts spanning "break far blueward of every band" through "break inside
#: FUV" to "break through FUV entirely".
PROBE_REDSHIFTS = [0.05, 0.5, 0.8, 1.0, 1.5, 2.0, 3.0]

#: Largest |error| the exact fold may show on the bare-stellar control, in
#: percent. Measured at 0.0000% across z=0.8-1.5 on a single band, so this is
#: three orders of headroom over what a correct fold produces, and still far
#: below the node fold's 0.3-0.4% at the same redshifts.
EXACT_FOLD_TOLERANCE_PCT = 0.01

SSP_GRID = "fsps_mist_c3k_a_chabrier"


def build(ssp, obs, z, approx, *, bare=False):
    """One model, identical in every respect but the approximation path.

    ``bare=True`` drops dust and nebular emission. That arm is the only one
    that isolates the IGM: see :func:`igm_attributable_error`.
    """
    if bare:
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={
                "type": "delayed",
                "all_params": Fixed(DEFAULT),
                "tau_gyr": Fixed(2.0),
                "age_gyr": Fixed(3.0),
                "log_total_mass": Fixed(10.0),
                "met_logzsol": Fixed(0.0),
            },
            redshift=Fixed(z),
            igm={"type": "inoue"},
            approx=approx,
        )
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "tau_gyr": Fixed(2.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "met_logzsol": Fixed(0.0),
        },
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Fixed(0.3),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Fixed(-2.5)},
        redshift=Fixed(z),
        igm={"type": "inoue"},
        approx=approx,
    )


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--flag",
        default="igm_fold",
        help="WavePrecomp keyword selecting the fold (default: igm_fold)",
    )
    parser.add_argument(
        "--redshifts",
        default=None,
        help="Comma-separated subset of redshifts (default: the full probe set). "
        "Each redshift costs three model builds, so a subset is the cheap first look.",
    )
    parser.add_argument("--node-value", default="node")
    parser.add_argument("--exact-value", default="exact")
    args = parser.parse_args(argv)

    ssp = tengri.load_ssp(SSP_GRID)
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(PROBE_FILTERS))

    print(f"SSP {SSP_GRID}; bands {PROBE_FILTERS}")
    print(f"selecting the fold with WavePrecomp({args.flag}=...)\n")
    redshifts = (
        [float(x) for x in args.redshifts.split(",")] if args.redshifts else PROBE_REDSHIFTS
    )
    print(
        f"{'z':>5}  {'band':<10} {'node err %':>11} {'exact err %':>12}"
        f" {'IGM part %':>11}  verdict"
    )
    print("-" * 78)

    canceled_at = []
    for z in redshifts:
        ref_model = build(ssp, obs, z, None)
        params = ref_model.spec.sample(key=jax.random.PRNGKey(0))
        reference = np.asarray(ref_model.predict_photometry(params))

        got = {}
        for label, value in ((args.node_value, "node"), (args.exact_value, "exact")):
            try:
                approx = WavePrecomp(**{args.flag: label})
            except TypeError as exc:
                print(f"\nWavePrecomp does not accept {args.flag}={label!r}: {exc}")
                print("Pass the real flag spelling with --flag / --exact-value.")
                return 2
            model = build(ssp, obs, z, approx)
            got[value] = np.asarray(model.predict_photometry(params))

        for i, band in enumerate(PROBE_FILTERS):
            if reference[i] <= 0:
                continue
            # SIGNED, not absolute. The node fold's error oscillates in sign as
            # the break sweeps a band, and an absolute value hides the zero
            # crossings -- which is what made an earlier version of this script
            # report a defect that was not one. See the module docstring.
            e_node = 100.0 * (got["node"][i] - reference[i]) / reference[i]
            e_exact = 100.0 * (got["exact"][i] - reference[i]) / reference[i]
            # The IGM-attributable part of the node fold's error. Both folds
            # carry the same sub-band quadrature error on the dust and nebular
            # shapes; it cancels in the difference, leaving the term that is
            # actually about the IGM.
            e_igm = e_node - e_exact
            verdict = "ok"
            if abs(e_exact) > abs(e_node) + 1e-6 and abs(e_exact) > 1e-4:
                verdict = "node smaller (cancellation)"
                canceled_at.append((z, band, e_node, e_exact))
            print(
                f"{z:>5.2f}  {band:<10} {e_node:>+10.4f}% {e_exact:>+11.4f}%"
                f" {e_igm:>+10.4f}%  {verdict}"
            )
        print()

    # THE GATE. Everything above is a rich model, in which both folds carry the
    # same sub-band quadrature error on the dust and nebular shapes -- a term
    # that has nothing to do with the IGM and can be larger than the IGM error
    # itself. Comparing the two folds' absolute errors there measures that
    # shared term plus a sign coincidence, not correctness.
    #
    # Stripping dust and nebular emission removes the confounder: in a bare
    # stellar model the ONLY approximation left between the two folds is how
    # transmission enters the sub-band average, so an exact fold that is truly
    # exact must reproduce the exact integrator to the quadrature floor. That is
    # a claim with no cancellation mode, and it is the one worth failing on.
    print("=" * 78)
    print("CONTROL: bare stellar (no dust, no nebular) -- the IGM is the only")
    print("approximation left, so the exact fold must read ~0 here.")
    print(f"{'z':>5}  {'band':<10} {'node err %':>11} {'exact err %':>12}  verdict")
    print("-" * 62)

    failures = []
    for z in redshifts:
        ref_model = build(ssp, obs, z, None, bare=True)
        params = ref_model.spec.sample(key=jax.random.PRNGKey(0))
        reference = np.asarray(ref_model.predict_photometry(params))
        got = {}
        for label, value in ((args.node_value, "node"), (args.exact_value, "exact")):
            model = build(ssp, obs, z, WavePrecomp(**{args.flag: label}), bare=True)
            got[value] = np.asarray(model.predict_photometry(params))

        for i, band in enumerate(PROBE_FILTERS):
            if reference[i] <= 0:
                continue
            e_node = 100.0 * (got["node"][i] - reference[i]) / reference[i]
            e_exact = 100.0 * (got["exact"][i] - reference[i]) / reference[i]
            verdict = "ok"
            if abs(e_exact) > EXACT_FOLD_TOLERANCE_PCT:
                verdict = "FAIL"
                failures.append((z, band, e_node, e_exact))
            print(f"{z:>5.2f}  {band:<10} {e_node:>+10.4f}% {e_exact:>+11.4f}%  {verdict}")
        print()

    if canceled_at:
        print(
            f"note: in {len(canceled_at)} rich-model cell(s) the node fold's "
            "absolute error was the smaller one. That is not evidence against "
            "the exact fold: the node fold's error oscillates in sign, so where "
            "it crosses zero it can sit closer to the reference than a fold "
            "carrying only the shared quadrature term. Judge on the control."
        )

    if failures:
        print("\nFAIL: the exact fold departs from the exact integrator with no")
        print("dust or nebular emission present, where nothing else can explain it:")
        for z, band, en, ee in failures:
            print(f"  z={z} {band}: exact {ee:+.4f}% (node {en:+.4f}%)")
        print(
            "\nCheck the frame first: T_IGM takes OBSERVED-frame wavelength, so a "
            "rest-frame SSP grid needs lambda*(1+z) before the lookup. Omitting "
            "that gives a wrong answer that still varies with z, so it looks "
            "plausible."
        )
        return 1

    print("[ok] the exact fold reproduces the exact integrator on the bare-stellar")
    print("     control at every probed redshift; the IGM fold is correct.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
