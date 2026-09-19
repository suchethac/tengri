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
* The exact fold should never be *worse* than the node fold in any band at any
  redshift. One that is has almost certainly evaluated the transmission in the
  wrong frame: T_IGM takes observed-frame wavelength, so the rest-frame SSP
  grid needs a (1+z) before the lookup, and omitting it produces a plausible
  but wrong answer that still varies with redshift.

Usage::

    python -m paper1.validate_igm_fold                    # default flag spelling
    python -m paper1.validate_igm_fold --flag igm_fold --exact-value exact
"""

from __future__ import annotations

import argparse

import jax
import numpy as np

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp

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

SSP_GRID = "fsps_mist_c3k_a_chabrier"


def build(ssp, obs, z, approx):
    """One model, identical in every respect but the approximation path."""
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
    parser.add_argument("--node-value", default="node")
    parser.add_argument("--exact-value", default="exact")
    args = parser.parse_args(argv)

    ssp = tengri.load_ssp(SSP_GRID)
    obs = tengri.Observation(photometry=tengri.Photometry.from_names(PROBE_FILTERS))

    print(f"SSP {SSP_GRID}; bands {PROBE_FILTERS}")
    print(f"selecting the fold with WavePrecomp({args.flag}=...)\n")
    print(f"{'z':>5}  {'band':<10} {'node err %':>11} {'exact err %':>12}  verdict")
    print("-" * 62)

    worse_at = []
    for z in PROBE_REDSHIFTS:
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
            e_node = 100.0 * abs(got["node"][i] - reference[i]) / reference[i]
            e_exact = 100.0 * abs(got["exact"][i] - reference[i]) / reference[i]
            verdict = "ok"
            # The exact fold must never be worse than the node fold by more
            # than quadrature noise. Worse means a real defect, most likely a
            # frame error on T_IGM.
            if e_exact > e_node + 1e-6 and e_exact > 1e-4:
                verdict = "EXACT WORSE"
                worse_at.append((z, band, e_node, e_exact))
            print(f"{z:>5.2f}  {band:<10} {e_node:>10.4f}% {e_exact:>11.4f}%  {verdict}")
        print()

    if worse_at:
        print("FAIL: the exact fold is worse than the node fold somewhere:")
        for z, band, en, ee in worse_at:
            print(f"  z={z} {band}: node {en:.4f}% vs exact {ee:.4f}%")
        print(
            "\nCheck the frame first: T_IGM takes OBSERVED-frame wavelength, so a "
            "rest-frame SSP grid needs lambda*(1+z) before the lookup. Omitting "
            "that gives a wrong answer that still varies with z, so it looks "
            "plausible."
        )
        return 1

    print("[ok] the exact fold is nowhere worse than the node fold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
