"""Decompose what the exact IGM fold leaves behind, component by component.

``validate_igm_fold.py`` answers whether the exact fold is correct. This answers
a different question the appendix makes a claim about: once the IGM term is
folded exactly at build time, what is left?

The appendix said "leaves only the dust-node error". That is incomplete. The
sub-band quadrature applies the node approximation to the multiplicative screen
on the stellar *and nebular* continuum, and measured here the nebular continuum
is the larger of the two contributions -- with no dust in the model at all.

Four arms, each against ``approx=None`` (the exact wavelength-grid integrator):

    bare stellar   the IGM is the only approximation; the exact fold must read 0
    + dust         adds the dust screen's node error
    + nebular      adds the nebular continuum's node error, no dust present
    + both         the production configuration

One model build per process is NOT used here; if this OOMs on a busy machine,
run with ``--arm`` to do one arm at a time.

Usage::

    python -m paper1.igm_fold_error_budget
    python -m paper1.igm_fold_error_budget --arm bare --json out.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import tengri
from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp

SSP_GRID = "fsps_mist_c3k_a_chabrier"
#: The band the Lyman break is sweeping at the probe redshift, so the IGM term
#: is live rather than negligible.
PROBE_BAND = "galex_nuv"
PROBE_Z = 1.0

#: Arm name -> (dust on, nebular on).
ARMS = {
    "bare": (False, False),
    "dust": (True, False),
    "nebular": (False, True),
    "both": (True, True),
}


def build(ssp, obs, z, approx, *, dust: bool, nebular: bool) -> SEDModel:
    """One model, identical but for the approximation path and the two blocks."""
    kwargs = dict(
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
    if dust:
        kwargs["dust_attenuation"] = {
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Fixed(0.3),
        }
        kwargs["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    if nebular:
        kwargs["neb"] = {
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "neb_logU": Fixed(-2.5),
        }
    return SEDModel.build(**kwargs)


def measure(ssp, obs, arm: str, z: float) -> dict:
    """Signed percent error of each fold against the exact integrator, one arm."""
    dust, nebular = ARMS[arm]
    ref_model = build(ssp, obs, z, None, dust=dust, nebular=nebular)
    params = {}
    for name in ref_model.spec.free_params:
        dist = ref_model.spec.get_distribution(name)
        params[name] = float(dist.mean()) if hasattr(dist, "mean") else 0.0
    reference = float(np.asarray(ref_model.predict_photometry(params))[0])

    out = {"arm": arm, "dust": dust, "nebular": nebular, "z": z}
    for fold in ("node", "exact"):
        model = build(ssp, obs, z, WavePrecomp(igm_fold=fold), dust=dust, nebular=nebular)
        value = float(np.asarray(model.predict_photometry(params))[0])
        out[fold] = 100.0 * (value - reference) / reference
    return out


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arm", choices=sorted(ARMS), default=None)
    parser.add_argument("--z", type=float, default=PROBE_Z)
    parser.add_argument("--band", default=PROBE_BAND)
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args(argv)

    ssp = tengri.load_ssp(SSP_GRID)
    obs = tengri.Observation(photometry=tengri.Photometry.from_names([args.band]))
    arms = [args.arm] if args.arm else ["bare", "dust", "nebular", "both"]

    rows = []
    print(f"{args.band} at z={args.z}; signed % error against approx=None")
    print(f"{'arm':<10} {'dust':>5} {'neb':>5} {'node %':>10} {'exact %':>10}")
    print("-" * 44)
    for arm in arms:
        row = measure(ssp, obs, arm, args.z)
        rows.append(row)
        print(
            f"{arm:<10} {row['dust']!s:>5} {row['nebular']!s:>5} "
            f"{row['node']:>+10.4f} {row['exact']:>+10.4f}"
        )

    if args.json:
        args.json.write_text(json.dumps({"band": args.band, "rows": rows}, indent=2))
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
