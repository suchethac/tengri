#!/usr/bin/env python3
"""Report the nebular feature table's accuracy against the exact path.

Measuring tool, not a CI gate (the build-time gate is the shared cell-centre check).
Builds one model per configuration, and for every star-formation history listed
below reports the worst relative error of the grid-served lines and bands against
the exact path, at grid nodes and at cell centres, with where it occurred. The
numbers are compared with the declared bounds (1e-4 for line tables, 1e-3 for band
tables; #3505 owner ruling 2026-11-16, item 5).

Usage::

    JAX_PLATFORMS=cpu PYTHONPATH=src python tools/measure_table_accuracy.py \
        [--backend cue] [--dust | --no-dust] [--max-points N] [--qh-linearity]

The definitions live in ``tests/contract/_table_accuracy.py``.
"""

from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

#: Rest-frame vacuum wavelengths [Angstrom]: Lya, C IV, He II, C III], [O II],
#: [Ne III], H-beta, [O III], H-alpha, [N II], [S II].
LINES = (
    1215.67,
    1549.48,
    1640.42,
    1908.73,
    3727.09,
    3869.86,
    4862.69,
    5008.24,
    6564.61,
    6585.27,
    6718.29,
)
BANDS = ("galex_nuv", "sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1")

#: Star-formation histories (delayed tau with a burst or quench, Ciesla et al. 2017):
#: near-constant, rising, old-dominated, bursts of three ages on an old population,
#: and a recent quench. Random prior draws are added on the command line.
SFH_DRAWS = {
    "constant": {
        "sfh_delayed_bq_tau_main_gyr": 10.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_r_sfr": 1.0,
    },
    "rising": {
        "sfh_delayed_bq_tau_main_gyr": 10.0,
        "sfh_delayed_bq_age_main_frac": 0.15,
        "sfh_delayed_bq_r_sfr": 1.0,
    },
    "old_tau1": {
        "sfh_delayed_bq_tau_main_gyr": 1.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_r_sfr": 1.0,
    },
    "old_tau5": {
        "sfh_delayed_bq_tau_main_gyr": 5.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_r_sfr": 1.0,
    },
    "burst_10myr_on_old": {
        "sfh_delayed_bq_tau_main_gyr": 1.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_age_bq_gyr": 0.01,
        "sfh_delayed_bq_r_sfr": 10.0,
    },
    "burst_30myr_on_old": {
        "sfh_delayed_bq_tau_main_gyr": 1.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_age_bq_gyr": 0.03,
        "sfh_delayed_bq_r_sfr": 10.0,
    },
    "burst_300myr_on_old": {
        "sfh_delayed_bq_tau_main_gyr": 1.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_age_bq_gyr": 0.3,
        "sfh_delayed_bq_r_sfr": 10.0,
    },
    "quenched_100myr": {
        "sfh_delayed_bq_tau_main_gyr": 3.0,
        "sfh_delayed_bq_age_main_frac": 0.9,
        "sfh_delayed_bq_age_bq_gyr": 0.1,
        "sfh_delayed_bq_r_sfr": 0.0,
    },
}


def _builder(backend: str, dust: bool):
    from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, load_ssp_data

    ssp = load_ssp_data(str(ROOT / "data" / "fsps_prsc_miles_chabrier.h5"))
    obs = Observation(photometry=Photometry.from_names(list(BANDS)))
    groups = dict(
        sfh={"type": "delayed_bq", "all_params": FREE},
        met={"logzsol": FREE},
        neb={"type": backend, "logU": FREE, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
    )
    if dust:
        groups["dust_attenuation"] = {
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        }
        groups["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}

    def build(approx):
        kw = dict(ssp_data=ssp, observation=obs, **groups)
        if approx is not None:
            kw["approx"] = approx
        return SEDModel.build(**kw)

    return build


def _random_draws(build, n: int, seed: int) -> dict:
    import jax

    model = build(None)
    keys = [n for n in model.spec.free_params if n.startswith("sfh_")]
    draws = {}
    for i in range(n):
        sample = model.spec.sample(jax.random.PRNGKey(seed + i))
        draws[f"prior_{i}"] = {k: float(sample[k]) for k in keys if "log_total_mass" not in k}
    return draws


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--backend", default="cue")
    parser.add_argument("--dust", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--max-points", type=int, default=None)
    parser.add_argument("--random-draws", type=int, default=4)
    parser.add_argument("--qh-linearity", action="store_true")
    args = parser.parse_args(argv)

    warnings.simplefilter("ignore")
    from tests.contract._table_accuracy import measure_qh_linearity, measure_table_vs_exact

    build = _builder(args.backend, args.dust)
    draws = {**SFH_DRAWS, **_random_draws(build, args.random_draws, seed=20261116)}
    print(f"backend={args.backend} dust={args.dust} draws={len(draws)} lines={len(LINES)}")
    if args.qh_linearity:
        w = measure_qh_linearity(build, sfh_draws=draws, line_waves=LINES)
        print(f"Q_H linearity: worst {w.error:.3e} at {w.where} ({w.draw}), n={w.n_compared}")
        return 0
    result = measure_table_vs_exact(
        build, sfh_draws=draws, line_waves=LINES, max_points=args.max_points
    )
    bounds = {"lines": 1e-4, "bands": 1e-3}
    for (channel, at), w in sorted(result.items()):
        verdict = "within" if w.error <= bounds[channel] else "OVER"
        point = ", ".join(f"{k}={v:.3f}" for k, v in w.point.items())
        print(
            f"{channel:5s} {at:7s}: worst {w.error:.3e} ({verdict} {bounds[channel]:g}) "
            f"at {w.where} [{point}] draw={w.draw} n={w.n_compared}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
