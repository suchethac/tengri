"""Wall-clock probe: NSS against NUTS on one CANDELS cell.

Same galaxy, configuration, data floor and model build as ``fit_one`` -- the
setup is imported from it, not copied -- with the sampler swapped for nested
slice sampling. Reports wall time, log Z, and per-parameter posterior medians
and 16-84 widths so the NUTS run on the same cell can be sized against it.

    python -m analysis.paper1.probe_nss --galaxy 79 --config III --preset fast
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import jax
import numpy as np

from tengri import Data, ForwardModel, Observation, Photometry

from .candels_io import load_candels_z1
from .configs import config_I, config_II, config_III, config_IV, config_V, config_VI, load_ssp_for
from .fit_one import apply_systematic_error_floor, extract_photometry

logger = logging.getLogger(__name__)

BUILDERS = {
    "I": config_I,
    "II": config_II,
    "III": config_III,
    "IV": config_IV,
    "V": config_V,
    "VI": config_VI,
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--galaxy", type=int, required=True)
    parser.add_argument("--config", required=True, choices=sorted(BUILDERS))
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--preset", default="fast", choices=["fast", "accurate"])
    parser.add_argument("--n-live", type=int, default=None)
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    candels = load_candels_z1()
    _, z, filter_names, fnu, sigma = extract_photometry(
        args.galaxy, candels, candels["z"][candels["id"] == args.galaxy][0]
    )
    sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=0.05)
    obs = Observation(photometry=Photometry.from_names(filter_names))
    ssp = load_ssp_for(args.config)
    sed_model = BUILDERS[args.config](ssp, obs, z)
    forward = ForwardModel.build(sed=sed_model)
    data = Data(photometry=(fnu, sigma_floor))
    logger.info(
        f"Galaxy {args.galaxy} config {args.config}: D={sed_model.spec.n_free}, "
        f"{len(filter_names)} bands, NSS preset={args.preset}"
    )

    kwargs = {"preset": args.preset}
    if args.n_live is not None:
        kwargs["n_live"] = args.n_live
    t0 = time.perf_counter()
    posterior = forward.fit(data, method="nss", key=jax.random.PRNGKey(args.seed), **kwargs)
    wall = time.perf_counter() - t0

    diag = dict(posterior.diagnostics or {})
    samples = {k: np.asarray(v) for k, v in posterior.samples.items()}
    summary = {}
    for k, v in samples.items():
        p16, p50, p84 = np.percentile(v, [16, 50, 84])
        summary[k] = {"p16": float(p16), "p50": float(p50), "p84": float(p84), "n": int(v.size)}

    out = {
        "gal_id": args.galaxy,
        "config": args.config,
        "z": float(z),
        "n_free": int(sed_model.spec.n_free),
        "method": "nss",
        "preset": args.preset,
        "wall_time_s": wall,
        # np.isscalar is True for a str, and diagnostics carry the profile_mass
        # reason as one; float() on it lost the first run's summary.
        "diagnostics": {
            k: (float(v) if isinstance(v, (int, float, np.integer, np.floating)) else str(v))
            for k, v in diag.items()
        },
        "summary": summary,
    }
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"{args.galaxy}_{args.config}_nss.json"
    path.write_text(json.dumps(out, indent=2))
    logger.info(f"NSS complete in {wall:.0f}s; diagnostics: {out['diagnostics']}")
    for k, s in summary.items():
        logger.info(f"  {k:32} {s['p50']:+.4f}  [{s['p16']:+.4f}, {s['p84']:+.4f}]")
    logger.info(f"saved {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
