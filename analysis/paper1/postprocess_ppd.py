# SPDX-License-Identifier: BSD-3-Clause
"""No-refit posterior-predictive photometry pass over existing grid NPZs (#2089).

``fit_one.save_fit_outputs`` now writes ``model_photometry_median`` (and the new
``model_photometry_p16``/``model_photometry_p84``) as posterior-predictive
quantiles: ``PPD_N_DRAWS`` draws each pushed through ``predict_photometry``,
rather than a single ``predict`` at the componentwise-median parameter vector
(see ``fit_one.PPD_N_DRAWS`` for the measured basis -- stored chi2/n of 8.88 and
12.38 against a posterior-predictive 0.26 and 0.18 on two Config III cells).
Grid cells written before that fix still carry the old at-median-params array.
This script recomputes the three keys for those NPZs WITHOUT rerunning any
fit: it rebuilds the cell's model exactly as ``fit_one.run_fit`` does (from the
cell's own JSON for ``z`` and ``filter_names``, ``configs.config_I/II/III`` for
the physics, ``configs.load_ssp_for`` for the SSP grid), subsamples the
ALREADY-SAVED draws in the NPZ via ``fit_one.iter_draws``, and writes the three
keys back -- every other key in the NPZ is carried through unchanged, and the
write is atomic (``fit_one._atomic_replace_write``, the same tmp+``os.replace``
pattern the fit path uses).

CLI::

    python postprocess_ppd.py [--cells 13097_III,16049_III] [--n-draws 200] [--out-dir DIR]

``--cells`` is a comma-separated subset of ``run_candels_fits.GALAXIES`` x
``run_candels_fits.CONFIGS`` cells, e.g. ``13097_III,16049_III``; the default is
every cell of that grid. A cell with no NPZ on disk is skipped with a warning,
not an error -- the grid is not required to be complete. ``--out-dir`` defaults
to ``results/fits`` (the grid's own output directory) and this script never
descends into its archive subdirectories (``agn_24497/``,
``stale_met_ceiling_p03/``): it only ever opens ``{out_dir}/{gal_id}_{config}.npz``
for a (galaxy, config) pair drawn from the grid, never a directory listing.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import jax
import numpy as np
from configs import config_I, config_II, config_III, load_ssp_for
from fit_one import PPD_N_DRAWS, _atomic_replace_write, iter_draws
from run_candels_fits import CONFIGS as CONFIG_KEYS, GALAXIES

from tengri import Observation, Photometry

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

#: Default output directory, relative to this file -- the grid's own.
DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "fits"

#: Configuration key -> model builder, the same dispatch ``fit_one.run_fit`` uses.
CONFIG_BUILDERS = {"I": config_I, "II": config_II, "III": config_III}


def default_cells() -> list[tuple[int, str]]:
    """Every (gal_id, config_key) cell of ``run_candels_fits``'s grid, in its order."""
    return [(gal_id, config_key) for gal_id in GALAXIES for config_key in CONFIG_KEYS]


def parse_cells(value: str) -> list[tuple[int, str]]:
    """Parse a comma-separated ``--cells`` value like ``13097_III,16049_III``.

    Raises:
        argparse.ArgumentTypeError: the list is empty, or a token is not
            ``{gal_id}_{config_key}`` with ``config_key`` in ``{I, II, III}``.
    """
    cells: list[tuple[int, str]] = []
    for token in value.split(","):
        token = token.strip()
        if not token:
            continue
        gal_str, sep, config_key = token.rpartition("_")
        if not sep or not gal_str.isdigit() or config_key not in ("I", "II", "III"):
            raise argparse.ArgumentTypeError(
                f"malformed cell {token!r}; expected e.g. '13097_III'"
            )
        cells.append((int(gal_str), config_key))
    if not cells:
        raise argparse.ArgumentTypeError("--cells needs at least one cell")
    return cells


def rebuild_model(gal_id: int, config_key: str, out_dir: Path):
    """Rebuild one cell's ``SEDModel`` from its saved JSON, exactly as ``fit_one`` did.

    Reads ``z`` and ``filter_names`` from ``{out_dir}/{gal_id}_{config_key}.json``
    -- the cell's own record of what it was fit with -- rather than re-deriving
    them from the CANDELS catalog, so a re-selected filter set or an updated
    catalog cannot silently change the model a re-run postprocess pass builds.

    Returns:
        (sed_model, filter_names, z)
    """
    json_path = out_dir / f"{gal_id}_{config_key}.json"
    with open(json_path) as f:
        diagnostics = json.load(f)
    z = float(diagnostics["z"])
    filter_names = list(diagnostics["filter_names"])

    observation = Observation(photometry=Photometry.from_names(filter_names))
    ssp = load_ssp_for(config_key)
    sed_model = CONFIG_BUILDERS[config_key](ssp, observation, z)
    return sed_model, filter_names, z


def recompute_ppd_quantiles(sed_model, npz_payload: dict, n_draws: int) -> dict:
    """Posterior-predictive photometry quantiles from an already-loaded NPZ payload.

    Subsamples the free-parameter draws already in ``npz_payload`` (the saved,
    thinned record -- no sampler runs here) via ``fit_one.iter_draws`` and pushes
    each draw through ``predict_photometry``, mirroring
    ``fit_one.save_fit_outputs``'s fix exactly so a re-run grid cell and a
    postprocessed old one carry the identical definition of the three keys.

    Args:
        sed_model: The cell's rebuilt model (from :func:`rebuild_model`).
        npz_payload: Every array from the cell's NPZ, keyed by name.
        n_draws: Draws to subsample (``fit_one.PPD_N_DRAWS`` by default).

    Returns:
        ``{"model_photometry_median": ..., "model_photometry_p16": ...,
        "model_photometry_p84": ...}``.

    Raises:
        ValueError: the NPZ carries no draws for one of the model's free
            parameters, or the resulting draw count is zero.
    """
    free_params = list(sed_model.spec.free_params)
    missing = [name for name in free_params if name not in npz_payload]
    if missing:
        raise ValueError(f"NPZ is missing draws for free parameter(s) {missing}")
    samples_thin = {name: np.asarray(npz_payload[name], dtype=float) for name in free_params}
    fixed_values = sed_model.spec.get_fixed_values()

    ppd_draws = [
        np.asarray(sed_model.predict_photometry(params))
        for params in iter_draws(samples_thin, fixed_values, n_draws)
    ]
    if not ppd_draws:
        raise ValueError("no draws available for posterior-predictive photometry")

    ppd_stack = np.stack(ppd_draws)
    return {
        "model_photometry_median": np.median(ppd_stack, axis=0),
        "model_photometry_p16": np.percentile(ppd_stack, 16, axis=0),
        "model_photometry_p84": np.percentile(ppd_stack, 84, axis=0),
    }


def recomputed_chi2_per_dof(updated_median: np.ndarray, npz_payload: dict) -> float | None:
    """chi2/n of the recomputed median against the cell's observed photometry.

    ``fit_one.save_fit_outputs`` writes ``obs_fnu``/``obs_sigma`` into every
    grid NPZ (the flux and its 5%-floored error, per band) -- that is the only
    place this script finds them: the cell's JSON carries diagnostics, not
    per-band flux arrays. Reads from the NPZ; if a cell's NPZ predates those
    keys (nothing in the current grid does), chi2/n is not computed and the
    caller is told so explicitly rather than getting a silent ``None``.

    Returns:
        chi2/n, or ``None`` if ``obs_fnu``/``obs_sigma`` are not in the NPZ.
    """
    if "obs_fnu" not in npz_payload or "obs_sigma" not in npz_payload:
        return None
    obs_fnu = np.asarray(npz_payload["obs_fnu"], dtype=float)
    obs_sigma = np.asarray(npz_payload["obs_sigma"], dtype=float)
    chi2 = float(np.sum(((updated_median - obs_fnu) / obs_sigma) ** 2))
    return chi2 / len(obs_fnu)


def postprocess_cell(gal_id: int, config_key: str, out_dir: Path, n_draws: int) -> float | None:
    """Recompute and write one cell's posterior-predictive photometry keys, no refit.

    Loads ``{out_dir}/{gal_id}_{config_key}.npz``, computes the three keys from
    its own saved draws, and writes them back -- every other key already in the
    file is preserved verbatim. The write goes through
    ``fit_one._atomic_replace_write`` (tmp sibling + ``os.replace``), the same
    atomicity the fit path itself uses.

    Returns:
        The recomputed chi2/n, or ``None`` if the cell was skipped (no NPZ) or
        ``obs_fnu``/``obs_sigma`` were not available to compute it.
    """
    npz_path = out_dir / f"{gal_id}_{config_key}.npz"
    if not npz_path.exists():
        logger.warning(f"skipping {gal_id}_{config_key}: no NPZ at {npz_path}")
        return None

    sed_model, _filter_names, _z = rebuild_model(gal_id, config_key, out_dir)

    with np.load(npz_path, allow_pickle=False) as npz:
        npz_payload = {key: npz[key] for key in npz.files}

    quantiles = recompute_ppd_quantiles(sed_model, npz_payload, n_draws)
    updated_payload = {**npz_payload, **quantiles}

    _atomic_replace_write(
        npz_path,
        lambda tmp_path: np.savez(tmp_path, **updated_payload),
        tmp_suffix=".npz",
    )

    chi2_n = recomputed_chi2_per_dof(quantiles["model_photometry_median"], npz_payload)
    if chi2_n is None:
        logger.warning(
            f"{gal_id}_{config_key}: obs_fnu/obs_sigma not in the NPZ; chi2/n not computed"
        )
    else:
        logger.info(
            f"{gal_id}_{config_key}: recomputed chi2/n = {chi2_n:.3f} "
            f"(obs_fnu/obs_sigma read from the NPZ)"
        )
    return chi2_n


def main(argv: list[str] | None = None) -> int:
    """Parse arguments and run the no-refit posterior-predictive-photometry pass."""
    parser = argparse.ArgumentParser(
        description="Recompute posterior-predictive model photometry for existing grid NPZs"
    )
    parser.add_argument(
        "--cells",
        type=parse_cells,
        default=None,
        help="Comma-separated cells, e.g. '13097_III,16049_III' (default: every grid cell)",
    )
    parser.add_argument(
        "--n-draws",
        type=int,
        default=PPD_N_DRAWS,
        help=f"Draws pushed through predict_photometry per cell (default: {PPD_N_DRAWS})",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Directory holding the grid NPZ/JSON files (default: results/fits)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    cells = args.cells if args.cells is not None else default_cells()
    for gal_id, config_key in cells:
        postprocess_cell(gal_id, config_key, args.out_dir, args.n_draws)

    return 0


if __name__ == "__main__":
    sys.exit(main())
