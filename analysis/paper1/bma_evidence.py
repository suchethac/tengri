#!/usr/bin/env python3
# SPDX-License-Identifier: BSD-3-Clause
"""Compute Laplace evidence for models in the BMA space, one galaxy per process.

CLI:
    python analysis/paper1/bma_evidence.py --galaxy 79 --set factorial|named_grid|named_all \
        [--models key1,key2] [--route laplace|nss] [--out DIR] [--n-restarts 8] \
        [--seed 0] [--force] [--dry-run] [--max-models N]

``--route laplace`` (default) writes to ``results/bma_evidence``; ``--route nss`` fits
the same model and data with nested sampling and writes to ``results/bma_evidence_nss``,
so neither route overwrites the other.

For each model: build the SED model, run MAP with n_restarts keeping the best restart
and recording max-min loss spread, then run Laplace to get evidence and diagnostics.
Outputs are saved atomically to JSON and NPZ files.

JSON fields: galaxy, z, model_key, model_set, weight_sets, components, route, log_evidence,
    map_loss, n_free, newton_decrement, n_clipped_eigenvalues, condition_number,
    valid, n_map_restarts, map_restart_loss_spread, wall_time_s, peak_rss_gb,
    code_revision, seed, error. NSS cells (route "nss") add log_evidence_err, n_live,
    n_live_effective, n_iterations, n_dead, information_nats, ess and null Hessian fields.

NPZ arrays (Laplace draws, n<=500): log_stellar_mass_formed, log_stellar_mass_survived,
    log_sfr_100myr, log_sfr_10myr.

All fields match the BMA evidence cell contract in the SPEC.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import importlib
import json
import logging
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ANALYSIS_DIR = Path(__file__).resolve().parent
# ``configs.py`` (reached through ``xlike_configs``) imports ``config_metadata``
# flat, so this directory must stay on ``sys.path``. Modules are still imported
# through the package: a flat import of one that uses relative imports fails.
if str(ANALYSIS_DIR) not in sys.path:
    sys.path.insert(0, str(ANALYSIS_DIR))

from paper1._atomic_io import _atomic_replace_write
from paper1._bma_keys import (
    NSS_MAX_LOG_EVIDENCE_ERR,
    enumerate_factorial,
    enumerate_named_all,
    enumerate_named_grid,
    model_key as make_model_key,
    set_membership,
    ssp_entry,
    ssp_grid_for_config,
)
from paper1._grid_completeness import load_expected_galaxy_ids
from paper1.candels_io import load_candels_z1, photometry_for_row
from paper1.config_metadata import XLIKE_CONFIGS

# jax-dependent imports (paper1.bma_space, paper1.configs, paper1.fit_one) are
# deferred into fit_one_model / main, so this module's key, result-assembly and
# writer code path imports without loading a JAX backend.

logger = logging.getLogger(__name__)

_SYSTEMATIC_FLOOR_FRAC = 0.05

#: Per-galaxy redshift span of a batch's one compiled model (``catalog_z_range``):
#: the CANDELS z~1 sample, z = 1.012-1.097, padded on both sides.
BATCH_CATALOG_Z_RANGE = (1.00, 1.11)

#: The locked 20-galaxy sample a ``--batch-model`` run fits by default.
SAMPLE_SELECTION = Path(__file__).resolve().parent / "results" / "selected_galaxies_20.json"

ROUTES = ("laplace", "nss")
DEFAULT_OUT = {
    "laplace": "analysis/paper1/results/bma_evidence",
    "nss": "analysis/paper1/results/bma_evidence_nss",
}

#: Settings validated against 11 cells (seed-to-seed scatter 0.1-0.2 nats). The
#: iteration cap and live-set span guards inside the NSS backend raise by design and
#: are not weakened here.
NSS_FIT_KWARGS = {
    "preset": "accurate",
    "n_live": 500,
    "num_delete": 50,
    "max_iterations": 10000,
}

#: NSS diagnostics copied into an NSS cell (key in the cell -> key in posterior.diagnostics).
_NSS_DIAGNOSTIC_FIELDS = {
    "n_live": "n_live",
    "n_live_effective": "n_live_effective",
    "n_iterations": "n_iterations",
    "n_dead": "n_dead",
    "information_nats": "information_nats",
    "ess": "ess",
}


def code_revision() -> str | None:
    """Git HEAD of the tengri code this process is using."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent,
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    sha = out.stdout.strip()
    return sha if out.returncode == 0 and sha else None


def peak_rss_gb() -> float | None:
    """Peak resident set size in GB, or None on systems that don't report it."""
    try:
        usage = resource.getrusage(resource.RUSAGE_SELF)
        # maxrss is in KB on Linux, bytes on macOS
        maxrss_kb = usage.ru_maxrss
        if os.uname().sysname == "Darwin":
            return maxrss_kb / (1024 * 1024)
        else:
            return maxrss_kb / 1024.0
    except Exception:
        return None


#: Draws per vmapped forward pass when computing derived quantities. Mirrors
#: ``fit_one.DERIVED_CHUNK``: the exact-wave-grid state vmapped over a chunk
#: allocates (chunk, n_age, n_wave) intermediates, so the chunk bounds the
#: peak, not the number of draws. Not imported from fit_one.py so this
#: module's own import surface stays free of it.
_DERIVED_CHUNK = 25

#: Posterior draws used for the derived quantities NPZ (mass/SFR posteriors).
#: The Laplace fit itself still draws its full sample count (2000, tengri's
#: default); this only caps how many of those draws get pushed through the
#: derived-property forward pass.
_MAX_DERIVED_DRAWS = 500


def _draw_indices(samples: dict, n_draws: int) -> np.ndarray:
    """Indices of ``n_draws`` draws strided across the whole record.

    Same striding rule as ``fit_one.draw_indices`` (linspace over the full
    record, rounded to the nearest integer index) so the two paths never
    disagree about which draws a "first `n`" selection means.
    """
    n_available = int(next(iter(samples.values())).shape[0])
    n_take = min(n_draws, n_available)
    if n_take <= 0:
        return np.zeros(0, dtype=int)
    return np.linspace(0, n_available - 1, n_take).round().astype(int)


def _chunked_vmap(fn, samples: dict, idx: np.ndarray, chunk: int = _DERIVED_CHUNK):
    """Apply a jitted ``vmap(fn)`` to the selected draws, ``chunk`` at a time.

    ``fn`` maps one dict of sampled scalars to a pytree of derived
    quantities; the result is the same pytree with a leading draw axis, as
    numpy arrays. Chunks are padded to ``chunk`` by repeating the last draw
    so every call hits one compiled program; the padding is sliced off
    before concatenating. Mirrors ``fit_one._chunked_vmap`` -- reimplemented
    here (not imported) so this module's own jax usage stays confined to
    this function and ``fit_one_model``, matching its existing lazy-import
    convention.
    """
    import jax

    sampled = {k: np.asarray(v)[idx] for k, v in samples.items()}
    batched = jax.jit(jax.vmap(fn))
    pieces = []
    for start in range(0, idx.shape[0], chunk):
        sel = {k: v[start : start + chunk] for k, v in sampled.items()}
        n_real = next(iter(sel.values())).shape[0]
        pad = chunk - n_real
        if pad:
            sel = {k: np.concatenate([v, np.repeat(v[-1:], pad)]) for k, v in sel.items()}
        out = batched(sel)
        pieces.append(jax.tree_util.tree_map(lambda a, n=n_real: np.asarray(a)[:n], out))
    return jax.tree_util.tree_map(lambda *a: np.concatenate(a), *pieces)


def _import_xlike_configs():
    """Import ``xlike_configs`` as a submodule of this module's own package.

    ``xlike_configs`` uses package-relative imports, so it only imports as
    ``<package>.xlike_configs``; a flat ``import xlike_configs`` off a patched
    ``sys.path`` has no parent package and fails.
    """
    return importlib.import_module(f"{__package__ or 'paper1'}.xlike_configs")


def _load_xlike_builders() -> dict[str, callable]:
    """Load XLIKE_BUILDERS from xlike_configs module if it exists.

    Raises:
        ImportError: If xlike_configs.py exists but fails to import (don't swallow).
    """
    xlike_module_path = ANALYSIS_DIR / "xlike_configs.py"

    if not xlike_module_path.is_file():
        return {}

    try:
        module = _import_xlike_configs()
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        # If the file exists but import fails, raise (don't silently drop X-like)
        raise ImportError(f"Failed to import xlike_configs from {xlike_module_path}: {e}") from e

    builders = {}
    if hasattr(module, "XLIKE_BUILDERS"):
        builders.update(module.XLIKE_BUILDERS)
    return builders


def _load_ssp_for_xlike(key: str):
    """Load SSP for an X-like configuration via xlike_configs module."""
    try:
        module = _import_xlike_configs()
    except (ImportError, ModuleNotFoundError, AttributeError) as e:
        raise ImportError(f"Cannot load xlike_configs for {key}: {e}") from e

    if hasattr(module, "load_ssp_for_xlike"):
        return module.load_ssp_for_xlike(key)
    raise AttributeError(f"No load_ssp_for_xlike in xlike_configs for {key}")


def get_galaxy_data(galaxy_id: int) -> tuple[float, dict, np.ndarray, np.ndarray]:
    """Load redshift, photometry, and filters for a galaxy.

    Returns:
        (z, photometry dict with fnu/fnu_err, fnu array, sigma array)
    """
    catalog = load_candels_z1()
    row_idx = np.where(catalog["id"] == galaxy_id)[0]
    if len(row_idx) == 0:
        raise ValueError(f"Galaxy {galaxy_id} not found in CANDELS catalog")

    z = float(catalog["z"][row_idx[0]])
    names, fnu, fnu_err = photometry_for_row(catalog["header"], catalog["data"][row_idx[0]])

    return z, {"names": names, "fnu": fnu, "fnu_err": fnu_err}, np.array(fnu), np.array(fnu_err)


def build_cell_result(
    *,
    galaxy_id: int,
    z: float,
    model_dict: dict[str, str],
    components: dict[str, str] | None = None,
    laplace=None,
    nss=None,
    route: str = "laplace",
    map_loss: float | None = None,
    n_free: int | None = None,
    n_map_restarts: int = 0,
    map_restart_loss_spread: float | None = None,
    started: float,
    stage_times: dict | None = None,
    seed: int,
    error: str | None = None,
) -> dict:
    """Assemble one evidence cell's JSON dict (jax-free; the runner's only assembler).

    ``laplace`` is any object with ``log_evidence`` and a ``diagnostics`` mapping
    (``newton_decrement``, ``n_clipped_eigenvalues``, ``condition_number``), or
    ``None`` for a cell that failed before Laplace. ``nss`` is the nested-sampling
    counterpart (``log_evidence`` and a ``diagnostics`` mapping with
    ``log_evidence_err`` and the NSS bookkeeping); passing it makes an
    ``route: "nss"`` cell whose Hessian fields are ``null``. The model key, ``model_set``
    and ``weight_sets`` all come from ``_bma_keys``, so the combiner reads the
    same identity the runner wrote. A missing diagnostic stays ``null`` and makes
    the cell invalid.

    Parameters
    ----------
    galaxy_id : int
        Galaxy id.
    z : float
        Redshift.
    model_dict : dict
        Component dict of the model (``_bma_keys`` format).
    components : dict, optional
        Components to record; defaults to ``model_dict``.
    laplace : object, optional
        Laplace posterior (``log_evidence``, ``diagnostics``).
    nss : object, optional
        Nested-sampling posterior (``log_evidence``, ``diagnostics``); mutually
        exclusive with ``laplace``.
    route : {"laplace", "nss"}
        Route recorded on a cell that carries neither posterior (a failed fit);
        a cell with ``nss`` is always ``"nss"``.
    map_loss : float, optional
        Best MAP loss [nats].
    n_free : int, optional
        Number of free parameters.
    n_map_restarts : int
        Restarts that produced a loss.
    map_restart_loss_spread : float, optional
        Max minus min restart loss [nats].
    started : float
        ``time.time()`` at the start of the cell.
    stage_times : dict, optional
        Per-stage wall times [s].
    seed : int
        PRNG seed.
    error : str, optional
        Failure description; a cell with an error is never valid.

    Returns
    -------
    dict
        The cell JSON dict.
    """
    if laplace is not None and nss is not None:
        raise ValueError("build_cell_result takes laplace or nss, not both")
    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}; got {route!r}")
    route = "nss" if nss is not None else route
    key = make_model_key(model_dict)
    log_evidence_out = None
    nss_fields: dict = {}
    newton_decrement = None
    n_clipped = None
    condition_number = None
    valid = False
    if laplace is not None:
        diag = getattr(laplace, "diagnostics", None) or {}
        log_evidence = float(laplace.log_evidence)
        nd = diag.get("newton_decrement")
        nc = diag.get("n_clipped_eigenvalues")
        cn = diag.get("condition_number")
        newton_decrement = float(nd) if nd is not None else None
        n_clipped = int(nc) if nc is not None else None
        condition_number = float(cn) if cn is not None else None
        finite = bool(np.isfinite(log_evidence))
        log_evidence_out = log_evidence if finite else None
        valid = bool(
            finite
            and newton_decrement is not None
            and newton_decrement <= 0.1
            and n_clipped is not None
            and n_clipped == 0
            and error is None
        )
    if nss is not None:
        log_evidence_out, valid, nss_fields = _nss_evidence_fields(nss, error)
    return {
        "galaxy": int(galaxy_id),
        "z": float(z),
        "model_key": key,
        "model_set": "named" if "config" in model_dict else "factorial",
        "weight_sets": sorted(set_membership(key)),
        "components": dict(model_dict if components is None else components),
        "route": route,
        "log_evidence": log_evidence_out,
        **nss_fields,
        "map_loss": None if map_loss is None else float(map_loss),
        "n_free": n_free,
        "newton_decrement": newton_decrement,
        "n_clipped_eigenvalues": n_clipped,
        "condition_number": condition_number,
        "valid": valid,
        "n_map_restarts": int(n_map_restarts),
        "map_restart_loss_spread": map_restart_loss_spread,
        "systematic_floor_frac": _SYSTEMATIC_FLOOR_FRAC,
        "profile_mass": False,
        "wall_time_s": time.time() - started,
        "peak_rss_gb": peak_rss_gb(),
        "code_revision": code_revision(),
        "seed": seed,
        "stage_times_s": {} if stage_times is None else stage_times,
        "error": error,
    }


_NSS_INT_FIELDS = frozenset({"n_live", "n_iterations", "n_dead"})


def _nss_evidence_fields(nss, error: str | None) -> tuple[float | None, bool, dict]:
    """Evidence, validity and NSS bookkeeping fields of an NSS cell.

    Valid means a finite ln Z with a finite ``log_evidence_err`` within
    ``NSS_MAX_LOG_EVIDENCE_ERR`` and no error. The Laplace-only diagnostics
    (Newton decrement, clipped eigenvalues, condition number) do not exist for
    nested sampling; ``build_cell_result`` leaves them ``null``, never filled in.
    """
    diag = getattr(nss, "diagnostics", None) or {}
    log_evidence = float(nss.log_evidence)
    err_raw = diag.get("log_evidence_err")
    err = float(err_raw) if err_raw is not None else None
    finite = bool(np.isfinite(log_evidence))
    err_ok = err is not None and bool(np.isfinite(err)) and err <= NSS_MAX_LOG_EVIDENCE_ERR
    fields = {
        "log_evidence_err": err if err is not None and np.isfinite(err) else None,
    }
    for cell_key, diag_key in _NSS_DIAGNOSTIC_FIELDS.items():
        value = diag.get(diag_key)
        if value is None:
            fields[cell_key] = None
        else:
            fields[cell_key] = int(value) if cell_key in _NSS_INT_FIELDS else float(value)
    return (log_evidence if finite else None), bool(finite and err_ok and error is None), fields


def write_cell(out_dir: Path, result: dict, npz_data: dict | None) -> tuple[Path, Path | None]:
    """Write a cell's NPZ (if any) then its JSON, both atomically, named by model key.

    Returns the JSON path and the NPZ path (``None`` when no draws were written).
    """
    key = result["model_key"]
    json_path = Path(out_dir) / f"{key}.json"
    npz_path = Path(out_dir) / f"{key}.npz"
    written_npz = None
    t_write = time.perf_counter()
    if npz_data is not None:
        _atomic_replace_write(
            npz_path,
            lambda tmp_path: np.savez(tmp_path, **npz_data),
            tmp_suffix=".npz",
        )
        written_npz = npz_path
    if isinstance(result.get("stage_times_s"), dict):
        result["stage_times_s"]["write_s"] = time.perf_counter() - t_write
    _atomic_replace_write(
        json_path,
        lambda tmp_path: tmp_path.write_text(json.dumps(result, indent=2)),
    )
    return json_path, written_npz


def _fit_laplace(forward, data, n_restarts, seed, cell_key, stage_times, fit_kwargs=None):
    """MAP with ``n_restarts`` then Laplace about the best restart.

    Returns ``(fields, posterior)``: ``fields`` are the ``build_cell_result`` keywords
    describing the fit (MAP loss, restart count and spread, or the error), and
    ``posterior`` is the Laplace posterior, or ``None`` when the fit failed.
    ``fit_kwargs`` (e.g. ``params``, ``presence``) is passed to every ``forward.fit``.
    """
    import jax

    extra = dict(fit_kwargs or {})

    # MAP with n_restarts: run multiple times and track losses
    # Note: profile_mass=False because evidence must integrate log_total_mass under
    # its Uniform(8, 12.5) prior. NUTS profiling is a sampler optimization, not for evidence.
    map_losses = []
    map_restart_times_s = []
    best_map_posterior = None
    best_loss = float("inf")

    # Separate forward.fit(method="map") calls, not one n_restarts=n_restarts
    # call: tengri caches the compiled loss/gradient function on the SEDModel
    # object itself (fitter.py's _get_or_build_loss_fn/_get_or_build_grad_fn,
    # keyed by _engine_cache_key(), which excludes the PRNG key), so repeated
    # calls on this same sed_model/data already share one compiled kernel --
    # confirmed against the pilot log (bma_pilot.log): its 8 restarts show a
    # uniform ~80-115ms per scipy evaluation with no "expensive first restart"
    # outlier, and their printed times already exclude compile (map_dispatch's
    # _run_map_scipy warms up grad_fn before starting its own timer). Switching
    # to one n_restarts=n call would also lose the per-restart final_loss list
    # that map_restart_loss_spread needs: the shared multistart path
    # (_run_map_multistart_scipy) only returns the winning restart's
    # diagnostics, not every restart's. map_restarts_s below times the outer
    # forward.fit() boundary (before tengri's own warmup-exclusion), so a real
    # first-compile cost on this box would still show up as an outlier here.
    for restart_idx in range(n_restarts):
        restart_seed = seed + restart_idx
        t_restart = time.perf_counter()
        try:
            key = jax.random.PRNGKey(restart_seed)
            # Use ForwardModel.fit like fit_one.py (line 1086)
            # profile_mass=False: evidence integrates all parameters under their priors
            posterior = forward.fit(data, key=key, method="map", profile_mass=False, **extra)
            loss = float(posterior.diagnostics.get("final_loss", np.inf))
            map_losses.append(loss)
            if loss < best_loss:
                best_loss = loss
                best_map_posterior = posterior
        except Exception as e:
            logger.warning(f"{cell_key} MAP restart {restart_idx}: {e}")
            continue
        finally:
            elapsed = time.perf_counter() - t_restart
            map_restart_times_s.append(elapsed)
            logger.info(
                "%s MAP restart %d/%d: %.2fs", cell_key, restart_idx + 1, n_restarts, elapsed
            )
    stage_times["map_restarts_s"] = map_restart_times_s

    if best_map_posterior is None:
        return {
            "n_map_restarts": len(map_losses),
            "error": f"MAP failed on all {n_restarts} restarts",
        }, None

    map_restart_loss_spread = None
    if len(map_losses) > 1:
        map_restart_loss_spread = float(np.max(map_losses) - np.min(map_losses))

    # Laplace evidence from best MAP. n_samples is left at tengri's default
    # (2000, backends/laplace.py's run_laplace) -- not reduced here.
    t0 = time.perf_counter()
    try:
        key = jax.random.PRNGKey(seed + n_restarts)
        posterior = forward.fit(
            data,
            key=key,
            method="laplace",
            init_from=best_map_posterior,
            profile_mass=False,
            **extra,
        )
    except Exception as e:
        stage_times["laplace_s"] = time.perf_counter() - t0
        logger.error(f"{cell_key} Laplace failed: {e}")
        return {
            "map_loss": best_loss,
            "n_map_restarts": len(map_losses),
            "map_restart_loss_spread": map_restart_loss_spread,
            "error": f"Laplace failed: {e}",
        }, None
    stage_times["laplace_s"] = time.perf_counter() - t0
    logger.info("%s Laplace: %.2fs", cell_key, stage_times["laplace_s"])
    return {
        "map_loss": best_loss,
        "n_map_restarts": len(map_losses),
        "map_restart_loss_spread": map_restart_loss_spread,
    }, posterior


def _fit_nss(forward, data, seed, cell_key, stage_times):
    """Nested-sampling evidence on the same model and data as the Laplace route.

    Returns ``(fields, posterior)`` like ``_fit_laplace``. The NSS backend's own
    guards (iteration cap, live-set span) raise by design; the raise is recorded as
    the cell's ``error`` and the cell is invalid, never retried with weaker settings.
    """
    import jax

    t0 = time.perf_counter()
    try:
        posterior = forward.fit(
            data,
            key=jax.random.PRNGKey(seed),
            method="nss",
            profile_mass=False,
            **NSS_FIT_KWARGS,
        )
    except Exception as e:
        stage_times["nss_s"] = time.perf_counter() - t0
        logger.error(f"{cell_key} NSS failed: {e}")
        return {"error": f"NSS failed: {e}"}, None
    stage_times["nss_s"] = time.perf_counter() - t0
    logger.info("%s NSS: %.2fs", cell_key, stage_times["nss_s"])
    return {}, posterior


def _make_observation(band_names):
    """The photometry-only ``Observation`` on ``band_names`` (data passed separately)."""
    from tengri import Observation, Photometry

    return Observation(photometry=Photometry.from_names(band_names))


def _props_fn(sed_model, names, redshift=None):
    """One draw -> the ``names`` properties, at ``redshift`` when given.

    With ``redshift=None`` this is ``predict_properties`` itself. A runtime
    per-galaxy redshift cannot ride ``predict_properties``: it takes no
    ``fixed_values`` and refuses a ``redshift`` key on a Fixed-redshift model
    (#2296). So the same body runs with the redshift threaded through
    ``predict_state(fixed_values=...)``, as the bug-2748 regression test does.
    """
    if redshift is None:

        def one(sample, _sed_model=sed_model):
            return _sed_model.predict_properties(sample, names=names)

        return one

    fixed = {**sed_model.spec.get_fixed_values(), "redshift": float(redshift)}

    def one_at_z(sample, _sed_model=sed_model):
        _sed_model._ensure_property_catalog()
        state = _sed_model.predict_state(sample, fixed_values=fixed)
        return {n: _sed_model._property_catalog[n].fn(state, sample) for n in names}

    return one_at_z


def _derived_npz(sed_model, posterior, cell_key, *, redshift=None):
    """The derived-quantity NPZ arrays of a posterior, or ``None`` on failure.

    Free parameters only (the NSS posterior carries Fixed ones too), at most
    ``_MAX_DERIVED_DRAWS`` draws, evaluated at ``redshift`` when given. A failure
    is logged and leaves the cell without its NPZ, as before.
    """
    if posterior.samples is None:
        return None
    try:
        free = set(sed_model.spec.free_params)
        samples = {k: v for k, v in posterior.samples.items() if k in free}
        idx = _draw_indices(samples, _MAX_DERIVED_DRAWS)
        wanted = ("stellar_mass", "stellar_mass_surviving", "sfr_100myr", "sfr_10myr")
        available = tuple(n for n in wanted if n in sed_model.available_properties)
        got = (
            _chunked_vmap(_props_fn(sed_model, available, redshift), samples, idx)
            if available
            else {}
        )
        props = {
            n: np.asarray(got[n], dtype=float) if n in got else np.full(idx.shape[0], np.nan)
            for n in wanted
        }
        return {
            "log_stellar_mass_formed": np.log10(props["stellar_mass"]),
            "log_stellar_mass_survived": np.log10(props["stellar_mass_surviving"]),
            "log_sfr_100myr": np.log10(props["sfr_100myr"]),
            "log_sfr_10myr": np.log10(props["sfr_10myr"]),
        }
    except Exception as e:
        logger.warning(f"{cell_key} Failed to compute derived quantities: {e}")
        return None


def _union_bands(band_lists) -> list[str]:
    """Every band name across ``band_lists``, in order of first appearance."""
    ordered: dict[str, None] = {}
    for names in band_lists:
        for name in names:
            ordered.setdefault(name, None)
    return list(ordered)


def _place_on_union(union, names, fnu, sigma):
    """A galaxy's fluxes and errors on the union band order, with its presence mask.

    Absent bands get flux 0, error 1 and presence 0 (neutral under the presence mask).
    """
    flux = np.zeros(len(union))
    err = np.ones(len(union))
    presence = np.zeros(len(union))
    for name, f, s in zip(names, fnu, sigma, strict=True):
        col = union.index(name)
        flux[col] = f
        err[col] = s
        presence[col] = 1.0
    return flux, err, presence


def fit_model_over_galaxies(
    model_dict: dict[str, str],
    galaxies: list[int],
    *,
    n_restarts: int,
    seed: int,
    out_dir: Path,
    force: bool = False,
) -> dict[int, Path]:
    """Laplace evidence of one factorial model over many galaxies, one compile.

    Builds and compiles the model ONCE at ``Z_REF_FACTORIAL`` with a runtime
    per-galaxy redshift (``catalog_z_range=BATCH_CATALOG_Z_RANGE``), on the union of
    every galaxy's bands in order of first appearance across ``galaxies``. Each
    galaxy is then fit with the same MAP x ``n_restarts`` + Laplace sequence as
    ``route="laplace"``, its redshift passed as ``params={"redshift": z}`` and its
    band mask as ``presence``. Cells go to ``out_dir/<gid>/<model_key>.json|.npz``,
    with each galaxy's own ``z`` and a ``batch`` block. A galaxy whose cell exists is
    skipped unless ``force``; if every cell exists nothing is built.

    Returns
    -------
    dict
        Galaxy id -> JSON path of each cell written.
    """
    import jax
    from paper1.bma_space import Z_REF_FACTORIAL, build_model
    from paper1.configs import load_ssp_for
    from paper1.fit_one import apply_systematic_error_floor

    from tengri import Data, ForwardModel

    if "config" in model_dict:
        raise ValueError("fit_model_over_galaxies takes a factorial model, not a named config")
    jax.config.update("jax_enable_x64", True)

    model_key = make_model_key(model_dict)
    out_dir = Path(out_dir)
    written: dict[int, Path] = {}
    todo = [g for g in galaxies if force or not (out_dir / str(g) / f"{model_key}.json").exists()]
    if not todo:
        logger.info("%s: every galaxy's cell exists, nothing built", model_key)
        return written

    loaded = {g: get_galaxy_data(g) for g in galaxies}
    union = _union_bands(loaded[g][1]["names"] for g in galaxies)
    obs = _make_observation(union)
    ssp_data = load_ssp_for(ssp_entry(model_dict["ssp"]).config)
    sed_model = build_model(
        model_dict,
        ssp_data,
        obs,
        z=Z_REF_FACTORIAL,
        catalog_z_range=BATCH_CATALOG_Z_RANGE,
        prior_z=Z_REF_FACTORIAL,
    )
    forward = ForwardModel.build(sed=sed_model)
    n_free = len(sed_model.spec.free_params)

    for gid in todo:
        started = time.time()
        stage_times: dict[str, object] = {}
        z, phot_dict, fnu, sigma = loaded[gid]
        cell_key = f"{gid}_{model_key}"
        sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=_SYSTEMATIC_FLOOR_FRAC)
        flux, err, presence = _place_on_union(union, phot_dict["names"], fnu, sigma_floor)
        data = Data(photometry=(flux, err))
        fit_kwargs = {"params": {"redshift": z}, "presence": presence}

        fit_fields, posterior = _fit_laplace(
            forward, data, n_restarts, seed, cell_key, stage_times, fit_kwargs=fit_kwargs
        )
        npz_data = None
        if posterior is not None:
            t0 = time.perf_counter()
            npz_data = _derived_npz(sed_model, posterior, cell_key, redshift=z)
            stage_times["derived_draws_s"] = time.perf_counter() - t0
        result = build_cell_result(
            galaxy_id=gid,
            z=z,
            model_dict=model_dict,
            components=model_dict.copy(),
            laplace=posterior,
            n_free=n_free,
            started=started,
            stage_times=stage_times,
            seed=seed,
            route="laplace",
            **fit_fields,
        )
        result["batch"] = {
            "union_bands": list(union),
            "presence": [float(p) for p in presence],
            "catalog_z_range": list(BATCH_CATALOG_Z_RANGE),
            "prior_z": Z_REF_FACTORIAL,
        }
        cell_dir = out_dir / str(gid)
        cell_dir.mkdir(parents=True, exist_ok=True)
        json_path, _ = write_cell(cell_dir, result, npz_data)
        written[gid] = json_path
        logger.info("%s: wrote %s", cell_key, json_path)
    return written


def fit_one_model(
    galaxy_id: int,
    model_dict: dict[str, str],
    z: float,
    phot_dict: dict,
    fnu: np.ndarray,
    sigma: np.ndarray,
    n_restarts: int,
    seed: int,
    route: str = "laplace",
) -> tuple[dict, dict | None]:
    """Fit the evidence (Laplace or nested sampling) for one model on one galaxy.

    ``route="laplace"`` runs an ``n_restarts`` MAP and a Laplace approximation about
    the best restart; ``route="nss"`` runs nested sampling on the identical model and
    data (``n_restarts`` unused). Everything else, including the derived-quantity
    draws, is shared.

    Uses same systematic error floor (5%) and ForwardModel path as fit_one.py.
    Returns:
        (result_dict, npz_data_dict_or_none) where result_dict has BMA evidence cell fields
        and npz_data_dict is a dict of numpy arrays or None on failure.
        On error, result_dict has error field set and valid=False.
    """
    import jax
    from paper1.bma_space import Z_REF_FACTORIAL, build_model
    from paper1.configs import CONFIGS, load_ssp_for
    from paper1.fit_one import apply_systematic_error_floor

    from tengri import Data, ForwardModel

    if route not in ROUTES:
        raise ValueError(f"route must be one of {ROUTES}; got {route!r}")
    jax.config.update("jax_enable_x64", True)

    started = time.time()
    rss_start = peak_rss_gb()
    # _bma_keys.model_key() keys named configurations by their id
    # ("config-I", "xlike-cigale_like"), never by the display strings
    # config_metadata carries for attenuation/dust_emission/nebular ("Kriek+13,
    # 2-comp", "Draine+2014"), which are not filename-safe.
    model_key = make_model_key(model_dict)
    cell_key = f"{galaxy_id}_{model_key}"
    stage_times: dict[str, object] = {}

    try:
        t0 = time.perf_counter()
        # Build model: handle named configs (I-V), X-like models, and factorial
        if "config" in model_dict:
            cfg_key = model_dict["config"]
            if cfg_key in XLIKE_CONFIGS:
                # X-like model (will raise if import fails)
                xlike_builders = _load_xlike_builders()
                if cfg_key not in xlike_builders:
                    raise ValueError(f"X-like config {cfg_key} not found in xlike_configs")
                ssp_data = _load_ssp_for_xlike(cfg_key)
                builder = xlike_builders[cfg_key]
                components_to_report = {
                    "sfh": "xlike",
                    "ssp": "xlike",
                    "attenuation": "xlike",
                    "dust_emission": "xlike",
                    "nebular": "xlike",
                    "config": cfg_key,
                }
            else:
                # Named config I-V
                if cfg_key not in CONFIGS:
                    raise ValueError(f"Configuration {cfg_key} not found")
                ssp_data = load_ssp_for(cfg_key)
                cfg = CONFIGS[cfg_key]
                builder = getattr(
                    __import__("paper1.configs", fromlist=[f"config_{cfg_key}"]),
                    f"config_{cfg_key}",
                )
                components_to_report = {
                    "sfh": cfg["sfh_type"],
                    "ssp": ssp_grid_for_config(cfg_key),
                    "attenuation": cfg.get("attenuation", "unknown"),
                    "dust_emission": cfg.get("dust_ir", "unknown"),
                    "nebular": cfg.get("nebular", "unknown"),
                    "config": cfg_key,
                }
        else:
            # Factorial model
            ssp_data = load_ssp_for(ssp_entry(model_dict["ssp"]).config)
            builder = None
            components_to_report = model_dict.copy()

        # Create observation (filter list only; data passed separately)
        obs = _make_observation(phot_dict["names"])

        # Build the model
        if builder is None:
            # Factorial priors are shared across galaxies (as in the batched build).
            sed_model = build_model(model_dict, ssp_data, obs, z, prior_z=Z_REF_FACTORIAL)
        else:
            sed_model = builder(ssp_data, obs, z)

        n_free = len(sed_model.spec.free_params)
        stage_times["build_s"] = time.perf_counter() - t0
        logger.info("%s build: %.2fs", cell_key, stage_times["build_s"])

        t0 = time.perf_counter()
        # Apply systematic error floor (5%) like fit_one.py does (lines 956-957)
        sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=_SYSTEMATIC_FLOOR_FRAC)

        # Build data and forward model like fit_one.py does (lines 987, 985)
        data = Data(photometry=(fnu, sigma_floor))
        forward = ForwardModel.build(sed=sed_model)
        stage_times["forward_build_s"] = time.perf_counter() - t0
        logger.info("%s forward build: %.2fs", cell_key, stage_times["forward_build_s"])

        common = {
            "galaxy_id": galaxy_id,
            "z": z,
            "model_dict": model_dict,
            "components": components_to_report,
            "n_free": n_free,
            "started": started,
            "stage_times": stage_times,
            "seed": seed,
            "route": route,
        }
        if route == "nss":
            fit_fields, posterior = _fit_nss(forward, data, seed, cell_key, stage_times)
        else:
            fit_fields, posterior = _fit_laplace(
                forward, data, n_restarts, seed, cell_key, stage_times
            )
        if posterior is None:
            return build_cell_result(**common, **fit_fields), None

        # Diagnostics, validity and log_evidence nulling all live in
        # build_cell_result (a missing diagnostic stays null and makes the cell invalid).

        # Derived quantities via the jit/vmap surface, chunked (mirrors
        # fit_one.derived_over_draws): a pilot cell spent the bulk of its
        # 1253.5s outside MAP+Laplace, and an eager per-draw Python loop
        # calling predict_properties 500 times separately -- never batched
        # through one compiled program -- was one of the named suspects.
        t0 = time.perf_counter()
        npz_data = _derived_npz(sed_model, posterior, cell_key)
        stage_times["derived_draws_s"] = time.perf_counter() - t0
        logger.info("%s derived draws: %.2fs", cell_key, stage_times["derived_draws_s"])

        result = build_cell_result(
            **common,
            **fit_fields,
            **{"nss" if route == "nss" else "laplace": posterior},
        )

        # Clear JAX/tengri caches after model (issue 5)
        with contextlib.suppress(Exception):
            jax.clear_caches()
        # Delete model objects to free memory
        del sed_model, forward, data, posterior

        return result, npz_data

    except Exception as e:
        logger.error(f"{cell_key} Unexpected error: {e}", exc_info=True)
        return build_cell_result(
            galaxy_id=galaxy_id,
            z=z,
            model_dict=model_dict,
            components=model_dict
            if "config" not in model_dict
            else {"config": model_dict["config"]},
            started=started,
            stage_times=stage_times,
            seed=seed,
            route=route,
            error=str(e),
        ), None


def _run_batch(parser, args) -> int:
    """``--batch-model``: the factorial model ``args.batch_model`` over the galaxies."""
    if args.route != "laplace":
        parser.error("--batch-model fits the laplace route only; drop --route nss")
    if args.set != "factorial":
        parser.error("--batch-model fits the factorial set only; drop --set")
    matches = [m for m in enumerate_factorial() if make_model_key(m) == args.batch_model]
    if not matches:
        parser.error(f"--batch-model: {args.batch_model!r} is not a factorial model key")
    if args.galaxies:
        galaxies = [int(g) for g in args.galaxies.split(",") if g.strip()]
    else:
        galaxies = load_expected_galaxy_ids(SAMPLE_SELECTION)
    written = fit_model_over_galaxies(
        matches[0],
        galaxies,
        n_restarts=args.n_restarts,
        seed=args.seed,
        out_dir=Path(args.out),
        force=args.force,
    )
    print(f"Batch {args.batch_model}: {len(written)} of {len(galaxies)} cells written", flush=True)
    return 0


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Compute evidence (Laplace or NSS) for BMA models"
    )
    parser.add_argument("--galaxy", type=int, help="Galaxy ID (required without --batch-model)")
    parser.add_argument(
        "--set",
        choices=["factorial", "named_grid", "named_all"],
        default="factorial",
        help="Model set to evaluate",
    )
    parser.add_argument("--models", help="Comma-separated model keys to limit (default: all)")
    parser.add_argument(
        "--route",
        choices=ROUTES,
        default="laplace",
        help="Evidence estimator: MAP + Laplace, or nested sampling (NSS)",
    )
    parser.add_argument(
        "--out",
        default=None,
        help="Output directory (default: results/bma_evidence, or results/bma_evidence_nss "
        "for --route nss)",
    )
    parser.add_argument("--n-restarts", type=int, default=8, help="Number of MAP restarts")
    parser.add_argument("--seed", type=int, default=0, help="PRNG seed")
    parser.add_argument("--force", action="store_true", help="Re-run even if output exists")
    parser.add_argument(
        "--limit", type=int, help="Limit number of models (alias for --max-models)"
    )
    parser.add_argument("--max-models", type=int, help="Limit number of models")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List models that would be run without running them",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Run exactly one model (the first in the set), print its stage_times_s, and stop",
    )
    parser.add_argument(
        "--batch-model",
        metavar="MODEL_KEY",
        help="Fit one factorial model over many galaxies with ONE compiled model "
        "(laplace route, factorial set only). --galaxy is not used.",
    )
    parser.add_argument(
        "--galaxies",
        help="Comma-separated galaxy ids for --batch-model (default: the locked 20-galaxy "
        "sample in results/selected_galaxies_20.json)",
    )

    args = parser.parse_args()
    if args.out is None:
        args.out = DEFAULT_OUT[args.route]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.batch_model:
        return _run_batch(parser, args)
    if args.galaxy is None:
        parser.error("--galaxy is required unless --batch-model is given")

    # Resolve max_models (prefer --max-models over --limit)
    max_models = getattr(args, "max_models", None) or getattr(args, "limit", None)

    # Get models to evaluate
    if args.set == "factorial":
        all_models = enumerate_factorial()
    elif args.set == "named_grid":
        all_models = enumerate_named_grid()
    else:  # named_all
        all_models = enumerate_named_all()

    if args.models:
        model_keys = set(args.models.split(","))
        all_models = [m for m in all_models if make_model_key(m) in model_keys]

    if max_models:
        all_models = all_models[:max_models]

    n_models = len(all_models)

    # Dry-run: just list models
    if args.dry_run:
        print(f"Galaxy {args.galaxy}: {n_models} models in set '{args.set}'", flush=True)
        out_dir = Path(args.out) / str(args.galaxy)
        for i, model_dict in enumerate(all_models, 1):
            model_key_str = make_model_key(model_dict)
            json_path = out_dir / f"{model_key_str}.json"
            status = "exists" if json_path.exists() else "new"
            print(f"  [{i:3d}/{n_models}] {model_key_str} ({status})", flush=True)
        return 0

    # Load galaxy data
    try:
        z, phot_dict, fnu, sigma = get_galaxy_data(args.galaxy)
    except Exception as e:
        logger.error(f"Failed to load galaxy {args.galaxy}: {e}")
        return 1

    # Create output directory
    out_dir = Path(args.out) / str(args.galaxy)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"Galaxy {args.galaxy}: {n_models} models in set '{args.set}'", flush=True)

    for i, model_dict in enumerate(all_models, 1):
        model_key_str = make_model_key(model_dict)
        json_path = out_dir / f"{model_key_str}.json"

        # Skip if exists and not forced
        if json_path.exists() and not args.force:
            logger.info(f"[{i}/{n_models}] {model_key_str}: exists, skipping")
            continue

        print(f"[{i}/{n_models}] {model_key_str}: running...", end=" ", flush=True)

        # Fit model
        result, npz_data = fit_one_model(
            args.galaxy,
            model_dict,
            z,
            phot_dict,
            fnu,
            sigma,
            n_restarts=args.n_restarts,
            seed=args.seed,
            route=args.route,
        )

        # NPZ first, then JSON, each atomic; write_cell records the write time
        # into the JSON's stage_times_s.
        try:
            write_cell(out_dir, result, npz_data)
        except Exception as e:
            logger.error(f"Failed to write cell {model_key_str}: {e}")
            print("ERROR writing cell", flush=True)
            continue

        print(f"done ({result['wall_time_s']:.1f}s)", flush=True)

        if args.profile:
            print(json.dumps(result.get("stage_times_s", {}), indent=2), flush=True)
            return 0

        # Clear caches after each model to avoid OOM
        with contextlib.suppress(Exception):
            gc.collect()

    logger.info(f"Completed {n_models} models for galaxy {args.galaxy}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
