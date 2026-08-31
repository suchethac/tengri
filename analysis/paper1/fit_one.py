"""Fit a single galaxy with a specified SED model configuration via NUTS MCMC.

CLI: python fit_one.py --galaxy ID --config {I,II,III} --method mcmc_nuts --out DIR
     [--seed N] [--n-warmup N] [--n-samples N] [--n-chains N]

``--n-warmup`` / ``--n-samples`` / ``--n-chains`` default to the paper's 600 / 600 / 4;
they exist so the pipeline can be smoke-tested at a small budget. Every run writes the
NPZ and the JSON: an attempt that clears the adoption bar is adopted immediately, and
otherwise the best of DEFAULT_RETUNE_ATTEMPTS attempts (fewest divergences, then lowest
max R-hat) is saved with ``adoption_pass: false`` and the process still exits 0.

The best attempt so far is written after every attempt that misses the bar, not only
at the end, so a per-cell timeout during a retune cannot erase a completed attempt.
The retune ladder raises ``target_accept_rate`` twice before it lengthens anything:
attempt 2 at 0.95 and attempt 3 at 0.99, both on the base warmup, then attempt 4 and
each further attempt double the warmup at 0.99. A retune never switches the mass
matrix to dense.

Outputs to DIR/<ID>_<config>.npz (parameters, derived quantities, diagnostics) and
DIR/<ID>_<config>.json (diagnostics summary).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np
from candels_io import load_candels_z1, photometry_for_row
from configs import config_I, config_II, config_III, load_ssp_for

from tengri import Data, ForwardModel, Observation, Photometry

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

#: Draws kept per parameter in the saved NPZ (4 chains x 1000 draws).
MAX_SAVED_DRAWS = 4000

#: The paper's canonical NUTS budget (quickstart notebook). The CLI exposes all
#: three so the save path can be exercised end to end at a tiny budget without
#: editing this file; the defaults are the paper's and are what the grid runs.
DEFAULT_N_WARMUP = 600
DEFAULT_N_SAMPLES = 600
DEFAULT_N_CHAINS = 4

#: NUTS step-size adaptation targets. A retune raises the target rather than
#: switching to a dense mass matrix: measured on grid cell 13097/II (600 warmup
#: + 4x600 draws, D = 8), attempt 1 on a diagonal mass matrix gave 3/2400
#: divergences at max R-hat 1.0014, and the old dense-mass retune gave 79/2400
#: at 1.023 (#2089). ``DEFAULT_TARGET_ACCEPT`` is ``run_nuts``'s own default.
#: The target is raised TWICE before any warmup grows: cell 13097/III (D = 11)
#: still missed on 77/2400 divergences (max R-hat 1.012, min ESS 485) after
#: 5741 s at 0.85, and percent-level divergences are a step-size problem, so
#: 0.99 is tried at the base warmup -- one run -- before paying for two.
DEFAULT_TARGET_ACCEPT = 0.85
RETUNE_TARGET_ACCEPT_1 = 0.95
RETUNE_TARGET_ACCEPT_2 = 0.99

#: Attempts the adoption loop makes before it keeps the best one it has.
DEFAULT_RETUNE_ATTEMPTS = 3

#: Keys the NPZ carries beside the sampled parameters, one array each.
#: ``dust_tau`` is the configuration's dust optical depth whichever parameter
#: carries it, and ``dust_tau_name`` names that parameter. They are deliberately
#: NOT the parameter's own name: configuration I samples ``dust_tau_v``, so
#: writing the derived array under that name made ``np.savez`` raise
#: ``TypeError: got multiple values for keyword argument 'dust_tau_v'`` -- after
#: a 1463 s fit that had already passed the adoption bar -- while II and III
#: (which sample ``dust_tau_diff``) wrote a *different* schema silently (#2089).
DERIVED_KEYS = ("stellar_mass", "sfr_100myr", "sfr_10myr", "dust_tau")


def dust_parameter_name(config_key: str) -> str:
    """Name of the free parameter carrying this configuration's dust optical depth."""
    return "dust_tau_v" if config_key == "I" else "dust_tau_diff"


def retune_settings(attempt: int, base: dict) -> dict:
    """NUTS settings for attempt ``attempt`` (1-based) of the adoption loop.

    Attempt 1 is ``base`` (diagonal mass, target 0.85). Attempts 2 and 3 raise
    ``target_accept_rate`` -- to RETUNE_TARGET_ACCEPT_1, then to
    RETUNE_TARGET_ACCEPT_2 -- both on the SAME warmup, because divergences with
    R-hat near 1.00 are a step-size problem and a smaller step size is the
    standard remedy (Stan's ``adapt_delta``). Only from attempt 4 does the
    warmup double, and again per further attempt, since that is the expensive
    knob. ``dense_mass_matrix`` is never toggled: measured on 13097/II it turned
    3 divergences into 79.

    A new dict every call; ``base`` is never mutated.
    """
    if attempt < 1:
        raise ValueError(f"attempt is 1-based, got {attempt}")
    if attempt == 1:
        return dict(base)
    if attempt == 2:
        return {**base, "target_accept_rate": RETUNE_TARGET_ACCEPT_1}
    settings = {**base, "target_accept_rate": RETUNE_TARGET_ACCEPT_2}
    if attempt >= 4:
        settings["n_warmup"] = base["n_warmup"] * 2 ** (attempt - 3)
    return settings


#: Max R-hat below which an attempt counts as mixed for ranking purposes. An
#: attempt at or above it is unmixed and loses to every mixed one, however few
#: divergences it has: measured on cell 15336/I (D = 5), attempt 2 (target 0.95)
#: gave 0/2400 divergences at max R-hat 1.041 and min ESS 48 -- chains that never
#: explored the same region, so nothing diverged -- while attempt 1 (0.85) gave
#: 59/2400 at max R-hat 1.008 and min ESS 264 (#2089). Deliberately looser than
#: the adoption bar's own 1.01, which is unchanged: this only orders attempts
#: that have ALREADY missed that bar.
BEST_ATTEMPT_RHAT_GATE = 1.02


def select_best_attempt(attempts: list[dict]) -> int:
    """Index of the best attempt: mixed first, then fewest divergences, then lowest R-hat.

    An attempt is mixed when ``rhat_max < BEST_ATTEMPT_RHAT_GATE``; every mixed
    attempt outranks every unmixed one. Mixing gates the rest because divergence
    counts are only comparable between chains that sampled the same
    distribution: on cell 15336/I attempt 2 was divergence-free at max R-hat
    1.041 (min ESS 48) purely because its chains never reached the funnel, and
    ranking on divergences alone kept it over attempt 1's 59 divergences at
    R-hat 1.008 (min ESS 264) -- the better posterior by any standard.

    Within a class, divergences come first because they bias the posterior and
    ``rhat_max`` only breaks ties. On cell 13097/II both rules agree and either
    would pick attempt 1 (3 divergences, R-hat 1.0014) over the dense-mass
    retune (79, 1.023) -- which the old two-attempt cap discarded along with
    everything else (#2089) -- since at 1.023 that retune is unmixed as well as
    the more divergent. The divergence term is what separates two attempts that
    both mix, and ``rhat_max`` only breaks a tie on it.

    When no attempt is mixed the gate separates nothing and the rule falls back
    to fewest divergences, so a fully unmixed list still yields the least bad
    index rather than an arbitrary one. The adoption bar is unaffected.
    """
    if not attempts:
        raise ValueError("select_best_attempt needs at least one attempt, got no attempts")
    return min(
        range(len(attempts)),
        key=lambda i: (
            attempts[i]["rhat_max"] >= BEST_ATTEMPT_RHAT_GATE,
            attempts[i]["divergences"],
            attempts[i]["rhat_max"],
        ),
    )


def build_npz_payload(samples_thin: dict, *extras: dict) -> dict:
    """Merge sampled draws and every derived array into one NPZ payload.

    The single place the NPZ's keys are assembled, so a derived name can never
    silently shadow -- or, under ``np.savez(**a, **b)``, collide with -- a
    sampled parameter's name (#2089). A collision raises here, before any
    expensive save work, and names the offending key.
    """
    payload = dict(samples_thin)
    for extra in extras:
        for key, value in extra.items():
            if key in payload:
                raise ValueError(f"derived quantity {key!r} collides with a sampled parameter")
            payload[key] = value
    return payload


def thin_samples(samples: dict, max_draws: int = MAX_SAVED_DRAWS) -> dict:
    """Thin flattened ``(n_chains * n_samples,)`` draws to at most ``max_draws``.

    tengri returns every chain's kept draws concatenated into one 1-D array per
    parameter; the previous save path indexed them as ``(n_chains, n_samples)``
    and raised ``IndexError`` (#2089).
    """
    n_total = int(next(iter(samples.values())).shape[0])
    step = max(1, -(-n_total // max_draws))  # ceiling division: result <= max_draws
    return {k: np.asarray(v)[::step] for k, v in samples.items()}


def iter_draws(samples_thin: dict, fixed_values: dict, n_draws: int):
    """Yield parameter dicts (fixed values merged) for ``n_draws`` draws spanning the record.

    The draws are strided with :func:`numpy.linspace` across the whole flattened
    record rather than taken from its front. tengri concatenates the chains
    chain-major, so the first ``n_draws`` entries are chain 0's earliest draws
    alone -- every derived quantity computed from them would be a single-chain,
    early-draw estimate (#2089). ``n_draws >= n_available`` yields every draw in
    order.
    """
    n_available = int(next(iter(samples_thin.values())).shape[0])
    n_take = min(n_draws, n_available)
    if n_take <= 0:
        return
    idx = np.linspace(0, n_available - 1, n_take).round().astype(int)
    for i in idx:
        yield {**fixed_values, **{k: float(v[i]) for k, v in samples_thin.items()}}


def diagnostics_payload(
    diagnostics: dict,
    attempts: list[dict],
    retune_history: list[dict],
) -> dict:
    """Build the fit's JSON payload: one attempt's diagnostics plus the history.

    The single definition of the file's shape, so the payload
    :func:`write_diagnostics_json` writes and the dict :func:`run_fit` returns
    cannot drift apart.
    """
    return {**diagnostics, "attempts": attempts, "retune_history": retune_history}


def _atomic_replace_write(
    path: Path,
    write: Callable[[Path], object],
    *,
    tmp_suffix: str = "",
) -> Path:
    """Write through a temporary sibling and ``os.replace`` it onto ``path``.

    ``os.replace`` is atomic within one filesystem, so a reader -- or the next
    process to look, after the driver's per-cell timeout killed this one -- sees
    either the previous complete file or the new complete one, never a truncated
    one. Writing in place gave no such guarantee: the best attempt so far is now
    saved mid-run, and a timeout landing inside that write would destroy a file
    that had been complete a moment earlier, which is precisely the hours of NUTS
    the interim save exists to protect (#2089).

    The temporary file is a sibling, so the rename never crosses filesystems, and
    it is removed if ``write`` raises, leaving the directory as it was found.

    Args:
        path: Final path; only ever created by the rename.
        write: Called with the temporary path; must write the whole payload there.
        tmp_suffix: Appended to the temporary name for writers that insist on an
            extension. ``np.savez`` appends ``.npz`` to any path lacking it, so
            without ``tmp_suffix=".npz"`` the payload would land beside the name
            it was handed and the rename would find nothing to move.

    Returns:
        ``path``.
    """
    path = Path(path)
    tmp_path = path.with_name(f"{path.name}.tmp{tmp_suffix}")
    try:
        write(tmp_path)
        os.replace(tmp_path, path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise
    return path


def write_diagnostics_json(
    path: Path,
    diagnostics: dict,
    attempts: list[dict],
    retune_history: list[dict],
) -> dict:
    """Write one attempt's diagnostics to the fit's JSON and return the payload.

    Called after every attempt, so a per-cell timeout that kills the process
    during a retune still leaves attempt 1's R-hat, ESS and wall time on disk;
    until #2089 the file only appeared once an attempt passed the adoption bar,
    and a killed retune erased the evidence for why it was retuning.

    The payload keeps the final JSON's shape: every key ``run_candels_fits.py``
    reads (``gal_id``, ``config``, ``n_free``, ``divergences``, ``rhat_max``,
    ``ess_min``, ``wall_time_s``, ``adoption_pass``, ``retune_history``), plus
    ``attempts`` — one entry per attempt so far, in order.
    """
    payload = diagnostics_payload(diagnostics, attempts, retune_history)

    def write_json(tmp_path: Path) -> None:
        with open(tmp_path, "w") as f:
            json.dump(payload, f, indent=2)

    # Never opened on ``path`` itself: a kill mid-``dump`` would leave the file
    # unparseable, and this one is rewritten after every attempt (#2089).
    _atomic_replace_write(path, write_json)
    return payload


def extract_photometry(
    gal_idx: int,
    candels_data: dict,
    z: float,
    ssp_grid_name: str | None = None,
) -> tuple[int, float, list[str], np.ndarray, np.ndarray]:
    """Extract detected photometry for a galaxy from CANDELS catalog.

    Args:
        gal_idx: Galaxy ID (used to find row in catalog)
        candels_data: Dict from load_candels_z1() with 'id', 'z', 'bands', 'header'
            and 'data' (the full float matrix)
        z: Redshift (may override catalog)
        ssp_grid_name: SSP grid name for filter availability check (optional)

    Returns:
        (gal_id, z, detected_filter_names, fnu_array, sigma_array)
    """
    # Find galaxy in catalog
    id_array = candels_data["id"]
    idx = np.where(id_array == gal_idx)[0]
    if len(idx) == 0:
        raise ValueError(f"Galaxy {gal_idx} not found in CANDELS catalog")
    row = candels_data["data"][idx[0]]

    # AB zero point, column map, sentinel handling and the one-Ks rule all live
    # in candels_io (#2089): this function only selects the row.
    detected_filters, fnu, fnu_err = photometry_for_row(candels_data["header"], row)
    if len(detected_filters) == 0:
        raise ValueError(f"Galaxy {gal_idx} has no detected filters")

    return gal_idx, z, detected_filters, fnu, fnu_err


def apply_systematic_error_floor(
    sigma: np.ndarray, fnu: np.ndarray, floor_frac: float = 0.05
) -> np.ndarray:
    """Apply a fractional systematic error floor in quadrature.

    Args:
        sigma: Measurement errors [erg/s/cm2/Hz]
        fnu: Flux densities [erg/s/cm2/Hz]
        floor_frac: Fractional floor (default 0.05 = 5%)

    Returns:
        Updated error array with floor applied
    """
    sys_error = floor_frac * fnu
    return np.sqrt(sigma**2 + sys_error**2)


def save_fit_outputs(
    best_posterior,
    best_diagnostics: dict,
    attempts: list[dict],
    retune_history: list[dict],
    sed_model,
    config_key: str,
    gal_id: int,
    out_dir: Path,
    obs_fnu: np.ndarray,
    obs_sigma: np.ndarray,
    filter_names: list[str],
) -> tuple[Path, Path]:
    """Write one fit's NPZ and JSON and return their paths.

    Extracted from :func:`run_fit` so the save path can be exercised without a
    real sampler run: the first real grid cell fit correctly and then died here
    (#2089).
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    output_npz = out_dir / f"{gal_id}_{config_key}.npz"
    output_json = out_dir / f"{gal_id}_{config_key}.json"

    # Thin to at most MAX_SAVED_DRAWS flattened draws (#2089)
    samples_thin = thin_samples(best_posterior.samples)

    # Prepare derived quantities
    fixed_values = sed_model.spec.get_fixed_values()

    # Compute derived quantities (stellar mass, SFR, dust). The dust parameter's
    # name varies by configuration; the NPZ key does not (#2089).
    dust_param = dust_parameter_name(config_key)
    derived_samples = {key: [] for key in DERIVED_KEYS}

    for params in iter_draws(samples_thin, fixed_values, 500):
        pred = sed_model.predict(params)
        props = pred.properties

        derived_samples["stellar_mass"].append(float(props.get("stellar_mass", np.nan)))
        derived_samples["sfr_100myr"].append(float(props.get("sfr_100myr", np.nan)))
        derived_samples["sfr_10myr"].append(float(props.get("sfr_10myr", np.nan)))
        derived_samples["dust_tau"].append(float(params.get(dust_param, np.nan)))

    # Compute SFH posteriors on a common grid
    t_lbt_yr = np.logspace(6, 10.1, 100)  # 100 points, 1 Myr to ~13 Gyr
    sfr_posterior = []

    for params in iter_draws(samples_thin, fixed_values, 200):
        state = sed_model.predict_state(params)
        t_lbt_grid = np.asarray(state.derived["sfh_grid_lbt_yr"])
        sfr_grid = np.asarray(state.derived["sfr_history"])

        # Interpolate SFR onto common grid
        sfr_interp = np.interp(t_lbt_yr, t_lbt_grid, sfr_grid)
        sfr_posterior.append(sfr_interp)

    sfr_posterior = np.stack(sfr_posterior)
    sfr_median = np.median(sfr_posterior, axis=0)
    sfr_p16 = np.percentile(sfr_posterior, 16, axis=0)
    sfr_p84 = np.percentile(sfr_posterior, 84, axis=0)

    derived = {key: np.array(derived_samples[key]) for key in DERIVED_KEYS}
    # Which parameter ``dust_tau`` came from, so a reader never has to infer it
    # from the configuration.
    derived["dust_tau_name"] = np.array(dust_param)

    grids = {
        # SFH grid
        "sfh_lookback_time_yr": t_lbt_yr,
        "sfh_sfr_median": sfr_median,
        "sfh_sfr_p16": sfr_p16,
        "sfh_sfr_p84": sfr_p84,
        # Model photometry at posterior median
        "model_photometry_median": np.asarray(
            sed_model.predict_photometry(
                {
                    **fixed_values,
                    **{k: float(np.median(v)) for k, v in samples_thin.items()},
                }
            )
        ),
        # Observed photometry and errors
        "obs_fnu": np.asarray(obs_fnu),
        "obs_sigma": np.asarray(obs_sigma),
        # A str_ array, not ``dtype=object``: an object array in an NPZ can only
        # be read back with ``allow_pickle=True`` (#2089).
        "filter_names": np.asarray(filter_names, dtype=np.str_),
    }

    # Every key the NPZ carries goes through the collision guard (#2089).
    npz_payload = build_npz_payload(samples_thin, derived, grids)
    # ``tmp_suffix=".npz"``: ``np.savez`` appends that suffix to a path without it.
    _atomic_replace_write(
        output_npz, lambda tmp_path: np.savez(tmp_path, **npz_payload), tmp_suffix=".npz"
    )
    logger.info(f"Saved results to {output_npz}")

    # Save JSON with diagnostics (same shape as the per-attempt writes)
    write_diagnostics_json(output_json, best_diagnostics, attempts, retune_history)
    logger.info(f"Saved diagnostics to {output_json}")

    return output_npz, output_json


def save_best_so_far(
    posteriors: list,
    attempts: list[dict],
    retune_history: list[dict],
    sed_model,
    config_key: str,
    gal_id: int,
    out_dir: Path,
    obs_fnu: np.ndarray,
    obs_sigma: np.ndarray,
    filter_names: list[str],
) -> tuple[object, dict]:
    """Write the best attempt so far to the fit's NPZ and JSON; return it and its diagnostics.

    The single definition of the "no attempt passed, keep the best one" write, used
    both after a missed attempt inside the retune loop and once more when the loop
    ends without a pass. Every call overwrites the same two paths, and an adoption's
    own ``save_fit_outputs`` overwrites them a final time.

    Writing after every miss is what makes a killed cell survivable: cell 13097/III
    spent 5741 s on attempt 1 and missed the bar on 77/2400 divergences, and until
    #2089 the per-cell timeout could kill the process during a retune with nothing
    on disk but the per-attempt JSON -- hours of NUTS and no draws.

    ``posteriors`` and ``attempts`` run parallel (index i of one is index i of the
    other). ``best_attempt`` is the 1-based attempt NUMBER, not the list index.
    """
    best_index = select_best_attempt(attempts)
    best_posterior = posteriors[best_index]
    best_diagnostics = dict(attempts[best_index])
    best_diagnostics["adoption_pass"] = False
    best_diagnostics["best_attempt"] = best_diagnostics["retune_attempt"]
    save_fit_outputs(
        best_posterior,
        best_diagnostics,
        attempts,
        retune_history,
        sed_model,
        config_key,
        gal_id,
        out_dir,
        obs_fnu=obs_fnu,
        obs_sigma=obs_sigma,
        filter_names=filter_names,
    )
    return best_posterior, best_diagnostics


def run_fit(
    gal_id: int,
    config_key: str,
    method: str,
    out_dir: Path,
    seed: int = 42,
    retune_attempts: int = DEFAULT_RETUNE_ATTEMPTS,
    n_warmup: int = DEFAULT_N_WARMUP,
    n_samples: int = DEFAULT_N_SAMPLES,
    n_chains: int = DEFAULT_N_CHAINS,
) -> dict:
    """Run a single fit for a galaxy and configuration.

    Args:
        gal_id: Galaxy ID (e.g. 13097)
        config_key: Configuration key (I, II, or III)
        method: Inference method (e.g. 'mcmc_nuts')
        out_dir: Output directory for results
        seed: Random seed for reproducibility
        retune_attempts: Attempts made before the best one is kept (see
            :func:`retune_settings`; default: DEFAULT_RETUNE_ATTEMPTS)
        n_warmup: NUTS warmup draws per chain (default: the paper's 600)
        n_samples: NUTS kept draws per chain (default: the paper's 600)
        n_chains: NUTS chains (default: the paper's 4)

    Returns:
        Dict with fit result and diagnostics
    """
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    # The JSON path is needed before the loop: a failed attempt is persisted
    # before the retune starts (#2089). ``save_fit_outputs`` derives the same
    # two paths from ``out_dir`` for the final write.
    output_json = out_dir / f"{gal_id}_{config_key}.json"

    # Load CANDELS catalog
    candels = load_candels_z1()

    # Extract photometry for this galaxy
    _, z, filter_names, fnu, sigma = extract_photometry(
        gal_id, candels, candels["z"][candels["id"] == gal_id][0]
    )

    # Apply 5% systematic error floor
    floor_frac = 0.05
    sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=floor_frac)
    logger.info(
        f"Applied {floor_frac * 100:.1f}% systematic error floor in quadrature. "
        f"Mean floor contribution: {(floor_frac * fnu).mean():.3e} erg/s/cm2/Hz"
    )

    # Build observation
    obs = Observation(photometry=Photometry.from_names(filter_names))

    # Load SSP and build model
    ssp = load_ssp_for(config_key)
    config_builder = {"I": config_I, "II": config_II, "III": config_III}[config_key]
    sed_model = config_builder(ssp, obs, z)
    forward = ForwardModel.build(sed=sed_model)

    logger.info(f"Galaxy {gal_id} (z={z:.3f}): {len(filter_names)} detected bands")
    logger.info(f"Config {config_key}: {sed_model.spec.n_free} free parameters")

    # Prepare data
    data = Data(photometry=(fnu, sigma_floor))

    # Attempt 1's NUTS settings (from quickstart notebook); ``retune_settings``
    # derives every later attempt's settings from these (#2089).
    base_kwargs = dict(
        method=method,
        n_warmup=n_warmup,
        n_samples=n_samples,
        n_chains=n_chains,
        n_burnin=0,
        dense_mass_matrix=False,
        target_accept_rate=DEFAULT_TARGET_ACCEPT,
    )

    # Run fit with retune logic
    best_posterior = None
    best_diagnostics = None
    retune_history = []
    attempts: list[dict] = []
    # Posteriors parallel to ``attempts`` -- index i of one is index i of the
    # other -- so the best attempt's posterior can be saved when none passes.
    posteriors: list = []
    attempt = 0

    while attempt < retune_attempts:
        attempt += 1
        nuts_kwargs = retune_settings(attempt, base_kwargs)
        logger.info(
            f"Attempt {attempt}/{retune_attempts}: "
            f"target_accept {nuts_kwargs['target_accept_rate']}, "
            f"warmup {nuts_kwargs['n_warmup']}, "
            f"{'dense' if nuts_kwargs['dense_mass_matrix'] else 'diagonal'} mass"
        )

        key = jax.random.PRNGKey(seed + attempt)
        t_start = time.perf_counter()

        try:
            posterior = forward.fit(data, key=key, **nuts_kwargs)
            t_elapsed = time.perf_counter() - t_start

            # Extract diagnostics
            rhat_dict = posterior.rhat()
            rhat_max = max(float(v) for v in rhat_dict.values())
            ess_dict = posterior.effective_sample_size()
            ess_min = min(float(v) for v in ess_dict.values()) if ess_dict else None
            n_divergent = posterior.diagnostics.get("n_divergent", 0)

            diagnostics = {
                "gal_id": gal_id,
                "config": config_key,
                "z": float(z),
                "n_free": sed_model.spec.n_free,
                "n_bands": len(filter_names),
                "filter_names": filter_names,
                "n_warmup": nuts_kwargs["n_warmup"],
                "n_samples": nuts_kwargs["n_samples"],
                "n_chains": nuts_kwargs["n_chains"],
                "dense_mass_matrix": nuts_kwargs["dense_mass_matrix"],
                "target_accept_rate": nuts_kwargs["target_accept_rate"],
                "max_tree_depth": nuts_kwargs.get("max_tree_depth"),
                "divergences": int(n_divergent),
                "rhat_max": float(rhat_max),
                "rhat_dict": {k: float(v) for k, v in rhat_dict.items()},
                "ess_min": float(ess_min) if ess_min is not None else None,
                "ess_dict": {k: float(v) for k, v in ess_dict.items()},
                "wall_time_s": t_elapsed,
                "systematic_floor_frac": floor_frac,
                "systematic_floor_mean_erg": float((floor_frac * fnu).mean()),
            }

            # Check adoption bar: 0 divergences and max R̂ < 1.01
            adoption_pass = n_divergent == 0 and rhat_max < 1.01
            diagnostics["adoption_pass"] = adoption_pass
            diagnostics["retune_attempt"] = attempt
            attempts.append(dict(diagnostics))
            posteriors.append(posterior)

            if adoption_pass:
                logger.info(
                    f"✓ Fit passed adoption bar: "
                    f"divergences={n_divergent}, rhat_max={rhat_max:.4f}"
                )
                best_posterior = posterior
                best_diagnostics = diagnostics
                best_diagnostics["best_attempt"] = attempt
                break

            logger.warning(
                f"✗ Fit failed adoption bar: divergences={n_divergent}, rhat_max={rhat_max:.4f}"
            )
            retune_history.append(dict(diagnostics))

            # Persist this attempt before the retune starts: the driver's
            # per-cell timeout kills the process mid-retune, and attempt 1's
            # evidence went with it (#2089).
            write_diagnostics_json(output_json, diagnostics, attempts, retune_history)

            # And persist the DRAWS of the best attempt so far, after that JSON
            # write so the JSON on disk carries the best attempt's diagnostics
            # plus every attempt so far. The last attempt needs no interim write:
            # the post-loop save follows it immediately (#2089).
            if attempt < retune_attempts:
                # Its own handler: this call sits inside the attempt's ``try``, so
                # a raise here was logged "Fit attempt N failed" -- the wrong
                # subject entirely, the fit had just succeeded and the save had
                # not. A missed interim write costs nothing the next attempt does
                # not redo, so the loop continues; the post-loop save is outside
                # this ``try`` and still propagates if the problem persists.
                try:
                    _, interim_diagnostics = save_best_so_far(
                        posteriors,
                        attempts,
                        retune_history,
                        sed_model,
                        config_key,
                        gal_id,
                        out_dir,
                        obs_fnu=fnu,
                        obs_sigma=sigma_floor,
                        filter_names=filter_names,
                    )
                except Exception as exc:
                    logger.warning(
                        "interim save after attempt %d failed: %s", attempt, exc, exc_info=True
                    )
                else:
                    logger.info(
                        f"Saved the best attempt so far (attempt "
                        f"{interim_diagnostics['best_attempt']} of {len(attempts)}, "
                        f"divergences={interim_diagnostics['divergences']}) before retuning"
                    )

        except Exception as e:
            logger.error(f"Fit attempt {attempt} failed: {e}", exc_info=True)
            # Only a run in which EVERY attempt raised has no posterior to keep.
            if attempt >= retune_attempts and not posteriors:
                raise

    if best_posterior is None:
        if not posteriors:
            raise RuntimeError(
                f"Fit failed all {retune_attempts} attempts for galaxy {gal_id} "
                f"config {config_key}"
            )

        # No attempt cleared the adoption bar, so the best one is saved anyway
        # with ``adoption_pass: false``. Discarding a near-passing posterior --
        # 13097/II's attempt 1 was 3/2400 divergent at R-hat 1.0014 -- threw away
        # hours of NUTS and left the cell with nothing but diagnostics (#2089).
        # The same helper the loop's interim writes use, so the miss path has one
        # definition; this call overwrites whatever the last interim write left.
        best_posterior, best_diagnostics = save_best_so_far(
            posteriors,
            attempts,
            retune_history,
            sed_model,
            config_key,
            gal_id,
            out_dir,
            obs_fnu=fnu,
            obs_sigma=sigma_floor,
            filter_names=filter_names,
        )
        logger.warning(
            f"No attempt cleared the adoption bar for galaxy {gal_id} config {config_key}; "
            f"keeping attempt {best_diagnostics['best_attempt']} of {len(attempts)} "
            f"(divergences={best_diagnostics['divergences']}, "
            f"rhat_max={best_diagnostics['rhat_max']:.4f}) with adoption_pass=False"
        )
    else:
        # An adopted attempt overwrites every interim write with its own draws.
        save_fit_outputs(
            best_posterior,
            best_diagnostics,
            attempts,
            retune_history,
            sed_model,
            config_key,
            gal_id,
            out_dir,
            obs_fnu=fnu,
            obs_sigma=sigma_floor,
            filter_names=filter_names,
        )

    return diagnostics_payload(best_diagnostics, attempts, retune_history)


def main():
    """Parse arguments and run fit."""
    parser = argparse.ArgumentParser(description="Fit a single galaxy with tengri SED model")
    parser.add_argument("--galaxy", type=int, required=True, help="Galaxy ID (e.g. 13097)")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        choices=["I", "II", "III"],
        help="Configuration (I, II, or III)",
    )
    parser.add_argument(
        "--method",
        type=str,
        default="mcmc_nuts",
        help="Inference method (default: mcmc_nuts)",
    )
    parser.add_argument("--out", type=Path, required=True, help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--n-warmup",
        type=int,
        default=DEFAULT_N_WARMUP,
        help=f"NUTS warmup draws per chain (default: {DEFAULT_N_WARMUP})",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help=f"NUTS kept draws per chain (default: {DEFAULT_N_SAMPLES})",
    )
    parser.add_argument(
        "--n-chains",
        type=int,
        default=DEFAULT_N_CHAINS,
        help=f"NUTS chains (default: {DEFAULT_N_CHAINS})",
    )

    args = parser.parse_args()

    # Set up logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        result = run_fit(
            gal_id=args.galaxy,
            config_key=args.config,
            method=args.method,
            out_dir=args.out,
            seed=args.seed,
            n_warmup=args.n_warmup,
            n_samples=args.n_samples,
            n_chains=args.n_chains,
        )
        logger.info(f"✓ Fit complete for galaxy {args.galaxy} config {args.config}")
        return 0
    except Exception as e:
        logger.error(f"✗ Fit failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
