"""Fit a single galaxy with a specified SED model configuration via NUTS MCMC.

CLI: python fit_one.py --galaxy ID --config {I,II,III} --method mcmc_nuts --out DIR
     [--seed N] [--n-warmup N] [--n-samples N] [--n-chains N]

``--n-warmup`` / ``--n-samples`` / ``--n-chains`` default to 150 / 300 / 4, the recipe
``mcmc_nuts_fast`` advertises;
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
import functools
import json
import logging
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path

import jax
import numpy as np

from tengri import Data, ForwardModel, Observation, Photometry

from .candels_io import load_candels_z1, photometry_for_row
from .configs import (
    CONFIGS,
    config_I,
    config_II,
    config_III,
    config_IV,
    config_V,
    config_VI,
    load_ssp_for,
)

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

#: Draws kept per parameter in the saved NPZ (4 chains x 1000 draws).
MAX_SAVED_DRAWS = 4000

#: Draws pushed through ``predict_photometry`` for the posterior-predictive
#: ``model_photometry_median``/``_p16``/``_p84`` quantiles (#2089). Peer
#: verification (2026-09-01) found the previous point -- ``predict`` at the
#: componentwise-median parameter vector -- sits off the posterior ridge for
#: skewed, correlated posteriors: Config III's continuity ratios gave a stored
#: chi2/n of 8.88 (13097/III) and 12.38 (16049/III) against a true
#: posterior-predictive chi2/n of 0.26 and 0.18 from 200 draws pushed through
#: ``predict_photometry`` -- the stored model would draw ~15% below every
#: photometric point in Figure 5. (Sanity anchor 15336/III: 1.82 stored vs 1.32
#: predictive -- that metric was already consistent, so the fix is not
#: uniform.) Also used by ``postprocess_ppd.py``'s ``--n-draws`` default.
PPD_N_DRAWS = 200

#: The paper's canonical NUTS budget (quickstart notebook). The CLI exposes all
#: three so the save path can be exercised end to end at a tiny budget without
#: editing this file; the defaults are the paper's and are what the grid runs.
#: The recipe ``mcmc_nuts_fast`` actually advertises: 4 chains x (150 warmup +
#: 300 draws). This script drove 600 + 600 until 2026-09-14 -- four times the
#: documented work -- which made the demonstration quietly disagree with the
#: registry entry the paper cites for it, and put the full grid past 130 h.
#:
#: Owner's decision 2026-09-14, taken over relaxing the adoption bar or cutting
#: the sample: demonstrate the documented settings. Min ESS falls from ~341
#: (measured on 79/II at 600 + 600) to roughly 170, which is still adequate for
#: the reported posteriors, and the adoption bar is untouched at 0 divergences
#: and max split R-hat < 1.01.
DEFAULT_N_WARMUP = 150
DEFAULT_N_SAMPLES = 300
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

#: Per-configuration override of ``DEFAULT_RETUNE_ATTEMPTS`` (ruling R60).
#: Edge-mass diagnostics on the best-so-far draws showed pile-up against
#: Config III's ``met_logzsol`` ceiling (frac_hi 0.107 on 13097/III, 0.149 on
#: 15336/III, both at max R-hat ~1.002): the divergences are well-mixed,
#: irreducible edge geometry, not a step-size problem. Raising
#: ``target_accept_rate`` to 0.99 cannot clear boundary geometry and cost
#: 13097/III its entire 21600 s cell timeout for nothing, so III's ladder caps
#: at the 0.95 rung (2 attempts) (#2089).
#:
#: Corrected 2026-09-14. This note used to say the ceiling "sits at the SSP
#: grid extent -- an immovable edge, unlike Config II's movable prior edge".
#: That is wrong, and the ceiling it quoted (+0.5) is not the one in configs.py
#: (+0.48) either. Scanning ``met_logzsol`` from -3.0 to +2.0 and watching the
#: predicted photometry respond puts the real grid edge near +0.75: the
#: response is healthy through +0.50, decays across +0.60 to +0.70, and is
#: exactly flat from +0.80 up. So III's ceiling sits about 0.27 dex INSIDE the
#: grid and is as movable as II's.
#:
#: The capped ladder is still right -- target_accept cannot fix a boundary
#: whether prior or grid imposes it -- but the pile-up is prior truncation with
#: headroom available, not an immovable grid limit. That is a modeling question
#: (a truncated metallicity posterior propagates into correlated quantities
#: such as stellar mass), not a sampler question, and it is deliberately left
#: to the owner rather than changed silently mid-grid.
#: CLEARED on 2026-09-20 with the locked six-configuration suite. The entry
#: above capped "III" because *that* Configuration III was a continuity history
#: whose metallicity posterior piled up against a hardcoded ceiling. Under the
#: locked tab:configs, III is delayed-tau on FSPS MIST/MILES with Charlot+2000
#: attenuation and THEMIS dust -- it shares no component with the model the
#: ruling examined. A cap keyed by roman numeral does not follow the physics it
#: was written about, so leaving it would throttle a model nobody has measured.
#:
#: The pathology it described is also addressed at the source: every
#: configuration now takes its metallicity prior from its own library's grid,
#: held inside the outermost node, so the truncation-with-headroom this comment
#: describes no longer exists to diagnose (configs.met_prior_for).
#:
#: Restore a cap only from a measurement on the current suite.
RETUNE_ATTEMPTS_BY_CONFIG: dict[str, int] = {}
#: Config I is deliberately NOT capped, and the reason is worth recording
#: because I capped it on 2026-09-14 and had to revert within the hour.
#:
#: Cell 79/I, unprofiled, same seed throughout:
#:
#:     attempt 1  target 0.85   38 min    2 divergences  rhat 1.0013  FAIL
#:     attempt 2  target 0.95   50 min    4 divergences  rhat 1.0006  FAIL
#:     attempt 3  target 0.99   ~120 min  0 divergences  rhat 1.0044  ADOPTED
#:
#: I capped it after watching attempts 1 and 2 fail with divergences moving the
#: WRONG way (2 then 4) while each rung cost more than the last, and wrote that
#: the third rung "has not succeeded anywhere in this grid's data". It then
#: succeeded on that very cell, at 0 divergences and min ESS 363, and is the
#: only reason 79/I is adopted at all.
#:
#: The lesson is not about this configuration. Non-monotone divergences across
#: the first two rungs say nothing about the third, and an attempt's cost while
#: it is still running is not evidence about its outcome. Do not cap a rung
#: from the shape of the rungs below it; cap it only on observed failures of
#: that rung. Config III's cap stands because attempt 3 there was observed to
#: exhaust a 21600 s cell timeout without clearing (R60/#2089).

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
    """Name of the free parameter carrying this configuration's dust optical depth.

    Read off the configuration table rather than inferred from the key. Which
    name applies is a property of the attenuation family -- ``dust_tau_v`` for a
    single screen, ``dust_tau_diff`` for the diffuse half of a two-component
    model -- and the suite mixes both, in an order that has already changed once.
    The consumer is ``params.get(name, np.nan)``, which cannot tell a wrong name
    from a genuinely absent parameter: a stale mapping writes a full column of
    NaN into the derived quantities and nothing raises.
    """
    try:
        return CONFIGS[config_key]["dust_param"]
    except KeyError as exc:  # pragma: no cover - configuration wiring error
        raise KeyError(
            f"No dust_param declared for configuration {config_key!r}. "
            f"Every row of configs.CONFIGS must name the free parameter carrying "
            f"its dust optical depth; known rows: {sorted(CONFIGS)}."
        ) from exc


def is_chain_sampler(method: str) -> bool:
    """Whether ``method`` produces chains the NUTS adoption bar can judge.

    The retune ladder, the divergence count and split R-hat all presuppose
    an MCMC backend. Nested sampling (``"nss"``) returns weighted dead points
    with an evidence and its own ESS; splitting those in half compares early
    against late likelihood levels and reports a meaningless R-hat, so a
    non-chain method gets one attempt and is adopted on completion (the
    backend raises if the evidence integral is cut off, so completion is
    the bar).
    """
    return method.startswith("mcmc")


def sampler_kwargs_for(method: str, kwargs: dict) -> dict:
    """Drop the kwargs ``method``'s runner does not declare, with a warning.

    ``base_kwargs`` is written for NUTS (``n_warmup``, ``target_accept_rate``,
    ``dense_mass_matrix``, ...). The dispatch seam refuses any name a runner
    cannot take -- deliberately, as a typo guard (#1469) -- so ``--method nss``
    would die there on ``n_warmup``. This is the one caller that knows it is
    holding NUTS settings, so it filters against the runner's signature here
    and says what it dropped; the library guard is untouched. A runner that
    takes ``**kwargs`` receives everything.
    """
    import inspect

    from tengri.inference._backend_registry import get_backend

    sig = inspect.signature(get_backend(method).runner)
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values()):
        return dict(kwargs)
    accepted = {k: v for k, v in kwargs.items() if k in sig.parameters or k == "method"}
    dropped = sorted(set(kwargs) - set(accepted))
    if dropped:
        logger.warning(
            f"method={method!r} does not take {dropped}; running without them. "
            f"Accepted: {sorted(k for k in accepted if k != 'method')}"
        )
    return accepted


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


def divergent_draw_payload(posterior) -> dict:
    """The parameter values AT the divergent transitions, unthinned.

    A divergence count cannot tell 22 divergences spread over the posterior
    from 22 in one corner of it, and those have different causes: the first is
    an integrator that is marginally too coarse everywhere, the second is a
    region of the density the sampler cannot follow. Only the second is fixed
    by changing the model. Nothing on disk could distinguish them, so the
    retune ladder was the only available response and it is the wrong one --
    on galaxy 79 configuration I it bought 3 of 14 divergences for half the
    effective sample size (ESS 273 -> 131).

    Saved UNTHINNED and separately from the thinned draws on purpose.
    Divergences are sparse -- tens out of a thousand-odd draws -- so
    ``thin_samples``' ``[::step]`` would discard most of exactly the draws
    being kept for diagnosis. The full record is a few tens of floats per
    parameter; the thinning it bypasses exists to bound a much larger array.

    Returns an empty dict when the sampler published no mask, so a backend
    that does not report one (or an older tengri) still saves.
    """
    mask = (posterior.diagnostics or {}).get("divergent_mask")
    if mask is None:
        return {}
    mask = np.asarray(mask, dtype=bool)
    n_draws = int(next(iter(posterior.samples.values())).shape[0])
    if mask.shape != (n_draws,):
        raise ValueError(
            f"divergent_mask has shape {mask.shape} against {n_draws} flattened draws. "
            "The mask must be the burn-in-sliced, chain-flattened draw axis, or every "
            "parameter value selected by it belongs to a different transition."
        )
    n_div = (posterior.diagnostics or {}).get("n_divergent")
    if n_div is not None and int(mask.sum()) != int(n_div):
        raise ValueError(
            f"divergent_mask sums to {int(mask.sum())} but n_divergent is {int(n_div)}. "
            "One of them is counting a different set of draws -- most likely the mask "
            "was published before the burn-in slice."
        )
    payload = {"divergent_mask": mask}
    for name, values in posterior.samples.items():
        payload[f"divergent_{name}"] = np.asarray(values)[mask]
    return payload


def energy_trace_payload(posterior) -> dict:
    """The per-draw Hamiltonian energy, unthinned, on the same axis as the mask.

    Published by the backend since a3ff0e362 and, until this, computed and
    discarded by fit_one -- the same gap that lost ``tree_depth_mean`` to the
    warning text. It is one float per draw, so it rides along unthinned like
    the mask and joins it and ``samples`` row-wise. E-BFMI is NOT recomputed
    from it here: that must be done per chain (the flattened axis is
    chain-major, and differencing across a chain boundary drags a healthy
    chain under the 0.3 line), and the backend already publishes
    ``ebfmi_per_chain`` -- recorded into the attempt's JSON, not here.

    Refuses a trace whose length disagrees with the draw count rather than
    saving one that cannot be joined. Empty dict when nothing was published.
    """
    energy = (posterior.diagnostics or {}).get("energy")
    if energy is None:
        return {}
    energy = np.asarray(energy, dtype=float)
    n_draws = int(next(iter(posterior.samples.values())).shape[0])
    if energy.shape != (n_draws,):
        raise ValueError(
            f"energy has shape {energy.shape} against {n_draws} flattened draws; "
            "it must be the burn-in-sliced, chain-flattened draw axis to join the mask."
        )
    return {"energy": energy}


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

    # Check the declared name against the model that was actually built. The
    # consumer below is params.get(dust_param, nan), which cannot distinguish a
    # wrong name from an absent parameter, so a mismatch is a full column of NaN
    # and nothing raises. The declaration is a property of the attenuation
    # family -- dust_tau_v for a single screen, dust_tau_diff for the diffuse
    # half of a two-component model -- but it is stored per configuration ID,
    # and an ID is a label: this suite already had one tuning rule silently
    # reattach itself to different physics when the numerals were reassigned.
    # Verify against the physics, here, on the machine running the fit.
    if dust_param not in sed_model.spec.free_params:
        raise KeyError(
            f"Configuration {config_key} declares its dust optical depth as "
            f"{dust_param!r}, but the model it builds has no such free "
            f"parameter. Free parameters are {sorted(sed_model.spec.free_params)}. "
            f"Either configs.CONFIGS[{config_key!r}]['dust_param'] is stale or "
            f"the attenuation family changed; the derived dust column would "
            f"otherwise be silently NaN for every draw of this cell."
        )
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

    # Posterior-predictive model photometry: median/p16/p84 of PPD_N_DRAWS draws
    # each pushed through ``predict_photometry``, NOT ``predict`` at the
    # componentwise-median parameter vector (see PPD_N_DRAWS docstring, #2089).
    # A posterior with no draws (``samples_thin`` empty) falls back to the old
    # single-point prediction for the median key alone, and the p16/p84 keys
    # are omitted -- there is nothing to take a quantile of.
    ppd_draws = [
        np.asarray(sed_model.predict_photometry(params))
        for params in iter_draws(samples_thin, fixed_values, PPD_N_DRAWS)
    ]
    if ppd_draws:
        ppd_stack = np.stack(ppd_draws)
        model_photometry_median = np.median(ppd_stack, axis=0)
        model_photometry_p16 = np.percentile(ppd_stack, 16, axis=0)
        model_photometry_p84 = np.percentile(ppd_stack, 84, axis=0)
    else:
        model_photometry_median = np.asarray(
            sed_model.predict_photometry(
                {
                    **fixed_values,
                    **{k: float(np.median(v)) for k, v in samples_thin.items()},
                }
            )
        )
        model_photometry_p16 = None
        model_photometry_p84 = None

    grids = {
        # SFH grid
        "sfh_lookback_time_yr": t_lbt_yr,
        "sfh_sfr_median": sfr_median,
        "sfh_sfr_p16": sfr_p16,
        "sfh_sfr_p84": sfr_p84,
        # Model photometry: posterior-predictive quantiles over PPD_N_DRAWS
        # draws (median key kept its name; its meaning is now the honest one).
        "model_photometry_median": model_photometry_median,
        # Observed photometry and errors
        "obs_fnu": np.asarray(obs_fnu),
        "obs_sigma": np.asarray(obs_sigma),
        # A str_ array, not ``dtype=object``: an object array in an NPZ can only
        # be read back with ``allow_pickle=True`` (#2089).
        "filter_names": np.asarray(filter_names, dtype=np.str_),
    }
    if model_photometry_p16 is not None:
        grids["model_photometry_p16"] = model_photometry_p16
        grids["model_photometry_p84"] = model_photometry_p84

    # Every key the NPZ carries goes through the collision guard (#2089).
    # Divergent draws ride along unthinned; see divergent_draw_payload.
    npz_payload = build_npz_payload(
        samples_thin,
        derived,
        grids,
        divergent_draw_payload(best_posterior),
        energy_trace_payload(best_posterior),
    )
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
    profile_mass: bool = False,
) -> dict:
    """Run a single fit for a galaxy and configuration.

    Args:
        gal_id: Galaxy ID (e.g. 13097)
        config_key: Configuration key (I, II, or III)
        method: Inference method (e.g. 'mcmc_nuts')
        out_dir: Output directory for results
        seed: Random seed for reproducibility
        retune_attempts: Attempts made before the best one is kept (see
            :func:`retune_settings`; default: DEFAULT_RETUNE_ATTEMPTS, unless
            ``config_key`` has an override in RETUNE_ATTEMPTS_BY_CONFIG). An
            explicitly passed value always wins over the per-config default.
        n_warmup: NUTS warmup draws per chain (default: 150, the advertised recipe)
        n_samples: NUTS kept draws per chain (default: 300, the advertised recipe)
        n_chains: NUTS chains (default: the paper's 4)

    Returns:
        Dict with fit result and diagnostics
    """
    # An explicit caller override (a value other than the module default) wins;
    # otherwise the per-config table applies (RETUNE_ATTEMPTS_BY_CONFIG) --
    # e.g. Config III caps at 2 (#2089, ruling R60).
    if retune_attempts == DEFAULT_RETUNE_ATTEMPTS:
        retune_attempts = RETUNE_ATTEMPTS_BY_CONFIG.get(config_key, DEFAULT_RETUNE_ATTEMPTS)

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
    config_builder = {
        "I": config_I,
        "II": config_II,
        "II_taucap": functools.partial(config_II, tau_cap=True),
        "III": config_III,
        "IV": config_IV,
        "V": config_V,
        "VI": config_VI,
    }[config_key]
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
    chain_sampler = is_chain_sampler(method)
    if not chain_sampler and retune_attempts != 1:
        logger.info(
            f"method={method!r} is not a chain sampler: one attempt, no retune ladder "
            f"(was {retune_attempts})"
        )
        retune_attempts = 1
    attempt = 0

    while attempt < retune_attempts:
        attempt += 1
        nuts_kwargs = retune_settings(attempt, base_kwargs)
        if chain_sampler:
            logger.info(
                f"Attempt {attempt}/{retune_attempts}: "
                f"target_accept {nuts_kwargs['target_accept_rate']}, "
                f"warmup {nuts_kwargs['n_warmup']}, "
                f"{'dense' if nuts_kwargs['dense_mass_matrix'] else 'diagonal'} mass"
            )
        else:
            logger.info(f"Attempt {attempt}/{retune_attempts}: method {method}")
        fit_kwargs = sampler_kwargs_for(method, nuts_kwargs)

        key = jax.random.PRNGKey(seed + attempt)
        t_start = time.perf_counter()

        try:
            # profile_mass=False, and this is the expensive choice, taken
            # deliberately. Profiling is 4.1x faster (cell 79/I, same seed:
            # 470 s profiled against 1941 s sampled) with far better geometry,
            # so this costs the grid roughly 24.5 h instead of 6.0 h.
            #
            # tengri#2358 (merged 77a202be) fixed the unbounded allocation of
            # #2356 -- mass_profile.py here is grafted from that commit and its
            # 21 tests pass against this pinned tree -- and the fix works: the
            # two worst cells fell from a predicted 21.3 and 32.3 GB to measured
            # peaks of 10.56 and 11.90 GB. But both were still SIGKILLed after
            # converging at 0/2400 divergences, so the acceptance test failed.
            #
            # The binding constraint moved rather than closing. Chunking bounded
            # the post-fit SPIKE to about 3.5 GB, but the profiled path's
            # steady-state footprint is ~8.5 GB against ~2.2 GB unprofiled, and
            # this box runs with ~4.7 GB free and ~12.5 GB held by the memory
            # compressor. An 8.5 GB floor does not fit, whatever the spike does.
            #
            # So the choice here is not about the fix being wrong. Unprofiled
            # cells peak near 4 GB and complete; profiled cells are faster and
            # die. Flip this back to True on a machine with real headroom, and
            # re-run the two acceptance cells (9884/IV, 9884/V) before trusting
            # a full grid to it.
            #
            # One caution when flipping it: profile_mass=True RAISES if the
            # linearity guard refuses the model, it does not fall back to
            # sampling the mass. So enabling it is not purely "the same fits,
            # faster" -- a configuration the guard rejects fails its cell
            # outright. These six should be safe, because the guard's failures
            # trace to the coarse dsps age kernel and all six take the
            # cloud-in-cell default (tengri#2368 measures cic at 1.2e-13
            # against a 1e-8 tolerance, four orders inside it), but verify
            # rather than assume if a configuration is ever added or its age
            # kernel changed.
            posterior = forward.fit(data, key=key, profile_mass=profile_mass, **fit_kwargs)
            t_elapsed = time.perf_counter() - t_start

            # Extract diagnostics
            if chain_sampler:
                rhat_dict = posterior.rhat()
                rhat_max = max(float(v) for v in rhat_dict.values())
                ess_dict = posterior.effective_sample_size()
                ess_min = min(float(v) for v in ess_dict.values()) if ess_dict else None
                n_divergent = posterior.diagnostics.get("n_divergent", 0)
                sampler_extra = {}
            else:
                # No chains to split: R-hat and per-parameter ESS are undefined.
                # NSS publishes one ESS for the weighted set, and the evidence.
                rhat_dict, rhat_max, ess_dict = {}, None, {}
                ess_min = posterior.diagnostics.get("ess")
                ess_min = float(ess_min) if ess_min is not None else None
                n_divergent = None
                sampler_extra = {
                    k: (float(v) if v is not None else None)
                    for k, v in posterior.diagnostics.items()
                    if k
                    in ("log_evidence", "log_evidence_err", "n_iterations", "n_dead", "n_live")
                }

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
                "profile_mass": profile_mass,
                "target_accept_rate": nuts_kwargs["target_accept_rate"],
                "max_tree_depth": nuts_kwargs.get("max_tree_depth"),
                "divergences": int(n_divergent) if n_divergent is not None else None,
                "ebfmi_per_chain": posterior.diagnostics.get("ebfmi_per_chain"),
                "ebfmi_min": posterior.diagnostics.get("ebfmi_min"),
                "rhat_max": float(rhat_max) if rhat_max is not None else None,
                "rhat_dict": {k: float(v) for k, v in rhat_dict.items()},
                "ess_min": float(ess_min) if ess_min is not None else None,
                "ess_dict": {k: float(v) for k, v in ess_dict.items()},
                "wall_time_s": t_elapsed,
                "systematic_floor_frac": floor_frac,
                "systematic_floor_mean_erg": float((floor_frac * fnu).mean()),
                **sampler_extra,
            }

            # Check adoption bar: 0 divergences and max R̂ < 1.01. A non-chain
            # sampler has neither; its backend raises when the run is cut off,
            # so reaching here is the bar.
            if chain_sampler:
                adoption_pass = n_divergent == 0 and rhat_max < 1.01
                bar = f"divergences={n_divergent}, rhat_max={rhat_max:.4f}"
            else:
                adoption_pass = True
                bar = ", ".join(f"{k}={v}" for k, v in sampler_extra.items()) + f", ess={ess_min}"
            diagnostics["adoption_pass"] = adoption_pass
            diagnostics["retune_attempt"] = attempt
            attempts.append(dict(diagnostics))
            posteriors.append(posterior)

            if adoption_pass:
                logger.info(f"✓ Fit passed adoption bar: {bar}")
                best_posterior = posterior
                best_diagnostics = diagnostics
                best_diagnostics["best_attempt"] = attempt
                break

            logger.warning(f"✗ Fit failed adoption bar: {bar}")
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
        choices=sorted(CONFIGS),
        help="Configuration (I, II, III, IV, V, or VI)",
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
    parser.add_argument(
        "--profile-mass",
        action="store_true",
        help="Profile log_total_mass analytically instead of sampling it (see the"
        " comment above forward.fit for why the grid default is off).",
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
            profile_mass=args.profile_mass,
        )
        logger.info(f"✓ Fit complete for galaxy {args.galaxy} config {args.config}")
        return 0
    except Exception as e:
        logger.error(f"✗ Fit failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
