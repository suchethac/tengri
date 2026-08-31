"""Run backend sweep: test all inference methods on galaxy 13097, configuration II.

Runs the paper's backend list: map, laplace, mcmc (automatic selector), mcmc_nuts,
mcmc_hmc, mcmc_raytrace. Saves results to <out-dir>/<method>.npz + .json and a
summary in <out-dir>/summary.json.

Every row's NPZ carries that method's thinned draws -- one array per parameter,
the same schema ``fit_one.save_fit_outputs`` writes -- beside the diagnostics,
so Figure 7 can overlay the backends' marginal posteriors; ``map`` (and
``laplace`` when its backend returns no draws) contributes its point estimate as
length-1 arrays. The file loads with ``allow_pickle=False``. Every row also
records ``dispatched_to``, the backend the fitter actually ran, which is the
point of the ``mcmc`` row: it is the automatic selector. The ``mcmc`` and
``mcmc_nuts`` rows run the grid cell's NUTS budget (600 warmup + 4 x 600 draws),
so they are the same run as this galaxy/configuration's grid fit (#2089).

CLI::

    python run_backend_sweep.py [--methods map,laplace] [--out-dir DIR]

``--methods`` is a comma-separated subset of ``SWEEP_METHODS``, run in the order
given; the default is all six. ``--out-dir`` defaults to
``results/backend_sweep``. Both exist so the two cheap rows can be smoke-tested
into a scratch directory without touching the paper's results (#2089).
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import jax
import numpy as np
from candels_io import load_candels_z1
from configs import config_II, load_ssp_for
from fit_one import (
    MAX_SAVED_DRAWS,
    apply_systematic_error_floor,
    build_npz_payload,
    extract_photometry,
    iter_draws,
    thin_samples,
)

from tengri import Data, ForwardModel, Observation, Photometry

jax.config.update("jax_enable_x64", True)

logger = logging.getLogger(__name__)

#: The paper's backend list (owner decision 2026-08-30, #2089): variational
#: inference and nested sampling are out of scope, and ``mcmc`` (the automatic
#: selector, NUTS at this dimensionality) is in.
SWEEP_METHODS = ("map", "laplace", "mcmc", "mcmc_nuts", "mcmc_hmc", "mcmc_raytrace")

#: Default output directory, relative to this file.
DEFAULT_OUT_DIR = Path(__file__).parent / "results" / "backend_sweep"


def sweep_npz_payload(posterior, results_dict: dict, sed_model) -> dict:
    """Build one row's NPZ payload: the posterior's draws plus the diagnostics.

    Figure 7 overlays every backend's marginal posteriors, so the NPZ has to
    carry samples; the sweep used to save ``**results_dict`` alone and no
    method's draws ever reached disk (#2089). The draws are thinned by
    :func:`fit_one.thin_samples` at the grid's ``MAX_SAVED_DRAWS`` cap and
    stored under the parameters' own names -- the schema
    :func:`fit_one.save_fit_outputs` writes, so a reader opens a sweep file and
    a grid file the same way. A point estimate (``samples is None``: ``map``,
    and ``laplace`` whenever its backend returns no draws) contributes
    ``posterior.params`` as length-1 arrays: one draw, same names.

    The diagnostics ride along as they do in the JSON, minus the two changes an
    ``allow_pickle=False`` load needs: a ``None`` value (``map`` and ``laplace``
    have no warm run) is dropped, since ``np.savez`` would store it as a pickled
    object array, and a string is stored as a ``np.str_`` array rather than
    ``dtype=object``. The JSON keeps both.

    Args:
        posterior: The fitted :class:`~tengri.inference.posterior.Posterior`.
        results_dict: The row's diagnostics, exactly as the JSON records them.
        sed_model: The fitted model, for the free-parameter names.

    Returns:
        The keyword arguments for ``np.savez``.

    Raises:
        ValueError: if the posterior carries no draws for a free parameter
            (Figure 7 would silently lose that marginal), or if a diagnostics
            key collides with a parameter name (from ``build_npz_payload``).
    """
    if posterior.samples is not None:
        draws = thin_samples(posterior.samples, MAX_SAVED_DRAWS)
    else:
        draws = {
            name: np.asarray([value], dtype=float) for name, value in posterior.params.items()
        }

    missing = [name for name in sed_model.spec.free_params if name not in draws]
    if missing:
        raise ValueError(f"the posterior carries no draws for free parameter(s) {missing}")

    diagnostics = {
        key: (np.asarray(value, dtype=np.str_) if isinstance(value, str) else value)
        for key, value in results_dict.items()
        if value is not None
    }
    return build_npz_payload(draws, diagnostics)


def run_backend_sweep(
    methods: tuple[str, ...] = SWEEP_METHODS,
    out_dir: Path | None = None,
):
    """Run the requested inference backends on galaxy 13097 with config II.

    Args:
        methods: Backend names to run, in order. Every name must be in
            ``SWEEP_METHODS``.
        out_dir: Directory for the per-method NPZ/JSON and ``summary.json``.
            Defaults to ``DEFAULT_OUT_DIR``.
    """
    gal_id = 13097
    config_key = "II"

    # Load CANDELS data
    candels = load_candels_z1()
    _, z, filter_names, fnu, sigma = extract_photometry(
        gal_id, candels, candels["z"][candels["id"] == gal_id][0]
    )

    # Apply error floor
    sigma_floor = apply_systematic_error_floor(sigma, fnu, floor_frac=0.05)
    logger.info(f"Galaxy {gal_id} (z={z:.3f}): {len(filter_names)} detected bands")

    # Build model
    obs = Observation(photometry=Photometry.from_names(filter_names))
    ssp = load_ssp_for(config_key)
    sed_model = config_II(ssp, obs, z)
    forward = ForwardModel.build(sed=sed_model)
    logger.info(f"Config {config_key}: {sed_model.spec.n_free} free parameters")

    # Prepare data
    data = Data(photometry=(fnu, sigma_floor))

    # Output directory
    out_dir = Path(DEFAULT_OUT_DIR if out_dir is None else out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []
    key = jax.random.PRNGKey(42)

    for method in methods:
        logger.info(f"\n{'=' * 60}")
        logger.info(f"Testing method: {method}")
        logger.info(f"{'=' * 60}")

        try:
            t_start = time.perf_counter()

            # Method-specific parameters
            if method == "map":
                posterior = forward.fit(data, key=key, method="map", n_steps=500, n_restarts=8)
                t_cold = time.perf_counter() - t_start
                t_warm = None

            elif method == "laplace":
                # No covariance-shape option here: ``ForwardModel.fit(approx=...)``
                # is the precompute/approximation policy, not a Laplace knob (#2089).
                posterior = forward.fit(data, key=key, method="laplace")
                t_cold = time.perf_counter() - t_start
                t_warm = None

            elif method == "mcmc":
                # The automatic selector: NUTS at this dimensionality, so the same
                # settings as the explicit NUTS row; the row measures the selector,
                # and ``dispatched_to`` records what it picked. The budget is the
                # grid cell's (600 + 4 x 600), so the row IS the paper's NUTS fit
                # for this galaxy and configuration, not a cheaper stand-in.
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=4,
                )
                t_cold = time.perf_counter() - t_start

                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=4,
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_nuts":
                # The paper's canonical NUTS budget, the grid cell's own
                # (``fit_one``'s 600 warmup + 4 x 600 draws).
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_nuts",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=4,
                )
                t_cold = time.perf_counter() - t_start

                # Warm run (second run in same process after compile)
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_nuts",
                    n_warmup=600,
                    n_samples=600,
                    n_chains=4,
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_hmc":
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_hmc",
                    n_warmup=200,
                    n_samples=300,
                    n_chains=4,
                    n_leapfrog_steps=50,
                    dense_mass_matrix=False,
                    precondition=True,  # 2026-08 benchmark: preconditioned L=50 ~10x ESS/s gain
                )
                t_cold = time.perf_counter() - t_start

                # Warm run
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_hmc",
                    n_warmup=200,
                    n_samples=300,
                    n_chains=4,
                    n_leapfrog_steps=50,
                    dense_mass_matrix=False,
                    precondition=True,  # 2026-08 benchmark: preconditioned L=50 ~10x ESS/s gain
                )
                t_warm = time.perf_counter() - t_start_warm

            elif method == "mcmc_raytrace":
                # Ray tracing: use the backend's mode-aware step size. The 0.05
                # constant came from the D~137 benchmark; the backend defaults to
                # 0.03*sqrt(D) (D<=10) times 0.3 when starting from a MAP point
                # estimate, correcting for the high-curvature mode that would
                # otherwise collapse acceptance to ~0%. Passing the constant
                # disabled this correction and caused 0.0% acceptance.
                posterior = forward.fit(
                    data,
                    key=key,
                    method="mcmc_raytrace",
                    n_burnin=400,
                    n_steps=400,
                    n_chains=2,
                )
                t_cold = time.perf_counter() - t_start

                # Warm run
                t_start_warm = time.perf_counter()
                posterior_warm = forward.fit(
                    data,
                    key=jax.random.fold_in(key, 1),
                    method="mcmc_raytrace",
                    n_burnin=400,
                    n_steps=400,
                    n_chains=2,
                )
                t_warm = time.perf_counter() - t_start_warm

            else:
                logger.error(f"Unknown method: {method}")
                continue

            # Extract diagnostics
            results_dict = {
                "method": method,
                # The backend the fitter dispatched to, in its own words (e.g.
                # "NUTS (BlackJAX)"). ``method="mcmc"`` is the automatic
                # selector, so without this its row does not say what ran; it is
                # informative for every other row too (#2089).
                "dispatched_to": posterior.method,
                "gal_id": gal_id,
                "config": config_key,
                "wall_time_cold_s": t_cold,
                "wall_time_warm_s": t_warm,
                "n_params": sed_model.spec.n_free,
            }

            # Add sample-based diagnostics. Both ``effective_sample_size()`` and
            # ``rhat()`` raise ValueError when there are no samples, so the guard
            # is on the samples, not on the method name (#2089). A ``hasattr``
            # guard told us nothing: True for a method that exists, False for a
            # misspelled one, either way independent of the fit.
            if posterior.samples is not None:
                ess_dict = posterior.effective_sample_size()
                ess_min = min(float(v) for v in ess_dict.values()) if ess_dict else None
                results_dict["ess_min"] = ess_min

                if t_cold > 0 and ess_min is not None:
                    results_dict["s_per_ess_cold"] = t_cold / ess_min
                if t_warm is not None and t_warm > 0 and ess_min is not None:
                    results_dict["s_per_ess_warm"] = t_warm / ess_min

                rhat_dict = posterior.rhat()
                rhat_max = max(float(v) for v in rhat_dict.values()) if rhat_dict else None
                results_dict["rhat_max"] = rhat_max

            # Add backend-specific diagnostics: raytrace exposes acceptance rate
            # and resolved step size (#2089).
            if method == "mcmc_raytrace" and posterior.diagnostics is not None:
                if "accept_rate" in posterior.diagnostics:
                    results_dict["acceptance"] = posterior.diagnostics["accept_rate"]
                if "step_size" in posterior.diagnostics:
                    results_dict["step_size"] = posterior.diagnostics["step_size"]

            # Extract marginal samples for log M*, log SFR100, dust optical depth
            fixed_values = sed_model.spec.get_fixed_values()

            # Point estimates (MAP, Laplace) live in ``posterior.params``; the
            # old ``covariance``/``mean`` guards were both permanently False and
            # fell through to an all-zeros parameter vector (#2089).
            if method in ("map", "laplace"):
                params = dict(posterior.params)
                params_full = {**fixed_values, **params}
                pred = sed_model.predict(params_full)
                props = pred.properties

                results_dict["log_stellar_mass"] = float(np.log10(props.get("stellar_mass", 1e10)))
                results_dict["log_sfr_100myr"] = float(np.log10(props.get("sfr_100myr", 1.0)))
                results_dict["dust_tau"] = float(params.get("dust_tau_diff", 0.0))

            elif method in ("mcmc", "mcmc_nuts", "mcmc_hmc", "mcmc_raytrace"):
                # Sample-based methods: tengri returns flattened (n_chains * n_samples,) draws.
                m_star_samples = []
                sfr_samples = []
                dust_samples = []

                for params in iter_draws(posterior.samples, fixed_values, 200):
                    pred = sed_model.predict(params)
                    props = pred.properties

                    m_star_samples.append(float(np.log10(props.get("stellar_mass", 1e10))))
                    sfr_samples.append(float(np.log10(props.get("sfr_100myr", 1.0))))
                    dust_samples.append(float(params.get("dust_tau_diff", 0.0)))

                results_dict["log_stellar_mass"] = float(np.median(m_star_samples))
                results_dict["log_sfr_100myr"] = float(np.median(sfr_samples))
                results_dict["dust_tau"] = float(np.median(dust_samples))

            # Save to NPZ: the thinned draws (or the point estimate as one draw)
            # plus the diagnostics, loadable with ``allow_pickle=False`` (#2089).
            npz_file = out_dir / f"{method}.npz"
            np.savez(npz_file, **sweep_npz_payload(posterior, results_dict, sed_model))
            logger.info(f"Saved {method} results to {npz_file}")

            # Save to JSON
            json_file = out_dir / f"{method}.json"
            with open(json_file, "w") as f:
                json.dump(results_dict, f, indent=2)

            results.append(results_dict)
            # Build the warm string outside the f-string: ``{t_warm:.2f}`` on the
            # ``None`` that map/laplace produce is a TypeError (#2089).
            warm_str = f"{t_warm:.2f}s" if t_warm is not None else "N/A"
            logger.info(f"✓ {method}: cold={t_cold:.2f}s, warm={warm_str}")

        except Exception as e:
            logger.error(f"✗ {method} failed: {e}")
            raise

    # Print summary table
    print("\n" + "=" * 120)
    print(f"BACKEND SWEEP SUMMARY — Galaxy 13097, Config II (D={sed_model.spec.n_free})")
    print("=" * 120)
    print(
        f"{'Method':<15} {'Wall(s) cold':<15} {'Wall(s) warm':<15} "
        f"{'s/ESS cold':<12} {'s/ESS warm':<12} {'log M*':<10} {'log SFR100':<12} {'dust':<8}"
    )
    print("-" * 120)

    for row in results:
        method = row["method"]
        cold = row.get("wall_time_cold_s", 0)
        warm = row.get("wall_time_warm_s")
        s_ess_cold = row.get("s_per_ess_cold")
        s_ess_warm = row.get("s_per_ess_warm")
        m_star = row.get("log_stellar_mass", 0)
        sfr = row.get("log_sfr_100myr", 0)
        dust = row.get("dust_tau", 0)

        warm_str = f"{warm:.2f}" if warm is not None else "N/A"
        s_ess_cold_str = f"{s_ess_cold:.2f}" if s_ess_cold is not None else "N/A"
        s_ess_warm_str = f"{s_ess_warm:.2f}" if s_ess_warm is not None else "N/A"

        print(
            f"{method:<15} {cold:<15.2f} {warm_str:<15} "
            f"{s_ess_cold_str:<12} {s_ess_warm_str:<12} {m_star:<10.2f} {sfr:<12.2f} {dust:<8.3f}"
        )

    print("=" * 120)

    # Save summary JSON
    summary_file = out_dir / "summary.json"
    summary_dict = {
        "gal_id": gal_id,
        "config": config_key,
        "n_params": sed_model.spec.n_free,
        "backends": results,
    }
    with open(summary_file, "w") as f:
        json.dump(summary_dict, f, indent=2)

    logger.info(f"\nSummary saved to {summary_file}")
    return 0


def parse_methods(value: str) -> tuple[str, ...]:
    """Parse a comma-separated ``--methods`` value into a validated tuple.

    Raises:
        argparse.ArgumentTypeError: if the list is empty or names a backend
            outside ``SWEEP_METHODS``.
    """
    names = tuple(name.strip() for name in value.split(",") if name.strip())
    if not names:
        raise argparse.ArgumentTypeError("--methods needs at least one backend name")
    unknown = [name for name in names if name not in SWEEP_METHODS]
    if unknown:
        raise argparse.ArgumentTypeError(
            f"unknown backend(s) {unknown}; choose from {list(SWEEP_METHODS)}"
        )
    return names


def main():
    """Parse arguments and run the backend sweep."""
    parser = argparse.ArgumentParser(description="Run the paper's inference-backend sweep")
    parser.add_argument(
        "--methods",
        type=parse_methods,
        default=SWEEP_METHODS,
        help=f"Comma-separated subset of {','.join(SWEEP_METHODS)} (default: all six)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=DEFAULT_OUT_DIR,
        help="Output directory (default: results/backend_sweep)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    try:
        return run_backend_sweep(methods=args.methods, out_dir=args.out_dir)
    except Exception as e:
        logger.error(f"Backend sweep failed: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    sys.exit(main())
