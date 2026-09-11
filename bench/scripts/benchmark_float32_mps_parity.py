# SPDX-License-Identifier: BSD-3-Clause
"""Pure-float32 forward/gradient/MAP parity sweep for Apple GPU via jax-mps (#1206).

#1206's acceptance criterion asks for "a documented inference subset that runs in pure
float32 on JAX-Metal". Apple's own ``jax-metal`` last released 0.1.1 on 2024-10-08 and
pins ``jax == jaxlib >= 0.4.34``, which is not viable against tengri's JAX 0.11 (see
``docs/internal/getting_started/gpu.md``). The community ``jax-mps`` plugin
(https://github.com/tillahoffmann/jax-mps, built on MLX) is the viable path instead, and
it has **no float64 at all** -- ``notebooks/apple_mps.py`` Rule 1: a float64 array does
not downcast, it raises ``MLX does not support float64 (F64)``. Nobody has run an
accuracy check on Apple hardware yet; the notebook only records throughput.

This script is that check. It is self-contained (no test-tree imports) so it runs
outside pytest, on a Mac under the ``jax-mps`` venv exactly as on CPU or CUDA, and it
never allocates anything at module scope that would care about ``jax_enable_x64`` --
the whole point being that this file itself imports cleanly under
``JAX_ENABLE_X64=0`` with nothing baked into float64 before the check below runs.

Two modes
---------
``--write-reference PATH``
    Runs the six seams in float64 on CPU and writes a JSON reference: per seam, the
    noiseless photometry vector at a fixed truth, the chi-squared gradient vector at
    that truth, a *converged* MAP fit's final loss and optimum (L-BFGS, ``init_from``
    pinned to that same truth, run to convergence or a 200-iteration cap), and the
    mock flux/noise the truth was scored against -- stored explicitly so the float32
    arm fits the **same numbers** rather than regenerating its own mock (float32 and
    float64 RNG draws differ; see ``docs/dev/`` boundary notes on this).

Default (float32 parity sweep)
    Loads ``--reference PATH``, rebuilds each seam in float32, and reports per seam the
    max relative forward error, gradient error, MAP-loss deviation, and MAP-optimum
    parameter-vector deviation, against PASS/FAIL thresholds. Exit code 1 if any seam
    FAILs.

Seams
-----
Six seams, each one physics block richer than the last, so a regression can be pinned
to the block that introduced it: ``stellar_dust``, ``+dust IR``, ``+Cue``, ``+AGN``,
``+radio+xray``, ``panchromatic``. Photometry-only (no line-flux channel): #1859 and the
``predict_line_fluxes`` xfail in ``tests/regression/precision/test_float32_fitting_path_
seams.py`` record that the discrete line-catalog operator overflows float32 today (a
separate PR's work); a seam this script cannot build is skipped with a printed reason
rather than failing the sweep.

Usage
-----
::

    # 1. Generate the float64 reference (CPU; this machine has no Apple GPU).
    JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu taskset -c 4 \\
        PYTHONPATH=$PWD/src .venv/bin/python \\
        bench/scripts/benchmark_float32_mps_parity.py \\
        --write-reference bench/results/float32_parity_reference_<sha>.json

    # 2. Verify the float32 arm (CPU here; identical invocation on a Mac under jax-mps,
    #    with JAX_PLATFORMS=mps instead of cpu).
    JAX_ENABLE_X64=0 JAX_PLATFORMS=cpu taskset -c 4 \\
        PYTHONPATH=$PWD/src .venv/bin/python \\
        bench/scripts/benchmark_float32_mps_parity.py \\
        --reference bench/results/float32_parity_reference_<sha>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
import warnings

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

#: PASS/FAIL thresholds, in the order the table reports them.
_TOL_FORWARD = 3e-3
_TOL_GRAD = 1e-2
_TOL_LOSS = 1e-4
_TOL_PARAM = 1e-2

_BANDS = ["sdss_g", "sdss_r", "wise_w1", "herschel_250"]
_SSP = "data/fsps_prsc_miles_chabrier.h5"
_Z = 0.1
_SNR = 30.0
_TRUTH_SEED = 0
_NOISE_SEED = 1
_MAP_SEED = 2
#: L-BFGS iteration cap (``scipy.optimize.minimize`` ``maxiter``) -- not a fixed
#: few-step trajectory. See ``_map_loss``.
_N_STEPS = 200


def _refuse_wrong_precision(write_reference: bool) -> None:
    """Exit with the one-line environment fix if ``jax_enable_x64`` is set wrong.

    Reference generation needs float64; the parity sweep refuses float64 outright,
    because setting ``JAX_ENABLE_X64`` after ``import jax`` is too late -- constants
    allocated during import are already on the device (``notebooks/apple_mps.py``
    Rule 2). This check runs before any tengri import, so it is the first thing that
    can go wrong and the only chance to fail loudly instead of silently.
    """
    x64 = bool(jax.config.jax_enable_x64)
    if write_reference and not x64:
        sys.exit(
            "--write-reference needs float64. Fix: JAX_ENABLE_X64=1 before Python "
            "starts, e.g. `JAX_ENABLE_X64=1 python bench/scripts/"
            "benchmark_float32_mps_parity.py --write-reference ...`."
        )
    if not write_reference and x64:
        sys.exit(
            "The float32 parity sweep refuses to run with jax_enable_x64=True. "
            "Fix: JAX_ENABLE_X64=0 before Python starts (setting it after import is "
            "too late -- notebooks/apple_mps.py Rule 2)."
        )


def _seam_groups(DEFAULT, Fixed, Uniform):
    """Build the six progressive seam group dicts (`SEDModel.build` kwargs)."""
    dust_attenuation = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_diff": Uniform(0.0, 1.5),
        "tau_bc": 0.0,
    }
    dust_emission = {"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)}
    neb = {"type": "cue", "all_params": Fixed(DEFAULT)}
    agn = {
        "type": "composable",
        "all_params": Fixed(DEFAULT),
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "norm": "cigale_joint",
        "log_lbol": Fixed(10.5),
        "fracAGN": 0.1,
    }
    radio = {"sf": {"type": "bell2003"}, "agn": {"type": "powerlaw"}}
    xray = {"type": "simple"}
    shock = {"frac": 0.1}

    groups = {"stellar_dust": dict(dust_attenuation=dust_attenuation)}
    groups["+dust IR"] = dict(groups["stellar_dust"], dust_emission=dust_emission)
    groups["+Cue"] = dict(groups["+dust IR"], neb=neb)
    groups["+AGN"] = dict(groups["+Cue"], agn=agn)
    groups["+radio+xray"] = dict(groups["+AGN"], radio=radio, xray=xray)
    groups["panchromatic"] = dict(groups["+radio+xray"], shock=shock)
    return groups


def _base(Fixed, DEFAULT, Uniform, zspec):
    """The shared SFH + redshift block every seam builds on."""
    return dict(
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        redshift=zspec,
    )


def _rel(a, b):
    """Max relative deviation of ``a`` from reference ``b``, componentwise."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _map_loss(ForwardModel, model, obs, flux, noise, truth, free_names, dtype, n_steps):
    """A *converged* MAP fit's final loss and optimum, starting from the shared truth.

    Uses ``optimizer="lbfgs"`` -- scipy's L-BFGS-B, the same quasi-Newton path
    ``docs/dev/float32-tier-b-boundary.md`` measured MAP parity against float64 on
    (its own convergence tolerance ``gtol``, default ``tol=1e-5``, zero JAX compilation
    for the optimizer itself; only the loss+grad evaluation is JIT-compiled). ``n_steps``
    is a ``maxiter`` cap, not a fixed trajectory length: the fit runs to convergence or
    to the cap, whichever comes first, and ``diagnostics["converged"]`` says which.

    An early version of this ran a fixed few Adam steps instead, which tests an
    *unconverged intermediate* -- float32 rounding compounds along the step sequence,
    so the comparison is dominated by trajectory divergence rather than the model's own
    float32 accuracy. It also inherited a real bug: ``init_from`` still matters here
    (see its docstring note below), because without it ``Fitter``'s own random init
    draws from ``jax.random`` at a dtype that tracks ``jax_enable_x64``, and a float32
    draw is not bit-identical to a float64 one from the same key -- that alone sent an
    unconverged 5-step trajectory to a final loss differing by ~15%.

    Returns ``(final_loss, params_vector, converged)``; ``params_vector`` is
    ``posterior.params`` read off in ``free_names`` order, physical space.
    """
    from tengri import Posterior

    forward = ForwardModel.build(sed=model, observation=obs)
    # Pins both precision arms to the same physical starting point, not merely the
    # same PRNGKey (see the module note above on why that is not the same thing).
    init_from = Posterior(
        samples=None,
        params={k: jnp.asarray(v, dtype=dtype) for k, v in truth.items()},
        method="truth",
        wall_time_s=0.0,
        diagnostics={},
        loss_history=None,
        _model=model,
    )
    posterior = forward.fit(
        jnp.asarray(flux, dtype=dtype),
        jnp.asarray(noise, dtype=dtype),
        method="map",
        approx=None,
        init_from=init_from,
        key=jax.random.PRNGKey(_MAP_SEED),
        n_steps=n_steps,
        optimizer="lbfgs",
        verbose=False,
    )
    final_loss = float(np.asarray(posterior.loss_history)[-1])
    params_vector = [float(np.asarray(posterior.params[k])) for k in free_names]
    converged = bool(posterior.diagnostics.get("converged", False))
    return final_loss, params_vector, converged


def _run_seam(
    name, kwargs, ssp, obs, z, n_steps, write_reference, ref_row, stage="full", checkpoint_row=None
):
    """Build one seam, then compute (or check) its forward/gradient/MAP-loss triple.

    Returns a dict record. ``write_reference=True`` computes everything fresh in
    float64; ``write_reference=False`` rebuilds in float32 and compares against
    ``ref_row`` (which must carry ``truth``, ``mock_flux``, ``mock_noise``,
    ``photometry``, ``grad``, ``free_names``, ``map_loss``, ``map_params``,
    ``map_converged``).

    ``stage`` splits the two expensive compiles (the gradient, and the MAP fit) across
    separate invocations for a seam heavy enough that both together risk a wall-clock
    budget -- SKIRTOR's torus grid makes ``+AGN`` and everything built on top of it the
    seams that need this. ``"grad"`` computes forward+gradient only; ``"map"`` computes
    only the MAP loss, reusing gradient-stage results carried over rather than
    recomputing them: from ``ref_row`` when ``write_reference=True`` (a prior ``"grad"``
    stage's partial reference record), or from ``checkpoint_row`` when checking float32
    (a prior ``"grad"`` stage's partial comparison record). ``"full"`` (default) does
    both in one call, as every seam light enough not to need splitting does.
    """
    from tengri import DEFAULT, Fixed, ForwardModel, SEDModel, Uniform

    dtype = jnp.float64 if write_reference else jnp.float32
    do_grad = stage in ("full", "grad")
    do_map = stage in ("full", "map")

    try:
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            approx=None,
            **_base(Fixed, DEFAULT, Uniform, Fixed(z)),
            **kwargs,
        )
    except Exception as exc:  # a seam that cannot even be built is a result, not a crash
        return {"seam": name, "built": False, "skip_reason": f"{type(exc).__name__}: {exc}"}

    if write_reference and stage == "map":
        # ref_row here is the partial record a prior ``--stage grad`` run wrote, not a
        # comparison target -- write-reference mode never has a ``--reference`` file.
        free_names = ref_row["free_names"]
        truth = ref_row["truth"]
        mock_flux = np.asarray(ref_row["mock_flux"], dtype=np.float64)
        mock_noise = np.asarray(ref_row["mock_noise"], dtype=np.float64)
    elif write_reference:
        free_names = sorted(model.spec.free_params)
        sampled = model.spec.sample(jax.random.PRNGKey(_TRUTH_SEED))
        truth = {k: float(v) for k, v in sampled.items()}
        mock = model.mock(truth, snr=_SNR, key=jax.random.PRNGKey(_NOISE_SEED))
        mock_flux = np.asarray(mock.flux_obs, dtype=np.float64)
        mock_noise = np.asarray(mock.noise, dtype=np.float64)
    else:
        free_names = ref_row["free_names"]
        truth = ref_row["truth"]
        ref_photometry = np.asarray(ref_row["photometry"], dtype=np.float64)
        mock_flux = np.asarray(ref_row["mock_flux"], dtype=np.float64)
        mock_noise = np.asarray(ref_row["mock_noise"], dtype=np.float64)

    photometry_here = grad_here = map_loss_here = map_params_here = map_converged_here = None
    try:
        if do_grad:
            fixed_extra = {k: v for k, v in truth.items() if k not in free_names}
            flux_arr = jnp.asarray(mock_flux, dtype=dtype)
            noise_arr = jnp.asarray(mock_noise, dtype=dtype)
            fixed_j = {k: jnp.asarray(v, dtype=dtype) for k, v in fixed_extra.items()}

            def chi2(values):
                # ``has_aux`` returns the forward pass alongside the gradient so the
                # parity check needs only one compiled program per seam, not two.
                params = {**fixed_j, **dict(zip(free_names, values))}
                pred = model.predict_photometry(params)
                resid = (pred - flux_arr) / noise_arr
                return jnp.sum(resid**2), pred

            values = [jnp.asarray(truth[k], dtype=dtype) for k in free_names]
            grad_vals, pred_here = jax.grad(chi2, has_aux=True)(values)
            photometry_here = np.asarray(pred_here, dtype=np.float64)
            grad_here = [float(np.asarray(g)) for g in grad_vals]
        if do_map:
            map_loss_here, map_params_here, map_converged_here = _map_loss(
                ForwardModel, model, obs, mock_flux, mock_noise, truth, free_names, dtype, n_steps
            )
    except Exception as exc:
        if write_reference:
            return {"seam": name, "built": False, "skip_reason": f"{type(exc).__name__}: {exc}"}
        return {
            "seam": name,
            "built": True,
            "error": f"{type(exc).__name__}: {exc}",
        }

    if write_reference:
        rec = {
            "seam": name,
            "built": True,
            "free_names": free_names,
            "truth": truth,
            "mock_flux": mock_flux.tolist(),
            "mock_noise": mock_noise.tolist(),
            "photometry": (
                photometry_here.tolist() if photometry_here is not None else ref_row["photometry"]
            ),
            "grad": grad_here if grad_here is not None else ref_row["grad"],
        }
        if map_loss_here is not None:
            rec["map_loss"] = map_loss_here
            rec["map_params"] = map_params_here
            rec["map_converged"] = map_converged_here
        elif ref_row is not None and "map_loss" in ref_row:
            rec["map_loss"] = ref_row["map_loss"]
            rec["map_params"] = ref_row["map_params"]
            rec["map_converged"] = ref_row["map_converged"]
        else:
            rec["partial"] = True  # no map_loss yet; a later ``--stage map`` fills it in
        return rec

    if do_grad:
        rel_fwd = _rel(photometry_here, ref_photometry)
        rel_grad = _rel(grad_here, ref_row["grad"])
    else:
        rel_fwd = checkpoint_row["rel_forward"]
        rel_grad = checkpoint_row["rel_grad"]

    if not do_map:
        return {"seam": name, "built": True, "rel_forward": rel_fwd, "rel_grad": rel_grad}

    ref_loss = float(ref_row["map_loss"])
    rel_loss = abs(map_loss_here - ref_loss) / max(abs(ref_loss), 1e-300)
    rel_param = _rel(map_params_here, ref_row["map_params"])
    passed = (
        rel_fwd < _TOL_FORWARD
        and rel_grad < _TOL_GRAD
        and rel_loss < _TOL_LOSS
        and rel_param < _TOL_PARAM
    )
    return {
        "seam": name,
        "built": True,
        "rel_forward": rel_fwd,
        "rel_grad": rel_grad,
        "rel_loss": rel_loss,
        "rel_param": rel_param,
        "map_converged": map_converged_here,
        "map_converged_ref": bool(ref_row["map_converged"]),
        "passed": passed,
    }


def _make_observation(bands):
    from tengri import Observation, Photometry

    return Observation(photometry=Photometry.from_names(bands))


def _load_ssp(path):
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

    return load_ssp_data(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-reference", default=None, help="write a float64 reference JSON here")
    ap.add_argument("--reference", default=None, help="float64 reference JSON to check against")
    ap.add_argument("--ssp", default=_SSP)
    ap.add_argument("--bands", nargs="+", default=_BANDS)
    ap.add_argument("--z", type=float, default=_Z)
    ap.add_argument("--snr", type=float, default=_SNR)
    ap.add_argument("--n-steps", type=int, default=_N_STEPS)
    ap.add_argument(
        "--seams",
        nargs="+",
        default=None,
        help="restrict to these seam names (default: all six); "
        "--write-reference merges into an existing file rather than overwriting it",
    )
    ap.add_argument(
        "--stage",
        choices=["full", "grad", "map"],
        default="full",
        help="split a heavy seam's gradient compile from its MAP-fit compile across two "
        "invocations ('grad' then 'map') instead of paying both in one wall-clock "
        "budget; merges into --write-reference, or into --checkpoint for the parity "
        "sweep",
    )
    ap.add_argument(
        "--checkpoint",
        default=None,
        help="parity sweep only: persist/merge per-seam comparison rows here across "
        "--stage grad/map invocations; the final printed table is read from this file "
        "(covering every seam ever written to it, not just this invocation's)",
    )
    args = ap.parse_args()

    write_reference = args.write_reference is not None
    if not write_reference and args.reference is None:
        ap.error("pass --write-reference PATH or --reference PATH")
    if not write_reference and args.stage != "full" and args.checkpoint is None:
        ap.error("--stage grad/map for the parity sweep needs --checkpoint PATH")

    _refuse_wrong_precision(write_reference)

    warnings.filterwarnings("ignore")

    dtype_label = "float64" if write_reference else "float32"
    meta = {
        "mode": "write-reference" if write_reference else "parity-sweep",
        "dtype": dtype_label,
        "jax": jax.__version__,
        "jaxlib": jaxlib.__version__,
        "backend": jax.default_backend(),
        "devices": [str(d) for d in jax.devices()],
        "z": args.z,
        "snr": args.snr,
        "n_steps": args.n_steps,
        "bands": args.bands,
        "ssp": args.ssp,
        "tolerances": {
            "forward": _TOL_FORWARD,
            "grad": _TOL_GRAD,
            "loss": _TOL_LOSS,
            "param": _TOL_PARAM,
        },
    }
    print(json.dumps(meta, indent=2), flush=True)

    ref = None
    if not write_reference:
        with open(args.reference) as fh:
            ref = json.load(fh)

    ssp = _load_ssp(args.ssp)
    obs = _make_observation(args.bands)

    from tengri import DEFAULT, Fixed, Uniform

    groups = _seam_groups(DEFAULT, Fixed, Uniform)
    if args.seams is not None:
        unknown = set(args.seams) - set(groups)
        if unknown:
            ap.error(f"unknown seam(s) {sorted(unknown)}; choices are {list(groups)}")
        groups = {k: v for k, v in groups.items() if k in args.seams}

    def _load_seams(path):
        try:
            with open(path) as fh:
                return json.load(fh).get("seams", {})
        except (FileNotFoundError, json.JSONDecodeError):
            return {}

    prior = _load_seams(args.write_reference) if write_reference and args.stage == "map" else {}
    checkpoint = _load_seams(args.checkpoint) if args.checkpoint else {}

    rows = []
    for name, kwargs in groups.items():
        t0 = time.time()
        ref_row = None
        checkpoint_row = None
        if ref is not None:
            ref_row = ref["seams"].get(name)
            if ref_row is None or not ref_row.get("built", False):
                reason = (ref_row or {}).get("skip_reason", "no reference row")
                print(f"{name:14s} SKIP  (reference: {reason})")
                rows.append({"seam": name, "built": False, "skip_reason": reason})
                continue
            if args.stage == "map":
                checkpoint_row = checkpoint.get(name)
                if checkpoint_row is None or "rel_forward" not in checkpoint_row:
                    ap.error(f"--stage map needs a prior '--stage grad' record for {name!r}")
        if write_reference and args.stage == "map":
            ref_row = prior.get(name)
            if ref_row is None or "grad" not in ref_row:
                ap.error(f"--stage map needs a prior '--stage grad' record for {name!r}")

        rec = _run_seam(
            name,
            kwargs,
            ssp,
            obs,
            args.z,
            args.n_steps,
            write_reference,
            ref_row,
            args.stage,
            checkpoint_row,
        )
        rows.append(rec)
        jax.clear_caches()
        gc.collect()

        dt = time.time() - t0
        if not rec.get("built", False):
            print(f"{name:14s} SKIP  {rec.get('skip_reason', 'unknown')}  ({dt:.1f}s)")
        elif "error" in rec:
            print(f"{name:14s} FAIL  {rec['error']}  ({dt:.1f}s)")
        elif write_reference:
            state = "partial" if rec.get("partial") else "wrote"
            print(f"{name:14s} {state}  D={len(rec['free_names'])}  ({dt:.1f}s)")
        elif "passed" not in rec:
            print(
                f"{name:14s} partial  fwd={rec['rel_forward']:.2e}  "
                f"grad={rec['rel_grad']:.2e}  ({dt:.1f}s)"
            )
        else:
            status = "PASS" if rec["passed"] else "FAIL"
            print(
                f"{name:14s} {status}  fwd={rec['rel_forward']:.2e}  "
                f"grad={rec['rel_grad']:.2e}  loss={rec['rel_loss']:.2e}  "
                f"param={rec['rel_param']:.2e}  ({dt:.1f}s)"
            )

    if write_reference:
        # --seams lets a slow full sweep be built up one seam (or a few) per
        # invocation; merge into whatever the file already holds rather than
        # clobbering seams a previous invocation already wrote.
        existing = _load_seams(args.write_reference)
        existing.update({r["seam"]: r for r in rows})
        out = {"meta": meta, "seams": existing}
        with open(args.write_reference, "w") as fh:
            json.dump(out, fh, indent=2)
        print(f"wrote {args.write_reference} ({len(existing)} seam(s) total)")
        return 0

    if args.checkpoint:
        checkpoint.update({r["seam"]: r for r in rows})
        with open(args.checkpoint, "w") as fh:
            json.dump({"meta": meta, "seams": checkpoint}, fh, indent=2)
        report_rows = list(checkpoint.values())
    else:
        report_rows = rows

    print()
    header = f"{'seam':14s} {'fwd':>10s} {'grad':>10s} {'loss':>10s} {'param':>10s}  status"
    print(header)
    print("-" * len(header))
    any_fail = False
    for rec in report_rows:
        if not rec.get("built", False):
            print(f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s}  SKIP")
            continue
        if "error" in rec:
            print(
                f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s}  "
                f"FAIL ({rec['error']})"
            )
            any_fail = True
            continue
        if "passed" not in rec:
            print(f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>10s} {'--':>10s}  PENDING")
            continue
        status = "PASS" if rec["passed"] else "FAIL"
        any_fail = any_fail or not rec["passed"]
        both_converged = rec.get("map_converged") and rec.get("map_converged_ref")
        conv = "" if both_converged else "  (unconverged)"
        print(
            f"{rec['seam']:14s} {rec['rel_forward']:10.2e} {rec['rel_grad']:10.2e} "
            f"{rec['rel_loss']:10.2e} {rec['rel_param']:10.2e}  {status}{conv}"
        )

    print()
    print(f"device: {meta['devices']}  jax {meta['jax']}  jaxlib {meta['jaxlib']}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
