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

Three modes
-----------
``--write-reference PATH``
    Runs the six seams in float64 on CPU and writes a JSON reference: per seam, the
    noiseless photometry vector at a fixed truth, the chi-squared gradient vector at
    that truth, a *converged* MAP fit's final loss and optimum (L-BFGS, ``init_from``
    pinned to that same truth, run to convergence -- see ``_map_loss`` on why that can
    take more than one L-BFGS call), and the mock flux/noise the truth was scored
    against -- stored explicitly so the float32 arm fits the **same numbers** rather
    than regenerating its own mock (float32 and float64 RNG draws differ; see
    ``docs/dev/`` boundary notes on this). **Exits non-zero if any seam's MAP fit does
    not converge** -- an unconverged reference is refused, never silently committed.

Default (float32 parity sweep)
    Loads ``--reference PATH``, rebuilds each seam in float32, and reports per seam the
    max relative forward error, gradient error, and MAP-optimum parameter-vector
    deviation, against PASS/FAIL thresholds -- the MAP-loss deviation is printed as an
    **informational** column only (see the note below on why it is not gated). Exit
    code 1 if any seam FAILs.

    **Why the loss column has no bar.** At a converged optimum the gradient is ~0, so
    a parameter perturbation of size epsilon moves the loss by order epsilon-squared
    (Taylor expansion around a stationary point) -- a parameter agreement of 1e-6
    implies a loss agreement of order 1e-12 *from that channel alone*. The loss
    deviation actually observed (~1e-4) is instead float32's
    own evaluation of chi-squared -- cancellation forming ``(data - model)`` at
    SNR 30, then squaring and summing -- which is exactly what the forward and
    gradient columns already bound. Gating PASS/FAIL on it a second time double-counts
    the same error source under a name that reads as "the fit disagrees," when the fit
    (the parameter vector) does not. The parameter vector is the scientific quantity a
    fit is for; it is what is gated.

``--self-check PATH``
    The reference is correct for the tree it was written on and nothing else. Every
    physics merge after it moves some seam, and the float32 sweep then reports a FAIL
    that has nothing to do with float32 or the device (#2300: #2260 moved the
    ``panchromatic`` seam's float64 photometry by 3.3e-3 in Herschel-250 and its
    ``tau_diff`` gradient by 67%; the FAIL was identical on CPU-float64, CPU-float32
    and MPS). This mode rebuilds every seam in float64 on CPU at the file's own truth
    and reports the per-seam drift of photometry and gradient against the file. Exit
    code 1 above ``_TOL_SELF_CHECK``: the file is stale, regenerate it. Run it before
    trusting any FAIL from the float32 sweep on a tree newer than the reference. The
    float32 sweep itself cannot do this (it runs with x64 off, on a device that may
    have no float64), so it prints a warning banner instead when the reference's tree
    SHA is not the current tree and ``src/tengri`` has commits since it.

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
import subprocess
import sys
import time
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import jaxlib
import numpy as np

#: PASS/FAIL thresholds. ``_TOL_LOSS`` is reported but does not gate ``passed`` -- see
#: the module docstring's "Why the loss column has no bar" note.
_TOL_FORWARD = 3e-3
_TOL_GRAD = 1e-2
_TOL_LOSS = 1e-4
_TOL_PARAM = 1e-2
#: ``--self-check`` bar: float64 on this tree against the file's float64. Same
#: arithmetic, same truth, so the only sources of drift are a physics change (what the
#: check exists to catch: #2260 was 3.3e-3) and summation-order noise across JAX/XLA
#: versions, measured at 1e-15..1e-11 on the six seams.
_TOL_SELF_CHECK = 1e-9

_REPO = Path(__file__).resolve().parents[2]

_BANDS = ["sdss_g", "sdss_r", "wise_w1", "herschel_250"]
_SSP = "data/fsps_prsc_miles_chabrier.h5"
_Z = 0.1
_SNR = 30.0
_TRUTH_SEED = 0
_NOISE_SEED = 1
_MAP_SEED = 2
#: L-BFGS iteration cap per attempt (``scipy.optimize.minimize`` ``maxiter``) -- not
#: a fixed few-step trajectory. See ``_map_loss``.
_N_STEPS = 1000
#: Cap on L-BFGS restarts when a call stalls (scipy ``ABNORMAL_TERMINATION_IN_LNSRCH``)
#: without exhausting ``_N_STEPS`` -- more of *this* budget, not a bigger ``maxiter``,
#: is what a line-search stall needs. See ``_map_loss``.
_MAX_MAP_RESTARTS = 5

#: Sites rewritten for #2295 (jax-mps#232): reversed-operand trapezoids replaced
#: with negated-descending to avoid silent zero-out on MLX compile.
_REWRITTEN_SITES = (
    "polar_dust.py: anisotropic_polar_luminosity negated-descending (#2295)",
    "adaf.py: normalization integral (float32) negated-descending (#2295)",
    "adaf.py: normalization integral (float64) negated-descending (#2295)",
    "unified.py: disc L_bol negated-descending (#2295)",
)


def _refuse_wrong_precision(write_reference: bool) -> None:
    """Exit with the one-line environment fix if ``jax_enable_x64`` is set wrong.

    ``write_reference`` here means "needs float64": reference generation and
    ``--self-check`` both do. The parity sweep refuses float64 outright,
    because setting ``JAX_ENABLE_X64`` after ``import jax`` is too late -- constants
    allocated during import are already on the device (``notebooks/apple_mps.py``
    Rule 2). This check runs before any tengri import, so it is the first thing that
    can go wrong and the only chance to fail loudly instead of silently.
    """
    x64 = bool(jax.config.jax_enable_x64)
    if write_reference and not x64:
        sys.exit(
            "--write-reference and --self-check need float64. Fix: JAX_ENABLE_X64=1 "
            "before Python starts, e.g. `JAX_ENABLE_X64=1 python bench/scripts/"
            "benchmark_float32_mps_parity.py --write-reference ...`."
        )
    if not write_reference and x64:
        sys.exit(
            "The float32 parity sweep refuses to run with jax_enable_x64=True. "
            "Fix: JAX_ENABLE_X64=0 before Python starts (setting it after import is "
            "too late -- notebooks/apple_mps.py Rule 2)."
        )


def _git(*args: str) -> str | None:
    """``git <args>`` in the repository, or ``None`` when git cannot answer."""
    try:
        r = subprocess.run(
            ["git", "-C", str(_REPO), *args], capture_output=True, text=True, timeout=30
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    return r.stdout.strip() if r.returncode == 0 else None


def _tree_sha() -> str | None:
    """The tree's HEAD SHA, suffixed ``-dirty`` when the working tree has changes."""
    sha = _git("rev-parse", "HEAD")
    if sha is None:
        return None
    dirty = _git("status", "--porcelain", "--untracked-files=no")
    return sha + ("-dirty" if dirty else "")


def _src_commits_since(ref_sha: str) -> int | None:
    """How many commits touching ``src/tengri`` HEAD carries past ``ref_sha``.

    ``None`` when git cannot say (a shallow clone, an unknown SHA): that is not zero.
    """
    out = _git("rev-list", "--count", f"{ref_sha}..HEAD", "--", "src/tengri")
    return int(out) if out is not None and out.isdigit() else None


def reference_staleness(ref_sha: str | None, src_commits_since: int | None) -> str | None:
    """A warning banner when the reference may not describe this tree, else ``None``.

    The float32 sweep cannot measure staleness itself (x64 off, possibly no float64
    device), so it can only correlate: a reference written on another tree with
    ``src/tengri`` commits in between *may* be stale, and the measurement is
    ``--self-check`` on CPU. Warn-only on purpose: a reference from a sibling branch is
    a legitimate thing to compare against during review.
    """
    remedy = (
        "measure before trusting a FAIL: JAX_ENABLE_X64=1 JAX_PLATFORMS=cpu python "
        "bench/scripts/benchmark_float32_mps_parity.py --self-check <reference>"
    )
    if ref_sha is None:
        return f"REFERENCE HAS NO tree_sha (written before #2300); {remedy}"
    if src_commits_since is None:
        return f"CANNOT COUNT src commits since reference {ref_sha[:9]} (shallow clone?); {remedy}"
    if src_commits_since == 0:
        return None
    return (
        f"REFERENCE MAY BE STALE: {src_commits_since} commit(s) touch src/tengri since "
        f"{ref_sha[:9]}; {remedy}"
    )


def self_check_drift(ref_rows: dict, here_rows: dict) -> dict[str, float]:
    """Per-seam max relative deviation of float64 rows recomputed now from the file's.

    Photometry and gradient together, so a physics change that moves only a derivative
    (a screen swap on a free parameter) is caught. A seam the file holds but
    ``here_rows`` does not is ``nan``: unmeasured is not clean.
    """
    drift = {}
    for name, ref in ref_rows.items():
        if not ref.get("built", False):
            continue
        here = here_rows.get(name)
        if here is None:
            drift[name] = float("nan")
            continue
        drift[name] = max(
            _rel(here["photometry"], ref["photometry"]), _rel(here["grad"], ref["grad"])
        )
    return drift


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
    is a ``maxiter`` cap, not a fixed trajectory length: each call runs to convergence
    or to the cap, whichever comes first.

    An early version of this ran a fixed few Adam steps instead, which tests an
    *unconverged intermediate* -- float32 rounding compounds along the step sequence,
    so the comparison is dominated by trajectory divergence rather than the model's own
    float32 accuracy. It also inherited a real bug: ``init_from`` still matters here
    (see its docstring note below), because without it ``Fitter``'s own random init
    draws from ``jax.random`` at a dtype that tracks ``jax_enable_x64``, and a float32
    draw is not bit-identical to a float64 one from the same key -- that alone sent an
    unconverged 5-step trajectory to a final loss differing by ~15%.

    **Restart, not a bigger cap, is what a line-search stall needs.** One seam
    (``panchromatic``) terminated at iteration 3 of a 1000-iteration budget with
    ``grad_norm`` still ~0.08 -- scipy's L-BFGS-B abandoning the line search, not
    exhausting ``maxiter``. A bigger cap cannot fix that: the call is not running out
    of steps, it is stuck. Restarting a fresh L-BFGS-B call from the stalled point
    (``init_from`` set to that result) resets the internal Hessian approximation and
    routinely clears the stall in one more call. This loop does that up to
    ``_MAX_MAP_RESTARTS`` times and reports the true cost: total iterations summed
    across every attempt, and whether the *last* attempt actually converged.

    Returns ``(final_loss, params_vector, converged, n_iters)``; ``params_vector`` is
    ``posterior.params`` read off in ``free_names`` order, physical space; ``n_iters``
    is the sum of ``diagnostics["n_steps"]`` over every restart attempt.
    """
    from tengri import Posterior

    forward = ForwardModel.build(sed=model, observation=obs)
    # Pins both precision arms to the same physical starting point, not merely the
    # same PRNGKey (see the module note above on why that is not the same thing).
    cur_init = Posterior(
        samples=None,
        params={k: jnp.asarray(v, dtype=dtype) for k, v in truth.items()},
        method="truth",
        wall_time_s=0.0,
        diagnostics={},
        loss_history=None,
        _model=model,
    )
    n_iters = 0
    posterior = None
    for _attempt in range(_MAX_MAP_RESTARTS):
        posterior = forward.fit(
            jnp.asarray(flux, dtype=dtype),
            jnp.asarray(noise, dtype=dtype),
            method="map",
            approx=None,
            init_from=cur_init,
            key=jax.random.PRNGKey(_MAP_SEED),
            n_steps=n_steps,
            optimizer="lbfgs",
            verbose=False,
        )
        n_iters += int(posterior.diagnostics["n_steps"])
        if posterior.diagnostics.get("converged", False):
            break
        cur_init = posterior  # restart from the stall, not from the shared truth again

    final_loss = float(np.asarray(posterior.loss_history)[-1])
    params_vector = [float(np.asarray(posterior.params[k])) for k in free_names]
    converged = bool(posterior.diagnostics.get("converged", False))
    return final_loss, params_vector, converged, n_iters


def _run_seam(
    name,
    kwargs,
    ssp,
    obs,
    z,
    n_steps,
    write_reference,
    ref_row,
    stage="full",
    checkpoint_row=None,
    dtype=None,
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

    # ``dtype`` overrides the mode's default: ``--self-check`` runs the parity path in
    # float64 so the file's own arithmetic is compared against itself (#2300).
    if dtype is None:
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

    photometry_here = grad_here = None
    map_loss_here = map_params_here = map_converged_here = map_n_iter_here = None
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
            map_loss_here, map_params_here, map_converged_here, map_n_iter_here = _map_loss(
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
            rec["map_n_iter"] = map_n_iter_here
        elif ref_row is not None and "map_loss" in ref_row:
            rec["map_loss"] = ref_row["map_loss"]
            rec["map_params"] = ref_row["map_params"]
            rec["map_converged"] = ref_row["map_converged"]
            rec["map_n_iter"] = ref_row.get("map_n_iter")
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
        return {
            "seam": name,
            "built": True,
            "rel_forward": rel_fwd,
            "rel_grad": rel_grad,
            # the recomputed values themselves, so ``--self-check`` can report drift
            # per seam from real rows rather than only the two relative numbers
            "photometry": photometry_here.tolist(),
            "grad": grad_here,
        }

    # Informational only -- see the module docstring's "Why the loss column has no
    # bar" note. Not part of ``passed``.
    ref_loss = float(ref_row["map_loss"])
    rel_loss = abs(map_loss_here - ref_loss) / max(abs(ref_loss), 1e-300)
    rel_param = _rel(map_params_here, ref_row["map_params"])
    passed = rel_fwd < _TOL_FORWARD and rel_grad < _TOL_GRAD and rel_param < _TOL_PARAM
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


def _self_check(ref, groups, ssp, obs, args, banner) -> int:
    """``--self-check``: the parity path in float64 against the file's own float64.

    Every seam the file holds is rebuilt on this tree at the file's truth and scored
    against the file's mock, exactly as the float32 sweep does but in the reference's
    own dtype, so the only thing that can differ is the physics of the tree. Drift is
    the max over photometry and gradient (``self_check_drift``). Exit 1 above
    ``_TOL_SELF_CHECK`` or for a seam this tree could not rebuild: an unmeasured seam is
    not a clean one.
    """
    here = {}
    for name, kwargs in groups.items():
        t0 = time.time()
        ref_row = ref["seams"].get(name)
        if ref_row is None or not ref_row.get("built", False):
            print(f"{name:14s} SKIP  (reference: {(ref_row or {}).get('skip_reason', 'no row')})")
            continue
        rec = _run_seam(
            name, kwargs, ssp, obs, args.z, args.n_steps, False, ref_row, "grad", dtype=jnp.float64
        )
        jax.clear_caches()
        gc.collect()
        if not rec.get("built", False):
            dt = time.time() - t0
            print(f"{name:14s} FAIL  cannot rebuild: {rec.get('skip_reason')}  ({dt:.1f}s)")
            continue
        here[name] = rec
        print(
            f"{name:14s} fwd drift={rec['rel_forward']:.2e}  grad drift={rec['rel_grad']:.2e}  "
            f"({time.time() - t0:.1f}s)"
        )
    drift = self_check_drift({k: v for k, v in ref["seams"].items() if k in groups}, here)
    stale = {k: d for k, d in drift.items() if not (d < _TOL_SELF_CHECK)}
    print()
    print(
        f"reference tree_sha: {(ref.get('meta') or {}).get('tree_sha')}   this tree: {_tree_sha()}"
    )
    if banner:
        print(f"note: {banner}")
    if stale:
        print(
            f"STALE: {len(stale)} seam(s) drift above {_TOL_SELF_CHECK:g} in float64 -- "
            + ", ".join(f"{k}={d:.2e}" for k, d in stale.items())
            + ". The reference does not describe this tree; regenerate it "
            "(--write-reference) before reading any float32 FAIL on these seams (#2300)."
        )
        return 1
    print(f"OK: {len(drift)} seam(s) within {_TOL_SELF_CHECK:g} of the reference in float64.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--write-reference", default=None, help="write a float64 reference JSON here")
    ap.add_argument("--reference", default=None, help="float64 reference JSON to check against")
    ap.add_argument(
        "--self-check",
        default=None,
        metavar="PATH",
        help="float64 on CPU: rebuild every seam at this reference's own truth and report "
        "the per-seam drift of photometry and gradient against the file; exit 1 above "
        f"{_TOL_SELF_CHECK:g} (the reference is stale, regenerate it). See #2300",
    )
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
    self_check = args.self_check is not None
    if self_check and (write_reference or args.reference is not None):
        ap.error("--self-check PATH stands alone (it is its own reference)")
    if not write_reference and not self_check and args.reference is None:
        ap.error("pass --write-reference PATH, --reference PATH, or --self-check PATH")
    if not write_reference and args.stage != "full" and args.checkpoint is None:
        ap.error("--stage grad/map for the parity sweep needs --checkpoint PATH")

    _refuse_wrong_precision(write_reference or self_check)

    warnings.filterwarnings("ignore")

    dtype_label = "float64" if (write_reference or self_check) else "float32"
    meta = {
        "mode": "write-reference"
        if write_reference
        else ("self-check" if self_check else "parity-sweep"),
        "dtype": dtype_label,
        # The tree the file describes (#2300). ``reference_staleness`` reads it back.
        "tree_sha": _tree_sha(),
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
            "param": _TOL_PARAM,
        },
        "loss_informational_reference_point": _TOL_LOSS,
    }
    print(json.dumps(meta, indent=2), flush=True)

    ref = None
    banner = None
    if not write_reference:
        with open(args.self_check if self_check else args.reference) as fh:
            ref = json.load(fh)
        ref_sha = (ref.get("meta") or {}).get("tree_sha")
        ref_sha_plain = ref_sha.replace("-dirty", "") if ref_sha else None
        banner = reference_staleness(
            ref_sha, _src_commits_since(ref_sha_plain) if ref_sha_plain else None
        )
        if banner and not self_check:
            print(f"WARNING: {banner}", flush=True)

    ssp = _load_ssp(args.ssp)
    obs = _make_observation(args.bands)

    from tengri import DEFAULT, Fixed, Uniform

    groups = _seam_groups(DEFAULT, Fixed, Uniform)
    if args.seams is not None:
        unknown = set(args.seams) - set(groups)
        if unknown:
            ap.error(f"unknown seam(s) {sorted(unknown)}; choices are {list(groups)}")
        groups = {k: v for k, v in groups.items() if k in args.seams}

    if self_check:
        return _self_check(ref, groups, ssp, obs, args, banner)

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
            conv_note = ""
            if not rec.get("partial") and "map_converged" in rec:
                conv_note = f"  map_iters={rec.get('map_n_iter')} converged={rec['map_converged']}"
            print(f"{name:14s} {state}  D={len(rec['free_names'])}{conv_note}  ({dt:.1f}s)")
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

        # An unconverged reference is refused, not silently committed: every seam
        # with a finished MAP fit (not a "--stage grad"-only partial) must have
        # actually converged, on this file's full merged state -- not only the
        # seams this invocation touched -- or the file is unusable as a reference.
        unconverged = [
            s
            for s, r in existing.items()
            if r.get("built") and "map_loss" in r and not r.get("map_converged")
        ]
        if unconverged:
            print(f"UNCONVERGED: {unconverged} -- refusing to treat this as a valid reference")
            return 1
        return 0

    if args.checkpoint:
        checkpoint.update({r["seam"]: r for r in rows})
        with open(args.checkpoint, "w") as fh:
            json.dump({"meta": meta, "seams": checkpoint}, fh, indent=2)
        report_rows = list(checkpoint.values())
    else:
        report_rows = rows

    print()
    header = f"{'seam':14s} {'fwd':>10s} {'grad':>10s} {'loss (info)':>13s} {'param':>10s}  status"
    print(header)
    print("-" * len(header))
    any_fail = False
    for rec in report_rows:
        if not rec.get("built", False):
            print(f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>13s} {'--':>10s}  SKIP")
            continue
        if "error" in rec:
            print(
                f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>13s} {'--':>10s}  "
                f"FAIL ({rec['error']})"
            )
            any_fail = True
            continue
        if "passed" not in rec:
            print(f"{rec['seam']:14s} {'--':>10s} {'--':>10s} {'--':>13s} {'--':>10s}  PENDING")
            continue
        status = "PASS" if rec["passed"] else "FAIL"
        any_fail = any_fail or not rec["passed"]
        both_converged = rec.get("map_converged") and rec.get("map_converged_ref")
        conv = "" if both_converged else "  (unconverged)"
        print(
            f"{rec['seam']:14s} {rec['rel_forward']:10.2e} {rec['rel_grad']:10.2e} "
            f"{rec['rel_loss']:13.2e} {rec['rel_param']:10.2e}  {status}{conv}"
        )

    print()
    print(
        "loss (info): not gated -- at a converged optimum a parameter difference of "
        "1e-6 moves the loss by ~1e-12 (stationary-point Taylor expansion); the loss "
        "gap seen here is float32's own chi-squared evaluation (cancellation in "
        "data-minus-model at this SNR), already bounded by the fwd/grad columns."
    )
    print()
    print("Rewritten sites (#2295):")
    for site in _REWRITTEN_SITES:
        print(f"  {site}")
    print(f"device: {meta['devices']}  jax {meta['jax']}  jaxlib {meta['jaxlib']}")
    if banner:
        # Repeated under the table on purpose: the load-time copy has scrolled away by
        # now, and a FAIL read without it is #2300 again.
        print(f"WARNING: {banner}")
    return 1 if any_fail else 0


if __name__ == "__main__":
    sys.exit(main())
