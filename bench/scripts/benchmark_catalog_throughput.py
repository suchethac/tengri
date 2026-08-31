#!/usr/bin/env python
"""Catalog throughput: vectorized per-galaxy MCMC, galaxies/s vs K, method and precision.

Measures the Track-A catalog sampling path — ``CatalogFitter.run(method,
forward_chunk_size=K, devices=...)`` — which fits many independent galaxies in
parallel and returns a posterior per galaxy.

Reports, for a mock photometric catalog:

* the **forward_chunk_size (K) sweep**: warm galaxies/second as K grows, so you
  can find the K that saturates the accelerator (cold compile is reported
  separately — it is paid once and cached), together with the device memory
  high-water mark after each K so the saturation point is visible in bytes and
  not only in seconds.
* **convergence alongside every throughput number**: max split-R-hat over the
  catalog, min ESS, and the divergence count. A galaxies/second figure without
  these is not reportable — see
  ``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``.
* a **precision axis** (``--dtype f32|f64``) and an explicit **method axis**
  (``--method mcmc_nuts mcmc_hmc``, the two entries of
  ``CatalogFitter._MCMC_VMAPPABLE``).
* **device scaling** (only when more than one device is visible): single-device
  vs ``devices="all"`` throughput, i.e. how well the galaxy axis shards.
* ``--mode grad``: the log-posterior and its gradient in isolation, batched over
  galaxies. This is the number that prices float64 on a die whose FP64 units are
  rate-limited (GeForce: 1/64 of FP32); the sampler rows mix it with control
  flow and host dispatch and cannot separate the two.

Precision is a **process-global** JAX setting that must be chosen *before*
``import tengri``, not merely before the model is built, so ``--dtype`` takes a
single value and a two-precision sweep is two processes writing into the same
``--json``. ``jax.config.update("jax_enable_x64", False)`` in ``main()`` is not
enough and is actively misleading: ``tengri/__init__.py`` turns x64 back **on**
at import unless ``JAX_ENABLE_X64`` is set in the environment (#1840), and by
then five DSPS modules have already allocated float64 module-scope constants
(#1880). The only switch that binds is the environment variable, applied before
``import jax`` — which this script does for you, from its own argv, at module
scope. ``set_precision`` then *verifies* the result on a real array's dtype
rather than on the config flag, because a measurement that cannot fail is not a
measurement.

The LUT caveat, which every row inherits: ``CatalogFitter`` defaults to
``approx="auto"``, which resolves to ``WavePrecomp()`` — ``band_integration=
"quadrature"``, ``n_subbands=5``. Its forward bias is constant in SNR but enters
the posterior gradient multiplied by SNR (#1671). The harness records the
catalog's median SNR, the resolved ``band_integration``, and whatever
``PrecompBiasWarning`` the fit raises, in every JSON row. A fast wrong gradient
is not a result.

Device-agnostic: it does NOT force a platform, so on a CUDA box / Sherlock GPU
node it runs on the GPU. Emulate multiple devices on CPU with
``XLA_FLAGS=--xla_force_host_platform_device_count=N``.

Usage::

    # CPU (or GPU if JAX picks one):
    python bench/scripts/benchmark_catalog_throughput.py

    # Sherlock GPU node, bigger catalog:
    python bench/scripts/benchmark_catalog_throughput.py --n-gal 512 2048 --chunk 32 128 512

    # Both samplers, float32, appending to a shared JSON:
    python bench/scripts/benchmark_catalog_throughput.py \
        --method mcmc_nuts mcmc_hmc --dtype f32 --json bench/results/gpu.json

    # The precision question on its own (log-posterior + gradient, no sampler):
    python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f64
    python bench/scripts/benchmark_catalog_throughput.py --mode grad --dtype f32

    # Emulate 4 devices on CPU to smoke the shard path:
    XLA_FLAGS=--xla_force_host_platform_device_count=4 \
        python bench/scripts/benchmark_catalog_throughput.py --shard
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
import warnings

import numpy as np

#: Kept identical to ``CatalogFitter._MCMC_VMAPPABLE`` by a contract test
#: (``tests/contract/test_catalog_throughput_bench.py``), so a sampler that
#: reaches the batched path cannot go unmeasured here.
METHODS = ("mcmc_nuts", "mcmc_hmc", "mcmc_chees")
DTYPES = ("f64", "f32")


def _apply_precision_env(argv=None) -> str:
    """Translate ``--dtype`` into the environment, BEFORE ``import jax``.

    This has to run at module scope. ``jax`` latches ``JAX_ENABLE_X64`` when it
    is imported, and ``tengri/__init__.py`` re-enables x64 on import unless the
    variable is present, so any later ``jax.config.update`` is overwritten by
    the very import the benchmark needs. Setting the variable here also arms
    tengri's own import-time x64 guard (#1880), which is what stops the DSPS
    modules allocating float64 constants during the build.

    ``JAX_DEFAULT_MATMUL_PRECISION=highest`` rides along in float32: on Ampere
    XLA otherwise lowers float32 matmuls to TF32 (10-bit mantissa) and tengri's
    own float32 Fisher-matrix test fails by 4.5 % on the error bars
    (``docs/internal/getting_started/gpu.md``). An explicit setting in the
    environment always wins over this default.
    """
    argv = list(sys.argv[1:] if argv is None else argv)
    dtype = "f64"
    for i, tok in enumerate(argv):
        if tok == "--dtype" and i + 1 < len(argv):
            dtype = argv[i + 1]
        elif tok.startswith("--dtype="):
            dtype = tok.split("=", 1)[1]
    if dtype == "f32":
        os.environ.setdefault("JAX_ENABLE_X64", "0")
        os.environ.setdefault("JAX_DEFAULT_MATMUL_PRECISION", "highest")
    return dtype


_ARGV_DTYPE = _apply_precision_env()

# Noisy build-time chatter only. PrecompBiasWarning is deliberately NOT filtered
# here: it is a reported column, and `_run_and_time` captures it per row. The
# float32 request itself warns once (cosmological distances) — expected, kept.
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# NOTE: must stay below _apply_precision_env — JAX latches JAX_ENABLE_X64 here.
import jax
import jax.numpy as jnp

#: The adoption bar this benchmark publishes against. Vehtari+2021, and the bar
#: already used by ``bench/reports/2026-08-17_*`` and the spine notebooks.
#:
#: **Imported, not restated.** It was a literal 1.01 here and a literal 1.01 in
#: the library, which is two places for one number to drift; the per-galaxy
#: buckets this harness reports now come from the library too, so the row-level
#: bar has to be the same object as the galaxy-level one or a row can pass a bar
#: its own galaxies failed.
from tengri.inference.catalog_convergence import CATALOG_MAX_RHAT as MAX_RHAT


def set_precision(dtype: str) -> dict:
    """Confirm the process is actually in ``dtype``, and refuse if it is not.

    The switch itself was thrown at module scope by :func:`_apply_precision_env`
    (see its docstring for why it cannot be thrown here). What this does is the
    part that makes the number trustworthy: it proves the precision on a real
    array's dtype, and raises if the process is running in a precision the
    caller did not ask for — a silently-float64 "float32" row is worse than no
    row at all.

    Returns the flags in force, for the JSON record.
    """
    if dtype not in DTYPES:
        raise ValueError(f"--dtype must be one of {DTYPES}, got {dtype!r}")
    if dtype == "f64":
        # tengri's default; also the state when nothing was requested.
        jax.config.update("jax_enable_x64", True)
    probe = jnp.zeros(1, dtype=jnp.float64) if dtype == "f64" else jnp.zeros(1) + 1.0
    want = "float64" if dtype == "f64" else "float32"
    if str(probe.dtype) != want:
        raise RuntimeError(
            f"--dtype {dtype} asked for {want} but jnp allocates {probe.dtype}. "
            f"JAX_ENABLE_X64={os.environ.get('JAX_ENABLE_X64')!r}, "
            f"jax_enable_x64={jax.config.x64_enabled}. `import tengri` re-enables "
            f"x64 unless JAX_ENABLE_X64 is set in the environment (#1840), so run "
            f"this script as its own process (it sets the variable from its own "
            f"argv) rather than calling main() from a session already in float64."
        )
    return {
        "jax_enable_x64": bool(jax.config.x64_enabled),
        "JAX_ENABLE_X64": os.environ.get("JAX_ENABLE_X64"),
        "JAX_DEFAULT_MATMUL_PRECISION": os.environ.get("JAX_DEFAULT_MATMUL_PRECISION"),
        "probe_dtype": str(probe.dtype),
    }


def _load_or_synth_ssp():
    """Real SSP if on disk (realistic numbers), else a portable synthetic grid."""
    for name in (
        "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
        "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    ):
        path = os.path.join("data", name)
        if os.path.exists(path):
            from tengri.sps.dsps_wrapper import load_ssp_data

            return load_ssp_data(path), f"real:{name}"
    from tengri.sps.dsps_wrapper import SSPData

    wave = jnp.linspace(3000.0, 10000.0, 100)
    ages = jnp.linspace(-1.0, 1.14, 20)
    flux = jnp.abs(jax.random.normal(jax.random.PRNGKey(1), (3, 20, 100))) * 1e-3 + 1e-5
    ssp = SSPData(
        ssp_wave=wave, ssp_flux=flux, ssp_lg_age_gyr=ages, ssp_lgmet=jnp.array([-1.5, -0.5, 0.0])
    )
    return ssp, "synthetic"


def build_model(ssp, ssp_tag):
    """Small photometric dpl model: mass + alpha (+ metallicity) free, the rest pinned."""
    from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
    from tengri.observation.photometry import FilterCurve

    if ssp_tag.startswith("real"):
        phot = Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"])
    else:
        phot = Photometry(
            filters=tuple(
                FilterCurve(
                    wave=jnp.linspace(c * 0.9, c * 1.1, 40), trans=jnp.ones(40) * 0.5, name=f"b{i}"
                )
                for i, c in enumerate((3500.0, 4800.0, 6200.0, 7600.0, 9000.0))
            )
        )
    obs = Observation(photometry=phot)
    met = Fixed(1.0) if ssp_tag == "synthetic" else Uniform(-1.5, 0.2)
    spec = Parameters(
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_beta=Fixed(2.0),
        sfh_dpl_tau_gyr=Fixed(3.0),
        met_logzsol=met,
        dust_tau_bc=Fixed(0.3),
        dust_tau_diff=Fixed(0.2),
        redshift=Fixed(0.1),
        mean_sfh_type="dpl",
    )
    return SEDModel(spec, ssp, observation=obs)


def make_catalog(model, n_gal, key, noise_frac=0.05):
    """Mock catalog at a fixed fractional noise, i.e. a fixed per-band SNR."""
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        tp = dict(model.spec.sample(k))
        flux = model.predict_photometry(tp)
        noise = jnp.abs(flux) * noise_frac
        galaxies.append(
            {
                "flux_obs": flux + noise * jax.random.normal(jax.random.fold_in(k, 1), flux.shape),
                "noise": noise,
            }
        )
    return galaxies


def catalog_snr(galaxies) -> dict:
    """Median / min / max per-band SNR of the mock, the axis #1671's bias scales on."""
    snr = np.concatenate(
        [
            np.abs(np.asarray(g["flux_obs"], dtype=np.float64))
            / np.maximum(np.asarray(g["noise"], dtype=np.float64), np.finfo(float).tiny)
            for g in galaxies
        ]
    )
    return {
        "snr_median": float(np.median(snr)),
        "snr_min": float(np.min(snr)),
        "snr_max": float(np.max(snr)),
    }


def approx_tag(cat) -> dict:
    """What LUT the fit actually *resolved* to, read off the built model.

    ``ApproxState`` reports the resolved state, not the requested one. It does
    not carry ``band_integration`` directly, but ``n_subbands`` does identify
    the scheme: ``WavePrecomp`` sets it to 5 for ``"quadrature"`` and to 0 for
    ``"effective_wavelength"`` / ``"taylor"`` (see its docstring), so the count
    is the evidence and the name is derived from it, not assumed.
    """
    state = getattr(cat.model, "approx", None)
    n_sub = int(getattr(state, "n_subbands", 0) or 0)
    return {
        "approx_active": bool(state),
        "wave_precomp": bool(getattr(state, "wave_precomp", False)),
        "ztable": bool(getattr(state, "ztable", False)),
        "n_subbands": n_sub,
        "band_integration": ("quadrature" if n_sub > 0 else "effective_wavelength/taylor")
        if getattr(state, "wave_precomp", False)
        else None,
        "approx_repr": repr(state),
    }


def device_peak_bytes():
    """Device allocator high-water mark, or None on a backend without stats.

    ``peak_bytes_in_use`` is monotone for the life of the process, which is why
    the K sweep runs **ascending** and reports the increment: the K at which the
    peak stops climbing is the K at which the device saturates.
    """
    try:
        stats = jax.devices()[0].memory_stats()
    except Exception:
        return None
    if not stats:
        return None
    return stats.get("peak_bytes_in_use")


def _diagnostics(cp, max_gal):
    """Per-catalog convergence, as four disjoint counts plus the extremes.

    Delegates to :func:`tengri.inference.catalog_convergence.catalog_convergence`
    rather than re-deriving the buckets here. That is deliberate: this harness
    had its own copy of the rule, and a benchmark whose convergence definition
    can drift from the library's is a benchmark that will eventually publish a
    rate the library disagrees with. The library owns the definition; this
    function owns the column names.

    The four buckets, and why none can be folded into another:

    * **converged** -- max split-R-hat below
      :data:`~tengri.inference.catalog_convergence.CATALOG_MAX_RHAT` with no
      divergence.
    * **frozen** -- every kept draw diverged, or a free parameter took
      essentially no distinct values, or R-hat is undefined because a free
      parameter has zero variance. Split R-hat cannot see this (both halves have
      zero variance, so it reads ~1.0) and the divergence count may be 0, so a
      frozen galaxy would otherwise be counted as a converged win.
    * **refused** -- the sampler raised ``DeadFitError`` before sampling and the
      galaxy has no posterior at all. Only ever non-zero on the sequential
      engine: the batched engine's ``run_one`` is inside ``lax.map``, where a
      Python raise is not expressible, so a refusal there fails the whole cell
      and is recorded as ``row["refused"]`` instead.
    * **unconverged** -- it moved, and it did not mix.

    ``min_ess_converged`` is reported beside ``frac_converged`` and the two must
    be quoted together: Phase 0 measured 73 % of galaxies clearing R-hat < 1.01
    with zero divergences at a worst ESS of **2.63 of 500 draws**.
    """
    from tengri.inference.catalog_convergence import catalog_convergence

    report = catalog_convergence(
        cp.posteriors, refusals=getattr(cp, "refusals", None), max_galaxies=max_gal
    )
    worst_rhat_param = min_ess_param = None
    worst, least = -np.inf, np.inf
    for row in report.per_galaxy:
        if row.max_rhat is not None and row.max_rhat > worst:
            worst = row.max_rhat
            worst_rhat_param = row.max_rhat_param
        if row.min_ess is not None and row.min_ess < least:
            least = row.min_ess
            min_ess_param = row.min_ess_param
    return {
        "n_gal_checked": report.n_galaxies,
        "n_gal_converged": report.n_converged,
        "n_frozen_chains": report.n_frozen,
        "n_gal_refused": report.n_refused,
        "n_gal_unconverged": report.n_unconverged,
        "frac_converged": report.frac_converged,
        "divergence_rate": report.divergence_rate,
        "max_rhat": report.max_rhat,
        "max_rhat_param": worst_rhat_param,
        "min_ess": report.min_ess,
        "min_ess_param": min_ess_param,
        "min_ess_converged": report.min_ess_converged,
        "convergence_summary": report.summary(),
    }


def _run_and_time(cat, method, K, devices, key, run_kw):
    """One ``CatalogFitter.run``, timed, with the LUT-bias warning captured.

    A :class:`~tengri.config.exceptions.DeadFitError` is caught and returned as
    a refusal rather than allowed to abort the sweep. Since #2090 the
    window-adaptation backends refuse to sample when the final warmup window is
    >=90 % divergent, and the PR names this caller explicitly: *"Drivers that
    loop over galaxies should catch it and record the galaxy as a failed fit."*
    A benchmark that dies on one bad galaxy reports nothing; one that swallows
    the refusal reports a throughput number over a catalog it silently shrank.
    Neither is acceptable, so the refusal becomes a **column**.

    Note the catalog-vectorized path cannot raise *per galaxy* — ``run_one`` is
    inside ``lax.map``/``vmap``, where a Python raise is not expressible — so a
    refusal here fails the whole cell. That is itself worth recording, and it is
    why ``refused`` is a property of the row rather than a per-galaxy count.

    Returns ``(wall, cp, bias, refusal)``; on a refusal ``cp`` is ``None`` and
    ``refusal`` is a dict carrying the divergent fraction and the adapted step
    size the sampler refused on.
    """
    from tengri.config.exceptions import DeadFitError

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        t0 = time.perf_counter()
        try:
            cp = cat.run(
                method, key=key, forward_chunk_size=K, devices=devices, verbose=False, **run_kw
            )
            jax.block_until_ready(cp[0].samples)
        except DeadFitError as exc:
            return (
                time.perf_counter() - t0,
                None,
                None,
                {
                    "reason": "DeadFitError",
                    "warmup_divergence_frac": float(exc.warmup_divergence_frac),
                    "step_size": float(exc.step_size),
                    "message": str(exc)[:400],
                },
            )
        wall = time.perf_counter() - t0
    bias = None
    for w in caught:
        est = getattr(w.message, "gradient_error_estimate", None)
        if est is not None:
            bias = float(est)
    return wall, cp, bias, None


#: Fields stamped onto every row, not only into ``meta``. A campaign writes GPU
#: and CPU rows into one file, so a row that cannot say which device it ran on
#: is unreadable once merged — and ``meta`` holds only the last writer's answer.
_STAMP_FIELDS = (
    "platform",
    "device",
    "n_devices",
    "jax",
    "tag",
    "snr_median",
    "band_integration",
    "n_subbands",
    "approx_repr",
    "ssp",
    "jax_persistent_cache",
    # Both change the *result*, not merely the provenance, and neither can be
    # reconstructed from a merged file. ``precondition`` is the whole subject of
    # bench/reports/2026-08-31_catalog_preconditioning.md and ``chain_jitter``
    # decides whether a ChEES row's R-hat is an independent test or a
    # consistency check (PR #2097).
    "precondition",
    "chain_jitter",
)


def stamp(row, meta):
    """Copy the provenance fields a merged JSON cannot reconstruct into ``row``."""
    for field in _STAMP_FIELDS:
        if field in meta:
            row[field] = meta[field]
    return row


def _row_key(row):
    return (
        row.get("mode"),
        row.get("method"),
        row.get("dtype"),
        row.get("n_gal"),
        row.get("chunk"),
        row.get("devices"),
        row.get("batch"),
        row.get("max_num_doublings"),
        row.get("n_leapfrog"),
        row.get("n_warmup"),
        row.get("n_samples"),
        # Part of the key, not just of the row. A preconditioned cell and an
        # identity cell of the same (method, N, K) are different measurements;
        # leaving these out let the second silently overwrite the first, and a
        # merged file would then show one row where two were run.
        row.get("precondition"),
        row.get("chain_jitter"),
        row.get("platform"),
        row.get("tag"),
    )


def write_json(path, rows, meta):
    """Merge rows into ``path``, keyed on the configuration, newest wins."""
    existing = {"meta": {}, "rows": []}
    if os.path.exists(path):
        try:
            with open(path) as fh:
                existing = json.load(fh)
        except (OSError, ValueError):
            existing = {"meta": {}, "rows": []}
    merged = {_row_key(r): r for r in existing.get("rows", [])}
    for r in rows:
        merged[_row_key(r)] = r
    payload = {
        "meta": {**existing.get("meta", {}), **meta},
        "rows": sorted(merged.values(), key=lambda r: [str(x) for x in _row_key(r)]),
    }
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as fh:
        json.dump(payload, fh, indent=2, sort_keys=False)
        fh.write("\n")
    return path


# ── mode: grad ──────────────────────────────────────────────────────────
#
# The sampler rows cannot answer the precision question on their own: a NUTS
# step is a data-dependent number of leapfrogs wrapped in host dispatch. This
# mode times the *same* flat log-posterior the catalog sampler calls
# (``_get_flat_logdensity``) and its gradient, batched over galaxies, so the
# f32/f64 ratio is arithmetic and bandwidth only.


def _build_flat_logposterior(cat):
    from tengri.inference.backends.mcmc.catalog import _get_flat_logdensity, _make_substitute

    fitter = cat._get_dummy_fitter()
    init_params = fitter._initialize_unbounded(jax.random.PRNGKey(0))
    ld2, _unravel_fn, init_flat, template = _get_flat_logdensity(fitter, init_params)
    substitute = _make_substitute(template, False, False)

    def one(x, data, noise, presence):
        return ld2(x, substitute(data, noise, presence, 0.0, None, None))

    return one, init_flat


def _steady(fn, reps, runs):
    """Minimum-of-reps of the mean over ``runs`` calls, all blocked on."""
    jax.block_until_ready(fn())
    best = np.inf
    for _ in range(reps):
        t0 = time.perf_counter()
        for _ in range(runs):
            out = fn()
        jax.block_until_ready(out)
        best = min(best, (time.perf_counter() - t0) / runs)
    return best


def run_grad_mode(cat, galaxies, batches, dtype, reps, runs):
    one, init_flat = _build_flat_logposterior(cat)
    rows = []
    for n in batches:
        if n > len(galaxies):
            continue
        data = jnp.stack([jnp.asarray(g["flux_obs"]) for g in galaxies[:n]])
        noise = jnp.stack([jnp.asarray(g["noise"]) for g in galaxies[:n]])
        pres = jnp.ones_like(data)
        xs = jnp.tile(init_flat[None, :], (n, 1))

        lp = jax.jit(jax.vmap(one, in_axes=(0, 0, 0, 0)))
        gr = jax.jit(jax.vmap(jax.grad(one), in_axes=(0, 0, 0, 0)))

        t0 = time.perf_counter()
        lp_out = jax.block_until_ready(lp(xs, data, noise, pres))
        lp_compile = time.perf_counter() - t0
        t0 = time.perf_counter()
        gr_out = jax.block_until_ready(gr(xs, data, noise, pres))
        gr_compile = time.perf_counter() - t0

        args = (xs, data, noise, pres)
        lp_s = _steady(lambda f=lp, a=args: f(*a), reps, runs)
        gr_s = _steady(lambda f=gr, a=args: f(*a), reps, runs)
        rows.append(
            {
                "mode": "grad",
                "dtype": dtype,
                "batch": int(n),
                "logp_us": round(lp_s * 1e6, 3),
                "logp_us_per_gal": round(lp_s * 1e6 / n, 4),
                "grad_us": round(gr_s * 1e6, 3),
                "grad_us_per_gal": round(gr_s * 1e6 / n, 4),
                "logp_compile_s": round(lp_compile, 3),
                "grad_compile_s": round(gr_compile, 3),
                "logp_dtype": str(lp_out.dtype),
                "grad_dtype": str(gr_out.dtype),
                "grad_finite": bool(np.all(np.isfinite(np.asarray(gr_out, dtype=np.float64)))),
                "grad_all_zero": bool(np.all(np.asarray(gr_out) == 0)),
                "peak_bytes": device_peak_bytes(),
            }
        )
        print(
            f"  {n:>6} {lp_s * 1e6:>12.1f} {lp_s * 1e6 / n:>12.3f} "
            f"{gr_s * 1e6:>12.1f} {gr_s * 1e6 / n:>12.3f}  "
            f"{lp_out.dtype}/{gr_out.dtype}",
            flush=True,
        )
    return rows


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-gal", type=int, nargs="+", default=[16, 64])
    ap.add_argument("--chunk", type=int, nargs="+", default=[1, 8, 32])
    ap.add_argument("--warmup", type=int, default=30)
    ap.add_argument("--burnin", type=int, default=10)
    ap.add_argument("--samples", type=int, default=50)
    ap.add_argument(
        "--method",
        nargs="+",
        default=["mcmc_nuts"],
        choices=list(METHODS),
        help="catalog-vectorized samplers to sweep (CatalogFitter._MCMC_VMAPPABLE)",
    )
    ap.add_argument(
        "--dtype",
        default="f64",
        choices=list(DTYPES),
        help="process-global precision; set before the model is built, so one per process",
    )
    ap.add_argument(
        "--mode",
        default="throughput",
        choices=("throughput", "grad"),
        help="'throughput' sweeps the sampler; 'grad' times the log-posterior and its gradient",
    )
    ap.add_argument("--noise-frac", type=float, default=0.05, help="mock 1-sigma / flux -> SNR")
    ap.add_argument(
        "--max-doublings",
        type=int,
        default=None,
        help=(
            "NUTS tree-depth cap (default: tengri's DEFAULT_MAX_NUM_DOUBLINGS). "
            "This is the dominant cost knob for a vmapped catalog: every chain in "
            "the batch runs to the deepest tree ANY chain asks for, so the batch "
            "pays 2**depth leapfrogs whenever one galaxy is hard."
        ),
    )
    ap.add_argument(
        "--n-leapfrog",
        type=int,
        default=None,
        help="HMC trajectory length L (default 10). Ignored by mcmc_nuts.",
    )
    ap.add_argument(
        "--diag-max-gal",
        type=int,
        default=10_000,
        help="cap on how many galaxies the R-hat/ESS pass inspects",
    )
    ap.add_argument(
        "--n-ensemble",
        type=int,
        default=None,
        help="mcmc_chees: adaptation-ensemble width WITHIN each galaxy (default "
        "CATALOG_CHEES_ENSEMBLE=8). VRAM scales with K * n_ensemble.",
    )
    ap.add_argument(
        "--n-chains",
        type=int,
        default=None,
        help="mcmc_chees: sampling chains per galaxy. >1 gives a per-galaxy split "
        "R-hat over genuinely separate chains rather than two halves of one.",
    )
    ap.add_argument(
        "--max-leapfrog-steps",
        type=int,
        default=None,
        help="mcmc_chees: hard cap on the adapted trajectory length. This is what "
        "bounds a vmapped ChEES batch -- lanes batch to the widest adapted L.",
    )
    ap.add_argument(
        "--chain-jitter",
        type=float,
        default=None,
        help="mcmc_chees: overdispersion of the SAMPLING chains around the warm "
        "start. The default None seeds them from the adaptation ensemble's own "
        "final states, so their split R-hat is a consistency check rather than an "
        "independent test (PR #2097). 0.5 is the suggested width.",
    )
    ap.add_argument(
        "--precondition",
        type=float,
        default=None,
        metavar="ALPHA",
        help="analytic J^T N^-1 J + I metric, per galaxy, at whitening strength "
        "ALPHA in [0, 1]. Omit for off (the default). 0.5 is "
        "DEFAULT_WHITENING_STRENGTH; 1.0 is full whitening, which #1442 measured "
        "as worse whenever the modal Hessian misstates the bulk curvature. "
        "Applies to every sampler on the batched path.",
    )
    ap.add_argument("--reps", type=int, default=4, help="--mode grad: timing repetitions")
    ap.add_argument("--runs", type=int, default=20, help="--mode grad: calls per repetition")
    ap.add_argument("--shard", action="store_true", help="also time devices='all' scaling")
    ap.add_argument("--json", default=None, help="write/merge results into this JSON path")
    ap.add_argument("--tag", default=None, help="free-text platform label recorded in the JSON")
    args = ap.parse_args(argv)

    if args.dtype != _ARGV_DTYPE:
        raise RuntimeError(
            f"--dtype {args.dtype!r} was parsed, but the module-scope precision "
            f"switch saw {_ARGV_DTYPE!r}. The environment was already latched by "
            f"`import jax`; re-run this script as its own process."
        )

    with warnings.catch_warnings():
        # x64 off truncates the SSP tables, which is the entire point here, and
        # tengri warns once that it is honoring the float32 request.
        warnings.filterwarnings("ignore", message=".*Explicitly requested dtype float64.*")
        warnings.filterwarnings("ignore", message=".*JAX_ENABLE_X64=.*")
        from tengri.inference.catalog_fitter import CatalogFitter

        # AFTER the import, which is the step that would silently undo a
        # float32 request, and BEFORE the model is built.
        flags = set_precision(args.dtype)
        ssp, ssp_tag = _load_or_synth_ssp()
        model = build_model(ssp, ssp_tag)

    dev = jax.devices()
    print("tengri catalog-throughput benchmark")
    print(
        f"  backend: {dev[0].platform}  |  devices: {len(dev)}  |  "
        f"dtype: {args.dtype}  |  x64: {jax.config.x64_enabled}  "
        f"|  probe: {flags['probe_dtype']}"
    )
    print(f"  ssp: {ssp_tag}  |  free params: {list(model.spec.free_params)}")

    key = jax.random.PRNGKey(0)
    biggest = max(args.n_gal)
    full_catalog = make_catalog(model, biggest, key, noise_frac=args.noise_frac)
    snr = catalog_snr(full_catalog)
    print(
        f"  mock SNR per band: median {snr['snr_median']:.1f} "
        f"(min {snr['snr_min']:.1f}, max {snr['snr_max']:.1f})"
    )

    meta = {
        "platform": dev[0].platform,
        "device": str(dev[0]),
        "n_devices": len(dev),
        "jax": jax.__version__,
        "host": platform.platform(),
        "ssp": ssp_tag,
        "free_params": list(model.spec.free_params),
        "n_bands": len(model.observation.photometry.filters),
        "noise_frac": args.noise_frac,
        "tag": args.tag,
        # `import tengri` enables JAX's persistent compilation cache by default
        # (``tengri.utils.jax_cache``). With it on, the "cold" column is a
        # cache *load*, not a full XLA compile; set TENGRI_DISABLE_JAX_CACHE=1
        # to measure the real compile. Recorded so a compile_s number can never
        # be quoted without knowing which of the two it is.
        "jax_persistent_cache": not os.environ.get("TENGRI_DISABLE_JAX_CACHE"),
        **snr,
    }
    rows = []

    if args.mode == "grad":
        cat = CatalogFitter(model, full_catalog, data_type="photometry")
        meta.update(approx_tag(cat))
        print(f"  approx: {meta['band_integration']} n_subbands={meta['n_subbands']}")
        print(f"\n  log-posterior + gradient, batched over galaxies  [dtype={args.dtype}]")
        print(
            f"  {'batch':>6} {'logp_us':>12} {'logp/gal':>12} "
            f"{'grad_us':>12} {'grad/gal':>12}  dtypes"
        )
        rows = run_grad_mode(cat, full_catalog, args.n_gal, args.dtype, args.reps, args.runs)
        for r in rows:
            r["dtype_flags"] = flags
            stamp(r, meta)
        if args.json:
            print(f"\n  wrote {write_json(args.json, rows, meta)}")
        return 0

    run_kw = dict(n_warmup=args.warmup, n_burnin=args.burnin, n_samples=args.samples)
    if args.max_doublings is not None:
        run_kw["max_num_doublings"] = args.max_doublings
    if args.n_leapfrog is not None:
        run_kw["n_leapfrog_steps"] = args.n_leapfrog
    # ChEES knobs. Passed only to ChEES cells: ``max_num_doublings`` means nothing
    # to it and ``n_ensemble`` means nothing to NUTS/HMC, and the catalog engine
    # refuses n_chains > 1 for the window-adaptation samplers by name rather than
    # ignoring it.
    chees_kw = {}
    if args.n_ensemble is not None:
        chees_kw["n_ensemble"] = args.n_ensemble
    if args.n_chains is not None:
        chees_kw["n_chains"] = args.n_chains
    if args.max_leapfrog_steps is not None:
        chees_kw["max_leapfrog_steps"] = args.max_leapfrog_steps
    if args.chain_jitter is not None:
        chees_kw["chain_jitter"] = args.chain_jitter
    # Preconditioning is NOT a ChEES knob: the analytic metric threads through
    # every sampler on the batched path, so it goes in the shared kwargs. Keeping
    # it out of ``chees_kw`` is what lets an HMC row be measured with the same
    # metric, which is the only way to tell "ChEES needs the metric" apart from
    # "this posterior needs the metric".
    if args.precondition is not None:
        run_kw["precondition"] = args.precondition
    meta["precondition"] = args.precondition
    meta["chain_jitter"] = args.chain_jitter
    meta["n_warmup"] = args.warmup
    meta["n_burnin"] = args.burnin
    meta["n_samples"] = args.samples
    meta["max_num_doublings"] = args.max_doublings
    meta["n_leapfrog"] = args.n_leapfrog
    meta["n_ensemble"] = args.n_ensemble
    meta["n_chains"] = args.n_chains
    meta["max_leapfrog_steps"] = args.max_leapfrog_steps

    print(f"\n  forward_chunk_size (K) sweep  [warmup={args.warmup}, samples={args.samples}]")
    print(
        f"  {'method':<10} {'N':>6} {'K':>5} {'cold_s':>9} {'warm_s':>9} {'compile_s':>10} "
        f"{'gal/GPUmin':>11} {'conv/min':>9} {'conv/N':>10} {'maxRhat':>9} {'minESS':>8} "
        f"{'div':>5} {'peakGiB':>8}"
    )
    prev_peak = device_peak_bytes()
    for method in args.method:
        for n_gal in args.n_gal:
            cat = CatalogFitter(model, full_catalog[:n_gal], data_type="photometry")
            if not meta.get("approx_active"):
                meta.update(approx_tag(cat))
            for K in sorted(args.chunk):  # ascending: peak_bytes_in_use is monotone
                if n_gal < K:
                    continue
                cell_kw = {**run_kw, **(chees_kw if method == "mcmc_chees" else {})}
                if method == "mcmc_chees":
                    cell_kw.pop("max_num_doublings", None)
                    cell_kw.pop("n_leapfrog_steps", None)
                cold, _, bias, refused = _run_and_time(cat, method, K, None, key, cell_kw)
                if refused is None:
                    warm, cp, _, refused = _run_and_time(cat, method, K, None, key, cell_kw)
                if refused is not None:
                    # #2090: the sampler refused a dead warmup. Record the cell
                    # as a failed fit and carry on -- a sweep that aborts on one
                    # bad cell reports nothing, and one that swallows the
                    # refusal reports a rate over a catalog it silently shrank.
                    row = stamp(
                        {
                            "mode": "throughput",
                            "method": method,
                            "dtype": args.dtype,
                            "dtype_flags": flags,
                            "n_gal": n_gal,
                            "chunk": K,
                            "devices": 1,
                            "n_warmup": args.warmup,
                            "n_samples": args.samples,
                            "max_num_doublings": args.max_doublings,
                            "n_leapfrog": args.n_leapfrog,
                            "refused": refused,
                            "n_gal_converged": 0,
                            "converged": False,
                        },
                        meta,
                    )
                    rows.append(row)
                    if args.json:
                        write_json(args.json, [row], meta)
                    print(
                        f"  {method:<10} {n_gal:>6} {K:>5}   REFUSED (DeadFitError, "
                        f"warmup divergent fraction "
                        f"{refused['warmup_divergence_frac']:.2f}) — recorded as a "
                        f"failed cell, not dropped",
                        flush=True,
                    )
                    continue
                diag = _diagnostics(cp, args.diag_max_gal)
                peak = device_peak_bytes()
                dpeak = None if (peak is None or prev_peak is None) else peak - prev_peak
                prev_peak = peak if peak is not None else prev_peak
                row = {
                    "mode": "throughput",
                    "method": method,
                    "dtype": args.dtype,
                    "dtype_flags": flags,
                    "n_gal": n_gal,
                    "chunk": K,
                    "devices": 1,
                    "n_warmup": args.warmup,
                    "n_samples": args.samples,
                    "max_num_doublings": args.max_doublings,
                    "n_leapfrog": args.n_leapfrog,
                    "cold_s": round(cold, 3),
                    "warm_s": round(warm, 3),
                    "compile_s": round(cold - warm, 3),
                    "gal_per_s": round(n_gal / warm, 3),
                    "gal_per_gpu_min": round(60.0 * n_gal / warm, 1),
                    "s_per_gal": round(warm / n_gal, 5),
                    "divergences": int(cp.diagnostics.get("n_divergent_total", -1)),
                    "peak_bytes": peak,
                    "peak_bytes_delta": dpeak,
                    "lut_grad_error_est": bias,
                    **diag,
                }
                # What the metric bought, read off the fit rather than assumed.
                # ``preconditioned_gal`` is a COUNT, not a flag: a lane whose
                # metric could not be factorized falls back to the identity on
                # its own, so a row can be partly whitened and has to say so.
                if cp.diagnostics.get("whitening_strength") is not None:
                    row["preconditioned_gal"] = cp.diagnostics.get("preconditioned")
                    row["metric_cond_median"] = cp.diagnostics.get("metric_condition_median")
                    row["whitened_cond_median"] = cp.diagnostics.get("whitened_condition_median")
                    row["metric_cond_max"] = cp.diagnostics.get("metric_condition_max")
                    row["whitened_cond_max"] = cp.diagnostics.get("whitened_condition_max")
                if row["min_ess"]:
                    row["s_per_eff_sample"] = round(warm / (row["min_ess"] * n_gal), 6)
                    row["eff_samples_per_gpu_min"] = round(60.0 * row["min_ess"] * n_gal / warm, 1)
                if row["frac_converged"] is not None:
                    # The figure that is actually comparable to a published
                    # "posteriors per GPU-minute": the rate counting only the
                    # galaxies that cleared max split-R-hat < MAX_RHAT with no
                    # divergence. A rate over the whole catalog counts chains
                    # nobody can use.
                    row["converged_gal_per_gpu_min"] = round(
                        60.0 * n_gal * row["frac_converged"] / warm, 1
                    )
                row["converged"] = bool(
                    row["max_rhat"] is not None
                    and row["max_rhat"] < MAX_RHAT
                    and row["divergences"] == 0
                    and row["n_frozen_chains"] == 0
                )
                stamp(row, meta)
                rows.append(row)
                if args.json:
                    # Written after every row, not at the end: a sweep of NUTS
                    # cells is hours long and a row that finished is a result
                    # whether or not the process survives to the last one.
                    write_json(args.json, [row], meta)
                print(
                    f"  {method:<10} {n_gal:>6} {K:>5} {cold:>9.2f} {warm:>9.2f} "
                    f"{cold - warm:>10.2f} "
                    f"{60.0 * n_gal / warm:>11.0f} "
                    f"{row.get('converged_gal_per_gpu_min', float('nan')):>9.0f} "
                    f"{row['n_gal_converged']:>4}/{row['n_gal_checked']:<5} "
                    f"{(row['max_rhat'] if row['max_rhat'] is not None else float('nan')):>9.4f} "
                    f"{(row['min_ess'] if row['min_ess'] is not None else float('nan')):>8.1f} "
                    f"{row['divergences']:>5} "
                    f"{(peak / 2**30 if peak else float('nan')):>8.2f}"
                    + ("" if row["converged"] else "   CATALOG-MAX NON-CONVERGED"),
                    flush=True,
                )

    if args.shard and len(dev) > 1:
        n_gal = biggest
        K = sorted(args.chunk)[len(args.chunk) // 2]
        method = args.method[0]
        # pad-match n_gal to lcm(K, n_dev) already handled internally; just compare.
        cat = CatalogFitter(model, full_catalog[:n_gal], data_type="photometry")
        print(f"\n  device scaling  [{method}, N={n_gal}, K={K}, {len(dev)} devices]")
        _ = _run_and_time(cat, method, K, None, key, run_kw)  # warm the single-device compile
        s1, cp1, _ = _run_and_time(cat, method, K, None, key, run_kw)
        _ = _run_and_time(cat, method, K, "all", key, run_kw)  # warm the sharded compile
        sN, cpN, _ = _run_and_time(cat, method, K, "all", key, run_kw)
        d1 = _diagnostics(cp1, args.diag_max_gal)
        dN = _diagnostics(cpN, args.diag_max_gal)
        print(
            f"  {'single':>8}: {s1:>8.2f} s  ({n_gal / s1:>8.1f} gal/s)  maxRhat {d1['max_rhat']}"
        )
        print(
            f"  {'sharded':>8}: {sN:>8.2f} s  ({n_gal / sN:>8.1f} gal/s)  speedup {s1 / sN:.2f}x"
            f"  maxRhat {dN['max_rhat']}"
        )
        for wall, dd, ndev in ((s1, d1, 1), (sN, dN, len(dev))):
            rows.append(
                {
                    "mode": "throughput",
                    "method": method,
                    "dtype": args.dtype,
                    "dtype_flags": flags,
                    "n_gal": n_gal,
                    "chunk": K,
                    "devices": ndev,
                    "warm_s": round(wall, 3),
                    "gal_per_s": round(n_gal / wall, 3),
                    "gal_per_gpu_min": round(60.0 * n_gal / wall, 1),
                    "peak_bytes": device_peak_bytes(),
                    **stamp({}, meta),
                    **dd,
                    "converged": bool(
                        dd["max_rhat"] is not None
                        and dd["max_rhat"] < MAX_RHAT
                        and dd["n_frozen_chains"] == 0
                    ),
                }
            )
    elif args.shard:
        print(
            "\n  --shard requested but only 1 device visible; "
            "set XLA_FLAGS=--xla_force_host_platform_device_count=N to emulate."
        )

    bad = [r for r in rows if not r.get("converged")]
    if bad:
        print(
            f"\n  {len(bad)}/{len(rows)} rows did NOT clear max split-Rhat < {MAX_RHAT} "
            f"with 0 divergences. Those rows are NOT throughput results "
            f"(bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md)."
        )
    ok = [r for r in rows if r.get("converged") and r.get("s_per_eff_sample")]
    if ok:
        best = min(ok, key=lambda r: r["s_per_eff_sample"])
        print(
            f"  best s/effective-sample among converged rows: {best['method']} "
            f"{best['dtype']} N={best['n_gal']} K={best['chunk']} -> "
            f"{best['s_per_eff_sample']:.3g} s/ESS, {best['gal_per_gpu_min']:.0f} gal/GPU-min"
        )

    if args.json:
        print(f"  wrote {write_json(args.json, rows, meta)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
