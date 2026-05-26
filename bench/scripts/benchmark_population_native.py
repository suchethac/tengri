"""Benchmark PopulationFitter: vi_native_linear speed + memory vs vi (NIFTy).

Four checks:
  1. Speed: vi_native_linear vs vi_linear at N=4,10,20 (small budget, both methods)
  2. Speed: vi_native_nonlinear vs vi_nonlinear at N=4,10,20
  3. Chunk-size sweep: native_vi_linear and native_vi_nonlinear at K=1,2,4
     (K=1 nonlinear ~9 GB; K=4 ~36 GB — safe on 48 GB)
  4. Memory flatness: vi_native_linear peak RSS is O(1) in N up to N=500

Each (N, method, K) combination runs in a fresh subprocess so RSS is clean.

Usage:
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_population_native.py
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_population_native.py --smoke
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_population_native.py --chunk-only
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_population_native.py --large
    JAX_PLATFORMS=cpu .venv/bin/python bench/scripts/benchmark_population_native.py --worker 10 native_vi_linear 3 2 1
"""

from __future__ import annotations

import json
import os
import resource
import subprocess
import sys
import time

os.environ.setdefault("JAX_PLATFORMS", "cpu")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SSP_FILE = os.path.join(
    _REPO_ROOT, "data", "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
# MIST/FSPS SSPs include baked-in nebular emission lines in the continuum
# spectrum at line wavelengths, so joint-obs mode can extract line fluxes
# directly via predict_spectrum without a separate nebular backend.
SSP_FILE_JOINT = os.path.join(
    _REPO_ROOT, "data", "ssp_mist_c3k_a_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)

# ─── Worker ──────────────────────────────────────────────────────────────────


def rss_gb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024**3)


def run_worker(
    n_gal: int,
    method: str,
    n_iterations: int,
    n_samples: int,
    forward_chunk_size: int,
    rich_obs: bool = False,
    noise_frac: float = 0.10,
    spec_obs: bool = False,
    joint_obs: bool = False,
) -> None:
    """Run one (N, method, K) combination and print a JSON result line.

    rich_obs : bool
        If True, use 10-band photometry (GALEX FUV/NUV + SDSS ugriz +
        2MASS JHKs) — UV/NIR coverage carries far more PSD information
        than optical-only SDSS, so σ_PSD/τ_PSD constraints actually
        tighten with N. Otherwise: 5-band SDSS (legacy).
    spec_obs : bool
        If True, replace photometry with a spectroscopy window covering
        Hα, Hβ, [OIII], 4000Å break (rest 3000–7500 Å, R≈500 at z=0.1).
        Continuum + emission lines together give direct Myr-scale SFR
        information through Hα/UV (the burstiness signature).
        ``rich_obs`` is ignored when ``spec_obs=True``.
    joint_obs : bool
        If True, append rest-frame line luminosities (Hα, Hβ, [OIII]_5007,
        [OII]_3727) to a rich-obs photometry vector — the "broadband +
        Hα directly" experiment. Implemented by monkey-patching
        ``model.predict_photometry`` so PopulationFitter's photometry
        path consumes the concatenated joint signal. Mutually exclusive
        with ``spec_obs`` (spec wins).
    noise_frac : float
        Fractional noise (default 0.10).
    """
    import jax
    import jax.numpy as jnp

    jax.config.update("jax_enable_x64", True)

    sys.path.insert(0, os.path.join(_REPO_ROOT, "src"))

    from tengri import Fixed, Observation, Parameters, Photometry, SEDModel, Uniform
    from tengri.inference.hierarchical import PopulationFitter
    from tengri.sps.dsps_wrapper import load_ssp_data

    rss0 = rss_gb()
    ssp_data = load_ssp_data(SSP_FILE_JOINT if joint_obs else SSP_FILE)

    band_names: list[str] = []
    spec_npix = 0
    line_names: tuple[str, ...] = ()
    line_wavelengths_arr = jnp.array([])  # populated below for joint mode
    if spec_obs:
        joint_obs = False  # spec wins
    if joint_obs:
        rich_obs = True  # joint requires the rich photometric base.
    if spec_obs:
        # Rest 3000–7500 Å covers Hβ, [OIII], Hα and the 4000Å break.
        # At z=0.1 → observed 3300–8250 Å. R≈500 → log-spaced n_pix ≈ 460.
        from tengri import Observation as _Obs, Spectroscopy as _Spec

        z_fix = 0.1
        wave_rest = jnp.exp(jnp.linspace(jnp.log(3000.0), jnp.log(7500.0), 460))
        wave_obs_grid = wave_rest * (1.0 + z_fix)
        spec_npix = int(wave_obs_grid.shape[0])
        spec_cfg = _Spec(wave_obs=wave_obs_grid, resolution=500.0)
        obs = _Obs(spectroscopy=spec_cfg)
    else:
        if rich_obs:
            band_names = [
                "galex_fuv",
                "galex_nuv",
                "sdss_u",
                "sdss_g",
                "sdss_r",
                "sdss_i",
                "sdss_z",
                "2mass_j",
                "2mass_h",
                "2mass_ks",
            ]
        else:
            band_names = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]
        obs = Observation(photometry=Photometry.from_names(band_names))

    if joint_obs:
        # Direct emission-line luminosity matching: Hα (10 Myr SFR),
        # Hβ (Balmer decrement → dust), [OIII]_5007 (ionization state),
        # [OII]_3727 (cool-warm transition / metallicity).
        line_names = ("Halpha", "Hbeta", "OIII_5007", "OII_3727")
        # Vacuum rest-frame wavelengths in Angstrom (matches LineList).
        line_wavelengths_arr = jnp.array([6564.61, 4862.68, 5008.24, 3727.09])

    def make_spec(psd_sigma, psd_tau_myr):
        kw = dict(
            sfh_tsnorm_log_total_mass=Uniform(8.0, 12.0),
            sfh_tsnorm_peak_lbt_gyr=Uniform(0.5, 12.0),
            sfh_tsnorm_width_gyr=Uniform(0.3, 5.0),
            sfh_tsnorm_skew=Uniform(-3.0, 3.0),
            sfh_tsnorm_trunc=Uniform(1.0, 10.0),
            sfh_field_psd_sigma=Uniform(0.1, 4.0),
            sfh_field_psd_tau_myr=Uniform(1.0, 300.0),
            met_logzsol=Uniform(-2.0, 0.2),
            dust_tau_bc=Uniform(0.0, 2.0),
            dust_tau_diff=Uniform(0.0, 1.5),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.1),
            mean_sfh_type=["tsnorm", "field"],
            n_grid=128,
        )
        # Joint and spec modes both use the wNE SSPs' baked-in nebular
        # emission. We don't enable a separate nebular backend (Cue
        # double-counts; CB19 isn't wired into SEDModel; BakedIn returns
        # empty line predictions). Joint mode extracts line fluxes from
        # the spectrum at line centers directly — see _patch_joint_predict.
        return Parameters(**kw)

    # Line extraction window in observed-frame Angstrom (rest-frame ~30 Å).
    # 41 pixels × 4 lines = 164 wave points per joint call. Continuum is
    # the mean of the two outer pixels; line flux = trapezoid integral
    # of (flux − continuum). Self-consistent against the wNE SSP's own
    # baked-in line shapes.
    z_fix_joint = 0.1
    line_window_aa = 30.0
    line_npix = 41
    line_centers_obs = line_wavelengths_arr * (1.0 + z_fix_joint) if joint_obs else None

    # Pre-build one concatenated wave grid covering all line windows.
    if joint_obs:
        _line_waves_per = [
            jnp.linspace(lam_c - line_window_aa, lam_c + line_window_aa, line_npix)
            for lam_c in line_centers_obs
        ]
        _line_waves_concat = jnp.concatenate(_line_waves_per)
    else:
        _line_waves_concat = None

    def _line_flux_from_spectrum(m: SEDModel, params, mode: str = "auto"):
        """One predict_spectrum call across all line windows; reshape and integrate."""
        spec_all = m.predict_spectrum(params, _line_waves_concat, mode=mode)
        spec_per = spec_all.reshape((line_centers_obs.shape[0], line_npix))
        # Continuum estimate: mean of two outermost pixels per line window.
        cont = 0.5 * (spec_per[:, 0] + spec_per[:, -1])
        wave_per = _line_waves_concat.reshape((line_centers_obs.shape[0], line_npix))
        return jax.vmap(lambda f, w, c: jnp.trapezoid(f - c, w))(spec_per, wave_per, cont)

    def _patch_joint_predict(m: SEDModel) -> SEDModel:
        """Wrap predict_photometry to append integrated line fluxes."""
        if not joint_obs:
            return m
        orig_predict = m.predict_photometry

        def predict_joint(params, mode="auto"):
            phot = orig_predict(params, mode=mode)
            lines = _line_flux_from_spectrum(m, params, mode=mode)
            return jnp.concatenate([phot, lines])

        m.predict_photometry = predict_joint  # type: ignore[assignment]
        return m

    def model_factory(psd_sigma=1.0, psd_tau_myr=50.0):
        spec = make_spec(psd_sigma, psd_tau_myr)
        # wave_chunk_size=64 dramatically shrinks spec HLO (~7×). Only enabled
        # for spec mode where the wave grid is 460 points (= 7×64 + 12).
        # Joint mode's per-galaxy line-window grid is 164 points; chunking
        # there triggers a ConcretizationTypeError in the MAP jit path.
        wcs = 64 if spec_obs else None
        return _patch_joint_predict(SEDModel(spec, ssp_data, observation=obs, wave_chunk_size=wcs))

    # Build mock galaxies
    key = jax.random.PRNGKey(42)
    template_model = model_factory()
    galaxies = []
    for i in range(n_gal):
        k = jax.random.fold_in(key, i)
        true_params = template_model.spec.sample(k)
        true_params["sfh_field_psd_sigma"] = jnp.array(2.0)
        true_params["sfh_field_psd_tau_myr"] = jnp.array(20.0)
        if spec_obs:
            flux = template_model.predict_spectrum(true_params, wave_obs_grid)
            noise = jnp.abs(flux) * noise_frac + 1e-4 * jnp.median(jnp.abs(flux))
        else:
            # Joint mode picks up line fluxes via the patched predict_photometry.
            flux = template_model.predict_photometry(true_params)
            noise = jnp.abs(flux) * noise_frac + 1e-3
        flux_obs = flux + noise * jax.random.normal(k, shape=flux.shape)
        galaxies.append({"flux_obs": flux_obs, "noise": noise})

    rss1 = rss_gb()

    pop = PopulationFitter(
        model_factory,
        galaxies,
        data_type="spectroscopy" if spec_obs else "photometry",
    )

    base_kwargs = dict(
        n_iterations=n_iterations,
        n_samples=n_samples,
        n_posterior_samples=50,
        forward_chunk_size=forward_chunk_size,
        verbose=False,
    )
    if method in ("native_vi_linear", "native_vi_nonlinear"):
        base_kwargs["n_seeds"] = 1

    # ── First run: includes JIT compilation ───────────────────────────────────
    t0 = time.perf_counter()
    post1 = pop.run(method, key=jax.random.PRNGKey(0), **base_kwargs)
    t_first = time.perf_counter() - t0
    rss2 = rss_gb()

    # ── Second run: JIT cache warm, compute only ──────────────────────────────
    # MAP re-runs per galaxy (Python loop) but VI engine is already compiled.
    t1 = time.perf_counter()
    post2 = pop.run(method, key=jax.random.PRNGKey(1), **base_kwargs)
    t_second = time.perf_counter() - t1

    # Convergence telemetry from PopulationPosterior.diagnostics.
    diag = getattr(post2, "diagnostics", {}) or {}
    n_iters_used = int(diag.get("n_iterations", -1))
    n_iters_max = int(diag.get("n_iterations_max", n_iterations))
    best_h = float(diag.get("best_hamiltonian", float("nan")))
    converged = (n_iters_used >= 0) and (n_iters_used < n_iters_max)
    diag1 = getattr(post1, "diagnostics", {}) or {}
    n_iters_used_cold = int(diag1.get("n_iterations", -1))

    # Hyperparameter constraint summary (warm run).
    # Truth injected per galaxy in the mock loop above: σ=2.0, τ=20 Myr.
    import numpy as _np

    shared = getattr(post2, "shared_samples", {}) or {}

    def _summary(key: str) -> dict:
        if key not in shared:
            return {}
        arr = _np.asarray(shared[key])
        return {
            "median": float(_np.median(arr)),
            "mean": float(_np.mean(arr)),
            "std": float(_np.std(arr)),
            "p16": float(_np.percentile(arr, 16)),
            "p84": float(_np.percentile(arr, 84)),
        }

    psd_sigma_summary = _summary("psd_sigma")
    psd_tau_summary = _summary("psd_tau_myr")

    # compile_s ≈ t_first - t_second (rough: MAP also speeds up on second run).
    compile_s = max(0.0, t_first - t_second)

    print(
        json.dumps(
            {
                "n_gal": n_gal,
                "method": method,
                "forward_chunk_size": forward_chunk_size,
                "wall_s": round(t_first, 2),
                "wall_s_warm": round(t_second, 2),
                "compile_s_approx": round(compile_s, 2),
                "rss_baseline_gb": round(rss0, 2),
                "rss_after_data_gb": round(rss1, 2),
                "rss_after_run_gb": round(rss2, 2),
                "rss_delta_gb": round(rss2 - rss0, 2),
                "n_iterations": n_iterations,
                "n_samples": n_samples,
                "n_iters_used_warm": n_iters_used,
                "n_iters_used_cold": n_iters_used_cold,
                "n_iters_max": n_iters_max,
                "converged": bool(converged),
                "best_hamiltonian": round(best_h, 3) if best_h == best_h else None,
                "truth_psd_sigma": 2.0,
                "truth_psd_tau_myr": 20.0,
                "psd_sigma_summary": psd_sigma_summary,
                "psd_tau_summary": psd_tau_summary,
                "rich_obs": bool(rich_obs),
                "spec_obs": bool(spec_obs),
                "joint_obs": bool(joint_obs),
                "n_bands": len(band_names),
                "n_lines": len(line_names),
                "line_names": list(line_names),
                "spec_npix": spec_npix,
                "noise_frac": float(noise_frac),
            }
        ),
        flush=True,
    )


# ─── Driver ──────────────────────────────────────────────────────────────────


def spawn(
    n_gal: int,
    method: str,
    n_iterations: int,
    n_samples: int,
    forward_chunk_size: int = 1,
    compile_timeout: int | None = 600,
    rich_obs: bool = False,
    noise_frac: float = 0.10,
    spec_obs: bool = False,
    joint_obs: bool = False,
) -> dict:
    """Run one worker subprocess, returning its JSON result.

    Parameters
    ----------
    compile_timeout : int or None
        Seconds before the subprocess is killed. Protects against XLA compile
        hangs at large N or K. None disables the timeout.
    """
    env = {**os.environ, "JAX_PLATFORMS": "cpu"}
    cmd = [
        sys.executable,
        __file__,
        "--worker",
        str(n_gal),
        method,
        str(n_iterations),
        str(n_samples),
        str(forward_chunk_size),
        str(int(bool(rich_obs))),
        f"{noise_frac:.4f}",
        str(int(bool(spec_obs))),
        str(int(bool(joint_obs))),
    ]
    # Use Popen + own session so we can SIGKILL the whole process group when
    # subprocess.run's timeout fails (which it does for hung XLA workers on
    # macOS — multithreaded BLAS sometimes blocks the default kill path).
    import signal
    import threading

    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        env=env,
        start_new_session=True,
    )

    # Watchdog thread: polls worker RSS every 5s, SIGKILLs process group if
    # it crosses MEM_LIMIT_GB. Default 30 GB; override via WORKER_MEM_LIMIT_GB
    # env var. Set the flag so the parent records 'OOM-prevented' instead of
    # mistaking it for a normal timeout.
    mem_limit_gb = float(os.environ.get("WORKER_MEM_LIMIT_GB", "30.0"))
    oom_killed = {"flag": False, "rss_gb": 0.0}
    stop_watchdog = threading.Event()

    def _watchdog():
        import resource as _resource  # noqa: F401  (parity with worker units)

        while not stop_watchdog.is_set():
            try:
                ps = subprocess.run(
                    ["ps", "-o", "rss=", "-p", str(proc.pid)],
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                rss_kb = int(ps.stdout.strip() or "0")
            except (subprocess.TimeoutExpired, ValueError, ProcessLookupError):
                rss_kb = 0
            rss_gb = rss_kb / (1024.0 * 1024.0)
            if rss_gb > mem_limit_gb:
                oom_killed["flag"] = True
                oom_killed["rss_gb"] = rss_gb
                try:
                    os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
                return
            stop_watchdog.wait(5.0)

    wd = threading.Thread(target=_watchdog, daemon=True)
    wd.start()

    try:
        out, err = proc.communicate(timeout=compile_timeout)
        stdout_s, stderr_s, returncode = out, err, proc.returncode
    except subprocess.TimeoutExpired:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        try:
            out, err = proc.communicate(timeout=10)
        except subprocess.TimeoutExpired:
            out, err = "", ""
        stop_watchdog.set()
        return {
            "n_gal": n_gal,
            "method": method,
            "forward_chunk_size": forward_chunk_size,
            "wall_s": -1,
            "rss_delta_gb": -1,
            "error": f"TIMEOUT after {compile_timeout}s (likely XLA compile hang)",
        }
    finally:
        stop_watchdog.set()

    if oom_killed["flag"]:
        return {
            "n_gal": n_gal,
            "method": method,
            "forward_chunk_size": forward_chunk_size,
            "wall_s": -1,
            "rss_delta_gb": round(oom_killed["rss_gb"], 2),
            "error": (
                f"OOM-PREVENTED: worker exceeded {mem_limit_gb:.0f} GB "
                f"(peak {oom_killed['rss_gb']:.1f} GB), SIGKILLed by watchdog."
            ),
        }

    class _ProcShim:
        pass

    proc_shim = _ProcShim()
    proc_shim.stdout = stdout_s
    proc_shim.stderr = stderr_s
    proc_shim.returncode = returncode
    proc = proc_shim
    for line in proc.stdout.splitlines():
        line = line.strip()
        if line.startswith("{"):
            return json.loads(line)
    # If no JSON found, return error record
    return {
        "n_gal": n_gal,
        "method": method,
        "forward_chunk_size": forward_chunk_size,
        "wall_s": -1,
        "rss_delta_gb": -1,
        "error": proc.stderr[-500:] if proc.stderr else "no output",
    }


def _run_chunk_sweep(chunk_ns: list[int], chunk_ks: list[int], n_iter: int, n_samp: int) -> None:
    """Section 3: forward_chunk_size sweep for both native methods."""
    for sect, method in (("3a", "native_vi_linear"), ("3b", "native_vi_nonlinear")):
        print(f"\n{sect}. forward_chunk_size sweep: {method} K={chunk_ks} at N={chunk_ns}\n")
        chunk_hdr = (
            f"  {'N':>4}  {'K':>2}  {'cold (s)':>9}  {'warm (s)':>9}"
            f"  {'compile~(s)':>12}  {'ΔRSS (GB)':>10}  {'warm vs K=1':>11}"
        )
        print(chunk_hdr)
        print("  " + "-" * 65)

        k1_warm: dict[int, float] = {}

        for n in chunk_ns:
            for k in chunk_ks:
                if n % k != 0:
                    print(f"  {n:>4}  {k:>2}  (skipped: {n} % {k} != 0)")
                    continue
                print(f"  Running N={n}/K={k} {method}...", flush=True)
                row = spawn(n, method, n_iter, n_samp, forward_chunk_size=k)
                err = row.get("error", "")
                if err:
                    print(f"  {n:>4}  {k:>2}  ERROR: {err[:60]}")
                    continue
                wall = row.get("wall_s", -1)
                warm = row.get("wall_s_warm", -1)
                comp = row.get("compile_s_approx", -1)
                delta = row.get("rss_delta_gb", -1)
                if k == 1:
                    k1_warm[n] = warm
                rel = ""
                if warm > 0 and n in k1_warm and k1_warm[n] > 0:
                    ratio = k1_warm[n] / warm
                    rel = f"  {ratio:>8.2f}x"
                print(
                    f"  {n:>4}  {k:>2}  {wall:>9.1f}  {warm:>9.1f}"
                    f"  {comp:>12.1f}  {delta:>10.2f}{rel}"
                )


def _run_large() -> None:
    """Large-N scaling: N=128..1024 at K=1, properly converged posterior.

    Uses full convergence budget (n_iterations=20, n_samples=6) to measure
    realistic wall-clock and peak memory. K=1 throughout (O(1) peak memory;
    chunk parallelism is benchmarked separately in _run_chunk_sweep).
    Compile timeout: 900s per worker.
    """
    large_ns = [128, 256, 512, 1024]
    n_iter, n_samp = 20, 6
    timeout = 900

    print(f"\n4. Large-N scaling: native_vi_linear K=1 at N={large_ns}")
    print(f"   Convergence budget: n_iterations={n_iter}, n_samples={n_samp}")
    print(f"   compile timeout: {timeout}s per worker\n")
    hdr = f"  {'N':>5}  {'cold (s)':>9}  {'warm (s)':>9}  {'compile~(s)':>12}  {'ΔRSS (GB)':>10}"
    print(hdr)
    print("  " + "-" * 52)

    n1_warm: float | None = None

    for n in large_ns:
        print(f"  Running N={n}...", flush=True)
        row = spawn(
            n, "native_vi_linear", n_iter, n_samp, forward_chunk_size=1, compile_timeout=timeout
        )
        err = row.get("error", "")
        if err:
            print(f"  {n:>5}  ERROR: {err[:70]}")
            continue
        wall = row.get("wall_s", -1)
        warm = row.get("wall_s_warm", -1)
        comp = row.get("compile_s_approx", -1)
        delta = row.get("rss_delta_gb", -1)
        if n1_warm is None:
            n1_warm = warm
        print(f"  {n:>5}  {wall:>9.1f}  {warm:>9.1f}  {comp:>12.1f}  {delta:>10.2f}")


def main(smoke: bool = False) -> None:
    if smoke:
        speed_ns = [4, 10]
        memory_ns = [4, 10, 20]
        # Chunk sweep N must be divisible by max(K)=4
        chunk_ns = [4, 8]
        n_iter, n_samp = 3, 2
    else:
        speed_ns = [4, 10, 20]
        memory_ns = [4, 10, 20, 100, 500]
        chunk_ns = [4, 8, 20]
        n_iter, n_samp = 6, 3

    chunk_ks = [1, 2, 4]

    print("=" * 70)
    print("PopulationFitter benchmark: apples-to-apples VI comparisons")
    print("=" * 70)

    # ── 1a. MGVI: native_vi_linear vs vi_linear ───────────────────────────────
    # Same model (flat σ_PSD/τ_PSD), different backends: pure-JAX vs NIFTy.
    hdr = f"  {'N':>4}  {'method':<26}  {'cold (s)':>9}  {'warm (s)':>9}  {'compile~(s)':>12}  {'ΔRSS (GB)':>10}"
    sep = "  " + "-" * 76

    def _print_row(n, method, row, extra=""):
        err = row.get("error", "")
        if err:
            print(f"  {n:>4}  {method:<26}  ERROR: {err[:60]}")
        else:
            wall = row.get("wall_s", -1)
            warm = row.get("wall_s_warm", -1)
            comp = row.get("compile_s_approx", -1)
            delta = row.get("rss_delta_gb", -1)
            print(
                f"  {n:>4}  {method:<26}  {wall:>9.1f}  {warm:>9.1f}"
                f"  {comp:>12.1f}  {delta:>10.2f}{extra}"
            )

    print(f"\n1a. MGVI flat-param: native_vi_linear vs vi_linear at N={speed_ns}\n")
    print(hdr)
    print(sep)

    mgvi_rows: list[dict] = []
    for n in speed_ns:
        for method in ("native_vi_linear", "vi_linear"):
            print(f"  Running N={n}/{method}...", flush=True)
            row = spawn(n, method, n_iter, n_samp)
            mgvi_rows.append(row)
            _print_row(n, method, row)

    # ── 1b. geoVI: native_vi_nonlinear vs vi_nonlinear ───────────────────────
    # Same model (flat σ_PSD/τ_PSD), different backends: pure-JAX vs NIFTy.
    print(f"\n1b. geoVI flat-param: native_vi_nonlinear vs vi_nonlinear at N={speed_ns}\n")
    print(hdr)
    print(sep)

    geovi_rows: list[dict] = []
    for n in speed_ns:
        for method in ("native_vi_nonlinear", "vi_nonlinear"):
            print(f"  Running N={n}/{method}...", flush=True)
            row = spawn(n, method, n_iter, n_samp)
            geovi_rows.append(row)
            _print_row(n, method, row)

    # ── 2. Memory flatness ────────────────────────────────────────────────────
    for label, method in (
        ("native_vi_linear", "native_vi_linear"),
        ("native_vi_nonlinear", "native_vi_nonlinear"),
    ):
        print(
            f"\n2{'a' if method == 'native_vi_linear' else 'b'}. Memory flatness: {label} at N={memory_ns}\n"
        )
        print(
            f"  {'N':>4}  {'cold (s)':>9}  {'warm (s)':>9}  {'compile~(s)':>12}  {'ΔRSS (GB)':>12}  {'flat?':>6}"
        )
        print("  " + "-" * 60)

        mem_rows: list[dict] = []
        for n in memory_ns:
            print(f"  Running N={n} {label}...", flush=True)
            row = spawn(n, method, n_iter, n_samp)
            mem_rows.append(row)
            err = row.get("error", "")
            if err:
                print(f"  {n:>4}  ERROR: {err[:60]}")
            else:
                wall = row.get("wall_s", -1)
                warm = row.get("wall_s_warm", -1)
                comp = row.get("compile_s_approx", -1)
                delta = row.get("rss_delta_gb", -1)
                flat = "YES" if delta < 12.0 else "NO"
                print(
                    f"  {n:>4}  {wall:>9.1f}  {warm:>9.1f}  {comp:>12.1f}  {delta:>12.2f}  {flat:>6}"
                )

        valid_mem = [r for r in mem_rows if r.get("rss_delta_gb", -1) > 0]
        if len(valid_mem) >= 2:
            deltas = [r["rss_delta_gb"] for r in valid_mem]
            spread = max(deltas) - min(deltas)
            flat = spread < 2.0
            print(f"  → ΔRSS spread: {spread:.2f} GB  |  O(1) in N: {'YES' if flat else 'NO'}")

    # ── 3. Chunk-size sweep (both native methods, K=1/2/4) ───────────────────
    # All N values must be divisible by max(K)=4 so padding is zero across all K.
    # At K=1 native_vi_nonlinear uses ~9 GB; K=4 uses ~36 GB — safe on 48 GB.
    _run_chunk_sweep(chunk_ns, chunk_ks, n_iter, n_samp)

    # ── Summary ───────────────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)

    all_rows = mgvi_rows + geovi_rows
    for family, native_m, nifty_m in (
        ("MGVI", "native_vi_linear", "vi_linear"),
        ("geoVI", "native_vi_nonlinear", "vi_nonlinear"),
    ):
        print(
            f"\nSpeedup {family}: {native_m} vs {nifty_m} (same flat-param model, JAX vs NIFTy):"
        )
        for n in speed_ns:
            native_row = next(
                (r for r in all_rows if r["n_gal"] == n and r["method"] == native_m), None
            )
            nifty_row = next(
                (r for r in all_rows if r["n_gal"] == n and r["method"] == nifty_m), None
            )
            if native_row and nifty_row and native_row["wall_s"] > 0 and nifty_row["wall_s"] > 0:
                ratio = nifty_row["wall_s"] / native_row["wall_s"]
                print(
                    f"  N={n:>3}: {ratio:.1f}x  "
                    f"(native={native_row['wall_s']:.1f}s, nifty={nifty_row['wall_s']:.1f}s)"
                )


if __name__ == "__main__":
    if len(sys.argv) >= 4 and sys.argv[1] == "--worker":
        n_gal = int(sys.argv[2])
        method = sys.argv[3]
        n_iter = int(sys.argv[4]) if len(sys.argv) > 4 else 6
        n_samp = int(sys.argv[5]) if len(sys.argv) > 5 else 3
        fcs = int(sys.argv[6]) if len(sys.argv) > 6 else 1
        rich = bool(int(sys.argv[7])) if len(sys.argv) > 7 else False
        nf = float(sys.argv[8]) if len(sys.argv) > 8 else 0.10
        spec = bool(int(sys.argv[9])) if len(sys.argv) > 9 else False
        joint = bool(int(sys.argv[10])) if len(sys.argv) > 10 else False
        run_worker(
            n_gal,
            method,
            n_iter,
            n_samp,
            fcs,
            rich_obs=rich,
            noise_frac=nf,
            spec_obs=spec,
            joint_obs=joint,
        )
    elif "--chunk-only" in sys.argv:
        smoke = "--smoke" in sys.argv
        skip_k1 = "--skip-k1" in sys.argv
        # All N must be divisible by 4
        chunk_ns = [4, 8] if smoke else [4, 8, 128, 256, 512, 1024]
        n_iter, n_samp = (3, 2) if smoke else (6, 3)
        chunk_ks = [2, 4] if skip_k1 else [1, 2, 4]
        _run_chunk_sweep(chunk_ns, chunk_ks, n_iter, n_samp)
    elif "--large" in sys.argv:
        _run_large()
    else:
        smoke = "--smoke" in sys.argv
        main(smoke=smoke)
