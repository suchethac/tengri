"""Component-pair sweep + model-misspecification check.

(a) For 5 (SFH, dust, nebular) combinations: build mock, fit with
    *the same* model via mcmc_hmc, check that posterior recovers
    log10(M*) and log10(SFR_100Myr) within 0.5 dex.

(b) Misspecification: build mock with model A, fit with model B,
    check that derived stellar mass + SFR_100Myr still land within
    0.5 dex of the truth. This is the practical question of "does
    inference give the right mass/SFR even if we picked the wrong
    SFH parametrisation?"

All fits use WavePrecomp() so each takes ≤ 10 s on D ≤ 8.
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
import traceback
from pathlib import Path

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_platforms", "cpu")

from tengri import (
    FIXED,
    FREE,
    Fitter,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    builders,
    load_filter_set,
    load_ssp_data,
)
from tengri.forward.sed_model import WavePrecomp
from tengri.inference._backend_registry import _BACKENDS

DATA = Path(__file__).resolve().parents[1] / "data"
SSP_FILE = DATA / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
FILTERS_NAMES = [
    "sdss_u",
    "sdss_g",
    "sdss_r",
    "sdss_i",
    "sdss_z",
    "2mass_j",
    "2mass_h",
    "2mass_ks",
]
HMC_KW = dict(
    n_warmup=300,
    n_burnin=50,
    n_samples=1000,
    n_leapfrog_steps=10,
    dense_mass_matrix=True,
    verbose=False,
)


# ── model factories ────────────────────────────────────────────────────────


def model_dpl_calzetti():
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
    )


def model_dense_basis_calzetti():
    return dict(
        sfh=builders.sfh.dense_basis(defaults=FREE),
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
    )


def model_tsnorm_calzetti():
    return dict(
        sfh=builders.sfh.tsnorm(defaults=FREE),
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
    )


def model_dpl_smc():
    return dict(
        sfh=builders.sfh.dpl(defaults=FREE),
        dust_attenuation={"type": "two_component", "all_params": FIXED, "law": "smc", "tau_bc": Uniform(0.0, 1.0)},
        neb={"type": "none"},
    )


def model_dexp_calzetti():
    return dict(
        sfh=builders.sfh.dexp(defaults=FREE),
        dust_attenuation={
            "type": "two_component",
            "all_params": FIXED,
            "law": "calzetti",
            "tau_bc": Uniform(0.0, 1.0),
        },
        neb={"type": "none"},
    )


SCENARIOS = {
    "dpl+calzetti": model_dpl_calzetti,
    "dense_basis+calzetti": model_dense_basis_calzetti,
    "tsnorm+calzetti": model_tsnorm_calzetti,
    "dpl+smc": model_dpl_smc,
    "dexp+calzetti": model_dexp_calzetti,
}


def build(scenario_factory):
    ssp = load_ssp_data(str(SSP_FILE))
    _, _, filters = load_filter_set(FILTERS_NAMES)
    obs = Observation(photometry=Photometry(filters=tuple(filters)))
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.05),
        approx=WavePrecomp(),
        **scenario_factory(),
    )


def truth_at_midpoint(model):
    out = {}
    for n in model.spec.free_params:
        p = model.spec.get_distribution(n)
        lo = getattr(p, "low", getattr(p, "lo", None))
        hi = getattr(p, "high", getattr(p, "hi", None))
        if lo is not None and hi is not None:
            out[n] = float(0.5 * (lo + hi))
        else:
            out[n] = float(p.sample(jr.PRNGKey(0)))
    return out


def derived_truth(model, truth: dict) -> dict:
    """log10 M* and log10 SFR_100Myr for the truth parameter set."""
    pred = model.predict(truth)
    sm = float(pred.sfh.stellar_mass)
    sfr = float(pred.sfh.sfr_100myr)
    return {
        "log_stellar_mass": float(np.log10(sm)),
        "log_sfr_100myr": float(np.log10(max(sfr, 1e-30))),
    }


def posterior_derived(post, model) -> dict:
    """Compute log10 M* and log10 SFR_100Myr posterior summaries."""
    samples = post.samples
    if samples is None:
        # MAP/Laplace point
        pred = model.predict(post.params)
        return {
            "log_stellar_mass": {
                "median": float(np.log10(float(pred.sfh.stellar_mass))),
                "p16": None,
                "p84": None,
            },
            "log_sfr_100myr": {
                "median": float(np.log10(max(float(pred.sfh.sfr_100myr), 1e-30))),
                "p16": None,
                "p84": None,
            },
        }
    # Compute per-sample
    free = list(samples.keys())
    n = next(iter(samples.values())).shape[0]
    sm_log = np.empty(n)
    sfr_log = np.empty(n)
    # Subsample to keep this cheap
    idx = np.linspace(0, n - 1, min(200, n), dtype=int)
    for j, i in enumerate(idx):
        params = {k: float(np.asarray(samples[k])[i]) for k in free}
        pred = model.predict(params)
        sm_log[j] = float(np.log10(float(pred.sfh.stellar_mass)))
        sfr_log[j] = float(np.log10(max(float(pred.sfh.sfr_100myr), 1e-30)))
    sm_log = sm_log[: len(idx)]
    sfr_log = sfr_log[: len(idx)]
    return {
        "log_stellar_mass": {
            "median": float(np.median(sm_log)),
            "p16": float(np.percentile(sm_log, 16)),
            "p84": float(np.percentile(sm_log, 84)),
        },
        "log_sfr_100myr": {
            "median": float(np.median(sfr_log)),
            "p16": float(np.percentile(sfr_log, 16)),
            "p84": float(np.percentile(sfr_log, 84)),
        },
    }


# ── child entry: one (truth, fit) pair ────────────────────────────────────


def child_run(truth_key: str, fit_key: str, out_json: str) -> None:
    rec = {"truth_model": truth_key, "fit_model": fit_key}
    try:
        t0 = time.perf_counter()
        truth_model = build(SCENARIOS[truth_key])
        truth = truth_at_midpoint(truth_model)
        if truth_key == fit_key:
            try:
                truth_derived = derived_truth(truth_model, truth)
            except Exception as e:
                truth_derived = {
                    "log_stellar_mass": None,
                    "log_sfr_100myr": None,
                    "_error": f"{type(e).__name__}: {str(e)[:200]}",
                }
        else:
            # Misspec: avoid building two SEDModel instances in one process
            # (JIT cache collision on Uniform.unstandardize). The orchestrator
            # passes truth_derived in via the env / a sidecar JSON.
            sidecar = Path(out_json).with_suffix(".truth.json")
            if sidecar.exists():
                truth_derived = json.loads(sidecar.read_text())
            else:
                truth_derived = {
                    "log_stellar_mass": None,
                    "log_sfr_100myr": None,
                    "_error": "no sidecar",
                }
        mock = truth_model.mock(truth, snr=20.0, key=jr.PRNGKey(42))

        # Drop the truth model so the fit_model has a clean slate.
        del truth_model
        import gc

        gc.collect()
        jax.clear_caches()

        fit_model = build(SCENARIOS[fit_key])
        fitter = Fitter(fit_model, mock.flux_obs, mock.noise)
        rec["n_free_fit"] = fit_model.spec.n_free
        rec["truth_derived"] = truth_derived

        post = _BACKENDS["mcmc_hmc"].runner(fitter, key=jr.PRNGKey(0), **HMC_KW)
        jax.block_until_ready(jnp.asarray(0.0))
        rec["wall_s"] = time.perf_counter() - t0

        try:
            fit_derived = posterior_derived(post, fit_model)
        except Exception as e:
            fit_derived = {
                "log_stellar_mass": {"median": None, "p16": None, "p84": None},
                "log_sfr_100myr": {"median": None, "p16": None, "p84": None},
                "_error": f"{type(e).__name__}: {str(e)[:200]}",
            }
        rec["fit_derived"] = fit_derived

        if (
            truth_derived.get("log_stellar_mass") is not None
            and fit_derived["log_stellar_mass"]["median"] is not None
        ):
            rec["bias_dex"] = {
                "log_stellar_mass": fit_derived["log_stellar_mass"]["median"]
                - truth_derived["log_stellar_mass"],
                "log_sfr_100myr": fit_derived["log_sfr_100myr"]["median"]
                - truth_derived["log_sfr_100myr"],
            }
            rec["within_0.5_dex"] = {k: abs(v) < 0.5 for k, v in rec["bias_dex"].items()}
        else:
            rec["bias_dex"] = None
            rec["within_0.5_dex"] = None
        rec["status"] = "ok"
    except Exception as e:
        rec["status"] = "failed"
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:400]
        rec["traceback"] = traceback.format_exc()[-1200:]
    Path(out_json).write_text(json.dumps(rec, indent=2, default=str))


# ── orchestrator ──────────────────────────────────────────────────────────


def main():
    out_path = Path(__file__).parent / "_backend_component_pairs_results.json"
    tmp = Path("/tmp/validate_231_component_pairs")
    tmp.mkdir(exist_ok=True)
    TIMEOUT = 120

    # (a) self-consistency: each model fits its own mock
    self_pairs = [(k, k) for k in SCENARIOS]
    # (b) misspecification: dense_basis truth, dpl fit, etc.
    mismatch_pairs = [
        ("dense_basis+calzetti", "dpl+calzetti"),
        ("tsnorm+calzetti", "dpl+calzetti"),
        ("dpl+calzetti", "dense_basis+calzetti"),
        ("dpl+smc", "dpl+calzetti"),  # wrong dust law
    ]
    all_pairs = self_pairs + mismatch_pairs

    # Precompute truth_derived for each distinct truth model in its own
    # subprocess (so we never have two models in one process).
    print("=== precomputing truth_derived for each model ===", flush=True)
    truth_derived_cache: dict[str, dict] = {}
    truth_sidecar = tmp / "truth_derived"
    truth_sidecar.mkdir(exist_ok=True)
    for k in SCENARIOS:
        side_path = truth_sidecar / f"{k}.json"
        if side_path.exists():
            side_path.unlink()
        cmd = [sys.executable, __file__, "--truth-only", k, str(side_path)]
        subprocess.run(cmd, timeout=60, check=False)
        if side_path.exists():
            td = json.loads(side_path.read_text())
            truth_derived_cache[k] = td
            print(
                f"  {k:24s} M*={td.get('log_stellar_mass'):.2f}  "
                f"SFR={td.get('log_sfr_100myr'):.2f}",
                flush=True,
            )
        else:
            print(f"  {k:24s} FAILED", flush=True)

    results = []
    print(f"\n=== {len(all_pairs)} (truth, fit) pairs ===", flush=True)
    for truth_key, fit_key in all_pairs:
        same = truth_key == fit_key
        label = "self " if same else "miss "
        print(f"  {label}{truth_key:24s} → {fit_key:24s}", end="", flush=True)
        j = tmp / f"{truth_key}__{fit_key}.json"
        if j.exists():
            j.unlink()
        # Pass truth_derived sidecar for misspec pairs.
        if not same and truth_key in truth_derived_cache:
            sidecar = j.with_suffix(".truth.json")
            sidecar.write_text(json.dumps(truth_derived_cache[truth_key]))
        cmd = [sys.executable, __file__, "--child", truth_key, fit_key, str(j)]
        try:
            subprocess.run(cmd, timeout=TIMEOUT, check=False)
            if j.exists():
                r = json.loads(j.read_text())
            else:
                r = {"truth_model": truth_key, "fit_model": fit_key, "status": "crashed_no_output"}
        except subprocess.TimeoutExpired:
            r = {
                "truth_model": truth_key,
                "fit_model": fit_key,
                "status": "timeout",
                "wall_s": TIMEOUT,
            }
        results.append(r)
        if r["status"] == "ok":
            b = r["bias_dex"]
            ok = r["within_0.5_dex"]
            mark = "✓" if (ok["log_stellar_mass"] and ok["log_sfr_100myr"]) else "✗"
            print(
                f"  {r['wall_s']:5.1f}s  Δlog M*={b['log_stellar_mass']:+.2f}dex  "
                f"ΔlogSFR={b['log_sfr_100myr']:+.2f}dex  {mark}",
                flush=True,
            )
        else:
            print(f"  FAIL[{r['status']}] {r.get('error_type', '')}", flush=True)
        out_path.write_text(json.dumps(results, indent=2, default=str))

    print(f"\nWritten to {out_path}", flush=True)


def truth_only(model_key: str, out_json: str) -> None:
    """Compute log10 M* and log10 SFR_100Myr for a model's prior midpoint."""
    model = build(SCENARIOS[model_key])
    truth = truth_at_midpoint(model)
    out = derived_truth(model, truth)
    Path(out_json).write_text(json.dumps(out, default=str))


def mock_only(model_key: str, out_npz: str) -> None:
    """Generate a mock (flux_obs, noise) — used by misspec to isolate truth model."""
    model = build(SCENARIOS[model_key])
    truth = truth_at_midpoint(model)
    mock = model.mock(truth, snr=20.0, key=jr.PRNGKey(42))
    np.savez(out_npz, flux_obs=np.asarray(mock.flux_obs), noise=np.asarray(mock.noise))


def fit_only(fit_key: str, mock_npz: str, out_json: str) -> None:
    """Fit a pre-generated mock with fit_key model — single SEDModel in process."""
    rec = {"fit_model": fit_key}
    try:
        t0 = time.perf_counter()
        data = np.load(mock_npz)
        flux_obs = jnp.asarray(data["flux_obs"])
        noise = jnp.asarray(data["noise"])

        fit_model = build(SCENARIOS[fit_key])

        # Pre-warm: a single forward pass before HMC's MAP-init.
        # Without this, HMC's JIT-traced MAP step crashes with
        # ConcretizationTypeError in priors.py:Uniform.unstandardize.
        # Tracked as a separate tengri bug; this workaround is enough
        # for the validation harness.
        midpoint = truth_at_midpoint(fit_model)
        _ = fit_model.predict_photometry(midpoint)

        fitter = Fitter(fit_model, flux_obs, noise)
        rec["n_free_fit"] = fit_model.spec.n_free

        post = _BACKENDS["mcmc_hmc"].runner(fitter, key=jr.PRNGKey(0), **HMC_KW)
        jax.block_until_ready(jnp.asarray(0.0))
        rec["wall_s"] = time.perf_counter() - t0

        fit_derived = posterior_derived(post, fit_model)
        rec["fit_derived"] = fit_derived
        rec["status"] = "ok"
    except Exception as e:
        rec["wall_s"] = time.perf_counter() - t0
        rec["status"] = "failed"
        rec["error_type"] = type(e).__name__
        rec["error_msg"] = str(e)[:400]
        rec["traceback"] = traceback.format_exc()[-800:]
    Path(out_json).write_text(json.dumps(rec, indent=2, default=str))


if __name__ == "__main__":
    if len(sys.argv) >= 5 and sys.argv[1] == "--child":
        child_run(sys.argv[2], sys.argv[3], sys.argv[4])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--truth-only":
        truth_only(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 4 and sys.argv[1] == "--mock-only":
        mock_only(sys.argv[2], sys.argv[3])
    elif len(sys.argv) >= 5 and sys.argv[1] == "--fit-only":
        fit_only(sys.argv[2], sys.argv[3], sys.argv[4])
    else:
        main()
