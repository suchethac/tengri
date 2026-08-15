"""Cache / compile / memory probe for tengri using only the public API.

Run as:
    python probe.py <scenario>

Scenarios:
    map          — MAP fit, mock_recovery_minimal recipe (5 free params)
    vi           — vi_native, same recipe
    nuts         — small NUTS (n_warmup=50, n_samples=50), same recipe
    precomp      — MAP with WavePrecomp on, compare wave-grid path
    sf_photo     — star_forming_photometry recipe (Cue nebular, more params)
    repeat       — run MAP 5× in same process, watch RSS for leaks
    cache_warm   — second-run timing test (intended to be invoked twice)

Reports JSON to stdout and an artefact JSON at bench/cache_probe/results/<scenario>_<timestamp>.json
"""

from __future__ import annotations

import argparse
import gc
import json
import os
import resource
import sys
import time
from pathlib import Path

# Note: import tengri inside main() so we can time it.

SSP_PATH = str(
    Path(__file__).resolve().parents[2]
    / "data"
    / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
)
# Bare SSP not available locally → sf_photo scenario relies on download via public API.

PROBE_DIR = Path(__file__).resolve().parent
RESULTS_DIR = PROBE_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def rss_gib() -> float:
    """Peak RSS so far in GiB (macOS reports in bytes)."""
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    # macOS ru_maxrss is bytes, Linux is KiB. Detect by magnitude.
    if r > 10**10:  # > 10 GiB in any unit; almost certainly bytes
        return r / (1024**3)
    if r > 10**7:  # bytes range plausible
        return r / (1024**3)
    return r / (1024**2)  # KiB → GiB


def current_rss_gib() -> float:
    """Current (not peak) RSS in GiB via /proc on linux or ps on darwin."""
    if sys.platform == "darwin":
        out = os.popen(f"ps -o rss= -p {os.getpid()}").read().strip()
        return int(out) / (1024**2) if out else 0.0
    try:
        with open(f"/proc/{os.getpid()}/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024**2)
    except Exception:
        pass
    return 0.0


class Phase:
    def __init__(self, name: str, log: list, tengri_mod=None):
        self.name = name
        self.log = log
        self.t = tengri_mod

    def __enter__(self):
        gc.collect()
        self.t0 = time.perf_counter()
        self.rss0 = current_rss_gib()
        self.cache0 = (
            self.t.cache_size_bytes() if (self.t and hasattr(self.t, "cache_size_bytes")) else 0
        )
        return self

    def __exit__(self, *exc):
        dt = time.perf_counter() - self.t0
        rss1 = current_rss_gib()
        cache1 = (
            self.t.cache_size_bytes() if (self.t and hasattr(self.t, "cache_size_bytes")) else 0
        )
        entry = {
            "phase": self.name,
            "wall_s": round(dt, 3),
            "rss_gib_before": round(self.rss0, 2),
            "rss_gib_after": round(rss1, 2),
            "rss_delta_gib": round(rss1 - self.rss0, 2),
            "cache_delta_mib": round((cache1 - self.cache0) / (1024**2), 1),
        }
        self.log.append(entry)
        print(
            f"[{self.name:32s}] {dt:7.2f}s   RSS {self.rss0:5.2f}→{rss1:5.2f} GiB"
            f"   cache+{entry['cache_delta_mib']:.1f} MiB"
        )


def build_quickstart_model(tengri, recipe_name: str, *, precomp: bool):
    bare_ssp_needed = recipe_name != "mock_recovery_minimal"
    if bare_ssp_needed:
        # Public API: tengri.download_ssp returns a local path, idempotent if already present.
        ssp_path = str(tengri.download_ssp("fsps_prsc_miles_chabrier"))
    else:
        ssp_path = SSP_PATH
    ssp = tengri.load_ssp_data(ssp_path)
    obs = tengri.Observation(
        photometry=tengri.Photometry.from_names(
            ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "wise_w1"]
        )
    )
    recipe = getattr(tengri.recipes, recipe_name)()
    kwargs = dict(ssp_data=ssp, observation=obs, **recipe)
    if precomp:
        kwargs["approx"] = tengri.WavePrecomp()
    return tengri.SEDModel.build(**kwargs)


def make_mock_and_fitter(tengri, model, key_seed: int = 7):
    import jax

    key = jax.random.PRNGKey(key_seed)
    k_truth, k_mock, k_fit = jax.random.split(key, 3)
    truth = model.spec.sample(k_truth)
    mock = tengri.generate_mock(model, truth, key=k_mock, snr=30.0)
    fitter = tengri.Fitter(model, mock["flux_obs"], mock["noise"], data_type="photometry")
    return fitter, truth, mock, k_fit


def run_scenario(name: str) -> dict:
    log: list = []
    meta = {"scenario": name, "pid": os.getpid(), "python": sys.version.split()[0]}

    # Phase: import
    t0 = time.perf_counter()
    rss0 = current_rss_gib()
    import tengri

    log.append(
        {
            "phase": "import_tengri",
            "wall_s": round(time.perf_counter() - t0, 3),
            "rss_gib_before": round(rss0, 2),
            "rss_gib_after": round(current_rss_gib(), 2),
            "cache_size_mib": round(tengri.cache_size_bytes() / (1024**2), 1),
            "cache_enabled": tengri.is_cache_enabled(),
        }
    )
    print(
        f"[import_tengri                  ] "
        f"{log[-1]['wall_s']:7.2f}s   RSS {log[-1]['rss_gib_before']:5.2f}→{log[-1]['rss_gib_after']:5.2f} GiB"
        f"   cache_total={log[-1]['cache_size_mib']:.0f} MiB"
    )

    # Map scenarios to (recipe, method, kwargs, precomp)
    if name == "map":
        recipe, method, kwargs, precomp = "mock_recovery_minimal", "map", {}, False
    elif name == "vi":
        recipe, method, kwargs, precomp = (
            "mock_recovery_minimal",
            "native_vi_nonlinear",
            {"n_iterations": 200},
            False,
        )
    elif name == "nuts":
        recipe, method, kwargs, precomp = (
            "mock_recovery_minimal",
            "mcmc_nuts",
            {"n_warmup": 50, "n_samples": 50, "dense_mass": False},
            False,
        )
    elif name == "precomp":
        recipe, method, kwargs, precomp = "mock_recovery_minimal", "map", {}, True
    elif name == "sf_photo":
        recipe, method, kwargs, precomp = "star_forming_photometry", "map", {}, False
    elif name == "repeat" or name == "cache_warm":
        recipe, method, kwargs, precomp = "mock_recovery_minimal", "map", {}, False
    else:
        raise SystemExit(f"unknown scenario: {name}")

    with Phase("build_model", log, tengri):
        model = build_quickstart_model(tengri, recipe, precomp=precomp)

    with Phase("first_predict (compile)", log, tengri):
        truth = model.spec.sample(__import__("jax").random.PRNGKey(0))
        _ = model.predict_observables(truth)

    with Phase("second_predict (warm)", log, tengri):
        _ = model.predict_observables(truth)

    with Phase("mock+fitter setup", log, tengri):
        fitter, truth, mock, k_fit = make_mock_and_fitter(tengri, model)

    with Phase(f"fit#1 {method}", log, tengri):
        post = fitter.run(method=method, key=k_fit, **kwargs)

    with Phase(f"fit#2 {method} (warm)", log, tengri):
        post2 = fitter.run(method=method, key=k_fit, **kwargs)

    # Leak loop
    if name == "repeat":
        for i in range(4):
            with Phase(f"fit#{i + 3} repeat", log, tengri):
                fitter.run(method=method, key=k_fit, **kwargs)

    result = {"meta": meta, "log": log}

    # Brief posterior sanity
    try:
        keys = list(post.samples.keys()) if hasattr(post, "samples") else []
        result["meta"]["posterior_keys"] = keys[:8]
    except Exception:
        pass

    out_path = RESULTS_DIR / f"{name}_{int(time.time())}.json"
    out_path.write_text(json.dumps(result, indent=2))
    print(f"\n→ {out_path}")
    return result


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("scenario")
    args = p.parse_args()
    run_scenario(args.scenario)
