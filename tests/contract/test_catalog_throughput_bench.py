# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for ``bench/scripts/benchmark_catalog_throughput.py``.

The harness is the only place tengri publishes a galaxies-per-GPU-minute
number, and two of its properties are load-bearing rather than cosmetic:

1. **Precision is chosen before ``import jax``.** ``jax_enable_x64`` is
   process-global, and ``tengri/__init__.py`` turns it back **on** at import
   unless ``JAX_ENABLE_X64`` is in the environment (#1840), so a
   ``jax.config.update`` anywhere in ``main()`` leaves a float64 model wearing
   a "float32" label. The switch must be the environment variable, applied
   above the ``import jax`` line.
2. **A throughput row is inseparable from its R-hat / ESS columns.** The house
   rule (``bench/reports/2026-08-17_quickstart_nuts_vs_hmc.md``) is that the
   s/ESS column is a trap without the R-hat column, so ``converged`` must be
   false whenever R-hat is missing, above the bar, divergences are non-zero,
   or a chain never moved.

The script is not an importable module, so it is loaded by path, the same way
``tests/contract/test_notebook_renders.py`` loads its tool.
"""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.contract

SCRIPT = (
    Path(__file__).resolve().parents[2] / "bench" / "scripts" / "benchmark_catalog_throughput.py"
)


@pytest.fixture(scope="module")
def bench():
    if not SCRIPT.exists():  # pragma: no cover - source checkouts only
        pytest.skip(f"{SCRIPT} not present (wheel install)")
    spec = importlib.util.spec_from_file_location("_bench_catalog_throughput", SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ── 1. precision ordering ───────────────────────────────────────────


def test_precision_env_is_applied_before_jax_is_imported() -> None:
    """The only switch that binds is ``JAX_ENABLE_X64``, set before ``import jax``.

    The previous harness did ``jax.config.update("jax_enable_x64", True)`` at
    module scope, which made a ``--dtype`` flag impossible to honor. Moving that
    update into ``main()`` is *also* not enough and is the subtler trap:
    ``tengri/__init__.py`` re-enables x64 on import unless ``JAX_ENABLE_X64`` is
    in the environment (#1840), so a float32 row would silently be a float64 row.
    """
    lines = SCRIPT.read_text().splitlines()
    import_jax = next(i for i, ln in enumerate(lines) if ln.startswith("import jax"))
    call = next(i for i, ln in enumerate(lines) if ln.startswith("_ARGV_DTYPE ="))
    assert call < import_jax, (
        "the precision switch must be applied above `import jax`; JAX latches "
        "JAX_ENABLE_X64 at import and nothing later can move it."
    )


def test_apply_precision_env_sets_the_variables_for_f32(bench, monkeypatch) -> None:
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    monkeypatch.delenv("JAX_DEFAULT_MATMUL_PRECISION", raising=False)
    assert bench._apply_precision_env(["--dtype", "f32"]) == "f32"
    import os

    assert os.environ["JAX_ENABLE_X64"] == "0"
    # Ampere lowers float32 matmuls to TF32 without this; see gpu.md.
    assert os.environ["JAX_DEFAULT_MATMUL_PRECISION"] == "highest"


def test_apply_precision_env_leaves_f64_alone(bench, monkeypatch) -> None:
    monkeypatch.delenv("JAX_ENABLE_X64", raising=False)
    assert bench._apply_precision_env(["--dtype", "f64"]) == "f64"
    import os

    assert "JAX_ENABLE_X64" not in os.environ
    assert bench._apply_precision_env([]) == "f64"
    assert bench._apply_precision_env(["--dtype=f32"]) == "f32"


def test_set_precision_rejects_unknown_dtype(bench) -> None:
    with pytest.raises(ValueError, match="--dtype"):
        bench.set_precision("float32")  # not one of f32 / f64


def test_set_precision_refuses_a_mismatched_process(bench) -> None:
    """Asking for f32 in a float64 process must raise, not quietly report f64."""
    import jax

    if not jax.config.x64_enabled:
        pytest.skip("this process is already in float32")
    with pytest.raises(RuntimeError, match="allocates float64"):
        bench.set_precision("f32")


def test_methods_match_catalog_fitter_vmappable(bench) -> None:
    """The --method choices are exactly the samplers the catalog path vectorizes."""
    from tengri.inference.catalog_fitter import CatalogFitter

    assert set(bench.METHODS) == set(CatalogFitter._MCMC_VMAPPABLE)


# ── 2. the R-hat gate ───────────────────────────────────────────────


class _FakePosterior:
    def __init__(self, rhat, samples, dead=False, n_divergent=0, n_chains=1):
        self._rhat = rhat
        self.samples = samples
        self._dead = dead
        n_draws = len(next(iter(samples.values()))) if samples else 0
        self.diagnostics = {
            "n_divergent": n_divergent,
            "n_samples": n_draws,
            "n_chains": n_chains,
        }

    def rhat(self):
        if self._dead:
            raise ValueError("the chain did not move")
        return self._rhat


class _FakeCatalogPosterior:
    def __init__(self, posteriors, n_divergent=0):
        self.posteriors = posteriors
        self.diagnostics = {"n_divergent_total": n_divergent}


def _chain(n=200, seed=0):
    rng = np.random.default_rng(seed)
    return {"a": rng.normal(size=n), "b": rng.normal(size=n)}


def test_diagnostics_takes_the_worst_rhat_over_the_catalog(bench) -> None:
    cp = _FakeCatalogPosterior(
        [
            _FakePosterior({"a": 1.001, "b": 1.002}, _chain(seed=1)),
            _FakePosterior({"a": 1.004, "b": 1.211}, _chain(seed=2)),
            _FakePosterior({"a": 1.000, "b": 1.003}, _chain(seed=3)),
        ]
    )
    diag = bench._diagnostics(cp, 10_000)
    assert diag["max_rhat"] == pytest.approx(1.211)
    assert diag["max_rhat_param"] == "b"
    assert diag["n_gal_checked"] == 3
    assert diag["n_frozen_chains"] == 0
    assert diag["min_ess"] is not None and diag["min_ess"] > 0
    # Two galaxies below 1.01, one (R-hat 1.211) above.
    assert diag["n_gal_converged"] == 2
    assert diag["n_gal_unconverged"] == 1
    assert diag["frac_converged"] == pytest.approx(2 / 3)


def test_diagnostics_counts_frozen_chains_rather_than_dropping_them(bench) -> None:
    """A frozen chain scores R-hat ~1.0; it must not read as a converged win.

    #2027 Finding 14 measured 3.1 % of galaxies frozen with zero divergences.
    Frozen, converged and unconverged are three disjoint counts here precisely
    so a frozen galaxy cannot be silently absorbed into either of the others.
    """
    cp = _FakeCatalogPosterior(
        [
            _FakePosterior({"a": 1.001}, _chain(seed=1)),
            _FakePosterior({}, _chain(seed=2), dead=True),
        ]
    )
    diag = bench._diagnostics(cp, 10_000)
    assert diag["n_frozen_chains"] == 1
    assert diag["n_gal_converged"] == 1
    assert diag["n_gal_unconverged"] == 0
    assert (
        diag["n_gal_converged"] + diag["n_gal_unconverged"] + diag["n_frozen_chains"]
        == diag["n_gal_checked"]
    )
    assert diag["max_rhat"] == pytest.approx(1.001)


def test_a_galaxy_with_divergences_is_not_counted_converged(bench) -> None:
    cp = _FakeCatalogPosterior(
        [
            _FakePosterior({"a": 1.001}, _chain(seed=1), n_divergent=3),
            _FakePosterior({"a": 1.002}, _chain(seed=2)),
        ]
    )
    diag = bench._diagnostics(cp, 10_000)
    assert diag["n_gal_converged"] == 1
    assert diag["n_gal_unconverged"] == 1


def test_divergence_rate_uses_total_draws_not_n_samples(bench) -> None:
    """#2087: ``n_samples`` is per chain, ``n_divergent`` is summed over chains.

    Dividing one by the other over-reports by the chain count — it read 400 %
    on a 4-chain fit. The rate must come out of ``total_draws``.
    """
    cp = _FakeCatalogPosterior(
        [_FakePosterior({"a": 1.5}, _chain(n=100, seed=1), n_divergent=100, n_chains=4)]
    )
    diag = bench._diagnostics(cp, 10_000)
    # 100 divergences against 4 chains x 100 draws = 25 %, not 100 %.
    assert diag["divergence_rate"] == pytest.approx(0.25)


def test_diag_max_gal_caps_the_inspection(bench) -> None:
    cp = _FakeCatalogPosterior(
        [_FakePosterior({"a": 1.0 + i / 100}, _chain(seed=i)) for i in range(5)]
    )
    assert bench._diagnostics(cp, 2)["n_gal_checked"] == 2
    assert bench._diagnostics(cp, 2)["max_rhat"] == pytest.approx(1.01)


def test_max_rhat_bar_is_the_house_bar(bench) -> None:
    assert bench.MAX_RHAT == 1.01


# ── 2b. the #2090 refusal ───────────────────────────────────────────


class _RefusingCatalogFitter:
    """Stands in for a ``CatalogFitter`` whose sampler refuses a dead warmup."""

    def __init__(self):
        self.calls = 0

    def run(self, *args, **kwargs):
        from tengri.config.exceptions import DeadFitError

        self.calls += 1
        raise DeadFitError(
            "warmup ended >=90% divergent; refusing to sample",
            warmup_divergence_frac=1.0,
            step_size=1.2e-9,
        )


def test_dead_fit_error_becomes_a_row_not_a_crash(bench) -> None:
    """#2090: a driver looping over galaxies must record the failure, not die.

    The catalog-vectorized path cannot raise per galaxy (``run_one`` lives
    inside ``lax.map``), so a refusal fails the whole cell. Either way a sweep
    that propagates it reports nothing at all.
    """
    cat = _RefusingCatalogFitter()
    wall, cp, bias, refused = bench._run_and_time(cat, "mcmc_nuts", 32, None, None, {})
    assert cp is None
    assert bias is None
    assert wall >= 0.0
    assert refused["reason"] == "DeadFitError"
    assert refused["warmup_divergence_frac"] == pytest.approx(1.0)
    assert refused["step_size"] == pytest.approx(1.2e-9)
    assert cat.calls == 1


def test_run_and_time_returns_four_values(bench) -> None:
    """The refusal slot semantics: None on success, dict on refusal.

    _run_and_time must always return exactly (wall_clock, posterior, bias, refused)
    so callers can reliably unpack the result without checking arity. The fourth
    slot (refused) is always present and carries the semantic meaning: None means
    the fit succeeded (posterior and bias are meaningful); a dict means the fit
    was refused (posterior and bias are None, refused describes the failure).
    """
    # Test the refusal path: refused must be non-None and structured
    cat = _RefusingCatalogFitter()
    wall, _cp, _bias, refused = bench._run_and_time(cat, "mcmc_nuts", 32, None, None, {})

    # Refusal case: refused slot must contain failure details
    assert isinstance(wall, (int, float)) and wall >= 0.0
    assert refused is not None, "refusal must populate the refused slot"
    assert isinstance(refused, dict), "refused must be a dict with failure details"
    assert "reason" in refused, "refused dict must explain why the fit was rejected"


# ── 3. JSON bookkeeping ─────────────────────────────────────────────


def test_write_json_merges_on_configuration_key(tmp_path, bench) -> None:
    """A second process (the other dtype) must extend, not clobber."""
    path = tmp_path / "out.json"
    row = dict(mode="throughput", method="mcmc_nuts", dtype="f64", n_gal=64, chunk=32, devices=1)
    bench.write_json(str(path), [dict(row, gal_per_s=1.0)], {"platform": "gpu"})
    bench.write_json(str(path), [dict(row, dtype="f32", gal_per_s=2.0)], {"jax": "0.11.0"})
    payload = json.loads(path.read_text())
    assert payload["meta"] == {"platform": "gpu", "jax": "0.11.0"}
    assert len(payload["rows"]) == 2
    assert {r["dtype"] for r in payload["rows"]} == {"f64", "f32"}


def test_write_json_overwrites_the_same_configuration(tmp_path, bench) -> None:
    path = tmp_path / "out.json"
    row = dict(mode="throughput", method="mcmc_nuts", dtype="f64", n_gal=64, chunk=32, devices=1)
    bench.write_json(str(path), [dict(row, gal_per_s=1.0)], {})
    bench.write_json(str(path), [dict(row, gal_per_s=9.0)], {})
    payload = json.loads(path.read_text())
    assert len(payload["rows"]) == 1
    assert payload["rows"][0]["gal_per_s"] == 9.0


# ── 4. the SNR column every throughput row must carry ───────────────


def test_catalog_snr_reports_the_axis_the_lut_bias_scales_on(bench) -> None:
    """#1671: the LUT bias enters the gradient multiplied by SNR."""
    gal = [
        {"flux_obs": np.array([1.0, 2.0]), "noise": np.array([0.05, 0.1])},
        {"flux_obs": np.array([4.0]), "noise": np.array([0.1])},
    ]
    snr = bench.catalog_snr(gal)
    assert snr["snr_min"] == pytest.approx(20.0)
    assert snr["snr_max"] == pytest.approx(40.0)
    assert snr["snr_median"] == pytest.approx(20.0)


# ── 5. end to end, in a subprocess, on the synthetic SSP ────────────


REPO_ROOT = Path(__file__).resolve().parents[2]
_SSP = REPO_ROOT / "data" / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.mark.slow
def test_grad_mode_honors_dtype_end_to_end(tmp_path) -> None:
    """--dtype f32 must produce a float32 gradient, proven on the output array.

    This is the regression for the trap the harness was written around:
    ``jax.config.update("jax_enable_x64", False)`` inside ``main()`` is undone
    by ``import tengri`` (#1840), so a "float32" row silently ran in float64.
    Proving it on the array's dtype rather than on the config flag is the
    standard ``bench/reports/2026-08-20_cuda_device_matrix.md`` holds itself to.

    Runs from the repo root against the real SSP grid, on CPU. The script's
    portable synthetic grid is deliberately not used here: its flux
    normalization underflows in float32 and the likelihood guard raises, which
    would make this test fail for a reason that has nothing to do with dtype
    plumbing.
    """
    if not SCRIPT.exists():  # pragma: no cover
        pytest.skip("bench script not present")
    if not _SSP.exists():
        pytest.skip(f"{_SSP.name} not on disk")
    out = tmp_path / "grad.json"
    proc = subprocess.run(
        [
            sys.executable,
            str(SCRIPT),
            "--mode",
            "grad",
            "--dtype",
            "f32",
            "--n-gal",
            "4",
            "--reps",
            "1",
            "--runs",
            "2",
            "--json",
            str(out),
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "JAX_PLATFORMS": "cpu"},
        capture_output=True,
        text=True,
        timeout=900,
    )
    assert proc.returncode == 0, proc.stdout + proc.stderr
    payload = json.loads(out.read_text())
    assert payload["rows"], "no rows written"
    for row in payload["rows"]:
        assert row["dtype"] == "f32"
        assert row["logp_dtype"] == "float32"
        assert row["grad_dtype"] == "float32"
        assert row["grad_finite"]
        # The raw-observable gradient is identically zero in float32 (gpu.md);
        # the *posterior* gradient, which is what NUTS calls, must not be.
        assert not row["grad_all_zero"]
    assert payload["meta"]["snr_median"] > 0
