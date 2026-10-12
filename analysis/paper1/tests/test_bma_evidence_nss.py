# SPDX-License-Identifier: BSD-3-Clause
"""The runner's ``--route nss``: same model and data, nested-sampling evidence.

No SSP data and no real fit: the model builder, the forward model and the data
classes are replaced by fakes, so what is tested is the runner's own code path
(fit call, cell assembly, derived draws, output directory) and nothing else.

Pinned invariants:
- ``route="nss"`` calls ``forward.fit(method="nss", profile_mass=False, ...)`` with the
  validated settings and runs no MAP/Laplace;
- the NSS cell records ``route: "nss"``, ``log_evidence_err``, the NSS bookkeeping and
  ``null`` Hessian fields, and ships the derived-draw NPZ from the NSS samples;
- the NSS cell goes to ``bma_evidence_nss``, the Laplace cell to ``bma_evidence``, and
  the Laplace route is unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar

import numpy as np
import pytest

PAPER1 = Path(__file__).resolve().parents[1]
ANALYSIS = PAPER1.parent
for entry in (str(ANALYSIS), str(PAPER1)):
    if entry not in sys.path:
        sys.path.insert(0, entry)

import paper1.bma_evidence as be
from paper1 import _bma_keys as bk

pytestmark = pytest.mark.unit

GALAXY = 79
N_DRAWS = 700  # more than the 500-draw cap, to pin the cap
MODEL = {
    "sfh": "dpl",
    "ssp": bk.SSP_LABELS[0],
    "attenuation": "calzetti",
    "dust_emission": bk.DUST_EMISSION,
    "nebular": bk.NEBULAR,
}


class _FakeSED:
    available_properties = ("stellar_mass", "stellar_mass_surviving", "sfr_100myr", "sfr_10myr")
    spec = SimpleNamespace(free_params=["x"])

    def predict_properties(self, sample, names):
        import jax.numpy as jnp

        # As tengri does: a call-time override of a Fixed parameter is refused.
        extra = set(sample) - set(self.spec.free_params)
        if extra:
            raise ValueError(f"cannot override Fixed parameters {sorted(extra)}")
        return {n: jnp.exp(sample["x"]) for n in names}


class _FakeForward:
    """Records every fit call; answers per method."""

    calls: ClassVar[list[dict]] = []

    def fit(self, data, *, key, method, **kwargs):
        type(self).calls.append({"method": method, **kwargs})
        samples = {"x": np.linspace(0.0, 1.0, N_DRAWS)}
        if method == "map":
            return SimpleNamespace(diagnostics={"final_loss": 5.0}, samples=None)
        if method == "laplace":
            diagnostics = {
                "newton_decrement": 0.01,
                "n_clipped_eigenvalues": 0,
                "condition_number": 1e3,
            }
            return SimpleNamespace(log_evidence=-12.0, diagnostics=diagnostics, samples=samples)
        assert method == "nss"
        diagnostics = {
            "n_live": 500,
            "n_live_effective": 431.5,
            "n_iterations": 812,
            "n_dead": 40600,
            "information_nats": 21.0,
            "ess": 1500.0,
            "log_evidence_err": 0.15,
        }
        return SimpleNamespace(log_evidence=-10.5, diagnostics=diagnostics, samples=samples)


@pytest.fixture
def fakes(monkeypatch):
    import paper1.bma_space as bma_space
    import paper1.configs as configs

    import tengri

    _FakeForward.calls = []
    monkeypatch.setattr(configs, "load_ssp_for", lambda config: object())
    monkeypatch.setattr(bma_space, "build_model", lambda *args, **kwargs: _FakeSED())
    monkeypatch.setattr(tengri.Photometry, "from_names", staticmethod(lambda names: object()))
    monkeypatch.setattr(tengri, "Observation", lambda **kwargs: object())
    monkeypatch.setattr(tengri, "Data", lambda **kwargs: SimpleNamespace(**kwargs))
    monkeypatch.setattr(
        tengri.ForwardModel, "build", classmethod(lambda cls, **kw: _FakeForward())
    )
    return _FakeForward


def _run(route: str):
    return be.fit_one_model(
        GALAXY,
        MODEL,
        1.0,
        {"names": ["hst_f814w"]},
        np.array([1e-27]),
        np.array([1e-28]),
        n_restarts=2,
        seed=7,
        route=route,
    )


def test_nss_route_fits_with_the_validated_settings_and_no_map(fakes):
    result, _ = _run("nss")
    assert result["error"] is None
    assert [c["method"] for c in fakes.calls] == ["nss"]
    call = fakes.calls[0]
    assert call["profile_mass"] is False
    assert {k: call[k] for k in ("preset", "n_live", "num_delete", "max_iterations")} == {
        "preset": "accurate",
        "n_live": 500,
        "num_delete": 50,
        "max_iterations": 10000,
    }


def test_nss_cell_schema(fakes):
    result, npz = _run("nss")
    assert result["route"] == "nss"
    assert result["log_evidence"] == pytest.approx(-10.5)
    assert result["log_evidence_err"] == pytest.approx(0.15)
    assert result["n_live_effective"] == pytest.approx(431.5)
    assert result["n_iterations"] == 812 and result["n_dead"] == 40600
    for hessian_field in ("newton_decrement", "n_clipped_eigenvalues", "condition_number"):
        assert hessian_field in result and result[hessian_field] is None
    assert result["valid"] is True
    assert result["profile_mass"] is False
    assert result["systematic_floor_frac"] == 0.05
    assert result["model_key"] == bk.model_key(MODEL)
    assert set(npz) == {
        "log_stellar_mass_formed",
        "log_stellar_mass_survived",
        "log_sfr_100myr",
        "log_sfr_10myr",
    }
    assert all(v.shape == (500,) for v in npz.values())
    assert npz["log_sfr_10myr"][0] == pytest.approx(0.0)  # log10(exp(0))


def test_derived_draws_survive_fixed_parameters_in_the_nss_samples(fakes, monkeypatch):
    """The NSS posterior carries every Fixed parameter beside the free ones.

    predict_properties refuses a call-time override of a Fixed parameter, so the
    derived draws must be evaluated on the free parameters only — otherwise the
    cell is written without its NPZ and the failure is only a log warning.
    """
    original = fakes.fit

    def with_fixed(self, data, *, key, method, **kwargs):
        posterior = original(self, data, key=key, method=method, **kwargs)
        if method == "nss":
            n = posterior.samples["x"].shape[0]
            posterior.samples = {**posterior.samples, "redshift": np.full(n, 1.2)}
        return posterior

    monkeypatch.setattr(fakes, "fit", with_fixed)
    _, npz = _run("nss")
    assert npz is not None
    assert all(v.shape == (500,) and np.isfinite(v).all() for v in npz.values())


def test_nss_cell_with_large_error_is_not_valid(fakes, monkeypatch):
    original = fakes.fit

    def noisy(self, data, *, key, method, **kwargs):
        post = original(self, data, key=key, method=method, **kwargs)
        post.diagnostics["log_evidence_err"] = bk.NSS_MAX_LOG_EVIDENCE_ERR + 0.1
        return post

    monkeypatch.setattr(fakes, "fit", noisy)
    result, _ = _run("nss")
    assert result["valid"] is False and result["log_evidence"] == pytest.approx(-10.5)


def test_nss_guard_raise_is_recorded_not_swallowed_into_a_cell_with_evidence(fakes, monkeypatch):
    def refuse(self, data, *, key, method, **kwargs):
        raise RuntimeError("NSS hit the iteration cap")

    monkeypatch.setattr(fakes, "fit", refuse)
    result, npz = _run("nss")
    assert result["route"] == "nss" and result["valid"] is False
    assert result["log_evidence"] is None and "iteration cap" in result["error"]
    assert npz is None


def test_laplace_route_is_unchanged(fakes):
    result, npz = _run("laplace")
    assert [c["method"] for c in fakes.calls] == ["map", "map", "laplace"]
    assert result["route"] == "laplace" and result["log_evidence"] == pytest.approx(-12.0)
    assert result["newton_decrement"] == pytest.approx(0.01)
    assert result["n_clipped_eigenvalues"] == 0 and result["valid"] is True
    assert "log_evidence_err" not in result
    assert npz is not None and npz["log_stellar_mass_formed"].shape == (500,)


def test_default_output_dirs_are_separate_per_route():
    assert be.DEFAULT_OUT["laplace"].endswith("results/bma_evidence")
    assert be.DEFAULT_OUT["nss"].endswith("results/bma_evidence_nss")


@pytest.mark.parametrize("route", ["laplace", "nss"])
def test_main_writes_the_cell_to_its_route_directory(fakes, monkeypatch, tmp_path, route):
    monkeypatch.setitem(be.DEFAULT_OUT, "laplace", str(tmp_path / "lap"))
    monkeypatch.setitem(be.DEFAULT_OUT, "nss", str(tmp_path / "nss"))
    monkeypatch.setattr(
        be,
        "get_galaxy_data",
        lambda gid: (1.0, {"names": ["hst_f814w"]}, np.array([1e-27]), np.array([1e-28])),
    )
    key = bk.model_key(MODEL)
    argv = ["bma_evidence.py", "--galaxy", str(GALAXY), "--set", "factorial", "--models", key]
    monkeypatch.setattr(sys, "argv", [*argv, "--route", route])
    assert be.main() == 0

    other = "nss" if route == "laplace" else "lap"
    mine = tmp_path / ("nss" if route == "nss" else "lap") / str(GALAXY)
    assert (mine / f"{key}.json").exists() and (mine / f"{key}.npz").exists()
    assert not (tmp_path / other).exists()
    assert json.loads((mine / f"{key}.json").read_text())["route"] == route
