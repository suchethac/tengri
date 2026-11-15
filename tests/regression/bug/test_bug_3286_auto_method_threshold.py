# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #3286: the auto-method threshold has one source.

``method="auto"`` and ``method="mcmc"`` choose NUTS (low D) or a high-D sampler
by comparing D with ``[inference] mcmc_auto_d`` from the defaults file. Before
the fix, the fitter caught read errors and missing keys and silently used a
hard-coded ``20`` instead, so an unreadable config changed the sampler without
a word.

* (a) a configured threshold is the one auto-selection follows;
* (b) a defaults read that fails (unreadable file, absent file, absent key)
  raises ``ConfigError`` rather than falling back to ``20``.

Dispatch is observed at the backend registry with probe runners, so no
sampler runs.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import jax
import jax.numpy as jnp
import pytest

from tengri.config.exceptions import ConfigError
from tengri.inference import _backend_registry as registry
from tengri.inference.fitter import Fitter
from tengri.parameters import defaults

pytestmark = pytest.mark.regression_bug

_N_LATENT = 10  # sits between a configured threshold of 5 and one of 50


def _mock_fitter() -> Fitter:
    spec = MagicMock()
    spec.eline_broad = False
    spec.n_free = _N_LATENT
    spec.n_latent = _N_LATENT
    spec.free_params = [f"p{i}" for i in range(_N_LATENT)]
    model = MagicMock()
    model.spec = spec
    model.predict_photometry.return_value = jnp.ones(3) * 1e-18
    model.observation = None
    noise = jnp.ones(3) * 1e-19
    data = jnp.ones(3) * 1e-18
    return Fitter(model, data, noise, data_type="photometry")


@pytest.fixture
def probes(monkeypatch):
    """Replace the sampler backends with probes that count dispatches."""
    calls: dict[str, int] = {"mcmc_nuts": 0, "mcmc_raytrace": 0, "vi_nonlinear_fast": 0}

    def make_runner(name):
        def runner(context, *, key, init_from=None, **kw):
            calls[name] += 1
            return MagicMock(name=f"{name}_posterior")

        return runner

    for name in calls:
        monkeypatch.setitem(
            registry._BACKENDS,
            name,
            registry.BackendEntry(
                name=name,
                runner=make_runner(name),
                tier="primary",
                short_doc="",
                requires=(),
                legacy_fitter=False,
            ),
        )
    return calls


def _write_user_defaults(tmp_path, monkeypatch, text: str):
    path = tmp_path / "defaults.toml"
    path.write_text(text)
    monkeypatch.setattr(defaults, "_USER_DEFAULTS", path)
    return path


# ── (a) the configured threshold drives auto-selection ───────────────


def test_auto_follows_configured_threshold_low(tmp_path, monkeypatch, probes):
    """D=10 with mcmc_auto_d=5 is above the threshold: auto picks geoVI."""
    _write_user_defaults(tmp_path, monkeypatch, "[inference]\nmcmc_auto_d = 5\n")
    _mock_fitter().run("auto", key=jax.random.PRNGKey(0))
    assert probes["vi_nonlinear_fast"] == 1
    assert probes["mcmc_nuts"] == 0


def test_auto_follows_configured_threshold_high(tmp_path, monkeypatch, probes):
    """D=10 with mcmc_auto_d=50 is below the threshold: auto picks NUTS."""
    _write_user_defaults(tmp_path, monkeypatch, "[inference]\nmcmc_auto_d = 50\n")
    _mock_fitter().run("auto", key=jax.random.PRNGKey(0))
    assert probes["mcmc_nuts"] == 1
    assert probes["vi_nonlinear_fast"] == 0


def test_mcmc_follows_configured_threshold(tmp_path, monkeypatch, probes):
    """method="mcmc" uses the same configured threshold for NUTS vs ray tracing."""
    _write_user_defaults(tmp_path, monkeypatch, "[inference]\nmcmc_auto_d = 5\n")
    _mock_fitter().run("mcmc", key=jax.random.PRNGKey(0))
    assert probes["mcmc_raytrace"] == 1
    assert probes["mcmc_nuts"] == 0


# ── (b) a failed defaults read raises, it does not use 20 ────────────


@pytest.mark.parametrize("method", ["auto", "mcmc"])
def test_unparseable_defaults_file_raises(tmp_path, monkeypatch, probes, method):
    _write_user_defaults(tmp_path, monkeypatch, "[inference\nmcmc_auto_d = \n")
    with pytest.raises(ConfigError, match="mcmc_auto_d"):
        _mock_fitter().run(method, key=jax.random.PRNGKey(0))
    assert sum(probes.values()) == 0, "a sampler ran on a silently chosen threshold"


@pytest.mark.parametrize("method", ["auto", "mcmc"])
def test_absent_defaults_files_raise(tmp_path, monkeypatch, probes, method):
    monkeypatch.setattr(defaults, "_USER_DEFAULTS", tmp_path / "no-user.toml")
    monkeypatch.setattr(defaults, "_PACKAGE_DEFAULTS", tmp_path / "no-package.toml")
    with pytest.raises(ConfigError, match="cannot read"):
        _mock_fitter().run(method, key=jax.random.PRNGKey(0))
    assert sum(probes.values()) == 0


@pytest.mark.parametrize("method", ["auto", "mcmc"])
def test_missing_threshold_key_raises(tmp_path, monkeypatch, probes, method):
    _write_user_defaults(tmp_path, monkeypatch, '[inference]\nmethod = "auto"\n')
    with pytest.raises(ConfigError, match="mcmc_auto_d"):
        _mock_fitter().run(method, key=jax.random.PRNGKey(0))
    assert sum(probes.values()) == 0


@pytest.mark.parametrize("bad", ['"20"', "20.0", "true"])
def test_non_integer_threshold_raises(tmp_path, monkeypatch, probes, bad):
    _write_user_defaults(tmp_path, monkeypatch, f"[inference]\nmcmc_auto_d = {bad}\n")
    with pytest.raises(ConfigError, match="mcmc_auto_d"):
        _mock_fitter().run("auto", key=jax.random.PRNGKey(0))
    assert sum(probes.values()) == 0
