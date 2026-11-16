# SPDX-License-Identifier: BSD-3-Clause
"""The default inference method is ``mcmc_nuts_fast``, and it is what its name says.

Guards the 2026-09-11 default change: ``forward.fit(data)`` with no ``method``
runs four NUTS chains at the settings
``bench/reports/2026-09-11_profile_mass_20s.md`` measured -- 150 warmup steps,
no separate burn-in, 300 draws, target acceptance 0.8 -- on the mass-profiled
posterior with the dense metric. Each clause below is one of those facts; a
change to any of them should have to edit this file.
"""

import inspect

import pytest

from tengri.inference._backend_registry import DEFAULT_METHOD, get_backend
from tengri.inference.fitter import _CANONICAL_METHODS, _many_eval_samplers, resolve_method
from tengri.parameters.defaults import get_inference_defaults

pytestmark = pytest.mark.contract


def test_default_method_is_the_fast_nuts_recipe():
    assert DEFAULT_METHOD == "mcmc_nuts_fast"


def test_fast_nuts_is_canonical_and_registered():
    assert "mcmc_nuts_fast" in _CANONICAL_METHODS
    assert "mcmc_nuts_fast" in _many_eval_samplers()
    assert resolve_method("mcmc_nuts_fast") == "mcmc_nuts_fast"
    entry = get_backend("mcmc_nuts_fast")
    assert entry.tier == "primary"
    assert entry.accepts_precondition


def test_fast_nuts_defaults_are_the_measured_recipe():
    defaults = get_inference_defaults("mcmc_nuts_fast")
    assert defaults == {
        "n_chains": 4,
        "n_warmup": 150,
        "n_burnin": 0,
        "n_samples": 300,
        "target_accept_rate": 0.8,
    }


def test_fast_nuts_runner_forwards_to_run_nuts_with_overrides():
    """The wrapper is a draw budget, not a fork: every setting is overridable."""
    from tengri.inference import _registration

    src = inspect.getsource(_registration._run_nuts_fast)
    assert "settings.update(kw)" in src
    assert "_ctx_run_nuts(" in src


@pytest.mark.parametrize("surface", ["forward_model", "fitter"])
def test_fit_surfaces_default_to_default_method(surface):
    if surface == "forward_model":
        from tengri.forward.forward_model import ForwardModel

        param = inspect.signature(ForwardModel.fit).parameters["method"]
    else:
        from tengri.inference.fitter import Fitter

        param = inspect.signature(Fitter.run).parameters["method"]
    assert param.default == DEFAULT_METHOD


def test_many_evaluation_set_is_what_the_registry_entries_declare():
    """A method is many-evaluation iff its entry carries a chain, ensemble or nested sampler."""
    from tengri.inference._backend_registry import all_backends

    sampled = {
        e.name
        for e in all_backends()
        if e.sampler is not None and e.sampler.kind in ("chain", "ensemble", "nested")
    }
    derived = _many_eval_samplers()
    assert sampled <= derived
    assert derived - sampled == {"native_vi_linear", "native_vi_nonlinear"}
    assert {
        "mcmc_barker",
        "mcmc_mala",
        "mcmc_hmc_lowrank",
        "hmc_is",
        "pathfinder",
    } <= derived


@pytest.mark.parametrize(
    "method", ["mcmc_barker", "mcmc_mala", "mcmc_hmc_lowrank", "hmc_is", "pathfinder"]
)
def test_the_exact_path_advisory_reaches_the_newly_listed_samplers(method):
    import types
    import warnings

    from tengri.inference.fitter import _warn_if_exact_forward_path

    model = types.SimpleNamespace(
        observation=types.SimpleNamespace(photometry=object(), spectroscopy=None),
        approx=types.SimpleNamespace(wave_precomp=None),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _warn_if_exact_forward_path(model, method)
    assert any(
        f"Fitting with '{method}' on the exact forward path" in str(w.message) for w in caught
    )
