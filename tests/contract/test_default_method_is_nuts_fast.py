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
from tengri.inference.fitter import _CANONICAL_METHODS, _MANY_EVAL_SAMPLERS, resolve_method
from tengri.parameters.defaults import get_inference_defaults

pytestmark = pytest.mark.contract


def test_default_method_is_the_fast_nuts_recipe():
    assert DEFAULT_METHOD == "mcmc_nuts_fast"


def test_fast_nuts_is_canonical_and_registered():
    assert "mcmc_nuts_fast" in _CANONICAL_METHODS
    assert "mcmc_nuts_fast" in _MANY_EVAL_SAMPLERS
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
