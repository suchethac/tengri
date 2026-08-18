# SPDX-License-Identifier: BSD-3-Clause
"""Contract: a mock SED model must declare which observation channels it lacks.

This pins the two-sided failure that produced #1931 and then #1942. Both were
the same disease — a hand-rolled stub restating a contract instead of deriving
it — and both hid in ``tests/inference/``, which ``tests/conftest.py``
auto-marks ``slow`` and the PR gate therefore never runs. #1931 fixed the
``tests/contract/`` half and merged green while the identical
``tests/inference/`` half stayed red for a week.

These tests live in ``tests/contract/`` deliberately: the *contract* belongs
where the gate can see it, even though the stubs that violate it live in a
tree the gate skips.

The two sides:

**Over-declaration** (the dangerous one, fails open). ``Fitter._build_data_args``
reads optional observation config as ``getattr(obs, "line_ratios", None)``. On a
bare ``MagicMock`` that default is unreachable — every attribute auto-vivifies —
so each ``is not None`` guard passes and a photometry-only stub silently claims
line fluxes, line ratios and spectral indices.

**Under-declaration** (the loud one). The eager channel-scale pre-check (#1495,
via #1905) calls ``spec.sample(key)`` at loss-build time on every fit, so a spec
stub that reaches the loss builder without ``sample`` dies with ``AttributeError``.

No SSP data required.
"""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import jax.numpy as jnp
import pytest

from tengri.inference.fitter import Fitter
from tengri.parameters.parameters import Parameters
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.contract


#: Optional observation channels Fitter._build_data_args probes with
#: ``getattr(obs, ..., None)``. A stub that declares none of them must produce
#: no data_args key from any of them.
_OPTIONAL_CHANNEL_KEYS = (
    "line_flux_obs",
    "line_flux_err",
    "line_flux_waves",
    "line_flux_limit_mask",
    "line_ratio_obs",
    "line_ratio_err",
    "index_obs",
    "index_err",
    "spec_cov_inv",
)


def _spec() -> Parameters:
    """A small photometry spec — three free parameters, no SSP data needed."""
    return Parameters(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Uniform(0.5, 5.0),
        sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
        met_logzsol=Uniform(-2.0, 0.5),
        dust_tau_diff=Fixed(0.3),
        redshift=Fixed(0.1),
    )


def _model(*, declare_observation: bool) -> MagicMock:
    """A photometry-only stub, with or without the declaration under test."""
    model = MagicMock()
    model.spec = _spec()
    model.predict_photometry.return_value = jnp.ones(3) * 1e-18
    if declare_observation:
        model.observation = SimpleNamespace(
            spectroscopy=None, line_fluxes=None, line_ratios=None, spectral_indices=None
        )
    return model


def _fitter(model: MagicMock) -> Fitter:
    return Fitter(model, jnp.ones(3) * 1e-18, jnp.ones(3) * 1e-19, data_type="photometry")


class TestObservationDeclaration:
    """A stub must say what it does NOT have; Mock cannot be asked."""

    def test_declared_stub_publishes_no_optional_channel(self):
        """The whole point of the declaration: no phantom data_args keys."""
        args = _fitter(_model(declare_observation=True))._data_args
        phantom = sorted(k for k in _OPTIONAL_CHANNEL_KEYS if k in args)
        assert phantom == [], (
            f"a photometry-only stub published optional channels {phantom}; "
            "Fitter._build_data_args found configuration the stub never declared"
        )

    def test_declaration_is_load_bearing(self):
        """Drop it and the phantom channels come back.

        Pinned as a mutation check on the test above: without this, a future
        change that stopped reading ``model.observation`` at all would leave
        ``test_declared_stub_publishes_no_optional_channel`` passing vacuously.
        """
        args = _fitter(_model(declare_observation=False))._data_args
        phantom = sorted(k for k in _OPTIONAL_CHANNEL_KEYS if k in args)
        assert phantom, (
            "a bare MagicMock model no longer fabricates optional channels. If "
            "Fitter stopped probing model.observation with getattr(..., None), "
            "this contract is obsolete — delete it. If it still probes, the "
            "sibling test above has gone vacuous."
        )

    def test_declared_stub_builds_a_jit_wrapped_objective(self):
        """The eager pre-check (#1495) evaluates every channel at build time.

        A stub claiming channels it cannot serve dies here, so reaching a
        jit-wrapped objective is the end-to-end statement that it claimed none.
        """
        fitter = _fitter(_model(declare_observation=True))
        loss_fn = fitter._get_or_build_loss_fn()
        assert hasattr(loss_fn, "lower"), "objective must stay jax.jit-wrapped"


class TestSpecSampleRequirement:
    """The under-declaring half: the pre-check needs a real ``sample``."""

    def test_spec_without_sample_fails_loudly(self, monkeypatch):
        """Loud is the design: the pre-check is deliberately fallback-free.

        Its docstring — "if a channel cannot be evaluated here, the same call
        fails inside the fit, so the error propagates instead of being
        swallowed" — is the reason this raises rather than skipping the check.
        A stub spec that reaches the loss builder must implement ``sample``.

        Build the fitter first, then remove ``sample``: this pins the pre-check
        at loss-build time specifically, rather than any earlier construction
        that happens to call it.
        """
        fitter = _fitter(_model(declare_observation=True))
        # Strip sample() the way a hand-rolled _MockSpec omits it. monkeypatch
        # restores the class attribute even if the assertion below fails.
        monkeypatch.delattr(Parameters, "sample")
        with pytest.raises(AttributeError, match="sample"):
            fitter._get_or_build_loss_fn()
