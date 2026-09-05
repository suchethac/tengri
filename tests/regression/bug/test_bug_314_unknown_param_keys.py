# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #314 — dict-merged unknown param keys silently dropped.

PR #315 added :func:`tengri.parameters.translate.check_unknown_params` to
:meth:`SEDModel.predict_observables_jit`. The dict-merge code path (any
caller that does ``{**baseline, 'maybe-typo': v}`` and passes the result to
:meth:`SEDModel.predict_rest_sed` or any other ``predict_state``-routed
method) was still silently dropping unknown keys, producing
plausible-looking but wrong physics — the worst class of bug because it
bypasses unit tests.

This fix validates at :meth:`SEDModel.predict_state` so the entire
orchestrator-path predict family (``predict_rest_sed``,
``predict_emission_lines``, ``predict_photometry``, …) gets the same
:class:`UnknownParameterError`.
"""

from __future__ import annotations

from pathlib import Path

import jax
import pytest

import tengri
from tengri.config.exceptions import UnknownParameterError

pytestmark = pytest.mark.regression_bug

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SSP_FILE = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


@pytest.fixture(scope="module")
def ssp():
    if not _SSP_FILE.is_file():
        pytest.skip(f"SSP file not present: {_SSP_FILE}")
    return tengri.load_ssp()


@pytest.fixture(scope="module")
def small_model(ssp):
    """Minimal SEDModel suitable for fast predict_rest_sed exercises."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.SEDModel.build(
            ssp,
            sfh={"type": "tsnorm", "all_params": tengri.Fixed(tengri.DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": tengri.Fixed(tengri.DEFAULT),
            },
            redshift=tengri.Fixed(0.01),
        )


@pytest.fixture
def baseline_params(small_model):
    return dict(small_model.spec.sample(jax.random.PRNGKey(0)))


class TestBug314UnknownKeyOnPredictRestSed:
    """The original #314 repro: predict_rest_sed must reject typos."""

    def test_typo_dust_key_raises(self, small_model, baseline_params):
        """``dust_two_component_tau_diff`` is a plausible-looking typo
        (real key is ``dust_tau_diff``) and used to silently drop."""
        params = {**baseline_params, "dust_two_component_tau_diff": 1.5}
        with pytest.raises(UnknownParameterError, match="dust_two_component_tau_diff"):
            small_model.predict_rest_sed(params)

    def test_did_you_mean_hint_in_message(self, small_model, baseline_params):
        """The error message points to the closest valid name."""
        params = {**baseline_params, "dust_two_component_tau_diff": 1.5}
        try:
            small_model.predict_rest_sed(params)
        except UnknownParameterError as exc:
            assert "did you mean" in str(exc) or "no close match" in str(exc)
        else:
            pytest.fail("UnknownParameterError not raised")

    def test_clean_call_still_works(self, small_model, baseline_params):
        """Don't false-positive on the legitimate call path."""
        out = small_model.predict_rest_sed(baseline_params)
        assert out.sed.shape == out.wavelength.shape

    def test_completely_made_up_key_raises(self, small_model, baseline_params):
        """A name with no close match still raises (with the
        ``no close match`` hint)."""
        params = {**baseline_params, "frob_nicate_quux": 1.5}
        with pytest.raises(UnknownParameterError, match="frob_nicate_quux"):
            small_model.predict_rest_sed(params)


class TestBug314CoversOtherPredictPaths:
    """The validator runs at predict_state, so every predict_* method
    routing through the orchestrator gets it."""

    def test_typo_blocked_in_predict(self, small_model, baseline_params):
        """``model.predict(...)`` builds a lazy Prediction object; the
        validator fires when the first property routes through
        ``predict_state``."""
        params = {**baseline_params, "typo_param": 1.5}
        with pytest.raises(UnknownParameterError, match="typo_param"):
            pred = small_model.predict(params)
            _ = pred.sed.l_bol  # force orchestrator evaluation


class TestBug314DoesNotBreakLegitimateOverrides:
    """Overrides on real param names must still work."""

    def test_real_dust_tau_diff_override(self, small_model, baseline_params):
        """The actual key (``dust_tau_diff``, no ``two_component_`` infix)
        is recognized and respected."""
        if "dust_tau_diff" not in baseline_params:
            pytest.skip("Baseline params don't include dust_tau_diff")
        # Swap to a different value — sweep should produce different SEDs.
        p_low = {**baseline_params, "dust_tau_diff": 0.0}
        p_high = {**baseline_params, "dust_tau_diff": 3.0}
        sed_low = small_model.predict_rest_sed(p_low).sed
        sed_high = small_model.predict_rest_sed(p_high).sed
        # Heavy dust at 3.0 must depress at least some wavelengths
        # relative to no dust.
        assert float(sed_high.sum()) < float(sed_low.sum()), (
            "dust_tau_diff override had no effect — regression in real-key path"
        )
