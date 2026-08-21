# SPDX-License-Identifier: BSD-3-Clause
"""Regression: predict_* must raise on unknown override keys, not silently ignore.

Issue #314: passing an override key that doesn't match a live free/fixed
parameter (typo, deleted param, stale name) was silently dropped — the JIT path
through ``predict_observables_jit`` bypassed the ``get_internal_params``
validator. Symptoms: 2-D age × tau_diff sweeps showed constant rows because the
age axis was a no-op; Z = 0.02 vs 0.001 sweeps produced identical Lick indices.

The fix wires :func:`tengri.parameters.translate.check_unknown_params` into the
JIT entry point so the user gets an :class:`UnknownParameterError` with a
"did you mean" hint instead of plausible-looking wrong physics.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

import tengri
from tengri.config.exceptions import UnknownParameterError
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug


@pytest.fixture(scope="module")
def model_with_filters():
    try:
        ssp = tengri.load_ssp()
    except FileNotFoundError:
        pytest.skip("default wNE SSP not available")
    obs = tengri.Observation(
        photometry=tengri.Photometry(
            filters=(
                FilterCurve(
                    wave=jnp.linspace(3500.0, 4500.0, 50), trans=jnp.ones(50) * 0.5, name="b"
                ),
            )
        )
    )
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": tengri.FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": tengri.FIXED,
            "tau_diff": 0.0,
            "tau_bc": 0.0,
        },
        redshift=tengri.Fixed(0.05),
    )


def test_unknown_override_key_raises(real_ssp_only, model_with_filters):
    # Needs the real grid: the expected suggestion set / valid-param list depends
    # on the real model's params; ``real_ssp_only`` skips on synthetic-only CI.
    # (``age`` is no longer a probe here — it resolves as a short-form alias of
    # the SFH age param instead of being flagged, which is correct behavior.)
    m = model_with_filters
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["not_a_param"] = 1.0
    p["nonsense_param"] = 99.0
    with pytest.raises(UnknownParameterError) as exc:
        m.predict_photometry(p)
    msg = str(exc.value)
    assert "not_a_param" in msg and "nonsense_param" in msg


def test_did_you_mean_suggestion(model_with_filters):
    m = model_with_filters
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["dust_tau_dif"] = 0.5  # typo of dust_tau_diff
    with pytest.raises(UnknownParameterError, match="dust_tau_diff"):
        m.predict_photometry(p)


def test_short_form_alias_still_recognized(model_with_filters):
    """Short-form aliases (alpha for sfh_dpl_alpha) must not be flagged."""
    m = model_with_filters
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p.pop("sfh_dpl_alpha", None)
    p["alpha"] = 0.5
    out = m.predict_photometry(p)  # no raise
    assert jnp.all(jnp.isfinite(out))


def test_clean_params_pass_through(model_with_filters):
    m = model_with_filters
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    out = m.predict_photometry(p)
    assert jnp.all(jnp.isfinite(out))
    assert jnp.all(out > 0.0)
