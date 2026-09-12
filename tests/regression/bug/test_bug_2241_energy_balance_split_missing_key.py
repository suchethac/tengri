# SPDX-License-Identifier: BSD-3-Clause
"""#2241, https://github.com/suchethac/tengri/issues/2241 -- a hand-built
params dict missing a key must raise, not fall back to a stale literal.

``EnergyBalanceSplitIRSEDComponent.predict`` used to read its six warm/cold +
AGN-IR knobs via ``p.get("f_cold", 0.5)`` and five siblings. On the grammar
path (``SEDModel.build`` / ``Parameters``) every declared ``dust_`` parameter
is always present in the sliced dict, so the fallback literal was dead code
there -- but a hand-built ``params`` dict that skips the grammar (the shape
``tests/contract/test_dust_emission_declared_params_apply.py`` documented as
its own "Known limit (#2241, not this census)") silently absorbed a missing
key into the fallback instead of raising, which is exactly the failure mode
that census exists to catch for every OTHER registered emission type.

This is a regression test for the fix: ``predict`` now indexes ``p["f_cold"]``
and siblings directly, so a missing key raises ``KeyError`` naming it.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.dust.emission.analytic.energy_balance_split import (
    EnergyBalanceSplitIRSEDComponent,
)

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.logspace(3.0, 7.0, 32)


def _predict_with(p: dict) -> None:
    component = EnergyBalanceSplitIRSEDComponent()
    sed_in = jnp.zeros_like(_WAVE)
    component.predict(p, sed_in, _WAVE, L_ir=1e44, log_L_ir=float(jnp.log10(1e44)))


def test_missing_f_cold_raises_keyerror_naming_it():
    """RED at HEAD (origin/main): ``p.get("f_cold", 0.5)`` silently returned a
    finite SED built from the stale 0.5 literal instead of raising.

    ``p`` supplies every OTHER key ``predict`` reads (``redshift``,
    ``L_agn_ir``) so the first and only missing key is ``f_cold``.
    """
    p = {"redshift": jnp.asarray(0.5), "L_agn_ir": jnp.asarray(0.0)}
    with pytest.raises(KeyError, match="f_cold"):
        _predict_with(p)


def test_missing_l_agn_ir_raises_keyerror_naming_it():
    """The AGN-IR knob (read before ``f_cold`` in ``predict``'s body) is the
    same shape of fix: ``p.get("L_agn_ir", 0.0)`` used to swallow a missing
    key just as silently."""
    p = {
        "redshift": jnp.asarray(0.5),
        "f_cold": jnp.asarray(0.5),
        "T_warm": jnp.asarray(45.0),
        "T_cold": jnp.asarray(20.0),
        "beta_warm": jnp.asarray(1.5),
        "beta_cold": jnp.asarray(2.0),
    }
    with pytest.raises(KeyError, match="L_agn_ir"):
        _predict_with(p)


def test_every_declared_key_present_does_not_raise():
    """The positive control: supplying every key ``predict`` reads must not
    raise, and must produce a finite, non-trivial SED (guards against a
    trivial "always raises" mutation passing the two tests above)."""
    p = {
        "redshift": jnp.asarray(0.5),
        "L_agn_ir": jnp.asarray(0.0),
        "f_cold": jnp.asarray(0.5),
        "T_warm": jnp.asarray(45.0),
        "T_cold": jnp.asarray(20.0),
        "beta_warm": jnp.asarray(1.5),
        "beta_cold": jnp.asarray(2.0),
    }
    component = EnergyBalanceSplitIRSEDComponent()
    sed_in = jnp.zeros_like(_WAVE)
    sed_out, published = component.predict(
        p, sed_in, _WAVE, L_ir=1e44, log_L_ir=float(jnp.log10(1e44))
    )
    assert jnp.all(jnp.isfinite(sed_out))
    assert float(jnp.max(sed_out)) > 0.0
    assert "sed_dust_ir" in published
