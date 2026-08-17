# SPDX-License-Identifier: BSD-3-Clause
"""Contract: an omitted component parameter names itself, not just its key (#1832).

When a params dict is missing a parameter a component indexes, the failure
surfaces from wherever that component happened to read first::

    KeyError: "xray_det_hmxb"

That is what all fifty of #1832's failures looked like, and it says nothing
about which component wanted the value or why it was the caller's to supply.
Diagnosing it took a traceback and a reading of ``_params.py``.

The named form already exists one layer up — ``predict()`` raises
``KeyError: "Free parameter 'xray_det_hmxb' not found in params dict"`` — but
two callers never reach it:

* ``predict_state`` on a flat ``Parameters(...)`` spec, because
  :func:`check_missing_free_params` exempts the ``agn_/radio_/xray_/igm_``
  families there (they are over-registered by that path, so requiring them all
  would be wrong — measured: 81 of 81 free params on a flat AGN spec);
* every ``run_components`` caller, which has no ``spec`` at all and so cannot
  use a spec-based check even in principle. Forty of #1832's fifty lived here.

Both share one seam — ``run_components`` calling ``component.apply`` — so the
message is fixed there, for both, once.

The type stays ``KeyError``: that is what ``predict()`` already raises for the
same condition, and callers catch it today.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri.components.xray.component import XRaySEDComponent
from tengri.forward.orchestrator import default_params_dict, run_components
from tengri.protocols import ForwardState


def _xray_state():
    wave = jnp.linspace(1e0, 1e3, 256)
    return ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived={"sfr": 5.0, "log_mstar": jnp.log10(5e10), "L_agn_bol": 1e44},
    )


def _params_missing(*names):
    component = XRaySEDComponent()
    full = default_params_dict([component], overrides={"redshift": 0.5})
    return component, {k: v for k, v in full.items() if k not in names}


def test_missing_parameter_names_the_parameter_and_the_component():
    """The message must carry both — which value, and who wanted it.

    Asserting ``component.name in message`` would pass on the *bare* message
    too: the component is called ``xray`` and the parameter ``xray_det_hmxb``,
    so the name is a substring of the key. A check that cannot go red is not a
    check — assert the explanatory phrasing the bare form does not have.
    """
    component, params = _params_missing("xray_det_hmxb")

    with pytest.raises(KeyError) as exc:
        run_components([component], _xray_state(), params)

    message = str(exc.value)
    assert "xray_det_hmxb" in message
    assert "requires parameter" in message, f"message does not explain itself: {message}"
    # The bare form is exactly the key and nothing else.
    assert message.strip("\"'") != "xray_det_hmxb"


def test_the_original_keyerror_is_chained():
    """``raise ... from`` — the component-level frame stays in the traceback."""
    component, params = _params_missing("xray_det_lmxb")

    with pytest.raises(KeyError) as exc:
        run_components([component], _xray_state(), params)

    assert isinstance(exc.value.__cause__, KeyError)
    assert exc.value.__cause__.args[0] == "xray_det_lmxb"


def test_an_unrelated_keyerror_is_not_relabeled():
    """A KeyError that is *not* about a declared parameter must pass through.

    Otherwise the fix mislabels every dict lookup inside every component — a
    guard that turns real bugs into confident wrong explanations is worse than
    the bare message it replaced.
    """

    class _Exploding:
        name = "exploding"
        parameter_prefix = "exploding_"

        def declared_parameters(self):
            return []

        def apply(self, state, params, *, ssp_data=None, template_data=None):
            raise KeyError("some_internal_bookkeeping_key")

    with pytest.raises(KeyError) as exc:
        run_components([_Exploding()], _xray_state(), {})

    assert exc.value.args[0] == "some_internal_bookkeeping_key"
    assert "declared" not in str(exc.value).lower()
