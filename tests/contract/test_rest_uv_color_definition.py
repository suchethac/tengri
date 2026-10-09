# SPDX-License-Identifier: BSD-3-Clause
"""``rest_uv_color`` is one definition: a U-V AB color, not FUV-NUV or f1500-f2300 (#2611).

The code computes an AB color between two rectangular top-hats on the rest-frame
L_nu, 3200-3900 A minus 5000-5800 A. Every declaration of the property must state
that definition, and none may describe the other two.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import tengri.components.stellar.component as stellar
import tengri.forward.prediction as prediction
import tengri.utils.sed_quantities as sq
from tengri.utils.sed_quantities import compute_rest_uv_color

pytestmark = pytest.mark.contract

_STALE = ("FUV", "NUV", "1500", "2300")

_DECLARATIONS = {
    "compute_rest_uv_color": sq.compute_rest_uv_color.__doc__,
    "_rest_uv_color_fn": stellar._rest_uv_color_fn.__doc__,
    "Prediction.rest_uv_color": prediction.Prediction.rest_uv_color.__doc__,
}


def test_computed_value_is_the_u_minus_v_color_not_fuv_minus_nuv():
    """A 4000 A break gives U-V = -2.5 log10(1/2) = +0.753 and FUV-NUV = 0."""
    wave = jnp.linspace(1000.0, 10000.0, 9001)
    sed_step = jnp.where(wave > 4000.0, 2.0, 1.0)
    got = float(compute_rest_uv_color(sed_step, wave))
    assert got == pytest.approx(-2.5 * np.log10(0.5), abs=5e-3)


def test_flat_spectrum_has_zero_color():
    wave = jnp.linspace(1000.0, 10000.0, 9001)
    assert float(compute_rest_uv_color(jnp.ones_like(wave), wave)) == pytest.approx(0.0, abs=1e-9)


@pytest.mark.parametrize("where", sorted(_DECLARATIONS))
def test_declaration_states_the_computed_definition(where):
    """The summary states the computed definition; a later note may say what it is not."""
    text = _DECLARATIONS[where]
    assert text, f"{where}: has no docstring"
    summary = text.strip().splitlines()[0]
    assert "3200" in text and "5800" in text, f"{where}: names no band edges"
    for stale in _STALE:
        assert stale not in summary, f"{where}: summary still describes {stale}"
