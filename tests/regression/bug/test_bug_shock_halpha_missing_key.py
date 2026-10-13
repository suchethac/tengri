# SPDX-License-Identifier: BSD-3-Clause
"""A shock ratio table without H-alpha must raise, not divide by 3.0.

Bug: ``components/nebular/shock.py`` read the normalizing line with
``ratios.get("HA_6563A", jnp.array(3.0))``. A missing key silently rescaled
every line by a made-up Halpha/Hbeta ratio of 3.0. Both the MAPPINGS grid path
and the Allen+2008 fallback always emit ``HA_6563A``, so a missing key is a
broken table and must fail loudly, naming the keys that were available.
"""

from __future__ import annotations

import jax.numpy as jnp
import pytest

from tengri.components.nebular import shock

pytestmark = pytest.mark.regression_bug


def test_missing_halpha_raises_keyerror_listing_available_keys(monkeypatch):
    """No H-alpha in the ratio table is a KeyError that names the keys present."""
    ratios = {"Hb_4861A": jnp.array(1.0), "O3_5007A": jnp.array(2.5)}
    monkeypatch.setattr(shock, "shock_line_ratios", lambda *args, **kwargs: ratios)

    with pytest.raises(KeyError) as excinfo:
        shock._shock_line_arrays(300.0, 1.0e40)

    message = str(excinfo.value)
    assert "HA_6563A" in message
    assert "Hb_4861A" in message
