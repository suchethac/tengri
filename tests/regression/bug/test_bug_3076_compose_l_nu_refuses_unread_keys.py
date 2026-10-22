# SPDX-License-Identifier: BSD-3-Clause
"""``compose_l_nu`` refuses a keyword its recipe does not read (#3076).

#2993 renamed the ``adaf_lopez2024`` disc's blend weight from ``agn_cigale_disk_delta`` to
``agn_cigale_adaf_delta``. The grammar refused the old spelling from #2994 on, but the keyword
path accepted it: ``compose_l_nu(..., agn_disc_block='adaf_lopez2024',
agn_cigale_disk_delta=0.3)`` returned the default blend, bit-identical to not passing it. The
keyword path now raises ``ParameterError`` naming the key, the keys the recipe reads, the
closest one and the CIGALE spelling hint of #2994. ``composable_agn_l_nu``, which the component
and the presets call with the full declared superset, and the documented pass-through keys keep
working.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks.runner import composable_agn_l_nu, compose_l_nu
from tengri.config.exceptions import ParameterError

pytestmark = pytest.mark.regression_bug

_WAVE = jnp.asarray(np.geomspace(1.0e3, 1.0e6, 64))
_RECIPE = dict(
    agn_disc_block="adaf_lopez2024",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="none",
    agn_attenuation_block="none",
)


def test_renamed_adaf_blend_key_is_refused_with_its_new_spelling():
    """The pre-#2993 key raises and points at ``agn_cigale_adaf_delta``."""
    with pytest.raises(ParameterError) as info:
        compose_l_nu(_WAVE, 10.0, agn_cigale_disk_delta=0.3, **_RECIPE)
    message = str(info.value)
    assert "'agn_cigale_disk_delta'" in message
    assert "agn_cigale_adaf_delta" in message
    assert "CIGALE's delta" in message


def test_the_read_key_is_accepted_and_moves_the_sed():
    """The new spelling is read: the SED changes with it (it was the silently dropped one)."""
    base = np.asarray(compose_l_nu(_WAVE, 10.0, **_RECIPE))
    moved = np.asarray(compose_l_nu(_WAVE, 10.0, agn_cigale_adaf_delta=0.3, **_RECIPE))
    assert not np.array_equal(base, moved)


def test_pass_through_and_runner_keys_are_accepted():
    """Documented pass-through keys and the runner's own reads are never refused."""
    compose_l_nu(
        _WAVE,
        10.0,
        agn_norm="conserving",
        agn_cos_inc=0.5,
        agn_torus_frac=0.3,
        agn_log_lbol_shape=10.0,
        **_RECIPE,
    )


def test_registry_entry_accepts_the_declared_superset():
    """``composable_agn_l_nu`` takes keys no block of the recipe reads (the component's call)."""
    sed = composable_agn_l_nu(_WAVE, 10.0, agn_cigale_disk_delta=0.3, **_RECIPE)
    assert np.all(np.isfinite(np.asarray(sed)))


def test_unknown_key_is_refused():
    """A key no AGN block declares is refused too, with the closest read key."""
    with pytest.raises(ParameterError, match="agn_torus_frak"):
        compose_l_nu(_WAVE, 10.0, agn_torus_frak=0.3, **_RECIPE)
