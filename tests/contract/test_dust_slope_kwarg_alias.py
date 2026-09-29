# SPDX-License-Identifier: BSD-3-Clause
"""Contract test for dust-slope keyword alias.

The attenuation-law keyword ``n_slope`` has been renamed to ``dust_slope`` across
all dust laws to make the law-function keywords align with registry names
(``dust_*``) and grammar stems (``slope``, ``delta``, etc.). The public law
functions like ``power_law`` and ``conroy2010`` retain backward compatibility via
the ``@renamed_kwarg`` decorator, so ``n_slope=`` works with a DeprecationWarning;
the registry callables, law-kwarg resolvers, and per-screen override dicts use
``dust_slope`` exclusively.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust.attenuation import conroy2010, power_law
from tengri.components.dust.laws._registry import (
    DUST_LAWS,
    law_kwarg_names,
    list_laws,
)

pytestmark = pytest.mark.contract


def test_old_keyword_warns_and_matches_new():
    """Old n_slope keyword works with a DeprecationWarning and matches new keyword."""
    wave = jnp.linspace(1000.0, 30000.0, 50)
    with pytest.warns(DeprecationWarning, match="dust_slope"):
        result_old = power_law(wave, n_slope=-0.9)
    result_new = power_law(wave, dust_slope=-0.9)
    np.testing.assert_array_equal(result_old, result_new)


def test_conroy2010_replaced_dust_slope_with_bump_strength():
    """conroy2010 (issue #2522) replaced dust_slope with dust_bump_strength.

    Unlike power_law, conroy2010 implements CCM89 (Cardelli curve) with a
    scalable 2175 Å bump. It takes dust_bump_strength (0-1 scale of bump)
    not dust_slope (power-law tail). The old test here is not applicable.
    """
    wave = jnp.linspace(1000.0, 30000.0, 50)
    # Verify conroy2010 takes dust_bump_strength, not dust_slope or n_slope
    result = conroy2010(wave, dust_Rv=3.1, dust_bump_strength=1.0)
    assert result.shape == wave.shape


def test_new_keyword_is_silent():
    """New dust_slope keyword produces no DeprecationWarning."""
    wave = jnp.linspace(1000.0, 30000.0, 50)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        power_law(wave, dust_slope=-0.9)
    for warning in w:
        assert not issubclass(warning.category, DeprecationWarning)


def test_both_keywords_raise():
    """Providing both n_slope and dust_slope raises TypeError."""
    wave = jnp.linspace(1000.0, 30000.0, 50)
    with pytest.raises(TypeError):
        power_law(wave, n_slope=-0.9, dust_slope=-0.9)


def test_registry_reports_only_the_new_name():
    """Registry law_kwarg_names reports dust_slope for power_law, dust_bump_strength for conroy2010.

    power_law keeps dust_slope; conroy2010 (#2522) replaced it with dust_bump_strength.
    """
    # power_law still uses dust_slope
    assert "dust_slope" in law_kwarg_names("power_law")
    assert "n_slope" not in law_kwarg_names("power_law")
    # conroy2010 uses dust_bump_strength (issue #2522)
    assert "dust_bump_strength" in law_kwarg_names("conroy2010")
    assert "dust_slope" not in law_kwarg_names("conroy2010")
    assert "n_slope" not in law_kwarg_names("conroy2010")


def test_resolver_dicts_stay_strict_after_the_swap():
    """The alias is for direct callers only: a dict carrying n_slope is not read.

    Registering the wrapper (#2257) must not loosen the resolver. ``select_law_kwargs``
    narrows to the declared names, so ``n_slope`` is dropped, and
    ``reject_unread_law_kwargs`` refuses it as read by none of the laws in play.

    For conroy2010 (#2522), dust_slope is no longer valid; bump_strength is used instead.
    """
    from tengri.components.dust.laws._registry import (
        reject_unread_law_kwargs,
        select_law_kwargs,
    )

    # Test power_law (still uses dust_slope)
    narrowed = select_law_kwargs("power_law", {"n_slope": -0.9, "dust_slope": -0.9, "dust_Rv": 3.1})
    assert "n_slope" not in narrowed
    assert narrowed["dust_slope"] == -0.9
    with pytest.raises(ValueError, match="n_slope"):
        reject_unread_law_kwargs({"n_slope": -0.9}, ("power_law",), context="test")

    # Test conroy2010 (uses dust_bump_strength, not dust_slope)
    narrowed = select_law_kwargs("conroy2010", {"dust_slope": 1.0, "dust_bump_strength": 0.8, "dust_Rv": 3.1})
    assert "dust_slope" not in narrowed  # conroy2010 doesn't read dust_slope anymore
    assert narrowed["dust_bump_strength"] == 0.8
    with pytest.raises(ValueError, match="dust_slope"):
        reject_unread_law_kwargs({"dust_slope": 1.0}, ("conroy2010",), context="test")


def test_registry_callable_accepts_old_name_with_warning():
    """Registry callable accepts old n_slope name with a DeprecationWarning.

    After the decorator swap, the wrapped function is stored in the registry,
    so the registry callable should accept the deprecated n_slope alias.
    This is (a) from issue #2257.
    """
    wave = jnp.linspace(1000.0, 30000.0, 50)
    with pytest.warns(DeprecationWarning, match="dust_slope"):
        result_old = DUST_LAWS["power_law"](wave, n_slope=-0.9)
    result_new = DUST_LAWS["power_law"](wave, dust_slope=-0.9)
    np.testing.assert_array_equal(result_old, result_new)


def test_registry_callable_accepts_old_name_conroy2010():
    """Registry callable for conroy2010 does NOT accept old n_slope (issue #2522).

    conroy2010 replaced dust_slope with dust_bump_strength (not a slope parameter,
    but a scaling factor for the 2175 Å bump). It does not have n_slope or dust_slope.
    """
    wave = jnp.linspace(1000.0, 30000.0, 50)
    # conroy2010 does not accept n_slope or dust_slope anymore
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        DUST_LAWS["conroy2010"](wave, dust_Rv=3.1, n_slope=-0.9)
    # conroy2010 uses dust_bump_strength instead
    result = DUST_LAWS["conroy2010"](wave, dust_Rv=3.1, dust_bump_strength=0.8)
    assert result.shape == wave.shape


def test_list_laws_callable_accepts_alias():
    """Callable from list_laws() accepts the deprecated n_slope alias.

    This is (b) from issue #2257: the callable from list_laws().to_dict("fn")
    is the same object as DUST_LAWS[name] and accepts the alias too.
    """
    wave = jnp.linspace(1000.0, 30000.0, 50)
    laws_dict = list_laws(headline=False).to_dict("fn")

    # Test power_law
    with pytest.warns(DeprecationWarning, match="dust_slope"):
        result_old = laws_dict["power_law"](wave, n_slope=-0.9)
    result_new = laws_dict["power_law"](wave, dust_slope=-0.9)
    np.testing.assert_array_equal(result_old, result_new)


def test_raw_registered_callable_has_alias_after_swap():
    """Callable stored in registry accepts the old n_slope alias.

    After the decorator swap (fixing #2257), the wrapped function is stored
    in the registry, so extracting the raw callable should still accept the alias.
    """
    # DUST_LAWS["power_law"] is a DustLawRegistryEntry; extract the callable
    entry = DUST_LAWS["power_law"]
    callable_fn = object.__getattribute__(entry, "callable")
    wave = jnp.linspace(1000.0, 30000.0, 50)

    # Should accept the old name with a warning
    with pytest.warns(DeprecationWarning, match="dust_slope"):
        result_old = callable_fn(wave, n_slope=-0.9)
    result_new = callable_fn(wave, dust_slope=-0.9)
    np.testing.assert_array_equal(result_old, result_new)


def test_list_laws_runs():
    """list_laws() returns a non-empty listing."""
    laws = list_laws()
    assert len(laws) > 0
