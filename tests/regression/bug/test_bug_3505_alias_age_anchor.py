# SPDX-License-Identifier: BSD-3-Clause
"""Regression: an SFH alias carries its family's age anchor (#3505).

``db`` and ``dbp`` are registry aliases of ``dense_basis`` and ``dense_basis_pure``
and share their implementation. The age-of-universe rule was keyed on the registry
*name*, so an alias received no ``age_universe_yr``. Its SFH then failed at
evaluation with ``missing 1 required keyword-only argument: 'age_universe_yr'``.
The rule is now keyed on the registry function as well as the name.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.stellar.component import age_universe_kwargs
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

Z_OBS = 0.1


@pytest.mark.parametrize("alias", ["db", "dbp"])
def test_alias_receives_the_age_of_the_universe(alias):
    """The alias gets ``age_universe_yr`` equal to age(z) in years, as its family does."""
    kwargs = age_universe_kwargs(alias, Z_OBS)
    assert set(kwargs) == {"age_universe_yr"}
    np.testing.assert_allclose(float(kwargs["age_universe_yr"]), float(age_at_z(Z_OBS)) * 1e9)


@pytest.mark.parametrize("alias", ["db", "dbp"])
def test_alias_prior_draws_evaluate(alias):
    """A prior fan of the alias evaluates to finite, non-negative SFR curves."""
    _age, curves = sample_sfh_prior(alias, jax.random.PRNGKey(0), n=2, redshift=Z_OBS)
    assert curves.shape[0] == 2
    assert bool(jnp.all(jnp.isfinite(curves)))
    assert bool(jnp.all(curves >= 0.0))
