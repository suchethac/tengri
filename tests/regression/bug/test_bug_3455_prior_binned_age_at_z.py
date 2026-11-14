# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #3455: binned SFH priors must put no mass older than age(z).

``sample_sfh_prior`` resolved the binned families (``continuity``,
``dirichlet``, ...) without bin edges, so the default ladder ran to 13.7 Gyr
whatever the redshift. Up to 91 % of the mass of a draw then sat at lookback
times older than the universe. The forward model builds its ladder with
``make_agebins_from_zred(z)``, whose last edge is age(z); the prior must use the
same edges.

At ``redshift=0`` the default ladder already ends inside age(0), so that case
is pinned to be unchanged.
"""

import jax
import numpy as np
import pytest

from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug

#: Families that take bin edges and whose default ladder the redshift now sets.
BINNED_FAMILIES = (
    "continuity",
    "continuity_young_fraction",
    "dirichlet",
    "bursty_continuity",
    "prospector_beta",
)

N_DRAWS = 50
REDSHIFTS = (0.5, 1.0, 6.0)


def _age_yr(z: float) -> float:
    return float(age_at_z(z)) * 1e9


@pytest.mark.parametrize("z", REDSHIFTS)
@pytest.mark.parametrize("family", BINNED_FAMILIES)
def test_binned_prior_has_no_mass_older_than_age_at_z(family, z):
    """Every draw forms no mass at lookback times beyond age(z)."""
    age_yr, curves = sample_sfh_prior(family, jax.random.PRNGKey(7), n=N_DRAWS, redshift=z)
    age_yr = np.asarray(age_yr, dtype=float)
    curves = np.asarray(curves, dtype=float)
    dt = np.gradient(age_yr)
    mass = curves * dt
    total = mass.sum(axis=1)
    old = mass[:, age_yr > _age_yr(z)].sum(axis=1)

    assert np.all(total > 0.0)
    assert np.max(old / total) <= 1e-10


@pytest.mark.parametrize("family", BINNED_FAMILIES)
def test_binned_prior_at_z0_is_unchanged(family):
    """At redshift 0 the prior draws are the same as with no redshift at all."""
    key = jax.random.PRNGKey(7)
    _, plain = sample_sfh_prior(family, key, n=N_DRAWS)
    _, at_zero = sample_sfh_prior(family, key, n=N_DRAWS, redshift=0.0)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(at_zero))
