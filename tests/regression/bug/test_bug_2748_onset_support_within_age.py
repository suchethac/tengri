# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2748: the SF-onset prior must not reach past age(z).

With a free redshift the onset prior's ceiling is narrowed only against the
lowest redshift the prior admits (z = 0.5, age 8.59 Gyr). A galaxy at z = 2
(age 3.28 Gyr) or z = 10 (age 0.47 Gyr) can then draw an onset older than the
universe at its own redshift. The mass-conserving normalisation zeroes the
support beyond age(z), which leaves a flat, zero-gradient region that kills
the HMC warmup of the population pilot (#2567 bisect, #2748).

The owner ruling reparametrises the onset as a fraction of age(z), so the
support is bounded by age(z) at every redshift. This test pins the
requirement: the onset prior, evaluated at z, never exceeds age(z).
"""

import pytest

import tengri
from tengri import FREE, Uniform
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.regression_bug


def _onset_upper_bound_gyr(redshift):
    """Ceiling of the dpl onset prior for a free-redshift build, in Gyr."""
    spec = tengri.parse_groups(
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=redshift,
    )
    lo, hi = spec.get_distribution("sfh_dpl_age_gyr").bounds
    assert lo > 0.0
    return hi


@pytest.mark.parametrize("z_at", [2.0, 10.0])
def test_onset_prior_support_never_exceeds_age_of_universe_at_z(z_at):
    """The free-z onset ceiling must not exceed age(z) at the galaxy's redshift."""
    hi = _onset_upper_bound_gyr(Uniform(0.5, 12.0, default=5.0))
    age_gyr = float(age_at_z(z_at))

    assert hi <= age_gyr * (1.0 + 1e-9), (
        f"onset prior reaches {hi:.3f} Gyr, past age({z_at:g}) = {age_gyr:.3f} Gyr"
    )
