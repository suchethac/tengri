# SPDX-License-Identifier: BSD-3-Clause
"""``FreeRedshiftOnsetCeilingWarning`` fires exactly when it should (#2521).

``_narrow_free_priors_to_z`` caps a z-capped onset/age/peak-time parameter's
free-prior ceiling at ``age_at_z(z_floor)``, the age of the universe at the
LOWEST redshift a build's own ``redshift`` prior admits. That cap says
nothing about the range's upper (younger-universe) end: a wide free
``redshift`` prior can still admit a draw near that end whose onset value,
though inside the z_floor-capped range, is not inside the cosmologically
valid range AT THAT DRAW. ``SEDModel.build`` -> ``parse_groups`` now emits
one ``FreeRedshiftOnsetCeilingWarning`` at build time when this is possible,
naming the affected parameter(s) and both ages, since the forward model
itself only truncates and mass-conserves silently there (no exception, and
under ``jax.jit`` not even the eager ``SFHBeforeBigBangWarning``).
"""

from __future__ import annotations

import warnings

import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri.config.exceptions import FreeRedshiftOnsetCeilingWarning

pytestmark = pytest.mark.contract


def _warnings_of(category, ssp, **build_kwargs):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        SEDModel.build(ssp_data=ssp, **build_kwargs)
    return [w for w in caught if issubclass(w.category, category)]


def test_fires_for_a_wide_free_redshift_with_the_default_dpl_prior(synthetic_ssp_wide):
    """redshift=Uniform(2, 6) leaves sfh_dpl_age_gyr's z_floor-capped ceiling
    (age_at_z(2) = 3.288 Gyr) above age_at_z(6) = 0.934 Gyr: exactly the gap.
    """
    fired = _warnings_of(
        FreeRedshiftOnsetCeilingWarning,
        synthetic_ssp_wide,
        sfh={"type": "dpl", "all_params": FREE},
        redshift=Uniform(2.0, 6.0),
    )
    assert len(fired) == 1, f"expected exactly one warning, got {len(fired)}"
    msg = str(fired[0].message)
    assert "sfh_dpl_age_gyr" in msg
    assert "0.9342" in msg or "0.934" in msg  # age_at_z(6), to 4 sig figs
    assert "3.288" in msg  # age_at_z(2)


def test_does_not_fire_for_a_fixed_redshift(synthetic_ssp_wide):
    """A Fixed redshift has bounds (z0, z0): no upper end to outrun the cap."""
    fired = _warnings_of(
        FreeRedshiftOnsetCeilingWarning,
        synthetic_ssp_wide,
        sfh={"type": "dpl", "all_params": FREE},
        redshift=Fixed(2.0),
    )
    assert fired == []


def test_does_not_fire_when_the_onset_ceiling_is_already_safe(synthetic_ssp_wide):
    """An explicit onset ceiling already below age_at_z(z_ceil) needs no warning.

    ``age_at_z(6) = 0.9342`` Gyr; a user-supplied ``Uniform(0.05, 0.5)`` never
    reaches it, so the same free-redshift range that triggers the default-prior
    case above must stay silent here. This also exercises the "final declared
    ceiling" read (post- :func:`_narrow_free_priors_to_z`), not the pre-cap
    declaration, for a parameter provenance the z_floor cap never touches
    (an explicit user prior).
    """
    fired = _warnings_of(
        FreeRedshiftOnsetCeilingWarning,
        synthetic_ssp_wide,
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_gyr": Uniform(0.05, 0.5),
            "log_total_mass": Fixed(10.0),
        },
        redshift=Uniform(2.0, 6.0),
    )
    assert fired == []
