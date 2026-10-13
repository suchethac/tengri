# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2748: the SF-onset prior must not reach past age(z).

With a free redshift the absolute-Gyr onset prior's ceiling is narrowed only
against the lowest redshift the prior admits (z = 0.5, age 8.59 Gyr). A
galaxy at z = 2 (age 3.28 Gyr) or z = 10 (age 0.47 Gyr) can then draw an onset
older than the universe at its own redshift. The mass-conserving normalisation
zeroes the support beyond age(z), which leaves a flat, zero-gradient region
that kills the HMC warmup of the population pilot (#2567 bisect, #2748).

The owner ruling reparametrises the onset as a fraction of age(z),
``sfh_dpl_age_frac``, so the support is bounded by age(z) at every redshift.
This test pins the requirement: the onset the fraction produces, evaluated at
z, never exceeds age(z), at the top and the bottom of the fraction's prior.
"""

import pytest

import tengri
from tengri import DEFAULT, Fixed, Uniform
from tengri.components.stellar.sfh.registry import _DPL_AGE_FRAC_PRIOR
from tengri.cosmology import age_at_z
from tengri.parameters.translate import _build_param_map, get_internal_params

pytestmark = pytest.mark.regression_bug


def _onset_gyr_at(frac, z_at):
    """Absolute dpl onset [Gyr] that a sampled ``sfh_dpl_age_frac`` produces at z_at."""
    spec = tengri.parse_groups(
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_frac": _DPL_AGE_FRAC_PRIOR,
        },
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Uniform(0.5, 12.0, default=5.0),
    )
    params = {"sfh_dpl_age_frac": frac, "redshift": z_at}
    internal = get_internal_params(params, _build_param_map(["dpl"]), spec, False)
    return float(internal["age"]) / 1e9


@pytest.mark.parametrize("z_at", [2.0, 10.0])
def test_onset_prior_support_never_exceeds_age_of_universe_at_z(z_at):
    """The onset at either end of the fraction's prior never exceeds age(z)."""
    age_gyr = float(age_at_z(z_at))
    lo, hi = _DPL_AGE_FRAC_PRIOR.bounds
    assert hi <= 1.0
    for frac in (lo, hi):
        onset = _onset_gyr_at(frac, z_at)
        assert onset <= age_gyr * (1.0 + 1e-9), (
            f"onset {onset:.3f} Gyr at frac={frac:.4f} is past age({z_at:g}) = {age_gyr:.3f} Gyr"
        )
