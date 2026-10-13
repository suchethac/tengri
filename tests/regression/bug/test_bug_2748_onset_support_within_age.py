# SPDX-License-Identifier: BSD-3-Clause
r"""Regression test for #2748: the SF-onset prior must not reach past age(z).

With a free redshift the absolute-Gyr onset prior's ceiling is narrowed only
against the lowest redshift the prior admits (z = 0.5, age 8.59 Gyr). A
galaxy at z = 2 (age 3.28 Gyr) or z = 10 (age 0.47 Gyr) can then draw an onset
older than the universe at its own redshift. The mass-conserving normalization
zeroes the support beyond age(z), which leaves a flat, zero-gradient region
that kills the HMC warmup of the population pilot (#2567 bisect, #2748).

The owner ruling reparametrises each onset as a fraction of age(z),
``sfh_<family>_<stem>_frac``, so the support is bounded by age(z) at every
redshift. This test pins the requirement for one representative of each
onset family: the onset the fraction produces, evaluated at z, never exceeds
age(z), at the top and the bottom of the fraction's prior.
"""

import pytest

import tengri
from tengri import DEFAULT, Fixed, Uniform
from tengri.components.stellar.sfh.registry import _AGE_FRAC_PRIORS
from tengri.cosmology import age_at_z
from tengri.parameters.translate import _build_param_map, get_internal_params

pytestmark = pytest.mark.regression_bug

#: One representative per onset family: (SFH type, onset stem). The fraction is
#: ``sfh_<type>_<stem>_frac`` and its absolute twin ``sfh_<type>_<stem>_gyr``.
FAMILIES = (
    ("dpl", "age"),
    ("delayed", "age"),
    ("exp", "start"),
    ("declining_exp", "age"),
    ("psb", "age"),
    ("delayed_bq", "age_main"),
    ("lnorm", "age"),
)


def _names(family, stem):
    return f"sfh_{family}_{stem}_frac", f"sfh_{family}_{stem}_gyr", f"{stem}_frac"


def _onset_gyr_at(family, stem, frac, z_at):
    """Absolute onset [Gyr] that a sampled fraction produces at z_at."""
    frac_name, gyr_name, short = _names(family, stem)
    prior = _AGE_FRAC_PRIORS[frac_name]
    spec = tengri.parse_groups(
        sfh={"type": family, "all_params": Fixed(DEFAULT), short: prior},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Uniform(0.5, 12.0, default=5.0),
    )
    param_map = _build_param_map([family])
    internal = get_internal_params({frac_name: frac, "redshift": z_at}, param_map, spec, False)
    _, gyr_scale, _ = param_map[gyr_name]
    return float(internal[param_map[gyr_name][0]]) / gyr_scale


@pytest.mark.parametrize("z_at", [2.0, 10.0])
@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f for f, _ in FAMILIES])
def test_onset_prior_support_never_exceeds_age_of_universe_at_z(family, stem, z_at):
    """The onset at either end of the fraction's prior never exceeds age(z)."""
    frac_name, _, _ = _names(family, stem)
    age_gyr = float(age_at_z(z_at))
    lo, hi = _AGE_FRAC_PRIORS[frac_name].bounds
    assert hi <= 1.0
    for frac in (lo, hi):
        onset = _onset_gyr_at(family, stem, frac, z_at)
        assert onset <= age_gyr * (1.0 + 1e-9), (
            f"{frac_name} onset {onset:.3f} Gyr at frac={frac:.4f} is past "
            f"age({z_at:g}) = {age_gyr:.3f} Gyr"
        )


@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f for f, _ in FAMILIES])
def test_prior_sampling_draws_are_finite_for_every_family(family, stem):
    """The SFH prior fan draws every onset family at a redshift, with and without a fraction.

    The default draw of a family with a Fixed fraction used to pass the fraction
    straight to the kernel (#2748).
    """
    import jax

    from tengri.components.stellar.sfh import sample_sfh_prior

    key = jax.random.PRNGKey(0)
    _, default_curves = sample_sfh_prior(family, key, n=2, redshift=2.0)
    assert bool(jax.numpy.isfinite(default_curves).all())
    frac_name = f"sfh_{family}_{stem}_frac"
    _, frac_curves = sample_sfh_prior(
        family, key, n=2, redshift=2.0, **{frac_name: Uniform(0.5, 1.0)}
    )
    assert bool(jax.numpy.isfinite(frac_curves).all())
