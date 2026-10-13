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

The two post-starburst quench times (``tlast`` and ``tflex``) are fractions
too. Their ordering is fixed by construction: the floor of the ``tflex``
fraction is the ceiling of the ``tlast`` fraction, and both scale with the same
age(z), so the flexible zone never has negative width at any redshift.
"""

import pytest

import tengri
from tengri import DEFAULT, FREE, ConfigError, Fixed, Uniform
from tengri.components.stellar.sfh.registry import (
    _AGE_FRAC_PRIORS,
    SFH_REGISTRY,
    _spec_public_prefix,
)
from tengri.cosmology import age_at_z
from tengri.parameters.translate import _build_param_map, get_internal_params

pytestmark = pytest.mark.regression_bug

#: One representative per onset family: (SFH type, onset stem). The fraction is
#: ``sfh_<type>_<stem>_frac`` and its absolute twin ``sfh_<type>_<stem>_gyr``.
#: The two post-starburst families are listed once per quench time.
FAMILIES = (
    ("dpl", "age"),
    ("delayed", "age"),
    ("exp", "start"),
    ("declining_exp", "age"),
    ("psb", "age"),
    ("delayed_bq", "age_main"),
    ("lnorm", "age"),
    ("tsnorm", "peak_lbt"),
    ("snorm", "peak_lbt"),
    ("snorm_burst", "peak_lbt"),
    ("tsnorm_burst", "peak_lbt"),
    ("norm", "peak_lbt"),
    ("psb_suess2022", "tflex"),
    ("psb_suess2022", "tlast"),
    ("psb_flex", "tflex"),
    ("psb_flex", "tlast"),
)

_PSB_TYPES = ("psb_suess2022", "psb_flex")
_PSB_PARTNER_STEM = {"tflex": "tlast", "tlast": "tflex"}


def _names(family, stem):
    """Full frac name, full Gyr name and short frac name of a timing."""
    prefix = _spec_public_prefix(SFH_REGISTRY[family])
    return f"{prefix}{stem}_frac", f"{prefix}{stem}_gyr", f"{stem}_frac"


def _sfh_for(family, stem, prior):
    """The sfh group that samples one timing fraction, with its PSB partner when there is one.

    A PSB pair must be sampled in full: a fraction given alone would pair with
    the absolute Gyr default of its partner, which is the mixed case refused at
    build (#2748).
    """
    sfh = {"type": family, "all_params": Fixed(DEFAULT), f"{stem}_frac": prior}
    if family in _PSB_TYPES:
        partner = _PSB_PARTNER_STEM[stem]
        sfh[f"{partner}_frac"] = _AGE_FRAC_PRIORS[_names(family, partner)[0]]
    return sfh


def _onset_gyr_at(family, stem, frac, z_at):
    """Absolute onset [Gyr] that a sampled fraction produces at z_at."""
    frac_name, gyr_name, _ = _names(family, stem)
    prior = _AGE_FRAC_PRIORS[frac_name]
    spec = tengri.parse_groups(
        sfh=_sfh_for(family, stem, prior),
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Uniform(0.5, 12.0, default=5.0),
    )
    param_map = _build_param_map([family])
    params = {frac_name: frac, "redshift": z_at}
    if family in _PSB_TYPES:
        partner_name, _, _ = _names(family, _PSB_PARTNER_STEM[stem])
        params[partner_name] = _AGE_FRAC_PRIORS[partner_name].default
    internal = get_internal_params(params, param_map, spec, False)
    _, gyr_scale, _ = param_map[gyr_name]
    return float(internal[param_map[gyr_name][0]]) / gyr_scale


@pytest.mark.parametrize("z_at", [2.0, 10.0])
@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f"{f}-{s}" for f, s in FAMILIES])
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


@pytest.mark.parametrize(("family", "stem"), FAMILIES, ids=[f"{f}-{s}" for f, s in FAMILIES])
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
    frac_name, _, _ = _names(family, stem)
    overrides = {frac_name: Uniform(0.5, 1.0)}
    if family in _PSB_TYPES:
        partner_name, _, _ = _names(family, _PSB_PARTNER_STEM[stem])
        overrides[partner_name] = _AGE_FRAC_PRIORS[partner_name]
    _, frac_curves = sample_sfh_prior(family, key, n=2, redshift=2.0, **overrides)
    assert bool(jax.numpy.isfinite(frac_curves).all())


@pytest.mark.parametrize("z_at", [0.0, 2.0, 10.0])
@pytest.mark.parametrize("family", _PSB_TYPES)
def test_psb_fraction_edges_keep_tlast_within_tflex(family, z_at):
    """tflex_frac at its floor and tlast_frac at its ceiling leave a non-negative zone (#2748).

    The fractions share the age(z) factor, so the Gyr ordering that holds at
    the edges holds at every redshift the history is built at.
    """
    tlast_frac, tlast_gyr, _ = _names(family, "tlast")
    tflex_frac, tflex_gyr, _ = _names(family, "tflex")
    spec = tengri.parse_groups(
        sfh=_sfh_for(family, "tflex", _AGE_FRAC_PRIORS[tflex_frac]),
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Uniform(0.5, 12.0, default=5.0),
    )
    param_map = _build_param_map([family])
    tflex_floor = _AGE_FRAC_PRIORS[tflex_frac].bounds[0]
    tlast_ceiling = _AGE_FRAC_PRIORS[tlast_frac].bounds[1]
    internal = get_internal_params(
        {tflex_frac: tflex_floor, tlast_frac: tlast_ceiling, "redshift": z_at},
        param_map,
        spec,
        False,
    )
    last = float(internal[param_map[tlast_gyr][0]])
    flex = float(internal[param_map[tflex_gyr][0]])
    assert last <= flex, f"{family} at z={z_at:g}: tlast {last:.4f} Gyr above tflex {flex:.4f} Gyr"


@pytest.mark.parametrize("family", _PSB_TYPES)
def test_psb_quench_units_cannot_be_mixed(family):
    """A fraction for one quench time with Gyr for the other is refused at build (#2748).

    The Gyr default of the unnamed time would be compared against a fraction of
    age(z) it cannot be ordered with at every redshift, so the grammar refuses
    the pair rather than guess.
    """
    tlast_frac, _, _ = _names(family, "tlast")
    with pytest.raises(ConfigError, match="mixed"):
        tengri.parse_groups(
            sfh={
                "type": family,
                "all_params": Fixed(DEFAULT),
                "tlast_frac": _AGE_FRAC_PRIORS[tlast_frac],
                "tflex_gyr": Fixed(2.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "none"},
            redshift=Uniform(0.5, 12.0, default=5.0),
        )


@pytest.mark.parametrize("family", _PSB_TYPES)
def test_psb_fraction_crossing_is_refused_in_fraction_units(family):
    """A tlast_frac whose ceiling exceeds the tflex_frac floor is refused (#2748)."""
    with pytest.raises(ValueError, match="negative width"):
        tengri.parse_groups(
            sfh={
                "type": family,
                "all_params": Fixed(DEFAULT),
                "tlast_frac": Uniform(0.08, 0.09),
                "tflex_frac": Uniform(0.01, 0.02),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "none"},
            redshift=Uniform(0.5, 12.0, default=5.0),
        )


@pytest.mark.parametrize(
    "sfh_spelling",
    ["grouped", "flat"],
)
@pytest.mark.parametrize(
    "prior",
    [Uniform(1, 10), FREE],
    ids=["uniform", "free"],
)
def test_free_prior_on_gyr_alias_is_refused_with_fraction_replacement(sfh_spelling, prior):
    """A free prior on an old ``*_gyr`` timing is a ConfigError naming its ``*_frac`` (#2748).

    The grouped and flat spellings both refuse it before the generic FREE error.
    The message converts the Gyr bounds to the fraction of age(z = 0) it replaces.
    """
    from tengri.parameters.parameters import Parameters

    gyr_name = "sfh_dpl_age_gyr"
    frac_name = "sfh_dpl_age_frac"
    with pytest.raises(ConfigError, match=frac_name) as excinfo:
        if sfh_spelling == "grouped":
            tengri.parse_groups(
                sfh={"type": "dpl", "age_gyr": prior},
                dust_attenuation={"type": "none"},
                neb={"type": "none"},
                redshift=Fixed(0.0),
            )
        else:
            Parameters(**{gyr_name: prior})
    message = str(excinfo.value)
    assert "Fixed-only alias" in message
    assert gyr_name in message
    if isinstance(prior, Uniform):
        assert "Uniform(0.07253, 0.7253)" in message


def _timing_gyr_model(synthetic_ssp_wide, sfh_type, timing, z):
    """A model whose one SFH timing is given as ``timing``, with every other parameter Fixed."""
    return tengri.SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": sfh_type, "all_params": Fixed(DEFAULT), **timing},
        dust_attenuation={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )


def test_published_gyr_of_a_fraction_is_the_fraction_of_age_at_z(synthetic_ssp_wide):
    """``sfh_dpl_age_gyr`` is frac * age(z) in Gyr, read through predict_properties (#2748)."""
    import jax

    z = 0.5
    model = _timing_gyr_model(synthetic_ssp_wide, "dpl", {"age_frac": Fixed(0.5)}, z)
    params = model.spec.sample(jax.random.PRNGKey(0))
    props = model.predict_properties(params, names=("sfh_dpl_age_gyr",))
    assert float(props["sfh_dpl_age_gyr"]) == pytest.approx(
        0.5 * float(age_at_z(z)), rel=0, abs=1e-10
    )

    jitted = jax.jit(
        lambda p: model.predict_properties(p, names=("sfh_dpl_age_gyr",))["sfh_dpl_age_gyr"]
    )
    assert float(jitted(params)) == pytest.approx(0.5 * float(age_at_z(z)), rel=0, abs=1e-10)


def test_published_gyr_of_a_fixed_gyr_shim_is_that_gyr(synthetic_ssp_wide):
    """A Fixed ``sfh_dpl_age_gyr`` is published as itself, unchanged by the redshift (#2748)."""
    import jax

    z = 0.5
    model = _timing_gyr_model(synthetic_ssp_wide, "dpl", {"age_gyr": Fixed(5.0)}, z)
    props = model.predict_properties(
        model.spec.sample(jax.random.PRNGKey(0)), names=("sfh_dpl_age_gyr",)
    )
    assert float(props["sfh_dpl_age_gyr"]) == pytest.approx(5.0, rel=0, abs=1e-10)


def test_timing_gyr_property_is_nan_for_an_sfh_without_that_timing(synthetic_ssp_wide):
    """A property named for another family's timing reads NaN, not a stale value (#2748)."""
    import jax

    model = _timing_gyr_model(synthetic_ssp_wide, "exp", {"start_frac": Fixed(0.5)}, 0.5)
    props = model.predict_properties(
        model.spec.sample(jax.random.PRNGKey(0)), names=("sfh_dpl_age_gyr",)
    )
    assert bool(jax.numpy.isnan(props["sfh_dpl_age_gyr"]))
