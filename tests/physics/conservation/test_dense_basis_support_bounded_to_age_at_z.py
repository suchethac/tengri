# SPDX-License-Identifier: BSD-3-Clause
"""dense_basis and dense_basis_pure: SFH support is [0, age(z)] (#2592).

The tx fractions of Iyer et al. (2019) are fractions of the cosmic time
available at the galaxy's redshift, so the star formation history of
``dense_basis`` and ``dense_basis_pure`` spans only [0, age(z)]: the star
formation rate vanishes at lookback times beyond age(z), and the declared
``log_total_mass`` is the mass formed inside that interval. The age of the
universe follows from the redshift and the cosmology; it is not a setting, and
writing either per-family settings key raises, in every spelling the grammar
could carry it.

age(z) is computed here with astropy's Planck18, which carries the same
parameters as tengri's default cosmology.

References
----------
.. [1] Iyer, K. G., Gawiser, E., et al. 2019, ApJ 879, 116 (Non-parametric
   star formation histories for 5 galaxy evolution states).
.. [2] Issue #2592.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from astropy.cosmology import Planck18

from tengri import DEFAULT, ConfigError, Fixed, SEDModel, Uniform
from tengri.components.stellar.component import SFHBeforeBigBangWarning, StellarSEDComponent

pytestmark = pytest.mark.conservation

#: Relative tolerance for kernel mass identity.
_RTOL_MASS = 1e-6

_FAMILIES = ("dense_basis", "dense_basis_pure")
_REDSHIFTS = (0.5, 2.0, 6.0)

_AGE_KEYS = {
    "dense_basis": "sfh_db_age_universe_gyr",
    "dense_basis_pure": "sfh_dbp_age_universe_gyr",
}


def _age_yr(z: float) -> float:
    """Age of the universe at ``z`` in yr, independently of tengri's cosmology module."""
    return float(Planck18.age(z).to_value("yr"))


def _sfh_dict(family) -> dict:
    """Fixed-parameter SFH block for ``family`` (string or composite list)."""
    sfh = {
        "type": family,
        "log_total_mass": Fixed(10.0),
        "tx_frac_0": Fixed(0.3),
        "tx_frac_1": Fixed(0.5),
        "tx_frac_2": Fixed(0.8),
        "all_params": Fixed(DEFAULT),
    }
    if family == "dense_basis":
        sfh["log_sfr_inst"] = Fixed(float(np.log10(2.0)))
    return sfh


def _build(ssp, z, family, **extra):
    extra.setdefault("redshift", Fixed(z))
    return SEDModel.build(
        ssp_data=ssp,
        sfh=_sfh_dict(family),
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        **extra,
    )


@pytest.mark.parametrize("z", _REDSHIFTS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_mass_formed_inside_age_at_z(family, z, synthetic_ssp_wide):
    """The declared mass forms inside [0, age(z)] and no SFR lies beyond age(z)."""
    state = _build(synthetic_ssp_wide, z, family).predict_state({})
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    age_yr = _age_yr(z)

    formed = 10.0 ** float(state.derived["log_mstar_formed"])
    assert formed == pytest.approx(10.0**10.0, rel=_RTOL_MASS), (
        f"{family} z={z}: formed mass {formed:.6e} vs declared {1e10:.6e}"
    )

    inside = lbt_yr <= age_yr
    mass_all = float(np.trapezoid(sfr, lbt_yr))
    mass_inside = float(np.trapezoid(np.where(inside, sfr, 0.0), lbt_yr))
    assert mass_inside == pytest.approx(mass_all, rel=1e-9), (
        f"{family} z={z}: mass inside age(z) {mass_inside:.6e} vs all lookback {mass_all:.6e}"
    )
    max_outside = float(np.max(np.abs(np.where(inside, 0.0, sfr))))
    assert max_outside == 0.0, f"{family} z={z}: SFR beyond age(z): {max_outside:.4e}"


@pytest.mark.parametrize("z", (0.5, 2.0))
@pytest.mark.parametrize("family", _FAMILIES)
def test_tx_axis_scales_with_age_at_z(family, z, synthetic_ssp_wide):
    """The oldest star formation sits at 0.9-1.0 x age(z), not at a fixed z = 0 age."""
    state = _build(synthetic_ssp_wide, z, family).predict_state({})
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    age_yr = _age_yr(z)

    assert np.any(sfr > 0.0), f"{family} z={z}: no SFR > 0"
    oldest = float(np.max(lbt_yr[sfr > 0.0]))
    assert 0.9 * age_yr <= oldest <= age_yr, (
        f"{family} z={z}: oldest SFR at {oldest / 1e9:.3f} Gyr, age(z) = {age_yr / 1e9:.3f} Gyr"
    )


def _refusal_spellings(family):
    """Every way the grammar could carry the per-family age-of-universe key."""
    key = _AGE_KEYS[family]
    return {
        "settings_kwarg": {"settings": {key: 10.0}},
        "flat_kwarg": {key: 10.0},
        "sfh_dict_key": {"sfh_extra": {key: 10.0}},
    }


@pytest.mark.parametrize("spelling", ("settings_kwarg", "flat_kwarg", "sfh_dict_key"))
@pytest.mark.parametrize("family", _FAMILIES)
def test_age_setting_is_refused(family, spelling, synthetic_ssp_wide):
    """Each spelling of the retired key raises one message naming redshift and cosmology."""
    kwargs = dict(_refusal_spellings(family)[spelling])
    sfh_extra = kwargs.pop("sfh_extra", {})
    sfh = {**_sfh_dict(family), **sfh_extra}
    with pytest.raises(ConfigError) as err:
        SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh=sfh,
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            neb={"type": "none"},
            redshift=Fixed(0.5),
            **kwargs,
        )
    message = str(err.value)
    assert _AGE_KEYS[family] in message
    assert "redshift" in message and "cosmology" in message


@pytest.mark.parametrize("family", _FAMILIES)
def test_registry_refuses_age_setting(family, monkeypatch):
    """A registry spec carrying the key (the registry layer) raises the same message."""
    from dataclasses import replace

    from tengri.components.stellar.sfh import registry

    entry = registry.SFH_REGISTRY[family]
    spec = entry.callable
    patched_spec = spec._replace(settings={**spec.settings, _AGE_KEYS[family]: 10.0})
    patched = replace(entry, callable=patched_spec)
    monkeypatch.setitem(registry.SFH_REGISTRY, family, patched)
    with pytest.raises(ConfigError, match=r"redshift.*cosmology"):
        registry.resolve_sfh(family)


@pytest.mark.parametrize("z", (2.0,))
def test_composite_member_gets_age_at_z(z, synthetic_ssp_wide):
    """A composite SFH containing dense_basis forms its declared mass inside age(z)."""
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        sfh={"type": ["dense_basis", "field"], "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=Fixed(z),
    )
    state = model.predict_state({})
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    age_yr = _age_yr(z)

    formed = 10.0 ** float(state.derived["log_mstar_formed"])
    assert formed == pytest.approx(10.0**10.0, rel=_RTOL_MASS)
    assert float(np.max(lbt_yr[sfr > 0.0])) <= age_yr


@pytest.mark.parametrize("z", _REDSHIFTS)
@pytest.mark.parametrize("family", _FAMILIES)
def test_no_before_big_bang_warning(family, z, synthetic_ssp_wide):
    """The history lies inside [0, age(z)], so no part of it precedes the Big Bang."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        _build(synthetic_ssp_wide, z, family).predict_state({})
    texts = [str(w.message) for w in caught if issubclass(w.category, SFHBeforeBigBangWarning)]
    assert not texts, f"{family} z={z}: {texts[0]}"


def _stellar_of(model):
    """The stellar component of ``model``'s built chain."""
    return next(c for c in model._build_component_chain() if isinstance(c, StellarSEDComponent))


@pytest.mark.parametrize("z", (0.5, 2.0))
@pytest.mark.parametrize("family", _FAMILIES)
def test_precompute_route_support_bounded_by_age_at_z(family, z, synthetic_ssp_wide):
    """The weights-only route has no weight above the age(z) bracketing node and keeps the mass."""
    model = _build(synthetic_ssp_wide, z, family)
    stellar = _stellar_of(model)
    joint_weights, total_mass, ages_yr = stellar.compute_joint_weights(
        model._evaluation_params({}, None)
    )
    weight_per_age = np.asarray(joint_weights).sum(axis=0)
    ages_yr = np.asarray(ages_yr)
    age_yr = _age_yr(z)

    assert ages_yr.max() > age_yr, "SSP grid must extend beyond age(z) for the bound to bite"
    bracket_yr = ages_yr[ages_yr >= age_yr].min()
    beyond = float(weight_per_age[ages_yr > bracket_yr].sum())
    assert beyond == 0.0, f"{family} z={z}: weight {beyond:.3e} above {bracket_yr / 1e9:.3f} Gyr"
    assert float(total_mass) == pytest.approx(10.0**10.0, rel=_RTOL_MASS), (
        f"{family} z={z}: precompute mass {float(total_mass):.6e} vs declared {1e10:.6e}"
    )
    # The CIC kernel truncates at age(z) on its own, so (a) and (b) hold even when the
    # tx axis is scaled to the wrong age; the weights' shape pins the injection.
    exact = np.asarray(model.predict_state({}).derived["joint_weights"])
    np.testing.assert_allclose(np.asarray(joint_weights), exact, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("family", _FAMILIES)
def test_free_redshift_uses_each_samples_age(family, synthetic_ssp_wide):
    """With a free redshift each sample's history spans [0, age(z)] of its own redshift."""
    model = _build(synthetic_ssp_wide, 0.5, family, redshift=Uniform(0.3, 3.0))
    assert "redshift" in model.spec.free_params
    for z in (0.5, 2.0):
        state = model.predict_state({"redshift": z})
        lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
        sfr = np.asarray(state.derived["sfr_history"])
        age_yr = _age_yr(z)
        oldest = float(np.max(lbt_yr[sfr > 0.0]))
        assert 0.9 * age_yr <= oldest <= age_yr, (
            f"{family} z={z}: oldest SFR at {oldest / 1e9:.3f} Gyr, "
            f"age(z) = {age_yr / 1e9:.3f} Gyr"
        )
