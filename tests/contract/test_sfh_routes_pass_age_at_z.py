# SPDX-License-Identifier: BSD-3-Clause
"""Every route that evaluates the SFH hands the age-anchored families age(z) (#2592).

``dense_basis``, ``dense_basis_pure`` and ``psb`` anchor their time axis to the
age of the universe at the galaxy's redshift (Iyer et al. 2019; Wild et al.
2020, eq. 5), so the SFH function requires ``age_universe_yr``. The stellar
component supplies it for the forward model; the SED model's own history
routes (``predict_sfh``, ``predict_sfh_quantities``) call the same function and
must supply the same value through the same rule
(:func:`tengri.components.stellar.component.age_universe_kwargs`).

The reference is the stellar component's own history,
``predict_state(p).derived["sfr_history"]`` on ``derived["sfh_grid_lbt_yr"]``.
Both come from one SFH function with one set of kwargs, so they agree to
rounding: the tolerance is ``rtol=1e-9``. An injection of the z = 0 age
(13.47 Gyr) in place of age(z = 2) = 3.28 Gyr moves the history by order
unity, so it fails every comparison by many orders of magnitude.

References
----------
.. [1] Iyer, K. G., Gawiser, E., et al. 2019, ApJ 879, 116.
.. [2] Wild, V., et al. 2020, MNRAS 494, 529.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest
from astropy.cosmology import Planck18

from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.component import (
    _AGE_FAMILIES,
    _FAST_PATH_UNSUPPORTED_SFH_FNS,
    StellarSEDComponent,
    age_universe_kwargs,
)
from tengri.components.stellar.sfh import sample_sfh_prior
from tengri.components.stellar.sfh.registry import SFH_REGISTRY, UNVALIDATED_SFH_TYPES
from tengri.utils.grid import interpolate_to_linear_time

pytestmark = pytest.mark.contract

#: Redshift at which age(z) = 3.28 Gyr, far from the z = 0 age of 13.47 Gyr.
_Z = 2.0
_Z_OTHER = 0.5

#: History routes are compared with the component's own history to rounding.
_RTOL = 1e-9

#: Largest SFR beyond age(z), as a fraction of the peak SFR.
_BEYOND_AGE_FLOOR = 1e-15

#: Families the builder accepts; the rest are covered by the rule's own test.
_BUILDABLE = tuple(f for f in _AGE_FAMILIES if f not in UNVALIDATED_SFH_TYPES)

#: Families whose SFH ends at age(z) for every parameter draw. ``psb`` does not:
#: its own age parameter is bounded by the prior, not by the cosmic age, so a
#: raw prior draw may form stars before the Big Bang (the model-level cases
#: above fix that parameter below age(z)).
_SUPPORT_ENDS_AT_AGE = ("dense_basis", "dense_basis_pure")

#: Per-family SFH settings of the model-level cases. Every buildable family in
#: ``_AGE_FAMILIES`` needs an entry (``{}`` when the defaults are fine), so a
#: family added later cannot be skipped silently.
_FAMILY_SETTINGS = {
    "dense_basis": {},
    "dense_basis_pure": {},
    "psb_suess2022": {},
    "psb_flex": {},
    "psb": {"age_gyr": Fixed(2.0), "tau_gyr": Fixed(1.0), "burstage_gyr": Fixed(0.3)},
}


def _age_yr(z: float) -> float:
    """Age of the universe at ``z`` [yr], independently of tengri's cosmology module."""
    return float(Planck18.age(z).to_value("yr"))


def test_every_buildable_age_family_has_settings():
    """A family added to ``_AGE_FAMILIES`` must be given settings here, not skipped."""
    missing = [f for f in _BUILDABLE if f not in _FAMILY_SETTINGS]
    assert not missing, f"add _FAMILY_SETTINGS entries for {missing}"


def _sfh_block(family) -> dict:
    block = {"type": family, "all_params": Fixed(DEFAULT)}
    if isinstance(family, str):
        block.update(_FAMILY_SETTINGS[family])
    return block


def _build(ssp, family: str, redshift):
    return SEDModel.build(
        ssp_data=ssp,
        sfh=_sfh_block(family),
        met={"type": "delta", "logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "none"},
        dust_emission={"type": "none"},
        neb={"type": "none"},
        redshift=redshift,
    )


def _component_history(model, params):
    """The stellar component's own SFR history on the model's lookback grid."""
    state = model.predict_state(params)
    lbt_yr = np.asarray(state.derived["sfh_grid_lbt_yr"])
    sfr = np.asarray(state.derived["sfr_history"])
    return np.interp(np.asarray(model.age_yr), lbt_yr, sfr)


def _window_mean(age_yr, sfr, window_yr):
    """Mean SFR over lookback <= window, with the weights of ``predict_sfh_quantities``."""
    dt = np.gradient(age_yr)
    mask = age_yr <= window_yr
    return float(np.sum(np.where(mask, sfr * dt, 0.0)) / max(np.sum(np.where(mask, dt, 0.0)), 1.0))


@pytest.mark.parametrize("family", _BUILDABLE)
def test_predict_sfh_native_equals_component_history(family, synthetic_ssp_wide):
    """``predict_sfh`` runs and reproduces the component's history at z = 2."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    out = model.predict_sfh({}, grid="native")
    ref = _component_history(model, {})

    sfr = np.asarray(out["sfr_mean"])
    np.testing.assert_allclose(sfr, ref, rtol=_RTOL, atol=0.0)
    assert np.any(sfr > 0.0), f"{family}: predict_sfh returned no star formation"


@pytest.mark.parametrize("family", _BUILDABLE)
def test_predict_sfh_linear_grid_equals_component_history(family, synthetic_ssp_wide):
    """The linear-time grid of ``predict_sfh`` is the component history, resampled alike."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    out = model.predict_sfh({})
    _, ref = interpolate_to_linear_time(model.log_age_grid, _component_history(model, {}), 1000)

    np.testing.assert_allclose(np.asarray(out["sfr_mean"]), np.asarray(ref), rtol=_RTOL, atol=0.0)


@pytest.mark.parametrize("family", _BUILDABLE)
def test_predict_sfh_vanishes_beyond_age_at_z(family, synthetic_ssp_wide):
    """No star formation at lookback beyond age(z): the grid is bounded by cosmic time."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    out = model.predict_sfh({}, grid="native")
    lbt_yr = np.asarray(out["t_gyr"]) * 1e9
    sfr = np.asarray(out["sfr_mean"])
    beyond = lbt_yr > _age_yr(_Z)

    assert beyond.any()
    # dense_basis is exactly zero there; psb carries a numerical floor ~1e-19
    # Msun/yr from its truncated exponential, ~1e-19 of the peak.
    assert float(np.max(np.abs(sfr[beyond]))) <= _BEYOND_AGE_FLOOR * float(np.max(sfr)), (
        f"{family}: SFR beyond age(z={_Z}) = {_age_yr(_Z) / 1e9:.3f} Gyr"
    )


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.parametrize("family", _BUILDABLE)
def test_predict_sfh_quantities_sfr_windows_equal_component_history(family, synthetic_ssp_wide):
    """``predict_sfh_quantities`` averages the component's history over 10 and 100 Myr."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    quantities = model.predict_sfh_quantities({})
    ref = _component_history(model, {})
    age_yr = np.asarray(model.age_yr)

    assert float(quantities.sfr_100myr) == pytest.approx(_window_mean(age_yr, ref, 1e8), rel=_RTOL)
    assert float(quantities.sfr_10myr) == pytest.approx(_window_mean(age_yr, ref, 1e7), rel=_RTOL)


@pytest.mark.parametrize("family", _BUILDABLE)
def test_free_redshift_uses_the_age_of_each_evaluation(family, synthetic_ssp_wide):
    """A sampled redshift sets age(z) per call: the two histories differ, each matches."""
    model = _build(synthetic_ssp_wide, family, Uniform(0.1, 4.0))
    histories = {}
    for z in (_Z, _Z_OTHER):
        params = {"redshift": z}
        sfr = np.asarray(model.predict_sfh(params, grid="native")["sfr_mean"])
        np.testing.assert_allclose(sfr, _component_history(model, params), rtol=_RTOL, atol=0.0)
        histories[z] = sfr

    assert not np.allclose(histories[_Z], histories[_Z_OTHER], rtol=1e-3)


def test_burst_composite_equals_component_history(synthetic_ssp_wide):
    """A composite with an age-anchored member gets age(z) on every route, too."""
    model = _build(synthetic_ssp_wide, ["dense_basis", "burst"], Fixed(_Z))
    out = model.predict_sfh({}, grid="native")

    np.testing.assert_allclose(
        np.asarray(out["sfr_full"]), _component_history(model, {}), rtol=_RTOL, atol=0.0
    )


def test_field_composite_mean_vanishes_beyond_age_at_z(synthetic_ssp_wide):
    """The smooth part of a field composite is bounded by age(z), as the bare family is."""
    model = _build(synthetic_ssp_wide, ["dense_basis", "field"], Fixed(_Z))
    out = model.predict_sfh({}, grid="native")
    sfr = np.asarray(out["sfr_mean"])
    beyond = np.asarray(out["t_gyr"]) * 1e9 > _age_yr(_Z)

    assert np.any(sfr > 0.0)
    assert float(np.max(np.abs(sfr[beyond]))) == 0.0


@pytest.mark.parametrize("family", _AGE_FAMILIES)
def test_rule_returns_age_of_universe_at_redshift(family):
    """The one rule: age(z) [yr] under the model's cosmology, for every age family."""
    for z in (_Z, _Z_OTHER):
        kwargs = age_universe_kwargs(family, z)
        assert set(kwargs) == {"age_universe_yr"}
        assert float(kwargs["age_universe_yr"]) == pytest.approx(_age_yr(z), rel=1e-6)


def test_rule_is_silent_for_other_families_and_reads_composites():
    """Non-anchored families get nothing; a composite is anchored if any member is."""
    assert age_universe_kwargs("dpl", _Z) == {}
    assert age_universe_kwargs(["tsnorm", "burst"], _Z) == {}
    assert set(age_universe_kwargs(["dense_basis_pure", "burst"], _Z)) == {"age_universe_yr"}
    assert set(age_universe_kwargs(("dense_basis", "field"), _Z)) == {"age_universe_yr"}


def _has_fast_path(family: str) -> bool:
    """Whether the weights-only route supports ``family`` (closed-form SFHs only)."""
    return SFH_REGISTRY[family].fn not in _FAST_PATH_UNSUPPORTED_SFH_FNS


def _stellar_component(model):
    return next(c for c in model._build_component_chain() if isinstance(c, StellarSEDComponent))


@pytest.mark.parametrize("family", [f for f in _BUILDABLE if _has_fast_path(f)])
def test_precompute_route_equals_exact_route(family, synthetic_ssp_wide):
    """The weights-only route injects the same age(z) as the exact forward."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    weights, _, _ = _stellar_component(model).compute_joint_weights(
        model._evaluation_params({}, None)
    )
    exact = np.asarray(model.predict_state({}).derived["joint_weights"])

    np.testing.assert_allclose(np.asarray(weights), exact, rtol=1e-8, atol=1e-14)


@pytest.mark.parametrize("family", [f for f in _BUILDABLE if not _has_fast_path(f)])
def test_precompute_route_refuses_families_without_a_fast_path(family, synthetic_ssp_wide):
    """A family with no closed-form SFR is refused by the weights-only route, not mis-evaluated."""
    model = _build(synthetic_ssp_wide, family, Fixed(_Z))
    with pytest.raises(ValueError, match="does not support"):
        _stellar_component(model).compute_joint_weights(model._evaluation_params({}, None))


@pytest.mark.parametrize("family", _BUILDABLE)
def test_sample_sfh_prior_is_evaluated_at_age_of_the_redshift(family):
    """Prior draws vanish beyond age(z) and move with the redshift, same key."""
    key = jax.random.PRNGKey(3)
    age_yr, curves = sample_sfh_prior(family, key, n=6, redshift=_Z)
    _, other = sample_sfh_prior(family, key, n=6, redshift=_Z_OTHER)
    curves = np.asarray(curves)
    beyond = np.asarray(age_yr) > _age_yr(_Z)

    assert beyond.any()
    peak = np.max(curves, axis=1)
    assert np.all(peak > 0.0)
    if family in _SUPPORT_ENDS_AT_AGE:
        assert np.all(np.max(np.abs(curves[:, beyond]), axis=1) <= _BEYOND_AGE_FLOOR * peak)
    assert not np.allclose(curves, np.asarray(other), rtol=1e-3)


@pytest.mark.parametrize("family", _BUILDABLE)
def test_sample_sfh_prior_requires_redshift_for_age_families(family):
    """No silent default age: an age-anchored family without a redshift raises."""
    with pytest.raises(ValueError, match="redshift"):
        sample_sfh_prior(family, jax.random.PRNGKey(0), n=2)


def test_sample_sfh_prior_other_families_need_no_redshift():
    """A family with no age anchor is unchanged by the new argument."""
    _, plain = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=3)
    _, with_z = sample_sfh_prior("dpl", jax.random.PRNGKey(0), n=3, redshift=_Z)
    np.testing.assert_array_equal(np.asarray(plain), np.asarray(with_z))
