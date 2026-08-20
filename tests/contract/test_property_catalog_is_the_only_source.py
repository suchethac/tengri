# SPDX-License-Identifier: BSD-3-Clause
r"""Contract §1: one property name, one number — whatever surface you read it from.

``pred.stellar_mass`` is specified as sugar for ``pred.properties["stellar_mass"]``.
It was not. ``Prediction`` carried 31 flat ``@property`` accessors, 28 of which
duplicated a catalog property with an **independent implementation** — and
``__getattr__`` (which does consult the catalog) never fires for a name that
already resolves as a real attribute.

So the catalog was shadowed for every legacy name. Eight had drifted (#1131):

===========================  =====================  ====================
property                     ``pred.properties``    ``pred.X`` (sugar)
===========================  =====================  ====================
``q_h``                      2.098e+49              **NaN**
``xi_ion``                   1.990e+07              **NaN**
``irx``                      4.0501                 0.9270
``mass_weighted_age_gyr``    8.3997                 8.0282
``sfr_100myr``               0.27509                0.27424
===========================  =====================  ====================

The other 20 agreed only by luck. This suite makes the contract true by
construction rather than by coincidence: it sweeps **every** property on
**every** surface and demands one number. A future accessor that recomputes
instead of delegating fails here immediately.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.contract


@pytest.fixture(scope="module")
def model(synthetic_ssp_wide, synthetic_tophat_obs):
    return SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )


@pytest.fixture(scope="module")
def params(model):
    return {k: jnp.asarray(v) for k, v in model.spec.sample(jax.random.PRNGKey(0)).items()}


@pytest.fixture(scope="module")
def pred(model, params):
    return model.predict(params)


def _scalar(x):
    return float(np.asarray(x, dtype=float).ravel()[0])


def test_the_sweep_is_not_vacuous(model, pred):
    """It must actually cover the names that broke, or it proves nothing."""
    names = set(model.available_properties)
    assert len(names) > 20
    for known in ("irx", "q_h", "xi_ion", "mass_weighted_age_gyr", "sfr_100myr"):
        assert known in names, f"{known} vanished from the catalog — the sweep has a hole"


def test_the_catalog_works_on_a_fresh_model(synthetic_ssp_wide, synthetic_tophat_obs, params):
    """``pred.properties[...]`` must work without a warm-up call (#1131).

    The catalog is assembled lazily, and only ``available_properties`` and
    ``predict_properties`` used to trigger it — while ``Prediction.properties``
    read the attribute directly. So on a freshly built model this raised
    ``AttributeError: 'SEDModel' object has no attribute '_property_catalog'``,
    and it *worked* only if you happened to have touched one of the other two
    first. Build the model here and touch nothing else beforehand.
    """
    fresh = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "*": FIXED, "log_total_mass": Uniform(9.0, 11.0)},
        dust_attenuation={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(0.1),
    )
    pred = fresh.predict(params)  # no available_properties / predict_properties first

    assert np.isfinite(_scalar(pred.properties["stellar_mass"]))
    assert np.isfinite(_scalar(pred.stellar_mass))  # the sugar, and the groups it forwards to
    assert np.isfinite(_scalar(pred.sfh.stellar_mass))
    assert "stellar_mass" in pred.properties
    assert len(list(pred.properties)) > 20


def test_attribute_sugar_is_the_catalog(model, pred):
    """Contract §1: ``pred.X`` IS ``pred.properties["X"]`` — for every X, no exceptions.

    Coverage note, stated rather than hidden: a property that is NaN on *both*
    surfaces carries no signal, so it cannot detect a broken accessor. On this
    fixture that is the emission-line family — the synthetic CI SSP has no nebular
    backend publishing a per-line catalog (that is #361's territory, and the
    warning contract is tested in
    ``tests/regression/bug/test_bug_361_nebular_line_silent_nan.py``). The
    ``compared`` counter below fails loudly if this sweep ever degenerates to
    comparing nothing.
    """
    split, compared = [], 0
    for name in sorted(model.available_properties):
        catalog = _scalar(pred.properties[name])
        sugar = _scalar(getattr(pred, name))
        if np.isnan(catalog) and np.isnan(sugar):
            continue  # no signal — see the coverage note above
        compared += 1
        if not np.isclose(catalog, sugar, rtol=1e-9, equal_nan=True):
            split.append((name, catalog, sugar))

    # Two families are legitimately NaN on this fixture and carry no signal:
    # the emission lines (no nebular backend publishes a per-line catalog) and
    # stellar_mass_surviving (the synthetic SSP has no mass-remaining table, so
    # "how much mass survives" genuinely has no answer — see #1131). That leaves
    # ~23 with signal; the floor guards against the sweep silently degenerating
    # further, which is the only way it could pass while proving nothing.
    assert compared >= 20, (
        f"only {compared} properties carried any signal — this sweep has gone "
        "vacuous and can no longer detect a shadowed accessor"
    )
    assert not split, (
        "pred.X and pred.properties['X'] returned DIFFERENT numbers — the accessor "
        "is recomputing instead of reading the catalog (#1131):\n"
        + "\n".join(f"  {n}: catalog={c!r} sugar={s!r}" for n, c, s in split)
    )


def test_the_sugar_never_turns_a_finite_value_into_nan(model, pred):
    """A NaN with no error is the failure mode this whole campaign is about.

    ``pred.q_h`` and ``pred.xi_ion`` returned NaN while the catalog returned finite
    values. Note the assertion is one-directional on purpose: a property the model
    genuinely cannot produce (line fluxes with ``neb={'type': 'none'}``) is NaN on
    *both* surfaces, and that is honest. What is not allowed is the catalog knowing
    the answer and the accessor handing back NaN.
    """
    lost = [
        n
        for n in sorted(model.available_properties)
        if np.isfinite(_scalar(pred.properties[n])) and np.isnan(_scalar(getattr(pred, n)))
    ]
    assert not lost, (
        f"the catalog computes these, but attribute access returns NaN (#1131): {lost}"
    )


def test_predict_properties_agrees_with_the_prediction(model, params, pred):
    """The JIT surface (contract §2) and the exploration surface are one catalog."""
    batch = model.predict_properties(params)
    for name, value in batch.items():
        np.testing.assert_allclose(
            _scalar(value),
            _scalar(pred.properties[name]),
            rtol=1e-9,
            err_msg=f"predict_properties()[{name!r}] != pred.properties[{name!r}]",
        )


def test_the_group_accessors_agree_with_the_catalog(model, pred):
    """``pred.sfh.X`` is a third surface — it must not be a third implementation."""
    split = []
    for group in ("sfh", "sed", "lines", "radio", "xray", "ionizing"):
        obj = getattr(pred, group, None)
        if obj is None:
            continue
        for name in sorted(model.available_properties):
            if not hasattr(type(obj), name):
                continue
            a, b = _scalar(getattr(obj, name)), _scalar(pred.properties[name])
            if np.isnan(a) and np.isnan(b):
                continue
            if not np.isclose(a, b, rtol=1e-9, equal_nan=True):
                split.append((f"pred.{group}.{name}", a, b))

    assert not split, "a group accessor disagrees with the catalog (#1131):\n" + "\n".join(
        f"  {n}: group={a!r} catalog={b!r}" for n, a, b in split
    )


def test_irx_uses_the_speed_of_light_from_physics_constants(model, pred):
    """Regression (#1131): ``irx`` hardcoded c = 2.998e15, 1000x too small in [A/s].

    That inflated every reported IRX by exactly log10(1000) = 3 dex. The two UV
    anchors are now separate, correctly-named properties — but neither may invent
    its own speed of light.
    """
    from tengri.utils.physics_constants import C_AA
    from tengri.utils.sed_quantities import compute_fuv_flux, compute_irx, compute_l_tir

    assert pytest.approx(2.99792458e18) == C_AA

    state = pred._ensure_state()
    sed, wave = state.sed_intrinsic, state.wave
    expected = compute_irx(compute_l_tir(sed, wave), compute_fuv_flux(sed, wave) * C_AA / 1500.0)

    got = _scalar(pred.properties["irx_fuv"])
    np.testing.assert_allclose(got, _scalar(expected), rtol=1e-12)

    # Vacuity guard: the OLD (buggy) constant must give a visibly different answer,
    # or this test could pass against the bug.
    buggy = _scalar(
        compute_irx(compute_l_tir(sed, wave), compute_fuv_flux(sed, wave) * 2.998e15 / 1500.0)
    )
    assert abs(buggy - got) == pytest.approx(3.0, abs=0.01), (
        "the 1000x constant error should shift IRX by exactly 3 dex; if it does not, "
        "this test cannot detect the regression"
    )


def test_both_uv_anchors_are_available_and_distinct(model, pred):
    """Two definitions, two names — the fix for one name carrying two numbers.

    ``irx``     : monochromatic nu*L_nu at 1600 A (Meurer+99, the IRX-beta anchor)
    ``irx_fuv`` : band-averaged FUV (1000-1700 A), pivoted at 1500 A
    """
    names = set(model.available_properties)
    assert {"irx", "irx_fuv"} <= names

    irx = _scalar(pred.properties["irx"])
    irx_fuv = _scalar(pred.properties["irx_fuv"])

    assert np.isfinite(irx) and np.isfinite(irx_fuv)
    # They must be genuinely different quantities — if they collapsed to the same
    # number, one of them is not measuring what its name says.
    assert abs(irx - irx_fuv) > 1e-3
    # ...but the same order of magnitude: a 3 dex gap means the c bug is back.
    assert abs(irx - irx_fuv) < 1.0
