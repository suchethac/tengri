# SPDX-License-Identifier: BSD-3-Clause
"""Regression: the no-state line screen must reach the SAME law kwargs the
continuum does, not a private ``{dust_slope, dust_bump_strength}`` dict (#2223).

``SEDModel._attenuate_line_catalog`` (the no-state fallback used when
``dust_model`` is ``off``/``wg00``, and by the #950 ``enable_fast_nebular()``
grid path, which reconstructs lines with no :class:`ForwardState`) built its
own law kwargs from exactly two scalars, ``dust_slope`` and
``dust_bump_strength``, via ``emission_helpers.attenuate_emission``. Every
attenuation law that reads ``dust_delta`` (kriek_conroy, salim, noll09,
salim_sbl18, tea), ``dust_Rv`` (cardelli, conroy2010) or ``redshift``
(narayanan_z) therefore evaluated the LINE channel at the law's *published
default* while the CONTINUUM used the *fitted* value. Per-screen overrides
(``slope_bc``/``slope_diff``, carried on ``self._dust_law_overrides``) were
ignored on this path too, and the fallback never applied the Lyman clip or
the covering fraction ``dust_f_obscuration`` that the live component path
applies.

The fix (#2223) deletes ``attenuate_emission`` and routes
``_attenuate_line_catalog`` through the SAME dust-component method the live
forward pass already uses for its continuum
(``DustSEDComponent.attenuate_line_catalog`` /
``DustAttenuationSEDComponent.attenuate_line_catalog``), so there is exactly
one implementation of the two-component line screen (#1867, #1858).

Every test below therefore compares the FALLBACK's line transmission against
an independent "continuum" evaluation and asserts they agree to floating-point
precision, with an in-test control demonstrating the compared quantity
actually MOVES under the parameter being tested (so the parity assertion is
not vacuously true at a trivial value like 1.0).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.observation.photometry import FilterCurve
from tengri.utils.scale import pow10

HALPHA_AA = 6564.72
HBETA_AA = 4862.71

#: Arbitrary rest-frame line wavelengths/luminosities used to exercise
#: ``_attenuate_line_catalog`` directly. The fallback is a pure function of
#: (params, line_waves, line_lums); it never reads a real backend catalog, so
#: these do not need to correspond to any actual emission line.
_LINE_WAVE = jnp.asarray([1300.0, 2200.0, 4862.71, 6564.72])
_LINE_LUM = jnp.asarray([3.0e40, 1.5e41, 8.0e40, 2.4e41])


@pytest.fixture(scope="module")
def ssp_bare():
    """The bare-stellar grid Cue requires, resolved the way users get it.

    Mirrors ``test_bug_1867_lines_are_reddened.py``: deliberately not a
    hardcoded ``data/*.h5`` path.
    """
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


@pytest.fixture(scope="module")
def observation():
    def band(center, n=24):
        wave = np.linspace(center * 0.85, center * 1.15, n)
        trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(band(c) for c in (1500.0, 2200.0, 6200.0)))
    )


def _build_model(ssp_bare, observation, *, dust, redshift=0.05, neb=True):
    """Two-component dusty model, Cue nebular backend, #1867-style fixture."""
    kwargs = dict(
        ssp_data=ssp_bare,
        observation=observation,
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        dust_attenuation=dust,
        redshift=Fixed(redshift),
    )
    if neb:
        kwargs["neb"] = {"type": "cue", "all_params": Fixed(DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _dust_group(law, **shape) -> dict:
    """A two-component ``dust_attenuation`` group with both screens on one law."""
    return {
        "type": "two_component",
        "law_bc": law,
        "law_diff": law,
        "tau_bc": Uniform(0.1, 1.0),
        "tau_diff": Uniform(0.1, 3.0),
        **shape,
    }


def _sampled_params(model, seed: int = 0) -> dict:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return dict(model.spec.sample(jax.random.PRNGKey(seed)))


def _resolved_transmission(model, params, wave) -> np.ndarray:
    r"""The two-component screen's own resolver and law calls, evaluated
    directly -- "the component's own curve evaluation" from the #2223 plan.

    Re-derives ``tau = tau_bc * k_bc + tau_diff * k_diff`` from
    :func:`resolve_bc_diff_law_params` and the registry's own law functions:
    the SAME resolver and laws :meth:`DustSEDComponent.attenuate_line_catalog`
    and ``apply()``'s §2b nebular-continuum block both call. This is NOT the
    per-age birth-cloud sigmoid (:meth:`DustSEDComponent.compute_transmission`
    never reaches EXACTLY weight=1 at any finite age -- ``sigmoid(23.33)`` is
    ``1 - 7e-11``, not ``1.0`` -- which put ~1e-10-level noise into an earlier
    draft of this oracle that this function avoids by construction).

    Independent of ``attenuate_line_catalog`` in the sense that matters for
    #2223: it is a SEPARATE call into the resolver and the law registry, not a
    call to the method under test, so a hand-built ``{dust_slope,
    dust_bump_strength}`` dict (the old ``attenuate_emission`` fallback) would
    diverge from it exactly as it diverges from the live continuum.
    """
    from tengri.components.dust.attenuation import (
        apply_lyman_cutoff,
        resolve_bc_diff_law_params,
    )
    from tengri.components.dust.laws._registry import resolve_dust_law, select_law_kwargs

    component = model._line_dust_component()
    config = component.config
    wave = jnp.asarray(wave)
    bc_params, diff_params = resolve_bc_diff_law_params(
        params,
        dict(config.bc_law_overrides),
        dict(config.diff_law_overrides),
        config.live_shape_params,
        bc_law=config.law_bc,
        diff_law=config.law_diff,
        redshift=params.get("redshift"),
    )
    neb_law = config.law_neb or config.law_bc
    neb_bc_params = select_law_kwargs(neb_law, {**bc_params, **dict(config.neb_law_overrides)})
    k_bc = apply_lyman_cutoff(
        resolve_dust_law(neb_law)(wave, **neb_bc_params), wave, config.lyman_cutoff_aa
    )
    k_diff = apply_lyman_cutoff(
        resolve_dust_law(config.law_diff)(wave, **diff_params), wave, config.lyman_cutoff_aa
    )
    tau = jnp.asarray(params["dust_tau_bc"]) * k_bc + jnp.asarray(params["dust_tau_diff"]) * k_diff
    f_obsc = jnp.asarray(params.get("dust_f_obscuration", 0.0))
    return np.asarray(f_obsc + (1.0 - f_obsc) * jnp.exp(-tau))


def _line_transmission(model, params, wave, lum) -> np.ndarray:
    """Fallback line transmission: attenuated / intrinsic, dimensionless."""
    atten = np.asarray(model._attenuate_line_catalog(params, wave, lum))
    return atten / np.asarray(lum)


# ── (1) Shared shape kwargs (dust_delta, dust_Rv) reach the fallback ───────


@pytest.mark.parametrize(
    "law, shaped, control",
    [
        ("kriek_conroy", {"dust_delta": Fixed(-0.5)}, {"dust_delta": Fixed(0.0)}),
        ("cardelli", {"dust_Rv": Fixed(5.0)}, {"dust_Rv": Fixed(3.1)}),
    ],
    ids=["kriek_conroy-dust_delta", "cardelli-dust_Rv"],
)
def test_line_screen_matches_continuum_for_shape_kwarg(
    ssp_bare, observation, law, shaped, control
):
    """The line channel must see the SAME shape kwarg as the continuum.

    Mutation to catch: a hand-built ``{dust_slope, dust_bump_strength}`` dict
    (the old ``attenuate_emission`` fallback) has no ``dust_delta``/``dust_Rv``
    slot at all, so the line side would stay pinned at the law's published
    default while the continuum moved with the shaped value -- the parity
    assertion would then fail, which is exactly what this test is for.
    """
    model_shaped = _build_model(ssp_bare, observation, dust=_dust_group(law, **shaped))
    model_control = _build_model(ssp_bare, observation, dust=_dust_group(law, **control))
    params_shaped = _sampled_params(model_shaped)
    params_control = _sampled_params(model_control)

    t_line_shaped = _line_transmission(model_shaped, params_shaped, _LINE_WAVE, _LINE_LUM)
    t_cont_shaped = _resolved_transmission(model_shaped, params_shaped, _LINE_WAVE)
    np.testing.assert_allclose(t_line_shaped, t_cont_shaped, rtol=1e-10, atol=0.0)

    t_line_control = _line_transmission(model_control, params_control, _LINE_WAVE, _LINE_LUM)

    # In-test control: the transmission must actually MOVE with the shape
    # kwarg -- otherwise the parity assertion above would pass trivially even
    # if the line screen ignored the shape kwarg entirely (both sides would
    # sit at the law's default and agree by coincidence, not by threading).
    assert not np.allclose(t_line_shaped, t_line_control, rtol=1e-6, atol=0.0), (
        f"{law} line transmission did not move when the shape kwarg changed "
        f"({shaped} vs {control}); the parity assertion above is meaningless "
        "until this moves."
    )


# ── (2) narayanan_z's redshift threading (#2199) reaches the fallback ──────


@pytest.mark.parametrize("redshift", [0.0, 2.0])
def test_line_screen_matches_continuum_for_narayanan_z_redshift(ssp_bare, observation, redshift):
    """narayanan_z's ONLY knob is the model redshift (#2199); the line channel
    must see it too, at both ends of the law's fitted range."""
    model = _build_model(ssp_bare, observation, dust=_dust_group("narayanan_z"), redshift=redshift)
    params = _sampled_params(model)
    t_line = _line_transmission(model, params, _LINE_WAVE, _LINE_LUM)
    t_cont = _resolved_transmission(model, params, _LINE_WAVE)
    np.testing.assert_allclose(t_line, t_cont, rtol=1e-10, atol=0.0)


def test_narayanan_z_line_transmission_moves_with_redshift(ssp_bare, observation):
    """Control for the test above: without this, z=0 and z=2 could trivially
    agree (e.g. if redshift never reached either side)."""
    model_z0 = _build_model(ssp_bare, observation, dust=_dust_group("narayanan_z"), redshift=0.0)
    model_z2 = _build_model(ssp_bare, observation, dust=_dust_group("narayanan_z"), redshift=2.0)
    params_z0 = _sampled_params(model_z0)
    params_z2 = _sampled_params(model_z2)
    t_z0 = _line_transmission(model_z0, params_z0, _LINE_WAVE, _LINE_LUM)
    t_z2 = _line_transmission(model_z2, params_z2, _LINE_WAVE, _LINE_LUM)
    assert not np.allclose(t_z0, t_z2, rtol=1e-6, atol=0.0), (
        "narayanan_z line transmission did not move between z=0 and z=2; the "
        "parity test above is meaningless until this moves."
    )


# ── (3) Per-screen overrides (self._dust_law_overrides) reach the fallback ─


def test_per_screen_override_reaches_the_fallback(ssp_bare, observation):
    """``slope_bc``/``slope_diff`` (config-level, #1833) must reach the LINE
    screen too, not just the continuum.

    ``slope_bc``/``slope_diff`` are plain floats in the grammar (not
    ``Fixed(...)``, unlike the shared ``dust_*`` keys): they route onto
    ``DustSEDComponentConfig.bc_law_overrides``/``diff_law_overrides``, a
    hashable static tuple, not a sampled parameter (see
    ``tests/regression/bug/test_bug_2185_attenuation_kwargs_catchall.py``).
    """
    dust_shaped = {
        **_dust_group("power_law"),
        "slope_bc": -0.3,
        "slope_diff": -1.2,
    }
    # Control: no per-screen override at all -> both screens share the one
    # ``dust_slope`` default (-0.7), which is the historical behavior the
    # override is layered on top of.
    dust_control = _dust_group("power_law")

    model_shaped = _build_model(ssp_bare, observation, dust=dust_shaped)
    model_control = _build_model(ssp_bare, observation, dust=dust_control)
    params_shaped = _sampled_params(model_shaped)
    params_control = _sampled_params(model_control)

    t_line_shaped = _line_transmission(model_shaped, params_shaped, _LINE_WAVE, _LINE_LUM)
    t_line_control = _line_transmission(model_control, params_control, _LINE_WAVE, _LINE_LUM)
    t_cont_shaped = _resolved_transmission(model_shaped, params_shaped, _LINE_WAVE)
    t_cont_control = _resolved_transmission(model_control, params_control, _LINE_WAVE)

    # Each side matches its own continuum...
    np.testing.assert_allclose(t_line_shaped, t_cont_shaped, rtol=1e-10, atol=0.0)
    np.testing.assert_allclose(t_line_control, t_cont_control, rtol=1e-10, atol=0.0)
    # ...and the DELTA the override makes on the line side equals the delta it
    # makes on the continuum side -- isolating the per-screen override itself
    # (the old fallback never read ``self._dust_law_overrides`` at all, so
    # this delta was zero on the line side while nonzero on the continuum).
    np.testing.assert_allclose(
        t_line_shaped - t_line_control, t_cont_shaped - t_cont_control, rtol=1e-10, atol=1e-12
    )
    assert not np.allclose(t_line_shaped, t_line_control, rtol=1e-6, atol=0.0), (
        "line transmission did not move when slope_bc/slope_diff were set; "
        "the delta assertion above is meaningless until this moves."
    )


# ── (4) Lyman clip + covering fraction reach the fallback ──────────────────


def test_lyman_clip_and_f_obscuration_reach_the_fallback(ssp_bare, observation):
    """The fallback must equal the LIVE component's published attenuated
    catalog exactly, including the Lyman clip and ``dust_f_obscuration``.

    The OLD fallback (``attenuate_emission``) never applied either: it has no
    Lyman-cutoff mask and no covering-fraction term at all, so this is RED
    before the fix even at the law's default slope.
    """
    dust = {
        **_dust_group("power_law"),
        "dust_f_obscuration": Fixed(0.3),
        "lyman_cutoff": True,
    }
    model = _build_model(ssp_bare, observation, dust=dust)
    params = _sampled_params(model)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(params)

    log_atten_live = state.derived["log_line_lums_attenuated"]
    line_waves = state.derived["line_waves"]
    log_line_lums = state.derived["log_line_lums"]

    fallback_atten = model._attenuate_line_catalog(
        params, jnp.asarray(line_waves), pow10(jnp.asarray(log_line_lums))
    )
    np.testing.assert_allclose(
        np.asarray(fallback_atten), np.asarray(pow10(log_atten_live)), rtol=1e-10, atol=0.0
    )


# ── (5) The fallback is genuinely exercised by the #950 fast-nebular path ──


def test_fallback_is_actually_exercised_by_fast_nebular(ssp_bare, observation, monkeypatch):
    """``enable_fast_nebular()`` + ``predict_line_fluxes(params)`` with NO
    ``state`` is the one live production route to ``_attenuate_line_catalog``
    that also carries a shaped law (kriek_conroy, ``dust_delta``). Spy on the
    method to confirm the fallback is genuinely taken, then check its dust
    TRANSMISSION (attenuated/intrinsic, which cancels the fast grid's
    independent Q_H-linear reconstruction of the intrinsic luminosity, itself
    only approximate at coarse ``n_grid``) against the exact forward path.
    """
    model = _build_model(
        ssp_bare,
        observation,
        dust=_dust_group("kriek_conroy", **{"dust_delta": Fixed(-0.5)}),
    )
    if not hasattr(model, "enable_fast_nebular"):  # pragma: no cover - defensive
        pytest.skip("SEDModel.enable_fast_nebular is not available")

    target_wavelengths = jnp.asarray([HBETA_AA, HALPHA_AA])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.enable_fast_nebular(target_wavelengths, n_grid=2)

    calls: list[int] = []
    original = type(model)._attenuate_line_catalog

    def _spy(self, params, line_waves, line_lums):
        calls.append(1)
        return original(self, params, line_waves, line_lums)

    monkeypatch.setattr(type(model), "_attenuate_line_catalog", _spy)

    params = _sampled_params(model)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast_atten = np.asarray(model.predict_line_fluxes(params, redden=True))
        fast_intr = np.asarray(model.predict_line_fluxes(params, redden=False))

    assert calls, (
        "predict_line_fluxes on the enable_fast_nebular() path (no state) did "
        "not call _attenuate_line_catalog; the fallback this file tests is not "
        "being exercised at all."
    )

    fast_transmission = fast_atten / fast_intr

    # Precise check: the fallback's dust transmission against the resolved-law
    # oracle, AT THE GRID'S OWN WAVELENGTHS (``enable_fast_nebular`` tabulates
    # exactly the ``target_wavelengths`` it was given -- it does not re-snap
    # them to the backend's own catalog line, e.g. Cue's Hbeta sits at
    # 4862.678 A, ~0.03 A off HBETA_AA). This is the same rtol=1e-10 parity the
    # other tests in this file assert, run through the actual production
    # call chain (enable_fast_nebular + predict_line_fluxes) instead of a
    # direct _attenuate_line_catalog call.
    resolved = _resolved_transmission(model, params, target_wavelengths)
    np.testing.assert_allclose(fast_transmission, resolved, rtol=1e-10, atol=0.0)

    # Looser, informational check against the exact forward path requested by
    # the #2223 test plan: it evaluates the SAME dust screen at the backend's
    # true (nearest-matched) line wavelength, ~0.03-0.1 A off the fast grid's
    # tabulated wavelength above, so a real (not a bug) difference of a few
    # 1e-5 in transmission is expected from k(lambda)'s slope alone -- nothing
    # to do with #2223's kwarg threading, which the tight check above already
    # isolates.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        state = model.predict_state(params)
    exact_atten = np.asarray(
        model.predict_line_fluxes(
            params, target_wavelengths=target_wavelengths, state=state, redden=True
        )
    )
    exact_intr = np.asarray(
        model.predict_line_fluxes(
            params, target_wavelengths=target_wavelengths, state=state, redden=False
        )
    )
    exact_transmission = exact_atten / exact_intr
    np.testing.assert_allclose(fast_transmission, exact_transmission, rtol=2e-4, atol=0.0)
