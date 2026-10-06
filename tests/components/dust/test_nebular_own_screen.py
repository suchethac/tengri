# SPDX-License-Identifier: BSD-3-Clause
r"""Own nebular screen: ``dust_attenuation={'nebular_screen': 'own'}`` (#2625).

The nebular continuum and every emission line see ONE dedicated screen

.. math::

    T_{\rm neb}(\lambda) = \exp\left[-\tau_{\rm neb}\,k_{\rm neb}(\lambda)\right],

with ``k_neb`` the ``law_neb`` curve (k(5500 A) = 1) and ``tau_neb`` the declared
parameter ``dust_tau_neb``: the TOTAL nebular attenuation (CIGALE's
``E(B-V)_lines`` convention), not cascaded with the diffuse screen, not mixed over
the age intervals, and without the obscuration floor.

Every test builds a model through ``SEDModel.build`` with the Cue nebular backend
on the bare-stellar SSP. They pin: the two identities that tie ``'own'`` to the
existing choices, the independence of the nebular channel from the stellar
screens, the dust energy budget, the refusals, the LUT seam, the gradients and the
grammar round trip.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform, WavePrecomp
from tengri.components.dust.laws._registry import resolve_dust_law
from tengri.components.lyc import edge_trapezoid
from tengri.config.exceptions import ParameterError
from tengri.inference import Fitter
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract

_BANDS = ["galex_fuv", "sdss_r", "Herschel_Pacs_green"]
_HALPHA = 6564.72
_Z = 0.05
_TAUS = {"dust_tau_bc": 0.8, "dust_tau_diff": 0.4}


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


def _build(ssp, dust, *, emission=None, sfh_extra=None, approx=None):
    kwargs = dict(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
            **(sfh_extra or {}),
        },
        dust_attenuation=dust,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(_Z),
        approx=approx,
    )
    if emission is not None:
        kwargs["dust_emission"] = emission
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _two(law_bc="calzetti", law_diff="power_law", **extra):
    """A two-component group with every optical depth free."""
    return {
        "type": "two_component",
        "law_bc": law_bc,
        "law_diff": law_diff,
        "tau_bc": Uniform(0.0, 4.0),
        "tau_diff": Uniform(0.0, 4.0),
        **extra,
    }


def _own(tau_neb=None, **kw):
    """An own-screen group; ``tau_neb`` defaults to a free ``Uniform(0, 4)``."""
    tau_neb = Uniform(0.0, 4.0) if tau_neb is None else tau_neb
    return _two(nebular_screen="own", tau_neb=tau_neb, **kw)


def _age_binned(**extra):
    return {
        "type": "age_binned",
        "screens": [{"law": "power_law", "window_log_yr": (None, None)}],
        "tau_0": Uniform(0.0, 4.0),
        **extra,
    }


def _full(model, **values):
    """Free-parameter dict: every free key at 0.5 unless ``values`` names it."""
    return {**{k: 0.5 for k in model.spec.free_params}, **values}


def _outputs(model, params):
    """``(rest SED, photometry, Halpha flux)``: the three channels the identities cover."""
    sed = model.predict(params).rest_sed()
    phot = model.predict_photometry(params)
    lines = model.predict_line_fluxes(params, target_wavelengths=jnp.asarray([_HALPHA]))
    return np.asarray(sed), np.asarray(phot), np.asarray(lines)


def _assert_bit_identical(a, b):
    for x, y in zip(a, b):
        np.testing.assert_array_equal(x, y)


def _assert_close(a, b, rtol):
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=rtol, atol=0.0)


# ── (a) 'own' with the diffuse law and depth IS the diffuse screen ──────────


def test_own_with_the_diffuse_law_and_depth_is_bit_identical_to_diffuse_two_component(ssp):
    """T_own = exp(-tau_diff k_diff) = T_diffuse when law_neb = law_diff, tau_neb = tau_diff.

    ``law_bc`` is a different law (calzetti) from ``law_diff`` (power_law), so an
    own screen that fell back to the birth-cloud law could not pass.
    """
    diffuse = _build(ssp, _two(nebular_screen="diffuse"))
    own = _build(ssp, _own(law_neb="power_law"))
    p = {**_TAUS}
    _assert_bit_identical(
        _outputs(diffuse, _full(diffuse, **p)),
        _outputs(own, _full(own, **p, dust_tau_neb=_TAUS["dust_tau_diff"])),
    )
    # Control: a different depth moves the nebular channel, so the identity is not vacuous.
    moved = _outputs(own, _full(own, **p, dust_tau_neb=2.0))
    assert not np.allclose(
        moved[2], _outputs(diffuse, _full(diffuse, **p))[2], rtol=1e-3, atol=0.0
    )


@pytest.mark.parametrize("law_neb", [None, "power_law"], ids=["screen-0-law", "explicit-law_neb"])
def test_own_with_the_screen_law_and_depth_is_bit_identical_to_the_mixture_age_binned(
    ssp, law_neb
):
    """One unbounded age_binned screen: the interval mixture is that screen alone.

    ``'own'`` with the same law (inherited from screen 0, or named through
    ``law_neb``, whose shape parameters are then ``dust_slope_neb``) and
    ``tau_neb = tau_0`` is the same transmission.
    """
    extra = {"nebular_screen": "own", "tau_neb": Uniform(0.0, 4.0)}
    if law_neb is not None:
        extra["law_neb"] = law_neb
    mixture = _build(ssp, _age_binned())
    own = _build(ssp, _age_binned(**extra))
    _assert_bit_identical(
        _outputs(mixture, _full(mixture, dust_tau_0=0.4)),
        _outputs(own, _full(own, dust_tau_0=0.4, dust_tau_neb=0.4)),
    )


# ── (b) 'own' = 'birth_cloud' when one law serves every screen and the mixture is young ──


def test_own_with_the_summed_depth_equals_birth_cloud_in_the_all_young_limit(ssp):
    """exp(-(tau_bc + tau_diff) k) = exp(-tau_bc k) exp(-tau_diff k) for ONE law.

    With ``t_birth_yr`` above every SSP node the whole ionizing luminosity is
    young (q = (1, 0)) and ``'birth_cloud'`` is the product of the two screens;
    ``dust_f_obscuration`` is 0 (its default), so the birth-cloud floor does not
    enter. The two sides differ only by the rounding of exp(a)exp(b) vs exp(a+b).
    """
    one_law = dict(law_bc="calzetti", law_diff="calzetti", t_birth_yr=1.0e12)
    birth = _build(ssp, _two(**one_law))
    own = _build(ssp, _own(**one_law))
    p = {**_TAUS}
    _assert_close(
        _outputs(birth, _full(birth, **p)),
        _outputs(own, _full(own, **p, dust_tau_neb=_TAUS["dust_tau_bc"] + _TAUS["dust_tau_diff"])),
        rtol=1e-12,
    )


# ── (c) independence of the nebular channel ─────────────────────────────────


def _lines_and_continuum(model, params):
    state = model.predict_state(params)
    d = state.derived
    k_line_shift = np.asarray(d["log_line_lums_attenuated"]) - np.asarray(d["log_line_lums"])
    return (
        np.asarray(state.wave),
        np.asarray(d["sed_dust_attenuated"]),
        np.asarray(d["sed_nebular"]),
        np.asarray(d["line_waves"]),
        k_line_shift,
    )


def test_tau_neb_moves_only_the_nebular_channel_by_exp_minus_delta_tau_k(ssp):
    """Stellar continuum bit-identical; nebular continuum and each line scale by exp(-dtau k)."""
    m = _build(ssp, _own(law_neb="smc"))
    base = _lines_and_continuum(m, _full(m, **_TAUS, dust_tau_neb=0.5))
    more = _lines_and_continuum(m, _full(m, **_TAUS, dust_tau_neb=1.7))
    wave, stellar0, neb0, line_waves, shift0 = base
    _wave, stellar1, neb1, _lw, shift1 = more
    delta = 1.7 - 0.5

    np.testing.assert_array_equal(stellar0, stellar1)
    law = resolve_dust_law("smc")
    k_cont = np.asarray(law(jnp.asarray(wave)))
    keep = neb0 > 0
    np.testing.assert_allclose(neb1[keep] / neb0[keep], np.exp(-delta * k_cont[keep]), rtol=1e-12)
    k_line = np.asarray(law(jnp.asarray(line_waves)))
    # the line shift is log10(T_neb): its change is -delta k / ln 10 at each line
    np.testing.assert_allclose(shift1 - shift0, -delta * k_line / np.log(10.0), atol=1e-12)
    assert np.all(shift1 < shift0)


def test_law_neb_moves_only_the_nebular_channel_by_the_ratio_of_the_two_laws(ssp):
    """Swapping law_neb changes T_neb by exp(-tau_neb [k2 - k1]); the stars do not move."""
    a = _build(ssp, _own(law_neb="smc"))
    b = _build(ssp, _own(law_neb="calzetti"))
    pa, pb = _full(a, **_TAUS, dust_tau_neb=1.2), _full(b, **_TAUS, dust_tau_neb=1.2)
    wave, stellar_a, neb_a, line_waves, shift_a = _lines_and_continuum(a, pa)
    _w, stellar_b, neb_b, _lw, shift_b = _lines_and_continuum(b, pb)

    np.testing.assert_array_equal(stellar_a, stellar_b)
    k_a, k_b = resolve_dust_law("smc"), resolve_dust_law("calzetti")
    dk = np.asarray(k_b(jnp.asarray(wave))) - np.asarray(k_a(jnp.asarray(wave)))
    keep = neb_a > 0
    np.testing.assert_allclose(neb_b[keep] / neb_a[keep], np.exp(-1.2 * dk[keep]), rtol=1e-12)
    dk_line = np.asarray(k_b(jnp.asarray(line_waves))) - np.asarray(k_a(jnp.asarray(line_waves)))
    np.testing.assert_allclose(shift_b - shift_a, -1.2 * dk_line / np.log(10.0), atol=1e-12)
    assert np.abs(shift_b - shift_a).max() > 1e-3


def test_tau_bc_leaves_the_nebular_channel_unchanged_under_own(ssp):
    """Under 'own' the stellar screens do not touch the nebular continuum or the lines."""
    m = _build(ssp, _own())
    lo = _lines_and_continuum(m, _full(m, dust_tau_bc=0.1, dust_tau_diff=0.2, dust_tau_neb=0.9))
    hi = _lines_and_continuum(m, _full(m, dust_tau_bc=2.5, dust_tau_diff=1.5, dust_tau_neb=0.9))
    np.testing.assert_array_equal(lo[2], hi[2])  # nebular continuum
    np.testing.assert_array_equal(lo[4], hi[4])  # line attenuation
    assert not np.array_equal(lo[1], hi[1])  # control: the stars did move


def test_tau_neb_moves_only_the_nebular_channel_on_age_binned(ssp):
    """The same independence on the other attenuator, with a young screen present."""
    screens = [
        {"law": "calzetti", "window_log_yr": (None, 7.0)},
        {"law": "power_law", "window_log_yr": (None, None)},
    ]
    m = _build(
        ssp,
        {
            "type": "age_binned",
            "screens": screens,
            "tau_0": Uniform(0.0, 4.0),
            "tau_1": Uniform(0.0, 4.0),
            "nebular_screen": "own",
            "tau_neb": Uniform(0.0, 4.0),
            "law_neb": "smc",
        },
    )
    base = _lines_and_continuum(m, _full(m, dust_tau_neb=0.5))
    more = _lines_and_continuum(m, _full(m, dust_tau_neb=1.7))
    np.testing.assert_array_equal(base[1], more[1])
    k_line = np.asarray(resolve_dust_law("smc")(jnp.asarray(base[3])))
    np.testing.assert_allclose(more[4] - base[4], -1.2 * k_line / np.log(10.0), atol=1e-12)


# ── (d) energy balance ──────────────────────────────────────────────────────


def _absorbed_nebular_power(wave, sed_neb, tau_neb, law):
    """Independent  int (1 - T_neb) L_neb dnu  over the non-ionizing side [erg/s].

    The integrand is built here from the intrinsic nebular SED and the law; only
    the Lyman-edge quadrature (shared by every dust energy-balance integral) is
    the library's.
    """
    wave = jnp.asarray(wave)
    t = jnp.exp(-tau_neb * law(wave))
    integrand = (1.0 - t) * jnp.asarray(sed_neb)
    return float(edge_trapezoid(integrand, wave, variable="nu", side="nonionizing"))


@pytest.mark.parametrize("eta", [1.0, 0.6])
@pytest.mark.parametrize("emission", ["dl07", "dale2014"])
def test_own_screen_energy_enters_the_ir_budget(ssp, emission, eta):
    r"""L_IR('own') - L_IR('none') = eta * int (1 - T_neb) L_neb dnu.

    Conservation: the nebular light the own screen removes is re-radiated by the
    dust, through the same seam as the other screens; ``eta_balance`` scales the
    budget. The 'none' model leaves the nebular light unabsorbed, so the
    difference is exactly the own screen's absorption.
    """
    em = {"type": emission, "all_params": Fixed(DEFAULT), "eta_balance": Uniform(0.1, 1.5)}
    own = _build(ssp, _own(law_neb="smc"), emission=em)
    none = _build(ssp, _two(nebular_screen="none"), emission=em)
    tau_neb = 1.3
    p_own = _full(own, **_TAUS, dust_tau_neb=tau_neb, dust_eta_balance=eta)
    p_none = _full(none, **_TAUS, dust_eta_balance=eta)

    s_own, s_none = own.predict_state(p_own), none.predict_state(p_none)
    delta = float(s_own.derived["L_ir"] - s_none.derived["L_ir"])
    want = eta * _absorbed_nebular_power(
        s_none.wave, s_none.derived["sed_nebular"], tau_neb, resolve_dust_law("smc")
    )
    assert want > 0.0
    np.testing.assert_allclose(delta, want, rtol=1e-6)


# ── (e) refusals ────────────────────────────────────────────────────────────

_NON_OWN = ["birth_cloud", "diffuse", "none"]


@pytest.mark.parametrize("choice", _NON_OWN)
def test_tau_neb_with_another_screen_is_refused_in_the_grammar(choice):
    group = {"type": "two_component", "law": "calzetti", "nebular_screen": choice, "tau_neb": 1.0}
    with pytest.raises(ParameterError, match=r"nebular_screen='own'"):
        parse_groups(dust_attenuation=group)


@pytest.mark.parametrize("choice", _NON_OWN)
def test_tau_neb_with_another_screen_is_refused_on_the_flat_surface(choice):
    with pytest.raises(ParameterError, match=r"nebular_screen='own'"):
        Parameters(dust_tau_neb=Fixed(1.0), dust_nebular_screen=choice)


def test_tau_neb_without_a_screen_choice_is_refused_naming_own():
    """The default choice ('birth_cloud') does not read tau_neb either."""
    with pytest.raises(ParameterError, match=r"nebular_screen='own'"):
        parse_groups(dust_attenuation={"type": "two_component", "law": "calzetti", "tau_neb": 1})
    with pytest.raises(ParameterError, match=r"nebular_screen='own'"):
        Parameters(dust_tau_neb=Fixed(1.0))


@pytest.mark.parametrize("key", ["tau_neb", "law_neb", "slope_neb"])
def test_age_binned_refuses_the_own_screen_keys_without_own(key):
    group = _age_binned(**{key: 1.0 if key != "law_neb" else "smc"})
    with pytest.raises(ParameterError, match=r"nebular_screen='own'"):
        parse_groups(dust_attenuation=group)


@pytest.mark.parametrize("choice", ["diffuse", "none"])
def test_age_binned_refuses_a_nebular_screen_it_cannot_honor(choice):
    with pytest.raises(ParameterError, match=r"age_binned"):
        parse_groups(dust_attenuation=_age_binned(nebular_screen=choice))


def test_own_is_a_nebular_only_choice():
    """shock_screen / agn_screen take the three stellar-derived choices only."""
    with pytest.raises(ParameterError, match=r"shock_screen"):
        parse_groups(dust_attenuation=_two(shock_screen="own"))
    with pytest.raises(ParameterError, match=r"agn_screen"):
        parse_groups(dust_attenuation=_two(agn_screen="own"))


# ── (f) gradients ───────────────────────────────────────────────────────────


def _fd(fn, x, h=1e-5):
    return (fn(x + h) - fn(x - h)) / (2.0 * h)


def test_line_and_uv_band_gradients_wrt_tau_neb_match_finite_differences(ssp):
    """d(Halpha flux)/d tau_neb and d(galex_fuv)/d tau_neb are finite, negative, and FD-exact.

    A larger own depth removes nebular light, so both derivatives are negative;
    the FUV band holds nebular continuum and lines, the Halpha flux is pure line.
    """
    m = _build(ssp, _own(law_neb="smc"))
    base = _full(m, **_TAUS)

    def halpha(t):
        p = {**base, "dust_tau_neb": t}
        return m.predict_line_fluxes(p, target_wavelengths=jnp.asarray([_HALPHA]))[0]

    def fuv(t):
        return m.predict_photometry({**base, "dust_tau_neb": t})[0]

    for fn in (halpha, fuv):
        g = float(jax.grad(fn)(jnp.asarray(0.9)))
        fd = float(_fd(fn, jnp.asarray(0.9)))
        assert np.isfinite(g), "a non-finite own-depth gradient"
        assert g != 0.0 and g < 0.0, "own depth removes nebular light: the gradient is negative"
        np.testing.assert_allclose(g, fd, rtol=1e-6)


def test_line_gradient_wrt_tau_bc_is_exactly_zero_under_own(ssp):
    """Under 'own' the stellar birth-cloud depth never reaches a line flux."""
    m = _build(ssp, _own())
    base = _full(m, dust_tau_diff=0.4, dust_tau_neb=0.9)

    def halpha(tau_bc):
        p = {**base, "dust_tau_bc": tau_bc}
        return m.predict_line_fluxes(p, target_wavelengths=jnp.asarray([_HALPHA]))[0]

    assert float(jax.grad(halpha)(jnp.asarray(0.8))) == 0.0


# ── (g) grammar round trip and provenance ──────────────────────────────────


@pytest.mark.parametrize("attenuator", ["two_component", "age_binned"])
def test_own_screen_round_trips_through_to_groups(attenuator, ssp):
    group = (
        _own(tau_neb=Fixed(1.7))
        if attenuator == "two_component"
        else _age_binned(nebular_screen="own", tau_neb=Fixed(1.7))
    )
    spec = parse_groups(dust_attenuation=group, redshift=Fixed(_Z))
    out = spec.to_groups()["dust_attenuation"]
    assert out["nebular_screen"] == "own"
    assert out["tau_neb"] == 1.7 or getattr(out["tau_neb"], "value", None) == 1.7

    again = parse_groups(dust_attenuation=out, redshift=Fixed(_Z))
    assert again.dust_nebular_screen == "own"
    assert float(again.get_distribution("dust_tau_neb").bounds[0]) == 1.7


def test_summary_lists_dust_tau_neb_with_its_provenance(ssp):
    m = _build(ssp, _own())
    row = next(
        line
        for line in m.spec.summary_str().splitlines()
        if line.strip().startswith("dust_tau_neb")
    )
    assert "[user]" in row
    fixed = _build(ssp, _own(tau_neb=Fixed(1.7)))
    assert any(
        line.strip().startswith("dust_tau_neb") and "1.7" in line
        for line in fixed.spec.summary_str().splitlines()
    )


def test_dust_tau_neb_is_declared_only_under_own(ssp):
    """One declaration, present exactly when something reads it."""
    assert "dust_tau_neb" not in _build(ssp, _two()).spec.all_params
    assert "dust_tau_neb" in _build(ssp, _own()).spec.all_params
    assert "dust_tau_neb" in _build(ssp, _age_binned(nebular_screen="own")).spec.all_params


# ── (h) LUT seam ────────────────────────────────────────────────────────────

#: The WavePrecomp nebular-channel bound of tests/contract/test_precomp_channel_drift.py.
_WAVE_PRECOMP_BOUND = 2e-3


def test_wave_precomp_photometry_under_own_matches_the_exact_path(ssp):
    """WavePrecomp screens the nebular sub-bands through the same T_neb as the exact path."""
    own = _own(tau_neb=Fixed(1.8), law_neb="smc")
    exact = _build(ssp, own)
    lut = _build(ssp, own, approx=WavePrecomp())
    p = _full(exact, **_TAUS)
    a = np.asarray(exact.predict_photometry(p))
    b = np.asarray(lut.predict_photometry(p))
    gap = float(np.max(np.abs(b - a) / np.abs(a)))
    assert gap < _WAVE_PRECOMP_BOUND
    # control: the own depth is visible in the compared bands
    none = _build(ssp, _two(nebular_screen="none"), approx=WavePrecomp())
    assert not np.allclose(np.asarray(none.predict_photometry(p)), b, rtol=1e-3, atol=0.0)


def test_fitter_auto_builds_and_agrees_with_the_exact_model_under_own(ssp):
    """approx='auto' keeps working: with a free tau_neb it resolves to the exact nebular path."""
    exact = _build(ssp, _own())
    flux = np.asarray(exact.predict_photometry(_full(exact, **_TAUS, dust_tau_neb=0.9)))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = Fitter(exact, data=flux, noise=0.1 * flux, data_type="photometry", approx="auto")
    p = _full(exact, **_TAUS, dust_tau_neb=0.9)
    got = np.asarray(fit.model.predict_photometry(p))
    gap = float(np.max(np.abs(got - flux) / np.abs(flux)))
    assert gap < _WAVE_PRECOMP_BOUND


#: The dust-grid fast path's bound of tests/contract/test_dusty_nebular_grid_wiring.py.
_NEBULAR_GRID_BOUND = 3e-2


def _grid_model(ssp, tau_neb, approx=None):
    """Dusty Cue model whose dust can take the nebular continuum from the per-Q_H grid."""
    dust = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Uniform(0.0, 2.0),
        "tau_diff": Uniform(0.0, 2.0),
        "nebular_screen": "own",
        "tau_neb": tau_neb,
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=Observation(photometry=Photometry.from_names(_BANDS)),
            approx=approx,
            redshift=Fixed(0.15),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(8, 12)},
            dust_attenuation=dust,
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            neb={"type": "cue", "all_params": Fixed(DEFAULT), "neb_logU": Uniform(-3.5, -2.0)},
        )


def test_nebular_grid_serves_the_own_screen_and_matches_the_exact_path(ssp):
    """The per-Q_H grid is built from the intrinsic nebular light and screened at run time.

    With ``dust_tau_neb`` Fixed the grid's absorbed-energy table does not depend
    on ``(tau_bc, tau_diff)`` (the own screen reads neither) and is exact at the
    build depth; a free ``dust_tau_neb`` is outside the energy-balance LUT's
    optical-depth axes, so the fit stays on the exact nebular path.
    """
    from tengri.inference.fitter import fast_nebular_can_engage

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        exact = _grid_model(ssp, Fixed(1.5))
        base = _grid_model(ssp, Fixed(1.5), approx=WavePrecomp())
        flux = np.asarray(exact.predict_photometry(_full(exact, sfh_dpl_log_total_mass=10.0)))
        fit = Fitter(base, data=flux, noise=0.1 * flux, data_type="photometry", approx="auto")
    assert fast_nebular_can_engage(fit.model)
    assert getattr(fit.model, "_nebular_grid_table", None) is not None

    worst = 0.0
    for tau_bc, tau_diff, logu in ((0.3, 0.2, -2.7), (1.2, 0.8, -3.2), (1.8, 1.5, -2.2)):
        p = _full(
            exact,
            sfh_dpl_log_total_mass=10.0,
            neb_logU=logu,
            dust_tau_bc=tau_bc,
            dust_tau_diff=tau_diff,
        )
        want = np.asarray(exact.predict_photometry(p))
        got = np.asarray(fit.model.predict_photometry(p))
        worst = max(worst, float(np.max(np.abs(got - want) / np.abs(want))))
    assert worst < _NEBULAR_GRID_BOUND

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        free = _grid_model(ssp, Uniform(0.0, 4.0), approx=WavePrecomp())
        fit_free = Fitter(free, data=flux, noise=0.1 * flux, data_type="photometry", approx="auto")
    assert getattr(fit_free.model, "_nebular_grid_table", None) is None
