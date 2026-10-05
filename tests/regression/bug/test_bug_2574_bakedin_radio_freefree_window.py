# SPDX-License-Identifier: BSD-3-Clause
"""
One thermal free-free term on a nebular-included SSP with a radio block (#2574).

Invariant: a model whose SSP flux holds the nebular continuum (a wNE grid,
``ssp_data.nebular == "included"``) plus a radio block counts thermal free-free
once at every wavelength: inside the SSP flux up to the SSP grid edge, and in
the radio block's Murphy+2011 term above it.

Mechanism: with the radio ``freefree`` switch unset, the factory sets
``RadioSEDComponentConfig.freefree_wave_min`` to the SSP grid edge (1 cm for the
grid used here) and ``RadioSEDComponent.emission_terms`` zeroes the ``"ff"``
term below it. The rule reads the SSP's nebular stamp, not the declared
``neb`` type: ``neb={'type': 'none'}`` on a wNE grid still has the continuum in
its flux, and the same declaration on a bare grid has none.

Measured before the fix: the default build's ``sed_radio`` exceeded the
``freefree: False`` build's at 250 of 250 master-grid nodes in (1 mm, 1 cm], and
a 3 mm band read 1.2575x the one-term value on both the exact and the
``WavePrecomp`` path.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.components.radio.component import RadioSEDComponentConfig
from tengri.config.exceptions import ConfigError
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_Z = 0.05
_RADIO_WAVE_MIN_AA = 1.0e7  # the radio module's own lower limit (1 mm)


def _tophat(lo_obs, hi_obs, name, n=60):
    wave = jnp.linspace(lo_obs, hi_obs, n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


# Observed-frame bands; rest = observed / (1 + _Z).
_OBS = Observation(
    photometry=Photometry(
        filters=(
            _tophat(5000.0, 6000.0, "opt"),
            _tophat(2.7e7, 3.2e7, "mm3"),  # rest 2.57-3.05 mm: below the SSP edge
            _tophat(2.0e8, 3.0e8, "cm2p5"),  # rest 1.90-2.86 cm: above the SSP edge
            # rest 1.01-1.10 cm: inside the first master-grid cell above the SSP edge
            _tophat(1.06e8, 1.15e8, "cm1p05"),
        )
    )
)
_UNSET = object()


def _build(
    ssp,
    *,
    freefree=_UNSET,
    neb=None,
    approx=None,
    sfh=None,
    sf_type="bell2003",
    agn_type="powerlaw",
):
    sf = {"type": sf_type}
    if freefree is not _UNSET:
        sf["freefree"] = freefree
    return SEDModel.build(
        ssp_data=ssp,
        observation=_OBS,
        sfh=sfh or {"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "draine_li2014", "all_params": Fixed(DEFAULT)},
        neb=neb if neb is not None else {"type": "ssp"},
        radio={"sf": sf, "agn": {"type": agn_type}, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(_Z),
        approx=approx,
    )


def _radio_config(model) -> RadioSEDComponentConfig:
    for name, config in model._component_configs:
        if name == "radio":
            return config
    raise AssertionError("radio component not found in model._component_configs")


def _state(model):
    params = dict(model.spec.sample(jax.random.PRNGKey(0)))
    return params, model.predict_state(params)


def _thermal_share():
    """Thermal share f of Bell's total at 1.4 GHz for the default q = 2.64 (#2590).

    Murphy+2011 (Eq. 4 into Eq. 11, T_e = 1e4 K) over Bell (2003) Eq. 1; L_IR cancels.
    The synchrotron term is (1 - f) of the ``freefree: False`` one at every frequency
    (same slope), and the thermal term is f * (nu / 1.4 GHz)**0.7 times it (alpha_ff =
    -0.1 against alpha_sf = 0.8).
    """
    ff_ref = 3.88e-44 / 4.6e-28 * 1.4**-0.1  # per erg/s of L_IR
    return ff_ref * 3.75e12 * 10.0**2.64


def _thermal_over_sync_off(wave):
    return _thermal_share() * (2.99792458e18 / wave / 1.4e9) ** 0.7


def test_factory_sets_the_window_from_the_ssp_nebular_stamp(ssp_data_wne, ssp_data_fsps):
    """The window follows the SSP's nebular stamp, and only an unset ``freefree``."""
    edge = float(np.max(np.asarray(ssp_data_wne.ssp_wave)))
    assert ssp_data_wne.nebular == "included"
    assert edge == 1e8

    # Nebular-included SSP, freefree unset: windowed, whatever neb type is declared.
    for neb in ({"type": "ssp"}, {"type": "none"}):
        cfg = _radio_config(_build(ssp_data_wne, neb=neb))
        assert cfg.include_freefree is True, neb
        assert cfg.freefree_wave_min == edge, neb

    # An explicit switch always wins: no window.
    cfg = _radio_config(_build(ssp_data_wne, freefree=True))
    assert (cfg.include_freefree, cfg.freefree_wave_min) == (True, None)
    cfg = _radio_config(_build(ssp_data_wne, freefree=False))
    assert (cfg.include_freefree, cfg.freefree_wave_min) == (False, None)

    # bell2003_split is the same model as bell2003 and gets the same window.
    cfg = _radio_config(_build(ssp_data_wne, sf_type="bell2003_split"))
    assert (cfg.include_freefree, cfg.freefree_wave_min) == (True, edge)

    # The same flux arrays stamped bare, or left unstamped: the SSP is not known
    # to hold a nebular continuum, so the term keeps its whole range.
    for stamp in ("bare", "unknown"):
        ssp = ssp_data_wne._replace(nebular=stamp)
        cfg = _radio_config(_build(ssp, neb={"type": "none"}))
        assert (cfg.include_freefree, cfg.freefree_wave_min) == (True, None), stamp

    # A backend that carries free-free itself owns the thermal term everywhere.
    cfg = _radio_config(_build(ssp_data_fsps, neb={"type": "cue", "all_params": Fixed(DEFAULT)}))
    assert (cfg.include_freefree, cfg.freefree_wave_min) == (False, None)


@pytest.mark.parametrize("agn_type", ["powerlaw", "none", "dpl"])
def test_thermal_term_is_zero_up_to_the_ssp_edge_and_on_above_it(ssp_data_wne, agn_type):
    """The thermal term is zero below the SSP edge and on from the edge node upward.

    Checked on every AGN-radio branch of ``emission_terms``.
    """
    m_default = _build(ssp_data_wne, agn_type=agn_type)
    m_off = _build(ssp_data_wne, freefree=False, agn_type=agn_type)

    _, state_default = _state(m_default)
    _, state_off = _state(m_off)

    off = np.asarray(state_off.derived["sed_radio"])
    wave = np.asarray(state_default.wave)
    # The thermal term is what the default build holds beyond the synchrotron share
    # (1 - f) of the Bell total (#2590).
    d = np.asarray(state_default.derived["sed_radio"]) - (1.0 - _thermal_share()) * off
    edge = 1e8

    # The master grid has a node exactly at the SSP edge.
    assert np.any(wave == edge), "the master grid must have a node at the SSP edge"

    # Below the edge the SSP owns the thermal continuum: no radio thermal term.
    below = wave < edge
    assert np.all(np.abs(d[below]) <= 1e-12 * off[below]), (
        "the thermal term must be zero below the SSP edge"
    )

    # At the edge node and above it the term is on, and equals f (nu/1.4 GHz)^0.7 of the
    # off-build synchrotron. The edge node must carry it:
    # the first cell above the edge is 12.5% wide and the SSP has no flux there.
    from_edge = wave >= edge
    assert from_edge.sum() >= 50
    assert np.all(d[from_edge] > 0.0), "the thermal term must be on at the SSP edge and above"
    np.testing.assert_allclose(
        d[from_edge], (_thermal_over_sync_off(wave) * off)[from_edge], rtol=1e-9
    )


def test_above_the_edge_the_term_is_the_unwindowed_one(ssp_data_wne):
    """From the edge upward the default equals explicit True; below it only True has the term."""
    m_default = _build(ssp_data_wne)
    m_true = _build(ssp_data_wne, freefree=True)
    m_false = _build(ssp_data_wne, freefree=False)

    _, state_default = _state(m_default)
    _, state_true = _state(m_true)
    _, state_false = _state(m_false)

    sed_radio_default = np.asarray(state_default.derived["sed_radio"])
    sed_radio_true = np.asarray(state_true.derived["sed_radio"])
    sed_radio_false = np.asarray(state_false.derived["sed_radio"])

    wave = np.asarray(state_default.wave)
    edge = 1e8

    # From the edge node upward the windowed default is the explicit-True term.
    assert np.array_equal(sed_radio_default[wave >= edge], sed_radio_true[wave >= edge])

    # Below the edge an explicit True keeps the term: the override is honored.
    below = (wave > _RADIO_WAVE_MIN_AA) & (wave < edge)
    share = 1.0 - _thermal_share()
    assert np.all((sed_radio_true - share * sed_radio_false)[below] > 0.0)
    np.testing.assert_allclose(
        sed_radio_default[below], (share * sed_radio_false)[below], rtol=1e-12
    )


def test_one_thermal_term_in_the_total_sed_below_the_edge(ssp_data_wne):
    """Below the edge the default total differs from ``freefree: False`` only by the
    synchrotron share the thermal term displaces (#2590); from the edge up it adds the thermal
    term beside that.
    """
    m_default = _build(ssp_data_wne)
    m_off = _build(ssp_data_wne, freefree=False)

    params_off = dict(m_off.spec.sample(jax.random.PRNGKey(0)))
    pred_default = m_default.predict(dict(m_default.spec.sample(jax.random.PRNGKey(0))))
    pred_off = m_off.predict(params_off)
    radio_off = np.asarray(m_off.predict_state(params_off).derived["sed_radio"])

    rest_sed_default = np.asarray(pred_default.rest_sed())
    rest_sed_off = np.asarray(pred_off.rest_sed())
    wave_rest = np.asarray(pred_default.wave_rest)
    assert radio_off.shape == wave_rest.shape
    edge = 1e8
    f = _thermal_share()
    diff = rest_sed_default - rest_sed_off
    atol = 1e-9 * float(np.max(np.abs(f * radio_off)))

    below = wave_rest < edge
    np.testing.assert_allclose(diff[below], (-f * radio_off)[below], rtol=1e-9, atol=atol)

    # From the edge upward the thermal term is on: f (nu/1.4 GHz)^0.7 of the off-build
    # synchrotron, against the share f it displaces. Above 1.4 GHz that is a gain.
    from_edge = wave_rest >= edge
    want = (f * radio_off * (_thermal_over_sync_off(wave_rest) / f - 1.0))[from_edge]
    np.testing.assert_allclose(diff[from_edge], want, rtol=1e-9, atol=atol)
    first = int(np.flatnonzero(from_edge)[0])
    assert diff[first] > 0.0


@pytest.mark.parametrize("approx", [None, WavePrecomp()], ids=["exact", "lut"])
def test_photometry_on_both_paths(ssp_data_wne, approx):
    """Photometry is consistent on exact and LUT paths."""
    m_default = _build(ssp_data_wne, approx=approx)
    m_off = _build(ssp_data_wne, freefree=False, approx=approx)
    m_on = _build(ssp_data_wne, freefree=True, approx=approx)
    m_base = _build(ssp_data_wne, sf_type="none", agn_type="none", approx=approx)

    def flux(m):
        return np.asarray(m.predict_photometry(dict(m.spec.sample(jax.random.PRNGKey(0)))))

    flux_default, flux_off, flux_on, flux_base = map(flux, (m_default, m_off, m_on, m_base))
    share = 1.0 - _thermal_share()
    sync_off = flux_off - flux_base  # the radio-only band flux of the off build

    # atol=0.0 throughout: the fluxes are ~1e-27 erg/s/cm2/Hz, below any default
    # absolute tolerance, so only a relative comparison can fail.

    # Band 1 (mm3, below the edge): the SSP owns the thermal term, so the radio block holds
    # only the synchrotron share (1 - f). The explicit-True build shows this band does see
    # the radio thermal term.
    np.testing.assert_allclose(
        flux_default[1], flux_base[1] + share * sync_off[1], rtol=1e-10, atol=0.0
    )
    assert flux_on[1] > flux_default[1]

    # Band 2 (cm2p5, above the edge): default == on, and above off (thermal beats the
    # synchrotron share it displaces above 1.4 GHz).
    np.testing.assert_allclose(flux_default[2], flux_on[2], rtol=1e-10, atol=0.0)
    assert flux_default[2] > flux_off[2]

    # Band 0 (opt, no radio emission): default == off.
    np.testing.assert_allclose(flux_default[0], flux_off[0], rtol=1e-10, atol=0.0)

    # Band 3 (cm1p05, inside the first grid cell above the edge): the SSP has no
    # flux there, so the term must be complete, equal to the explicit-True build.
    np.testing.assert_allclose(flux_default[3], flux_on[3], rtol=1e-10, atol=0.0)
    assert flux_default[3] > flux_off[3]


def test_gradient_through_the_windowed_term_is_finite_and_nonzero(ssp_data_wne):
    """The windowed thermal term carries a finite gradient.

    Its own share at the edge node, none one node below.
    """
    sfh = {"type": "const", "log_total_mass": Uniform(9.0, 11.0), "all_params": Fixed(DEFAULT)}
    m_default = _build(ssp_data_wne, sfh=sfh)
    m_off = _build(ssp_data_wne, freefree=False, sfh=sfh)

    params = dict(m_default.spec.sample(jax.random.PRNGKey(0)))
    assert "sfh_const_log_total_mass" in params

    wave = np.asarray(m_default.predict_state(params).wave)
    edge = 1e8
    i_edge = int(np.flatnonzero(wave == edge)[0])
    i_below = int(np.flatnonzero(wave < edge)[-1])

    def grad_at(model, idx):
        def loss(p):
            return model.predict_state(p).derived["sed_radio"][idx]

        return float(jax.grad(loss)(params)["sfh_const_log_total_mass"])

    # At the edge node the thermal term adds its own share to d sed_radio / d log M.
    g_edge, g_edge_off = grad_at(m_default, i_edge), grad_at(m_off, i_edge)
    assert np.isfinite(g_edge) and g_edge != 0.0
    assert g_edge > 1.3 * g_edge_off  # ratio 1 - f + f (nu_edge/1.4 GHz)^0.7 = 2.0

    # One node below, the window zeroes the term: the gradient is finite and is
    # the synchrotron share (1 - f) of the off build, with nothing leaking through the mask.
    g_below, g_below_off = grad_at(m_default, i_below), grad_at(m_off, i_below)
    assert np.isfinite(g_below) and g_below != 0.0
    np.testing.assert_allclose(
        g_below, (1.0 - _thermal_share()) * g_below_off, rtol=1e-12, atol=0.0
    )


def test_config_refuses_a_window_without_the_term_and_bad_values():
    """Config validation for freefree_wave_min."""
    # freefree_wave_min without the term
    with pytest.raises(ConfigError):
        RadioSEDComponentConfig(include_freefree=False, freefree_wave_min=1e8)

    # Invalid values
    with pytest.raises(ValueError):
        RadioSEDComponentConfig(freefree_wave_min=-1.0)

    with pytest.raises(ValueError):
        RadioSEDComponentConfig(freefree_wave_min=0.0)

    with pytest.raises(ValueError):
        RadioSEDComponentConfig(freefree_wave_min=float("nan"))

    with pytest.raises(ValueError):
        RadioSEDComponentConfig(freefree_wave_min=float("inf"))

    with pytest.raises(TypeError):
        RadioSEDComponentConfig(freefree_wave_min="1e8")

    with pytest.raises(TypeError):
        RadioSEDComponentConfig(freefree_wave_min=True)

    # Valid: just freefree_wave_min without include_freefree (should default to True)
    cfg = RadioSEDComponentConfig(freefree_wave_min=1e8)
    assert cfg.include_freefree is True
