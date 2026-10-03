# SPDX-License-Identifier: BSD-3-Clause
"""The AGN line-only SED joins the instrument-only kernel group (#2565).

``sed_agn_lines_attenuated`` is the AGN's line light exactly as it enters the
pipeline SED. Both kernel splits in ``observation.py`` put it beside
``sed_nebular`` and ``sed_shock`` in the instrument-only group, so

- the stellar group is ``sed - instrument_only`` by construction (sum identity),
- the published array equals what the SED carries (AGN's own screen, plus the
  host ``agn_screen`` when the AGN runs before the dust adapter),
- every spectrum surface, with an IGM, agrees to 1e-10, and
- a model with no AGN component is untouched.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    SpectrumPrecomp,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.observation.banded import gaussian_resolution_bands
from tengri.observation.observation import (
    _split_stellar_and_instrument_only_sed,
    _split_stellar_and_instrument_only_sed_pre_igm,
)
from tengri.observation.photometry import FilterCurve
from tengri.observation.spectroscopy import Spectroscopy

pytestmark = pytest.mark.regression_bug

_MILES_BARE = "data/fsps_prsc_miles_chabrier.h5"
_DISC = {"type": "powerlaw", "all_params": Fixed(DEFAULT)}
_TWO = {"type": "two_component", "law": "calzetti", "all_params": FREE}
_ALL_LINES = {
    "type": "composable",
    "norm": "independent",
    "disc": _DISC,
    "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
    "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
    "feii": {"type": "boroson_green", "all_params": Fixed(DEFAULT)},
    "all_params": Fixed(DEFAULT),
}
_LINE_BEARING = {
    "composable_nlr_blr_feii": (_ALL_LINES, _TWO),
    "composable_agn_screen_diffuse": (_ALL_LINES, {**_TWO, "agn_screen": "diffuse"}),
    "composable_single_screen": (
        _ALL_LINES,
        {"type": "single_component", "law": "calzetti", "all_params": FREE},
    ),
    "grahsp_monolithic": ({"type": "grahsp", "all_params": Fixed(DEFAULT)}, _TWO),
    "qsogen_preset": ({"type": "qsogen", "all_params": Fixed(DEFAULT)}, _TWO),
    "unified_nlr_blr_preset": ({"type": "unified_nlr_blr", "all_params": Fixed(DEFAULT)}, _TWO),
}
_LINELESS = {
    "skirtor_stalevski": ({"type": "skirtor_stalevski", "all_params": Fixed(DEFAULT)}, _TWO)
}


def _ssp_or_skip():
    if not Path(_MILES_BARE).is_file():
        pytest.skip(f"missing SSP grid {_MILES_BARE}")
    return load_ssp_data(_MILES_BARE)


def _build(
    agn,
    dust,
    wave_obs,
    *,
    z,
    igm=True,
    resolution=3000.0,
    resolution_matrix=None,
    approx=None,
    filters=(),
):
    obs = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=wave_obs,
            resolution=resolution,
            sigma_lib_kms=70.0,
            resolution_matrix=resolution_matrix,
        ),
        photometry=Photometry(filters=tuple(filters)) if filters else None,
    )
    kwargs = {"igm": {"type": "inoue14"}} if igm else {}
    if agn is not None:
        kwargs["agn"] = agn
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = SEDModel.build(
            ssp_data=_ssp_or_skip(),
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust,
            neb={"type": "none"},
            redshift=Fixed(z),
            approx=approx,
            **kwargs,
        )
        merged = base.spec.merge_observation_params(sigma_v_kms=Uniform(0.0, 2000.0))
        return SEDModel(merged, _ssp_or_skip(), observation=obs)


def _params(model, sigma_v=300.0):
    p = dict(model.spec.sample(jax.random.PRNGKey(5)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    p["sigma_v_kms"] = jnp.asarray(sigma_v)
    return p


def _rel_max(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ── sum identity on every AGN variant, both split functions ──


@pytest.mark.parametrize("variant", sorted(_LINE_BEARING) + sorted(_LINELESS))
@pytest.mark.parametrize("igm", [False, True])
def test_split_sum_identity_and_group_content(variant, igm):
    """stellar + instrument_only == sed to 1e-12 (relative), and the group is
    exactly nebular + shock + the published AGN line light.
    """
    agn, dust = {**_LINE_BEARING, **_LINELESS}[variant]
    z = 2.0
    wave_obs = jnp.linspace(4000.0, 5200.0, 300)
    model = _build(agn, dust, wave_obs, z=z, igm=igm)
    p = _params(model)
    state = model.predict_state(p)
    sed_rest = state.sed_intrinsic
    lines_att = state.derived.get("sed_agn_lines_attenuated")
    assert lines_att is not None, "every AGN variant publishes the key"
    if variant in _LINELESS:
        assert float(jnp.max(jnp.abs(lines_att))) == 0.0
    else:
        assert float(jnp.max(lines_att)) > 0.0

    expected_group = (
        jnp.asarray(state.derived.get("sed_nebular", 0.0))
        + jnp.asarray(state.derived.get("sed_shock", 0.0))
        + lines_att
    )
    stellar_rest, instrument_rest = _split_stellar_and_instrument_only_sed_pre_igm(state, sed_rest)
    chex.assert_trees_all_close(instrument_rest, expected_group, rtol=1e-14, atol=0.0)
    assert _rel_max(stellar_rest + instrument_rest, sed_rest) < 1e-12

    trans = state.derived.get("igm_transmission") if igm else None
    sed_spec = sed_rest if trans is None else sed_rest * trans
    stellar, instrument = _split_stellar_and_instrument_only_sed(state, sed_spec, trans)
    assert _rel_max(stellar + instrument, sed_spec) < 1e-12
    expected = expected_group if trans is None else expected_group * trans
    chex.assert_trees_all_close(instrument, expected, rtol=1e-14, atol=0.0)


# ── the published array is the light the SED carries ──


def test_published_lines_are_the_light_in_the_sed_unscreened():
    """agn_screen='none': the AGN runs after the dust adapter, the lines enter
    the SED untouched, and the published array equals ``sed_agn_lines``.
    """
    agn, dust = _LINE_BEARING["composable_nlr_blr_feii"]
    model = _build(agn, dust, jnp.linspace(4000.0, 5200.0, 300), z=2.0)
    state = model.predict_state(_params(model))
    chex.assert_trees_all_close(
        state.derived["sed_agn_lines_attenuated"],
        state.derived["sed_agn_lines"],
        rtol=0.0,
        atol=0.0,
    )


def test_published_lines_carry_the_host_screen_when_agn_runs_first():
    """agn_screen='diffuse': the AGN runs before the dust adapter, which
    multiplies the lines by the diffuse-screen transmission it applies to
    ``sed_agn``; the published full-grid diffuse transmission is the oracle.
    """
    agn, dust = _LINE_BEARING["composable_agn_screen_diffuse"]
    model = _build(agn, dust, jnp.linspace(4000.0, 5200.0, 300), z=2.0)
    p = _params(model)
    p["dust_tau_diff"] = jnp.asarray(0.8)
    state = model.predict_state(p)
    lines = state.derived["sed_agn_lines"]
    transmission = state.derived["dust_diff_transmission"]
    assert float(jnp.min(transmission)) < 0.9, "the screen must bite for the test to bind"
    chex.assert_trees_all_close(
        state.derived["sed_agn_lines_attenuated"], lines * transmission, rtol=1e-12, atol=0.0
    )


# ── cross-path agreement with an AGN model, IGM on ──


def _surfaces(model, p, wave_obs):
    pred = model.predict(p)
    return {
        "eager": model._spectrum_via_state(p, wave_obs=wave_obs),
        "predict_spectrum_grid": model.predict_spectrum(p, wave_obs=wave_obs),
        "compiled_kernel": model.predict_spectrum(p),
        "Prediction.spectrum()": pred.spectrum(),
        "Prediction.spectrum(wave_obs)": pred.spectrum(wave_obs=wave_obs),
    }


@pytest.mark.parametrize("variant", ["composable_nlr_blr_feii", "grahsp_monolithic"])
@pytest.mark.parametrize("banded", [False, True])
def test_cross_path_agreement_agn_igm_z2_sigma_v300(variant, banded):
    """Every spectrum surface returns the same flux to 1e-10 (AGN, IGM, z=2, sigma_v=300)."""
    agn, dust = _LINE_BEARING[variant]
    wave_obs = jnp.linspace(4000.0, 5200.0, 300)
    kwargs = {"resolution": 3000.0}
    if banded:
        kwargs = {
            "resolution": None,
            "resolution_matrix": gaussian_resolution_bands(wave_obs, resolution=3000.0, n_diag=15),
        }
    model = _build(agn, dust, wave_obs, z=2.0, **kwargs)
    p = _params(model, 300.0)
    surfaces = _surfaces(model, p, wave_obs)

    if not banded:
        state = model.predict_state(p)
        z = model._get_redshift(p)
        from tengri.cosmology import luminosity_distance

        dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
        out = model.observation.predict(
            state,
            {**p, "redshift": z},
            dl_cm=dl_cm,
            wave_obs=wave_obs,
            sigma_v_kms=300.0,
            lsf_resolution=3000.0,
            lsf_sigma_lib_kms=70.0,
        )
        surfaces["Observation.predict"] = out["spec_fnu"]

    reference = surfaces["eager"]
    assert float(jnp.max(jnp.abs(reference))) > 0.0
    for name, flux in surfaces.items():
        np.testing.assert_allclose(flux, reference, rtol=1e-10, atol=0.0, err_msg=name)


def test_spectrum_precomp_agrees_with_exact_path_for_agn_model():
    """SpectrumPrecomp applies no kinematics or LSF (documented), so the
    comparison is at sigma_v = 0 with no LSF: the spectrum LUT point-samples the
    AGN SED, and the split changes nothing there, since the two kernel groups
    sum linearly. Tolerance is the LUT's documented 1e-6.
    """
    agn, dust = _LINE_BEARING["composable_nlr_blr_feii"]
    wave_obs = jnp.linspace(4000.0, 5200.0, 200)
    exact = _build(agn, dust, wave_obs, z=2.0, resolution=None)
    lut = _build(agn, dust, wave_obs, z=2.0, resolution=None, approx=SpectrumPrecomp())
    p = _params(exact, 0.0)
    p["dust_tau_bc"] = jnp.asarray(0.0)
    p["dust_tau_diff"] = jnp.asarray(0.0)
    flux_exact = np.asarray(exact.predict_spectrum(p))
    flux_lut = np.asarray(lut.predict_spectrum(p))
    assert _rel_max(flux_lut, flux_exact) < 1e-6


# ── models without an AGN component are untouched ──


def test_model_without_agn_has_no_line_key_and_group_is_neb_plus_shock():
    """No AGN component: the key is absent and the instrument-only group is
    exactly ``sed_nebular + sed_shock`` as before.
    """
    wave_obs = jnp.linspace(4000.0, 5200.0, 300)
    model = _build(None, _TWO, wave_obs, z=2.0)
    state = model.predict_state(_params(model))
    assert state.derived.get("sed_agn_lines_attenuated") is None
    _, instrument_rest = _split_stellar_and_instrument_only_sed_pre_igm(state, state.sed_intrinsic)
    expected = jnp.asarray(state.derived.get("sed_nebular", 0.0)) + jnp.asarray(
        state.derived.get("sed_shock", 0.0)
    )
    chex.assert_trees_all_close(instrument_rest, expected, rtol=0.0, atol=0.0)


# ── the change is spectroscopy-only: photometry never reads the new key ──


def _band(center, n=24):
    wave = np.linspace(center * 0.85, center * 1.15, n)
    trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{center:.0f}")


def _build_photometry_only(agn, dust, *, z, approx):
    obs = Observation(
        photometry=Photometry(filters=tuple(_band(c) for c in (3000.0, 4500.0, 7000.0, 12000.0)))
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp_or_skip(),
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=dust,
            neb={"type": "none"},
            igm={"type": "inoue14"},
            agn=agn,
            redshift=Fixed(z),
            approx=approx,
        )


@pytest.mark.parametrize("approx", [None, WavePrecomp()], ids=["exact", "wave_precomp"])
def test_photometry_is_bit_identical_with_and_without_the_key(approx):
    """``sed_agn_lines_attenuated`` is a subset of the AGN light used only by the
    spectrum kernel split: the exact and precompute photometry paths neither
    read it nor sum a photometry LUT for it (no ``*_phot_lnu_precomp`` double
    count), so stripping it from the state changes no photometry bit.
    """
    agn, dust = _LINE_BEARING["composable_nlr_blr_feii"]
    model = _build_photometry_only(agn, dust, z=2.0, approx=approx)
    p = dict(model.spec.sample(jax.random.PRNGKey(5)))
    p["sfh_dpl_log_total_mass"] = jnp.asarray(10.0)
    state = model.predict_state(p)
    assert state.derived.get("sed_agn_lines_attenuated") is not None
    stripped = state.with_(derived=state.derived.with_(sed_agn_lines_attenuated=None))
    assert stripped.derived.get("sed_agn_lines_attenuated") is None

    lut_keys = [k for k in state.derived.field_names() if k.endswith("_phot_lnu_precomp")]
    assert not [k for k in lut_keys if "lines" in k], lut_keys
    if approx is not None:
        assert lut_keys, "the precompute path must have published its photometry tables"

    full_p = {**model.spec.get_fixed_values(), **p}
    project = (
        model.observation.predict if approx is None else model.observation.predict_via_precomp
    )
    with_key = np.asarray(project(state, full_p)["phot_fnu"])
    without_key = np.asarray(project(stripped, full_p)["phot_fnu"])
    np.testing.assert_array_equal(with_key, without_key)
    np.testing.assert_allclose(np.asarray(model.predict_photometry(p)), with_key, rtol=1e-10)
