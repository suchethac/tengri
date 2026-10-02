# SPDX-License-Identifier: BSD-3-Clause
"""THEMIS ``U^-alpha`` component: the alpha != 2 axis is tabulated per q_hAC.

The PDR emission is the U-integral of the single-U emission of the grains of one
q_hAC mixture, so the template shape and power weight at alpha != 2 depend on
q_hAC and alpha jointly.

References
----------
.. [1] Jones, A. P. et al. 2017, A&A, 602, A46 (THEMIS).
.. [2] Draine, B. T. & Li, A. 2007, ApJ, 657, 810, Eq. 23 (``dU/dM ~ U^-alpha``).
"""

from __future__ import annotations

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp, load_ssp
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve
from tests._data_skip import DATA_DIR

pytestmark = pytest.mark.regression_bug

C_AA = 2.99792458e18
FINE = np.logspace(4, 7, 20001)
#: 8-24, 24-70, 70-160, 160-500, 500-1000 um in Angstrom.
BANDS = ((8e4, 2.4e5), (2.4e5, 7e5), (7e5, 1.6e6), (1.6e6, 5e6), (5e6, 1e7))

#: (q_hAC, U_min, alpha, gamma) and the five band fractions of unit power in
#: 1 um-1 mm from pcigale ``themis.THEMIS(...).process``.
PUBLIC_NODES = (
    pytest.param(
        (0.02, 0.5, 1.0, 0.01),
        (
            0.6490117412330542,
            0.08170433312876643,
            0.001106415141096436,
            6.162261024891557e-05,
            1.3011576867341749e-06,
        ),
        id="q0.02-U0.5-a1-g0.01",
    ),
    pytest.param(
        (0.40, 30.0, 3.0, 0.5),
        (
            0.2752405514720881,
            0.24097820880302298,
            0.17465381416148637,
            0.022238535987217457,
            0.0002508506522407425,
        ),
        id="q0.40-U30-a3-g0.5",
    ),
)

REGISTRY_NODES = ((0.02, 0.5, 1.0, 0.01), (0.17, 1.0, 1.0, 0.1), (0.40, 30.0, 1.0, 0.5))


def _unit(wave, y):
    y = np.interp(FINE, wave, y, left=0.0, right=0.0)
    return y / np.trapezoid(y, FINE)


def _bands(y):
    grid = FINE
    masks = [(grid >= lo) & (grid <= hi) for lo, hi in BANDS]
    return np.array([np.trapezoid(y[m], grid[m]) for m in masks])


def _pcigale_bands(q, u, a, g):
    from pcigale.sed import SED
    from pcigale.sed_modules import themis

    sed = SED()
    sed.add_info("dust.luminosity", 1.0, True, unit="W")
    themis.THEMIS(name="themis", qhac=q, umin=u, alpha=a, gamma=g).process(sed)
    return _bands(_unit(sed.wavelength_grid * 10.0, sed.luminosity / 10.0))


def _model(node, observation=None, approx=None):
    q, u, a, g = node
    kwargs = {} if observation is None else {"observation": observation}
    if approx is not None:
        kwargs["approx"] = approx
    return SEDModel.build(
        ssp_data=load_ssp("fsps_prsc_miles_chabrier", download=False),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Fixed(1.0),
            "tau_diff": Fixed(1.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={
            "type": "themis",
            "dust_qhac": Fixed(q),
            "dust_umin": Fixed(u),
            "dust_alpha": Fixed(a),
            "dust_gamma_dl": Fixed(g),
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
        **kwargs,
    )


def _public_band_fractions(node):
    state = _model(node).predict_state({})
    wave = np.asarray(state.wave, dtype=np.float64)
    sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    return _bands(_unit(wave, sed * C_AA / wave**2))


@pytest.fixture(scope="module")
def templates():
    with h5py.File(DATA_DIR / "themis_templates.h5", "r") as h:
        yield {k: h[k][:] for k in h}


@pytest.mark.parametrize("node", REGISTRY_NODES)
def test_registry_band_powers_match_pcigale_at_alpha_1(node, templates):
    """Registry ``themis`` band powers equal pcigale's per-q_hAC library at alpha = 1."""
    pytest.importorskip("pcigale")
    from tengri.components.dust.emission import DUST_EMISSION_MODELS

    q, u, a, g = node
    wave = templates["wavelength_aa"]
    lnu = np.asarray(
        DUST_EMISSION_MODELS["themis"](
            wave, 1.0, dust_umin=u, dust_gamma_dl=g, dust_qhac=q, dust_alpha=a
        ),
        dtype=np.float64,
    )
    got = _bands(_unit(wave, lnu * C_AA / wave**2))
    np.testing.assert_allclose(got / _pcigale_bands(*node), 1.0, atol=2e-3)


def test_alpha_2_slice_of_the_ratio_is_exactly_one(templates):
    """The alpha = 2 slice of ``powerlaw_alpha_ratio`` is 1 at every (q_hAC, U_min, wave)."""
    ratio = templates["powerlaw_alpha_ratio"]
    k2 = int(np.argmin(np.abs(templates["alpha_grid"] - 2.0)))
    assert ratio.shape == (
        templates["qhac_grid"].size,
        templates["umin_grid"].size,
        templates["alpha_grid"].size,
        templates["wavelength_aa"].size,
    )
    assert np.array_equal(ratio[:, :, k2], np.ones_like(ratio[:, :, k2]))


def test_ratio_is_not_the_qhac_average(templates):
    """The alpha = 1 PDR power differs from the q_hAC-averaged-ratio power by 0.70-1.38x."""
    ratio = templates["powerlaw_alpha_ratio"].astype(np.float64)[:, :, 0, :]
    nu = C_AA / templates["wavelength_aa"]
    per_q = np.trapezoid(templates["powerlaw"] * ratio, nu, axis=-1)
    averaged = np.trapezoid(templates["powerlaw"] * ratio.mean(axis=0)[None], nu, axis=-1)
    assert (averaged / per_q).min() < 0.75
    assert (averaged / per_q).max() > 1.25


@pytest.mark.parametrize(
    ("q", "u", "a"),
    [(0.02, 0.5, 1.0), (0.17, 1.0, 1.0), (0.40, 30.0, 1.0), (0.02, 0.5, 3.0), (0.40, 30.0, 3.0)],
)
def test_pdr_power_weight_matches_pcigale(q, u, a, templates):
    """PDR power over single-U power equals pcigale's database ratio at fixed U_min."""
    pytest.importorskip("pcigale")
    from pcigale.data import SimpleDatabase

    nu = C_AA / templates["wavelength_aa"]
    iq = int(np.argmin(np.abs(templates["qhac_grid"] * 2.2 / 100.0 - q)))
    iu = int(np.argmin(np.abs(templates["umin_grid"] - u)))
    ia = int(np.argmin(np.abs(templates["alpha_grid"] - a)))
    plaw = templates["powerlaw"][iq, iu] * templates["powerlaw_alpha_ratio"][iq, iu, ia]
    weight = np.trapezoid(plaw, nu) / np.trapezoid(templates["single_u"][iq, iu], nu)

    with SimpleDatabase("themis") as db:
        single = db.get(qhac=q, umin=u, umax=u, alpha=1.0)
        pdr = db.get(qhac=q, umin=u, umax=1e7, alpha=a)
    reference = np.trapezoid(pdr.spec, pdr.wl) / np.trapezoid(single.spec, single.wl)
    assert weight / reference == pytest.approx(1.0, abs=2e-3)


@pytest.mark.parametrize(("node", "expected"), PUBLIC_NODES)
def test_public_sed_dust_ir_band_fractions_literals(node, expected):
    """``SEDModel.build`` THEMIS ``sed_dust_ir`` band fractions equal pcigale's literals."""
    np.testing.assert_allclose(_public_band_fractions(node), expected, rtol=3e-3, atol=0.0)


@pytest.mark.parametrize(("node", "expected"), PUBLIC_NODES)
def test_public_sed_dust_ir_band_fractions_live_pcigale(node, expected):
    """``SEDModel.build`` THEMIS ``sed_dust_ir`` band fractions equal a live pcigale run."""
    pytest.importorskip("pcigale")
    live = _pcigale_bands(*node)
    np.testing.assert_allclose(live, expected, rtol=1e-6)
    np.testing.assert_allclose(_public_band_fractions(node) / live, 1.0, atol=3e-3)


@pytest.mark.parametrize(("node", "expected"), PUBLIC_NODES)
def test_public_float32_matches_float64(node, expected):
    """Pure-float32 ``sed_dust_ir`` band fractions agree with float64 to 1e-4."""
    ref = _public_band_fractions(node)
    with jax.enable_x64(False):
        state = _model(node).predict_state({})
        assert state.derived["sed_dust_ir"].dtype == jnp.float32
        wave = np.asarray(state.wave, dtype=np.float64)
        sed = np.asarray(state.derived["sed_dust_ir"], dtype=np.float64)
    got = _bands(_unit(wave, sed * C_AA / wave**2))
    np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-4)


def test_precompute_photometry_matches_exact_in_far_ir_top_hats():
    """``WavePrecomp`` and exact THEMIS photometry agree to 1e-3 at alpha = 1."""

    def tophat(center, frac=0.16, n=40):
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    obs = Observation(
        photometry=Photometry(filters=tuple(tophat(c) for c in (2.4e5, 1.0e6, 3.5e6)))
    )
    node = PUBLIC_NODES[0].values[0]
    exact = np.asarray(_model(node, observation=obs).predict_photometry({}))
    fast = np.asarray(_model(node, observation=obs, approx=WavePrecomp()).predict_photometry({}))
    assert np.all(exact > 0.0)
    np.testing.assert_allclose(fast, exact, rtol=1e-3)
