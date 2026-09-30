# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the per-Q_H nebular grid applies the escape and dust-destruction
fractions at reconstruction, so a free neb_fesc or neb_fdust moves the
grid-served emission exactly as the exact path does. Every other free
nebular parameter that the grid bakes in is refused.

Pinned:
  * neb_fesc ~ Uniform(0, 0.8): -2.95e-2 (fesc = 0) ... +4.73e-2 (fesc = 0.8)
  * neb_fdust ~ Uniform(0, 0.5): -2.2e-3 ... +5.27e-2
"""

from __future__ import annotations

import functools
import types
import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
    load_ssp_data,
)
from tengri.components.nebular.nebular_grid_precompute import grid_baked_free_params
from tengri.forward.sed_model import FeaturePrecomp
from tengri.inference import Fitter
from tengri.inference.fitter import fast_nebular_can_engage

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_BANDS = ["galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1"]
Z = 0.15

_RTOL_PHOT = 4e-3
_RTOL_LINES = 1e-4
_RTOL_GRAD = 2e-2
_TAU_NAMES = ("dust_tau_bc", "dust_tau_diff", "dust_tau_v")
_BAKED = {
    "neb_fesc_lya": ("fesc_lya", Uniform(0.0, 0.9)),
    "ionspec_index1": ("ionspec_index1", Uniform(1.0, 20.0)),
    "gas_logco": ("gas_logco", Uniform(-1.0, 0.5)),
    "neb_eline_sigma_kms": ("eline_sigma_kms", Uniform(50.0, 300.0)),
}

_TWO = {
    "type": "two_component",
    "law": "calzetti",
    "all_params": Fixed(DEFAULT),
    "tau_bc": Uniform(0.0, 2.0),
    "tau_diff": Uniform(0.0, 2.0),
}

_SSP = None


def _require():
    for f in (_BARE, "data/cue_weights.npz"):
        if not Path(f).is_file():
            pytest.skip(f"missing {f}")


@pytest.fixture(scope="module")
def ssp():
    global _SSP
    _require()
    _SSP = load_ssp_data(_BARE)
    return _SSP


@pytest.fixture(scope="module")
def data():
    fnu = np.array([2.0e-6, 4.0e-6, 6.0e-6, 8.0e-6, 1.0e-5, 1.5e-5])
    return fnu, 0.1 * fnu


def _model(dusty, neb_extra):
    kw = dict(
        ssp_data=_SSP,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        approx=WavePrecomp(),
        redshift=Fixed(Z),
        sfh={
            "type": "dpl",
            "all_params": Fixed(DEFAULT),
            "age_gyr": Uniform(0.05, 10.0),
            "log_total_mass": Uniform(8, 12),
        },
        neb={
            "type": "cue",
            "all_params": Fixed(DEFAULT),
            "neb_logU": Uniform(-3.5, -2.0),
            **neb_extra,
        },
    )
    if dusty:
        kw["dust_attenuation"] = _TWO
        kw["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kw)


@functools.cache
def _views(dusty: bool, key: str, lo: float, hi: float):
    """(model, grid-served model, evaluation point) for one free nebular parameter."""
    m = _model(dusty, {key: Uniform(lo, hi)})
    p = _point(m)
    flux = np.asarray(m.predict_photometry(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = Fitter(
            m, data=flux, noise=0.05 * np.abs(flux), data_type="photometry", approx="auto"
        ).model
    return m, fast, p


def _point(m) -> dict:
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    assert "sfh_dpl_age_gyr" in p
    p["sfh_dpl_age_gyr"] = 0.1
    for name in _TAU_NAMES:
        if name in m.spec.free_params:
            p[name] = 1.0
    assert set(p) == set(m.spec.free_params)
    return p


def _nebular_share(m, p) -> np.ndarray:
    d = m.predict_state(p).derived
    neb = np.asarray(d["nebular_phot_lnu_precomp"], dtype=float)
    star = np.asarray(d["stellar_phot_lnu_precomp"], dtype=float)
    return neb / (neb + star)


@pytest.mark.parametrize("dusty", [False, True])
def test_free_escape_fraction_moves_the_grid_served_emission(ssp, dusty):
    m, fast, p = _views(dusty, "fesc", 0.0, 0.8)

    assert fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    shares = _nebular_share(m, p)
    assert np.max(shares) > 0.10
    sh_min, sh_max = np.min(shares), np.max(shares)
    print(f"test_free_escape_fraction[dusty={dusty}]: shares={sh_min:.4f}-{sh_max:.4f}")

    measured_worst = 0.0
    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    print(f"test_free_escape_fraction[dusty={dusty}]: measured worst={float(measured_worst):.2e}")

    for fesc in (0.0, 0.2, 0.4, 0.6, 0.8):
        phot_fast = fast.predict_photometry({**p, "neb_fesc": fesc})
        phot_exact = m.predict_photometry({**p, "neb_fesc": fesc})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=_RTOL_PHOT)

    sensitivities = [
        float(
            jnp.max(
                jnp.abs(
                    (
                        m.predict_photometry({**p, "neb_fesc": 0.8})
                        - m.predict_photometry({**p, "neb_fesc": 0.0})
                    )
                    / jnp.maximum(jnp.abs(m.predict_photometry({**p, "neb_fesc": 0.0})), 1e-30)
                )
            )
        )
    ]
    assert sensitivities[0] > 10 * _RTOL_PHOT


@pytest.mark.parametrize("dusty", [False, True])
def test_free_dust_fraction_moves_the_grid_served_emission(ssp, dusty):
    m, fast, p = _views(dusty, "fdust", 0.0, 0.5)

    assert fast._nebular_grid_table is not None
    assert fast_nebular_can_engage(m)

    measured_worst = 0.0
    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        rel_diff = jnp.abs((phot_fast - phot_exact) / jnp.maximum(jnp.abs(phot_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    print(f"test_free_dust_fraction[dusty={dusty}]: measured worst={float(measured_worst):.2e}")

    for fdust in (0.0, 0.1, 0.25, 0.4, 0.5):
        phot_fast = fast.predict_photometry({**p, "neb_fdust": fdust})
        phot_exact = m.predict_photometry({**p, "neb_fdust": fdust})
        np.testing.assert_allclose(phot_fast, phot_exact, rtol=_RTOL_PHOT)

    sensitivities = [
        float(
            jnp.max(
                jnp.abs(
                    (
                        m.predict_photometry({**p, "neb_fdust": 0.5})
                        - m.predict_photometry({**p, "neb_fdust": 0.0})
                    )
                    / jnp.maximum(jnp.abs(m.predict_photometry({**p, "neb_fdust": 0.0})), 1e-30)
                )
            )
        )
    ]
    assert sensitivities[0] > 10 * _RTOL_PHOT


def test_a_fixed_escape_fraction_is_applied_at_reconstruction(ssp):
    m = _model(True, {"fesc": Fixed(0.3), "fdust": Fixed(0.1)})
    p = _point(m)
    flux = np.asarray(m.predict_photometry(p))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast = Fitter(
            m, data=flux, noise=0.05 * np.abs(flux), data_type="photometry", approx="auto"
        ).model

    phot_fast = fast.predict_photometry(p)
    phot_exact = m.predict_photometry(p)
    np.testing.assert_allclose(phot_fast, phot_exact, rtol=_RTOL_PHOT)

    m_zero = _model(True, {"fesc": Fixed(0.0), "fdust": Fixed(0.0)})
    p_zero = _point(m_zero)
    flux_zero = np.asarray(m_zero.predict_photometry(p_zero))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fast_zero = Fitter(
            m_zero,
            data=flux_zero,
            noise=0.05 * np.abs(flux_zero),
            data_type="photometry",
            approx="auto",
        ).model

    sensitivity = float(
        jnp.max(
            jnp.abs(
                (fast.predict_photometry(p) - fast_zero.predict_photometry(p_zero))
                / jnp.maximum(jnp.abs(fast_zero.predict_photometry(p_zero)), 1e-30)
            )
        )
    )
    assert sensitivity > 10 * _RTOL_PHOT


def test_total_photon_loss_gives_no_nebular_emission(ssp):
    m, fast, p = _views(False, "fesc", 0.0, 1.0)

    phot_fast = fast.predict_photometry({**p, "neb_fesc": 1.0})
    phot_exact = m.predict_photometry({**p, "neb_fesc": 1.0})
    np.testing.assert_allclose(phot_fast, phot_exact, rtol=_RTOL_PHOT)
    assert jnp.all(jnp.isfinite(phot_fast))

    def loss_fast(q):
        return jnp.sum(fast.predict_photometry(q))

    grad_fast = jax.grad(loss_fast)
    grad_at_one = grad_fast({**p, "neb_fesc": 1.0})
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grad_at_one)
    )

    grad_at_half = grad_fast({**p, "neb_fesc": 0.5})
    assert all(
        bool(jnp.all(jnp.isfinite(leaf))) for leaf in jax.tree_util.tree_leaves(grad_at_half)
    )


def test_the_gradient_with_respect_to_the_escape_fraction_matches(ssp):
    m, fast, p = _views(True, "fesc", 0.0, 0.8)

    def loss_fast(q):
        return jnp.sum(fast.predict_photometry(q))

    def loss_exact(q):
        return jnp.sum(m.predict_photometry(q))

    grad_fast = jax.grad(loss_fast)
    grad_exact = jax.grad(loss_exact)

    q_test = {**p, "neb_fesc": 0.4}
    g_fast = grad_fast(q_test)
    g_exact = grad_exact(q_test)

    g_fesc_fast = g_fast.get("neb_fesc", 0.0)
    g_fesc_exact = g_exact.get("neb_fesc", 0.0)

    assert float(g_fesc_fast) != 0.0
    assert float(g_fesc_exact) != 0.0
    np.testing.assert_allclose(g_fesc_fast, g_fesc_exact, rtol=_RTOL_GRAD)


def test_line_fluxes_follow_the_escape_fraction(ssp):
    m, fast, p = _views(True, "fesc", 0.0, 0.8)

    m_lines = m.with_approx((WavePrecomp(), FeaturePrecomp(lines=[6564.6, 4862.7, 1215.67])))
    fast_lines = fast.with_approx((WavePrecomp(), FeaturePrecomp(lines=[6564.6, 4862.7, 1215.67])))

    target_wavelengths = jnp.asarray([6564.6, 4862.7, 1215.67])

    measured_worst = 0.0
    for neb_fesc in (0.0, 0.6):
        lines_fast = fast_lines.predict_line_fluxes(
            {**p, "neb_fesc": neb_fesc}, target_wavelengths=target_wavelengths
        )
        lines_exact = m_lines.predict_line_fluxes(
            {**p, "neb_fesc": neb_fesc}, target_wavelengths=target_wavelengths
        )
        rel_diff = jnp.abs((lines_fast - lines_exact) / jnp.maximum(jnp.abs(lines_exact), 1e-30))
        measured_worst = jnp.maximum(measured_worst, jnp.max(rel_diff))

    print(
        f"test_line_fluxes_follow_the_escape_fraction: measured worst={float(measured_worst):.2e}"
    )
    np.testing.assert_allclose(lines_fast, lines_exact, rtol=_RTOL_LINES)

    lya_fast_0 = fast_lines.predict_line_fluxes(
        {**p, "neb_fesc": 0.0}, target_wavelengths=target_wavelengths
    )[2]
    lya_fast_6 = fast_lines.predict_line_fluxes(
        {**p, "neb_fesc": 0.6}, target_wavelengths=target_wavelengths
    )[2]
    lya_exact_0 = m_lines.predict_line_fluxes(
        {**p, "neb_fesc": 0.0}, target_wavelengths=target_wavelengths
    )[2]
    lya_exact_6 = m_lines.predict_line_fluxes(
        {**p, "neb_fesc": 0.6}, target_wavelengths=target_wavelengths
    )[2]

    assert float(lya_fast_6) < float(lya_fast_0)
    assert float(lya_exact_6) < float(lya_exact_0)


@pytest.mark.parametrize("name", list(_BAKED.keys()))
def test_a_baked_free_parameter_is_refused(ssp, name):
    key, prior = _BAKED[name]
    m = _model(False, {key: prior})

    assert grid_baked_free_params(m.spec) == (name,)
    assert fast_nebular_can_engage(m) is False

    with pytest.raises(ValueError, match=name):
        m.with_approx((WavePrecomp(), FeaturePrecomp()))

    with pytest.raises(ValueError, match="reference value"):
        m.with_approx((WavePrecomp(), FeaturePrecomp()))


def test_every_nebular_parameter_has_exactly_one_disposition(ssp):
    from tengri.components.nebular.nebular_grid_precompute import (
        _CANDIDATE_AXES,
        _RECONSTRUCTION_MIXED,
        _RECONSTRUCTION_SCALED,
    )

    m = _model(False, {})
    declared_params = set(m.spec.free_params) | set(m.spec.get_fixed_values().keys())
    nebular_params = sorted(
        [p for p in declared_params if p.startswith(("neb_", "ionspec_", "gas_"))]
    )
    print(f"nebular params: {nebular_params}")

    assert "neb_fesc" in nebular_params
    assert "neb_fdust" in nebular_params
    assert "neb_fesc_lya" in nebular_params
    assert "neb_logU" in nebular_params
    assert "neb_dig_frac" in nebular_params

    for param_name in nebular_params:
        stand_in = types.SimpleNamespace(free_params=[param_name])
        count = sum(
            [
                param_name in _CANDIDATE_AXES,
                param_name in _RECONSTRUCTION_SCALED,
                param_name in _RECONSTRUCTION_MIXED,
                grid_baked_free_params(stand_in) == (param_name,),
            ]
        )
        assert count == 1, f"Parameter {param_name} has disposition count {count}, expected 1"

    axes_count = sum(1 for p in nebular_params if p in _CANDIDATE_AXES)
    scaled_count = sum(1 for p in nebular_params if p in _RECONSTRUCTION_SCALED)
    mixed_count = sum(1 for p in nebular_params if p in _RECONSTRUCTION_MIXED)
    baked_count = sum(
        1
        for p in nebular_params
        if grid_baked_free_params(types.SimpleNamespace(free_params=[p])) == (p,)
    )

    assert axes_count > 0
    assert scaled_count > 0
    assert mixed_count > 0
    assert baked_count > 0
