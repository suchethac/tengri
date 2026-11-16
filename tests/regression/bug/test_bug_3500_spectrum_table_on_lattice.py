# SPDX-License-Identifier: BSD-3-Clause
r"""The spectrum table holds the templates on the projection's ln-lambda lattice (#3500).

``approx=SpectrumPrecomp()`` used to hold the SSP templates at the pixel centers, broadened
on the native grid in 16 index segments, and to convolve the instrument kernel on the
pixels. The table now holds each template as the cell means the exact projection loads onto
its :math:`\ln\lambda` lattice, and ends in the same kernel and readout
(``lattice_pixel_read``), at the pixel centers or as the mean between the pixel edges.

The flux-conserving exact path with a Gaussian resolution moved onto the same lattice. Its
reference here is the pixel mean of ``tests/_lsf_reference.convolve_piecewise_linear``,
which integrates the model's interpolant segment by segment and shares no code with the
lattice.
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import simpson

from tengri import DEFAULT, Fixed, Observation, SEDModel, Uniform
from tengri.components.stellar.sps.precompute import (
    SpectrumLatticeTable,
    spectrum_lattice_table_nbytes,
)
from tengri.forward.sed_model import SpectrumPrecomp
from tengri.observation.spectroscopy import Spectroscopy
from tengri.observation.spectrum import (
    _pixel_edges,
    compute_spectrum_conserving_lsf,
    lattice_span,
    project_spectrum,
)
from tengri.units import lnu_to_fnu
from tengri.utils.physics_constants import C_KM_S, GAUSSIAN_FWHM_PER_SIGMA
from tests._lsf_reference import convolve_piecewise_linear

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_FINE = np.linspace(4000.0, 7900.0, 1800)
_COARSE = np.linspace(4000.0, 7900.0, 300)


def _build(ssp, *, approx, wave_obs, resample, resolution, sigma_v, redshift=None, **components):
    obs = Observation(
        spectroscopy=Spectroscopy(resample=resample, wave_obs=wave_obs, resolution=resolution)
    )
    base = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(_Z) if redshift is None else redshift,
        sfh={"type": "tsnorm", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        **components,
    )
    spec = base.spec.merge_observation_params(sigma_v_kms=sigma_v)
    return SEDModel(spec, ssp, observation=obs, approx=approx)


def _pair(ssp, **kwargs):
    """The exact model and its spectrum-table twin; the table must be active."""
    exact = _build(ssp, approx=None, **kwargs)
    table = _build(ssp, approx=SpectrumPrecomp(), **kwargs)
    assert table._approx.get("spectrum_precomp"), "the table declined; the test would be vacuous"
    return exact, table


def _max_rel(a, b) -> float:
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ── the table against the exact path ─────────────────────────────────────────


@pytest.mark.parametrize(
    ("resample", "wave_obs", "resolution", "bound"),
    [
        # Measured 9.5e-5: the per-pixel library width puts the instrument kernel in 16
        # segments, applied to the combined kernel on one side and after sigma_v on the other.
        ("point", _FINE, 2000.0, 3e-4),
        # Measured 1.2e-6.
        ("point", _FINE, None, 1e-5),
        # Measured 1.3e-5 and 1.2e-5.
        ("conserving", _COARSE, 2000.0, 5e-5),
        ("conserving", _COARSE, None, 5e-5),
    ],
)
def test_table_matches_the_exact_path(ssp_data_wne, resample, wave_obs, resolution, bound):
    exact, table = _pair(
        ssp_data_wne,
        wave_obs=wave_obs,
        resample=resample,
        resolution=resolution,
        sigma_v=Fixed(150.0),
    )
    assert _max_rel(table.predict_spectrum({}), exact.predict_spectrum({})) < bound


def test_table_with_dust_and_igm_matches_away_from_the_calzetti_branch_join(ssp_data_wne):
    """The table applies dust after the velocity kernel; the exact path before it.

    The two orders differ only where the curve is not smooth. The Calzetti law's two
    branches meet at 0.63 micron with a step of 0.025 in k, which the exact path smooths
    by sigma_v and the table does not.
    """
    exact, table = _pair(
        ssp_data_wne,
        wave_obs=_FINE,
        resample="point",
        resolution=2000.0,
        sigma_v=Fixed(150.0),
        igm={"type": "madau"},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
    )
    f_exact = np.asarray(exact.predict_spectrum({}), dtype=float)
    f_table = np.asarray(table.predict_spectrum({}), dtype=float)
    rest = _FINE / (1.0 + _Z)
    near_join = np.abs(rest - 6300.0) < 30.0
    peak = np.max(np.abs(f_exact))
    # Measured: 4.5e-4 of peak at the join, at most 1e-4 elsewhere.
    assert np.max(np.abs(f_table - f_exact)[near_join]) / peak < 1.5e-3
    assert np.max(np.abs(f_table - f_exact)[~near_join]) / peak < 3e-4


def test_conserving_table_with_a_single_screen_at_the_calzetti_join(ssp_data_wne):
    """One pixel straddles the 0.63 micron join; the worst pixel is reported against exact."""
    exact, table = _pair(
        ssp_data_wne,
        wave_obs=_COARSE,
        resample="conserving",
        resolution=2000.0,
        sigma_v=Fixed(150.0),
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
    )
    edges = np.asarray(_pixel_edges(jnp.asarray(_COARSE))) / (1.0 + _Z)
    assert np.any((edges[:-1] < 6300.0) & (edges[1:] > 6300.0)), "no pixel straddles the join"
    # Measured: see the PR body; the bound is the dust-order difference at the join.
    assert _max_rel(table.predict_spectrum({}), exact.predict_spectrum({})) < 5e-4


def test_one_table_serves_a_free_redshift(ssp_data_wne):
    kwargs = dict(
        wave_obs=_FINE,
        resample="point",
        resolution=2000.0,
        sigma_v=Fixed(150.0),
        redshift=Uniform(0.05, 0.3),
    )
    exact, table = _pair(ssp_data_wne, **kwargs)
    for z in (0.06, 0.171, 0.29):
        params = {"redshift": z}
        assert _max_rel(table.predict_spectrum(params), exact.predict_spectrum(params)) < 3e-4


# ── what the table holds ─────────────────────────────────────────────────────


def test_table_is_on_the_lattice_and_its_size_is_predicted(ssp_data_wne):
    model = _build(
        ssp_data_wne,
        approx=SpectrumPrecomp(),
        wave_obs=_FINE,
        resample="point",
        resolution=2000.0,
        sigma_v=Fixed(150.0),
    )
    table = model._get_component_chain()[0]._state.ssp_spec_lut
    assert isinstance(table, SpectrumLatticeTable)
    origin, n_cells = lattice_span(_FINE, _Z, _Z)
    assert (table.origin, table.n_cells) == (origin, n_cells)
    cube = ssp_data_wne.ssp_flux
    assert table.ssp_on_lattice.shape == (*cube.shape[:-1], n_cells)
    assert table.ssp_on_lattice_slope.shape == table.ssp_on_lattice.shape
    held = table.ssp_on_lattice.nbytes + table.ssp_on_lattice_slope.nbytes
    assert spectrum_lattice_table_nbytes(cube.shape, n_cells, cube.dtype.itemsize) == held


# ── where the table declines ─────────────────────────────────────────────────


def test_table_declines_a_kernel_the_lattice_does_not_resolve(ssp_data_wne):
    """MILES at R = 2000 with no dispersion: the library width cancels the instrument's."""
    with pytest.warns(UserWarning, match="lattice does not resolve"):
        model = _build(
            ssp_data_wne,
            approx=SpectrumPrecomp(),
            wave_obs=_FINE,
            resample="point",
            resolution=2000.0,
            sigma_v=Fixed(0.0),
        )
    assert not model._approx.get("spectrum_precomp")


def test_conserving_pixels_are_served_at_any_kernel_width(ssp_data_wne):
    """The pixel mean integrates the cell means exactly, so a narrow kernel is no limit."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        exact, table = _pair(
            ssp_data_wne,
            wave_obs=_COARSE,
            resample="conserving",
            resolution=2000.0,
            sigma_v=Fixed(0.0),
        )
    # Measured: see the PR body.
    assert _max_rel(table.predict_spectrum({}), exact.predict_spectrum({})) < 5e-5


# ── the conserving exact path against an analytic reference ──────────────────

_LINES = (
    (3934.8, 0.60, 1.2),
    (3969.6, 0.50, 1.2),
    (4861.3, 0.40, 1.0),
    (5175.0, 0.30, 1.5),
    (5895.0, 0.35, 1.3),
    (6564.6, 0.50, 1.0),
)
_WAVE_REST = np.concatenate(
    [
        np.linspace(1000.0, 3500.0, 400, endpoint=False),
        np.arange(3500.0, 7500.0, 0.9),
        np.linspace(7500.0, 20000.0, 600),
    ]
)
_SED = (
    1.0e28
    * (_WAVE_REST / 5500.0) ** -0.7
    * (1.0 - sum(d * np.exp(-0.5 * ((_WAVE_REST - c) / w) ** 2) for c, d, w in _LINES))
)
_DL_CM = 1.4e27


def _pixel_mean_reference(wave_obs, resolution, sigma_v_kms, n_sub=33):
    """Pixel mean of the closed-form convolution, by Simpson's rule over each pixel."""
    edges = np.asarray(_pixel_edges(jnp.asarray(wave_obs)))
    width = np.sqrt((C_KM_S / (GAUSSIAN_FWHM_PER_SIGMA * resolution)) ** 2 + sigma_v_kms**2)
    sub = np.linspace(edges[:-1], edges[1:], n_sub)
    values = convolve_piecewise_linear(
        _WAVE_REST, _SED, sub.ravel(), _Z, np.full(sub.size, width / C_KM_S), quantity="nufnu"
    ).reshape(sub.shape)
    dimming = float(lnu_to_fnu(jnp.ones(()), _DL_CM, _Z))
    return dimming * simpson(values, x=sub, axis=0) / (edges[1:] - edges[:-1])


@pytest.mark.parametrize(
    ("wave_obs", "resolution", "sigma_v_kms"),
    [(_FINE, 2000.0, 0.0), (_FINE, 2000.0, 150.0), (_COARSE, 500.0, 150.0), (_FINE, 5000.0, 30.0)],
)
def test_conserving_exact_path_matches_the_pixel_mean_reference(wave_obs, resolution, sigma_v_kms):
    reference = _pixel_mean_reference(wave_obs, resolution, sigma_v_kms)
    out = project_spectrum(
        jnp.asarray(_SED),
        jnp.asarray(_WAVE_REST),
        jnp.asarray(wave_obs),
        _Z,
        _DL_CM,
        resolution=resolution,
        sigma_v_kms=sigma_v_kms,
        conserving=True,
    )
    # Measured: 1.6e-5, 2.3e-6, 3.2e-7 and 2.9e-5 of peak.
    assert _max_rel(out, reference) < 1e-4


def test_the_kernel_on_the_model_grid_is_measurably_different():
    """The fixture can see the defect: the model-grid route is off by more than 1e-3."""
    reference = _pixel_mean_reference(_FINE, 2000.0, 150.0)
    old = compute_spectrum_conserving_lsf(
        jnp.asarray(_SED),
        jnp.asarray(_WAVE_REST),
        jnp.asarray(_FINE),
        _Z,
        _DL_CM,
        resolution=2000.0,
        sigma_v_kms=150.0,
    )
    # Measured: 2.8e-3 of peak.
    assert _max_rel(old, reference) > 1e-3


def test_conserving_gradient_in_sigma_v_matches_a_finite_difference():
    def loss(sigma_v):
        flux = project_spectrum(
            jnp.asarray(_SED),
            jnp.asarray(_WAVE_REST),
            jnp.asarray(_COARSE),
            _Z,
            _DL_CM,
            resolution=2000.0,
            sigma_v_kms=sigma_v,
            conserving=True,
            lattice_window=lattice_span(_COARSE, _Z, _Z),
        )
        return jnp.sum(flux**2) * 1.0e54

    sigma_v, step = 150.0, 1.0e-1
    grad = float(jax.grad(loss)(sigma_v))
    reference = float((loss(sigma_v + step) - loss(sigma_v - step)) / (2.0 * step))
    assert np.isfinite(grad), "a positive kernel width has a finite derivative"
    assert grad != 0.0, "150 km/s broadens every absorption line of the fixture"
    np.testing.assert_allclose(grad, reference, rtol=1e-5)
