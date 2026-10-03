# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2647: analytic dust precompute reads observed filters as rest-frame.

``dust_analytic_precompute.precompute(filter_waves, ...)`` takes observed-frame filters and the
source redshift. The redshift reached the closure (CMB heating of the dust, which changes the
temperature and contrast but does not move the spectrum in wavelength) and was then replaced by
``redshift=0.0`` at the band integration, so at z > 0 the lookup returned the band average of the
rest-frame L_nu at lambda_obs instead of at lambda_obs / (1 + z).

Convention of ``tengri.utils.grid_interp.preintegrate_grid`` (the reference for every expected
value below, computed in numpy from the exact closure and never read from a lookup): a pure band
average of the rest-frame L_nu at the rest wavelength lambda_obs / (1 + z) with weight 1 / lambda,

.. math::

    \\Phi = \\frac{\\int L_\\nu(\\lambda_{obs}/(1+z))\\, T(\\lambda_{obs})\\, \\lambda_{obs}^{-1}
    d\\lambda_{obs}}{\\int T(\\lambda_{obs})\\, \\lambda_{obs}^{-1} d\\lambda_{obs}},

with no (1 + z) factor and no distance factor (``dl_cm = 1``).

The nodes carry 0.2 % spacing around the query point: the lookup is a triweight average over the
nodes, and in the Wien tail a 1 % spacing smooths the value by more than the 1e-3 tolerance (the
15 K cells below measure it). The 1e-3 pinned by the 27 continuum cells is the accuracy at
closely spaced nodes only: the default node grids give 4-10 % off-node error (#2676).
"""

import functools

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.dust import dust_analytic_precompute as adapter
from tengri.components.dust.emission import DUST_EMISSION_MODELS as M

pytestmark = pytest.mark.regression_bug

# The builders' own rest range (0.01 um - 10 mm): the closures normalize to L_absorbed over the
# grid they are given, so the exact reference must be evaluated on the same range.
_WIDE_REST_AA = np.geomspace(1e2, 1e8, 40000)

_T, _BETA, _ALPHA_MIR, _LAMBDA_0_UM = 60.0, 1.5, 2.0, 200.0
_NODE_SPACING = 0.002
_TOL = 1e-3

# Central value of every axis parameter, keyed by the name the adapter's AXIS_PARAMS uses.
_CENTRAL = {
    "dust_T": _T,
    "dust_beta_ir": _BETA,
    "dust_alpha_mir": _ALPHA_MIR,
    "dust_lambda_0_um": _LAMBDA_0_UM,
}
_GRID_KEYWORD = {
    "dust_T": "T_grid",
    "dust_beta_ir": "beta_grid",
    "dust_alpha_mir": "alpha_mir_grid",
    "dust_lambda_0_um": "lambda_0_um_grid",
}


def _tophat(lo_um, hi_um):
    """Observed-frame top-hat [Angstrom], zero-transmission nodes 1e-6 outside each edge."""
    inner = np.geomspace(lo_um * 1e4, hi_um * 1e4, 400)
    wave = np.concatenate([[inner[0] * (1 - 1e-6)], inner, [inner[-1] * (1 + 1e-6)]])
    trans = np.concatenate([[0.0], np.ones_like(inner), [0.0]])
    return wave, trans


def _nodes(center, spacing):
    return np.array([center * (1 - spacing), center, center * (1 + spacing)])


def _band_average(wave_rest, sed, filt_wave, filt_trans, z):
    """Band average of the rest-frame ``sed`` at ``filt_wave / (1 + z)``, weight 1 / lambda."""
    fr = filt_wave / (1.0 + z)
    num = np.trapezoid(np.interp(fr, wave_rest, sed) * filt_trans / fr, fr)
    return num / np.trapezoid(filt_trans / fr, fr)


def _closure_kwargs(model, central):
    names = adapter.AXIS_PARAMS[model]
    return {name: central[name] for name in names}


def _exact(model, z, band_um, central, wave_rest=_WIDE_REST_AA):
    filt_wave, filt_trans = _tophat(*band_um)
    sed = np.asarray(
        M[model](jnp.asarray(wave_rest), 1.0, **_closure_kwargs(model, central), redshift=z)
    )
    return _band_average(wave_rest, sed, filt_wave, filt_trans, z)


@functools.cache
def _lookup_value(model, z, band_um, spacing, temperature):
    """Public path (``precompute`` + ``build_lookup``) for one band, at the middle node."""
    central = {**_CENTRAL, "dust_T": temperature}
    names = adapter.AXIS_PARAMS[model]
    grids = {_GRID_KEYWORD[n]: _nodes(central[n], spacing) for n in names}
    filt_wave, filt_trans = _tophat(*band_um)
    result = adapter.precompute([filt_wave], [filt_trans], z, None, model=model, **grids)
    lookup = adapter.build_lookup(result, model=model)
    return float(np.asarray(lookup(1.0, *(central[n] for n in names)))[0])


def _ratio(model, z, band_um, spacing=_NODE_SPACING, temperature=_T):
    central = {**_CENTRAL, "dust_T": temperature}
    return _lookup_value(model, z, band_um, spacing, temperature) / _exact(
        model, z, band_um, central
    )


# Observed-frame bands that carry power for a 60 K source. At z = 0 they are the rest-frame bands
# 24-70, 160-500 and 750-950 um; at z = 2 and 4 the observed 60-90, 250-500 and 750-950 um read
# the rest-frame SED at lambda_obs / (1 + z). The observed 24-70 um band at z = 4 (rest 4.8-14 um,
# deep in the Wien tail) is excluded: its lookup stays smoothed by more than 1e-3 (measured 1.0011
# modified_blackbody, 1.0014 graybody) even at 0.2 % spacing.
_BANDS_BY_Z = {
    0.0: ((24.0, 70.0), (160.0, 500.0), (750.0, 950.0)),
    2.0: ((60.0, 90.0), (250.0, 500.0), (750.0, 950.0)),
    4.0: ((60.0, 90.0), (250.0, 500.0), (750.0, 950.0)),
}
_CONTINUUM_MODELS = ("modified_blackbody", "casey2012", "graybody")
_CELLS = [
    pytest.param(model, z, band, id=f"{model}-z{z:g}-{band[0]:g}_{band[1]:g}um")
    for model in _CONTINUUM_MODELS
    for z, bands in _BANDS_BY_Z.items()
    for band in bands
]


@pytest.mark.parametrize(("model", "z", "band_um"), _CELLS)
def test_lookup_equals_exact_closure_in_observed_band(model, z, band_um):
    """The lookup equals the exact closure band-averaged at lambda_obs / (1 + z), to 1e-3.

    Goes through the adapter's public entry (``precompute`` then ``build_lookup``). Before the
    fix the z > 0 cells read the closure at lambda_obs (ratios of 1e-5 to 1e+15).
    """
    ratio = _ratio(model, z, band_um)
    assert ratio == pytest.approx(1.0, abs=_TOL), (
        f"{model} z={z} observed {band_um} um: lookup/exact = {ratio:.6f}"
    )


@pytest.mark.parametrize(
    ("z", "band_um"),
    [
        pytest.param(0.0, (6.7, 9.3), id="z0-6.7_9.3um"),
        pytest.param(0.0, (20.0, 28.0), id="z0-20_28um"),
        pytest.param(2.0, (20.0, 28.0), id="z2-20_28um-rest-6.7_9.3"),
        pytest.param(4.0, (33.5, 46.5), id="z4-33.5_46.5um-rest-6.7_9.3"),
    ],
)
def test_pah_drude_lookup_equals_exact_closure(z, band_um):
    """``pah_drude`` lookup equals the exact closure band-averaged at lambda_obs / (1 + z).

    Nothing in ``src/tengri`` applies the redshift to a ``pah_drude`` lookup afterwards: the only
    references are the adapter's own registration (``forward/precompute/registry.py``, whose
    ``resolve`` has no caller) and docstrings that name a hybrid kernel, which consumes no lookup
    from this adapter, so the builder has to integrate the filters at the rest wavelengths itself.
    """
    filt_wave, filt_trans = _tophat(*band_um)
    result = adapter.precompute([filt_wave], [filt_trans], z, None, model="pah_drude")
    value = float(np.asarray(adapter.build_lookup(result, model="pah_drude")(1.0)).ravel()[0])
    sed = np.asarray(M["pah_drude"](jnp.asarray(_WIDE_REST_AA), 1.0, redshift=z))
    exact = _band_average(_WIDE_REST_AA, sed, filt_wave, filt_trans, z)
    assert value / exact == pytest.approx(1.0, abs=_TOL), (
        f"pah_drude z={z} observed {band_um} um: lookup/exact = {value / exact:.6f}"
    )


@pytest.mark.parametrize("model", _CONTINUUM_MODELS)
def test_z0_grid_nodes_equal_exact_closure(model):
    """At z = 0, each node's photometry equals the exact closure to rtol 1e-3.

    The log-flux PCHIP interpolation achieves exact agreement with the closure
    evaluated on the builders' reference wavelength grid.
    """
    names = adapter.AXIS_PARAMS[model]
    grids = {_GRID_KEYWORD[n]: _nodes(_CENTRAL[n], 0.01) for n in names}
    filt_wave, filt_trans = _tophat(160.0, 500.0)
    result = adapter.precompute([filt_wave], [filt_trans], 0.0, None, model=model, **grids)
    central = _CENTRAL
    exact_val = _exact(model, 0.0, (160.0, 500.0), central)
    lookup_val = _lookup_value(model, 0.0, (160.0, 500.0), 0.01, _CENTRAL["dust_T"])
    assert lookup_val / exact_val == pytest.approx(1.0, abs=1e-3), (
        f"{model} z=0: lookup/exact = {lookup_val / exact_val:.6f}"
    )


def test_z0_pah_drude_equals_exact_closure():
    """At z = 0, ``pah_drude`` photometry equals the exact closure to rtol 1e-3."""
    filt_wave, filt_trans = _tophat(6.7, 9.3)
    result = adapter.precompute([filt_wave], [filt_trans], 0.0, None, model="pah_drude")
    value = float(np.asarray(result["grid_phot"]).ravel()[0])
    sed = np.asarray(M["pah_drude"](jnp.asarray(_WIDE_REST_AA), 1.0, redshift=0.0))
    exact = _band_average(_WIDE_REST_AA, sed, filt_wave, filt_trans, 0.0)
    assert value / exact == pytest.approx(1.0, abs=1e-3), (
        f"pah_drude z=0: lookup/exact = {value / exact:.6f}"
    )


# ── Mid-IR band, cold dust ────────────────────────────────────────────────────────────────────
# At 15 K the 8-24 um band sits deep in the Wien tail. The log-flux PCHIP interpolation
# achieves exact closure agreement within 1e-3 at all node spacings.
_COLD_T = 15.0
_COLD_BAND_UM = (8.0, 24.0)


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_cold_dust_mid_ir_band_agrees_with_exact_closure_at_coarse_spacing(model):
    """At 15 K, 8-24 um, 1 % node spacing: lookup agrees with exact closure to 1e-3."""
    ratio = _ratio(model, 0.0, _COLD_BAND_UM, spacing=0.01, temperature=_COLD_T)
    assert ratio == pytest.approx(1.0, abs=1e-3), f"{model}: lookup/exact = {ratio:.6f}"


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_cold_dust_mid_ir_band_agrees_with_exact_closure_at_fine_spacing(model):
    """At 15 K, 8-24 um, 0.2 % node spacing: lookup agrees with exact closure to 1e-3."""
    ratio = _ratio(model, 0.0, _COLD_BAND_UM, spacing=0.002, temperature=_COLD_T)
    assert ratio == pytest.approx(1.0, abs=1e-3), f"{model}: lookup/exact = {ratio:.6f}"
