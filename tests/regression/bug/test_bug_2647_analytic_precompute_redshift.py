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
from tengri.forward.precompute.templates import precompute_template_photometry
from tengri.utils.physics_constants import C_CGS

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


def _legacy_z0_phot(model, filt_wave, filt_trans, grids):
    """Photometry grid of the pre-change builders: closure on the rest grid, ``redshift=0.0``."""
    wave_rest = adapter._continuum_wave_rest()
    names = adapter.AXIS_PARAMS[model]
    axes = tuple(grids[_GRID_KEYWORD[n]] for n in names)
    templates = np.empty((*(a.size for a in axes), wave_rest.size))
    for idx in np.ndindex(*(a.size for a in axes)):
        kwargs = {n: float(ax[i]) for n, ax, i in zip(names, axes, idx, strict=True)}
        templates[idx] = np.asarray(M[model](jnp.asarray(wave_rest), 1.0, **kwargs, redshift=0.0))
    return precompute_template_photometry(
        templates=templates,
        wave_rest=wave_rest,
        filter_waves=[filt_wave],
        filter_trans=[filt_trans],
        axes=axes,
        redshift=0.0,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    ).phot


@pytest.mark.parametrize("model", _CONTINUUM_MODELS)
def test_z0_grid_is_bit_identical_to_the_unshifted_integration(model):
    """At z = 0 the redshift passed to the integration is 0.0, as before: the grids are identical.

    Protects the z = 0 cells of #2642 (the builders' integration range) from this change.
    """
    names = adapter.AXIS_PARAMS[model]
    grids = {_GRID_KEYWORD[n]: _nodes(_CENTRAL[n], 0.01) for n in names}
    filt_wave, filt_trans = _tophat(160.0, 500.0)
    result = adapter.precompute([filt_wave], [filt_trans], 0.0, None, model=model, **grids)
    legacy = _legacy_z0_phot(model, filt_wave, filt_trans, grids)
    assert np.array_equal(np.asarray(result["grid_phot"]), np.asarray(legacy))


def test_z0_pah_drude_grid_is_bit_identical_to_the_unshifted_integration():
    """``pah_drude`` at z = 0 equals the integration with ``redshift=0.0`` bit for bit."""
    filt_wave, filt_trans = _tophat(6.7, 9.3)
    result = adapter.precompute([filt_wave], [filt_trans], 0.0, None, model="pah_drude")
    wave_rest = np.logspace(2, 5.5, 1000, dtype=np.float64)
    from tengri.components.dust.drude_profiles import compute_pah_template

    pah = np.asarray(compute_pah_template(jnp.asarray(wave_rest * 1e-4)))
    lnu = pah * (wave_rest * 1e-8) ** 2 / C_CGS
    legacy = precompute_template_photometry(
        templates=np.array([lnu]),
        wave_rest=wave_rest,
        filter_waves=[filt_wave],
        filter_trans=[filt_trans],
        axes=(),
        redshift=0.0,
        dl_cm=1.0,
        energy_normalize=False,
        units="lnu",
    ).phot
    assert np.array_equal(np.asarray(result["grid_phot"]), np.asarray(legacy))


# ── Mid-IR band, cold dust ────────────────────────────────────────────────────────────────────
# At 15 K the 8-24 um band sits deep in the Wien tail, where two properties of the lookup show.
#  (1) The triweight kernel averages over the temperature nodes: at 1 % spacing the value is
#      high by about 3 % (measured 1.0292 modified_blackbody, 1.0305 graybody), at 0.2 % by
#      0.9-1 %.
#  (2) The builders integrate on a 1500-point log-spaced rest grid and interpolate linearly, which
#      overestimates a convex Wien tail by about 0.9-1.0 % whatever the node spacing: evaluating
#      the closure on that grid alone gives 1.0089 and 1.0097. At 0.2 % spacing the lookup sits on
#      that floor (1.0090, 1.0097); only the 1 % excess is lookup smoothing.
_COLD_T = 15.0
_COLD_BAND_UM = (8.0, 24.0)
_COLD_MEASURED_1PCT = {"modified_blackbody": 1.0292, "graybody": 1.0305}


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_cold_dust_mid_ir_band_is_smoothed_by_one_percent_nodes(model):
    """At 15 K, 8-24 um, 1 % node spacing: lookup/exact is the smoothing, not agreement.

    The ~1.03 is a measured limit of the lookup (triweight smoothing over the T nodes), recorded
    so a change is noticed and tracked in #2676; it is not intended behavior, and the cell is to
    be tightened when that issue closes.
    """
    ratio = _ratio(model, 0.0, _COLD_BAND_UM, spacing=0.01, temperature=_COLD_T)
    assert ratio == pytest.approx(_COLD_MEASURED_1PCT[model], abs=0.01)
    assert ratio > 1.02


@pytest.mark.parametrize("model", ["modified_blackbody", "graybody"])
def test_cold_dust_mid_ir_band_at_fine_node_spacing(model):
    """With 0.2 % nodes the lookup is within 1e-2 of exact, on the rest-grid floor.

    The ~1.009 is a measured limit of the lookup (the 1500-point rest grid interpolated linearly
    across the Wien tail), recorded so a change is noticed and tracked in #2676; it is not intended
    behavior, and the cell is to be tightened when that issue closes.
    """
    ratio = _ratio(model, 0.0, _COLD_BAND_UM, spacing=0.002, temperature=_COLD_T)
    assert ratio == pytest.approx(1.0, abs=1e-2)
    central = {**_CENTRAL, "dust_T": _COLD_T}
    floor = _exact(model, 0.0, _COLD_BAND_UM, central, wave_rest=adapter._continuum_wave_rest())
    on_grid = _lookup_value(model, 0.0, _COLD_BAND_UM, 0.002, _COLD_T) / floor
    assert on_grid == pytest.approx(1.0, abs=1e-3), (
        f"{model}: lookup / closure on the builder grid = {on_grid:.5f}"
    )
