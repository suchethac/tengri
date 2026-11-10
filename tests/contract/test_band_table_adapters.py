# SPDX-License-Identifier: BSD-3-Clause
"""Band-table adapters tested through their own ``band_table`` alone, without an SEDModel.

Each test hands an adapter a hand-built :class:`BandTableContext`: the spec, the rest-frame
grid, the WavePrecomp setting and the component chain. The chain holds stub components built
with ``object.__new__``, so no model, SSP grid or observation is constructed.
"""

from __future__ import annotations

import types

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    DEFAULT,
    Fixed,
    Observation,
    Parameters,
    Photometry,
    SEDModel,
    Uniform,
    WavePrecomp,
)
from tengri.components.nebular import cloudy_precompute, mappings_shock_precompute
from tengri.components.nebular._shared import render_nebular_lines
from tengri.components.nebular.cloudy_grid import CloudyGridBackend
from tengri.components.nebular.component import NebularSEDComponent
from tengri.components.nebular.shock_model import ShockNebular
from tengri.forward.precompute.band_table import BandTable, BandTableContext, Decline
from tengri.observation.photometry import lnu_filter_integral_batch
from tests.contract import _band_table_cases as cases

pytestmark = pytest.mark.contract

_WAVE = np.linspace(500.0, 25000.0, 6000)
_CENTERS = (3500.0, 4800.0, 6200.0, 7600.0, 9000.0)
_REDSHIFT = 0.1


def _padded_filters() -> tuple[np.ndarray, np.ndarray]:
    """Five top-hat bands on the shape of ``synthetic_tophat_obs``, zero-padded to one length."""
    curves = []
    for center in _CENTERS:
        wave = np.linspace(center * 0.84, center * 1.16, 40)
        trans = np.sin(np.linspace(0.0, np.pi, 40)) * 0.6
        curves.append((wave, trans))
    n_max = max(len(w) for w, _ in curves)
    fw = np.empty((len(curves), n_max))
    ft = np.zeros((len(curves), n_max))
    for i, (wave, trans) in enumerate(curves):
        fw[i, : len(wave)] = wave
        fw[i, len(wave) :] = wave[-1]
        ft[i, : len(trans)] = trans
    return fw, ft


def _cloudy_spec(**overrides) -> Parameters:
    """A CLOUDY-nebular spec with a fixed redshift and line width, the table's engaged case."""
    kwargs = {
        "mean_sfh_type": "dpl",
        "nebular": "cloudy",
        # The gates read the spec only and the backend is a stub, so the grid file is never
        # opened; this module stands in for it because the untracked grid is absent here.
        "cloudy_grid_path": __file__,
        "redshift": Fixed(_REDSHIFT),
        "neb_logU": Uniform(-3.0, -1.0),
        "neb_eline_sigma_kms": Fixed(100.0),
    }
    kwargs.update(overrides)
    return Parameters(**kwargs)


def _context(spec: Parameters, *, chain=(), wave_precomp=True) -> BandTableContext:
    """A context with the fields the two adapters read; the rest are neutral (see step 0)."""
    return BandTableContext(
        spec=spec,
        wave=_WAVE,
        filters=None,
        padded_filters=None,
        z_nodes=None,
        wave_precomp=wave_precomp,
        chain=list(chain),
        dust_ir_diffuse_screen=False,
        atten_free_ok=frozenset(),
        agn_axis_grids=None,
        ask=lambda name, slot: None,
    )


def _cloudy_backend(grid_interp: str = "pchip") -> CloudyGridBackend:
    backend = object.__new__(CloudyGridBackend)
    object.__setattr__(backend, "_grid_interp", grid_interp)
    return backend


def _nebular(backend, state=None) -> NebularSEDComponent:
    comp = object.__new__(NebularSEDComponent)
    object.__setattr__(comp, "config", types.SimpleNamespace(backend="cloudy_grid"))
    object.__setattr__(comp, "backend", backend)
    object.__setattr__(comp, "_state", state)
    return comp


def _cloudy_decline(ctx: BandTableContext) -> str:
    result = cloudy_precompute.band_table(ctx, component="nebular", slot="band_table")
    assert isinstance(result, Decline)
    assert result.reason is not None
    return result.reason


def test_cloudy_declines_without_wave_precomp():
    ctx = _context(_cloudy_spec(), chain=[_nebular(_cloudy_backend())], wave_precomp=None)
    assert _cloudy_decline(ctx) == "approx=WavePrecomp() not enabled"


def test_cloudy_declines_without_a_cloudy_nebular_component():
    assert _cloudy_decline(_context(_cloudy_spec(), chain=[])) == (
        "the nebular backend is not the CLOUDY grid"
    )


def test_cloudy_declines_on_a_non_linear_grid_interpolation():
    ctx = _context(_cloudy_spec(), chain=[_nebular(_cloudy_backend("triweight"))])
    assert _cloudy_decline(ctx) == (
        "the CLOUDY backend interpolates with grid_interp='triweight'; "
        "the band table reproduces only 'pchip' and 'linear'"
    )


def test_cloudy_declines_while_the_diffuse_gas_may_be_active():
    ctx = _context(_cloudy_spec(neb_dig_frac=Fixed(0.1)), chain=[_nebular(_cloudy_backend())])
    assert _cloudy_decline(ctx) == (
        "the diffuse ionized gas may be active (neb_dig_frac is not pinned to zero)"
    )


def test_cloudy_declines_on_a_free_redshift():
    ctx = _context(_cloudy_spec(redshift=Uniform(0.0, 1.0)), chain=[_nebular(_cloudy_backend())])
    assert _cloudy_decline(ctx) == ("redshift is free (the band table is built at one redshift)")


def test_cloudy_declines_on_a_free_line_width():
    ctx = _context(
        _cloudy_spec(neb_eline_sigma_kms=Uniform(50.0, 150.0)),
        chain=[_nebular(_cloudy_backend())],
    )
    assert _cloudy_decline(ctx) == (
        "neb_eline_sigma_kms is free (line band coefficients are fixed at one width)"
    )


def test_cloudy_declines_on_a_free_nebular_parameter_outside_its_axes():
    ctx = _context(
        _cloudy_spec(neb_dig_delta_logU=Uniform(-3.0, -1.0)),
        chain=[_nebular(_cloudy_backend())],
    )
    assert _cloudy_decline(ctx) == (
        "free nebular parameters outside the table's axes: ['neb_dig_delta_logU']"
    )


def test_cloudy_declines_when_the_filter_cache_is_absent():
    ctx = _context(_cloudy_spec(), chain=[_nebular(_cloudy_backend(), state=None)])
    assert _cloudy_decline(ctx) == ("the nebular filter cache is absent (no WavePrecomp filters)")


def _shock_context(fw: np.ndarray, ft: np.ndarray) -> BandTableContext:
    """An engaged shock context: a fixed redshift, a shock component and a photometry cache."""
    stellar = types.SimpleNamespace(
        name="stellar",
        _state=types.SimpleNamespace(phot_fw_padded=fw, phot_ft_padded=ft),
    )
    shock = object.__new__(ShockNebular)
    spec = Parameters(mean_sfh_type="dpl", redshift=Fixed(_REDSHIFT))
    return _context(spec, chain=[stellar, shock])


def test_shock_coefficients_equal_the_integral_of_each_delta_line():
    """Each row of the table is the band integral of one delta line, at z and at rest.

    The delta placement is ``render_nebular_lines`` with zero width, the call the per-call
    shock path makes. The table is built with one ``vmap`` over the lines and the reference
    one line at a time; the two differ in the last bit (measured max relative difference
    1.2e-16 over the 24 lines), so the bound is rtol = 1e-12 with atol = 0.
    """
    fw, ft = _padded_filters()
    result = mappings_shock_precompute.band_table(
        _shock_context(fw, ft), component="shock", slot="band_table"
    )
    assert isinstance(result, BandTable)
    coeff = np.asarray(result.arrays)

    grid = mappings_shock_precompute._load_mappings_grids()["mappings5"]
    line_waves = jnp.asarray(grid["line_wavelengths_aa"])
    n_lines = line_waves.shape[0]
    wave = jnp.asarray(_WAVE)
    fw_j, ft_j = jnp.asarray(fw), jnp.asarray(ft)
    assert coeff.shape == (n_lines, 2, len(_CENTERS))
    for i in range(n_lines):
        lum = jnp.eye(n_lines)[i]
        sed = render_nebular_lines(line_waves, lum, wave, 0.0, 0.0)
        expected = jnp.stack(
            [
                lnu_filter_integral_batch(sed, wave, fw_j, ft_j, _REDSHIFT),
                lnu_filter_integral_batch(sed, wave, fw_j, ft_j, 0.0),
            ]
        )
        np.testing.assert_allclose(coeff[i], np.asarray(expected), rtol=1e-12, atol=0)


def _dust_context(model, z_nodes=None) -> BandTableContext:
    """A context around one real dust emitter and the real stellar padded filters.

    The adapter reads only this context: the model's own memo is not consulted, and ``ask``
    answers nothing.
    """
    chain = model._build_component_chain()
    stellar = next(c for c in chain if getattr(c, "name", "") == "stellar")
    state = stellar._state
    return BandTableContext(
        spec=model.spec,
        wave=model._rest_wavelength,
        filters=(model.filter_waves, model.filter_trans),
        padded_filters=(state.phot_fw_padded, state.phot_ft_padded),
        z_nodes=z_nodes,
        wave_precomp=WavePrecomp(),
        chain=chain,
        dust_ir_diffuse_screen=False,
        atten_free_ok=SEDModel._BAND_RESPONSE_ATTEN_FREE_OK,
        agn_axis_grids=None,
        ask=lambda name, slot: None,
    )


def _dust_emitter(ctx: BandTableContext):
    return next(c for c in ctx.chain if getattr(c, "name", "") == "dust_emission")


def test_dust_band_table_equals_the_filter_integral_at_each_node_bit_for_bit():
    """Each R[z] row is the filter integral of the emitter's unit-luminosity SED at that z."""
    from tengri.components.dust import dust_emission_precompute as adapter

    ctx = _dust_context(cases.build_dust_fixed_z(), z_nodes=jnp.log1p(jnp.array([0.0, 0.5, 1.0])))
    entry = adapter.band_table(ctx, component="dust_emission", slot="emission_band_response")
    assert isinstance(entry, BandTable)
    values = np.asarray(entry.arrays["values"])

    emitter = _dust_emitter(ctx)
    fixed = {k: jnp.asarray(v) for k, v in ctx.spec.get_fixed_values().items()}
    fw, ft = ctx.padded_filters
    zeros = jnp.zeros_like(ctx.wave)
    takes_log_budget = "log_L_ir" in adapter._declared_input_names(emitter)

    def unit_response(z):
        budget = {"L_ir": 1.0}
        if takes_log_budget:
            budget["log_L_ir"] = jnp.log10(1.0)
        p = emitter.slice_params({**fixed, "redshift": z})
        sed, _ = emitter.predict(p, zeros, ctx.wave, **budget)
        return lnu_filter_integral_batch(sed, ctx.wave, fw, ft, z)

    assert values.shape == (3, fw.shape[0])
    for row, z in enumerate(jnp.expm1(ctx.z_nodes)):
        expected = np.asarray(jax.jit(unit_response)(z))
        assert np.array_equal(values[row], expected), f"R[z] row {row} differs from the integral"


def test_bosa_dust_band_table_is_declined_with_the_homogeneity_reason():
    """BOSA's template depends on L_ir, so no constant per-filter response is built."""
    from tengri.components.dust import dust_emission_precompute as adapter

    model = SEDModel.build(
        ssp_data=cases._bare_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(cases.FILTERS))),
        approx=WavePrecomp(),
        redshift=Fixed(0.1),
        sfh=cases._dpl_sfh(),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "bosa", "log_L_ir": Fixed(11.0), "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
    )
    entry = adapter.band_table(
        _dust_context(model), component="dust_emission", slot="emission_band_response"
    )
    assert isinstance(entry, Decline)
    assert entry.reason == (
        "dust emission template is not homogeneous in L_ir: its shape depends on luminosity, "
        "so a constant per-filter response cannot represent it"
    )
