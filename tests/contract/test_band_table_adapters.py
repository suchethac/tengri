# SPDX-License-Identifier: BSD-3-Clause
"""Band-table adapters tested through their own ``band_table`` alone, without an SEDModel.

Most tests hand an adapter a hand-built :class:`BandTableContext`: the spec, the rest-frame
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


class _AGNSpec:
    """The spec surface the AGN gate reads: free names, fixed values and prior bounds."""

    def __init__(self, free, fixed, bounds):
        self.free_params = list(free)
        self._fixed = dict(fixed)
        self._bounds = dict(bounds)

    def get_fixed_values(self):
        return dict(self._fixed)

    def get_distribution(self, name):
        return types.SimpleNamespace(bounds=self._bounds[name])


_AGN_RECIPE = {
    "norm": "cigale_joint",
    "polar_law": "smc",
    "disc": "powerlaw",
    "nlr": "none",
    "blr": "none",
    "feii": "none",
    "torus": "simple",
    "attenuation": "none",
}


def _agn_stub(model: str, recipe: dict):
    """A composable AGN component stub: the gate reads only its config and its resolved recipe."""
    from tengri.components.agn.component import AGNSEDComponent

    agn = object.__new__(AGNSEDComponent)
    object.__setattr__(agn, "config", types.SimpleNamespace(model=model))
    object.__setattr__(agn, "resolved_recipe", lambda: recipe)
    return agn


def _agn_context(
    *,
    model="composable",
    recipe=None,
    free=("agn_log_lbol",),
    fixed=None,
    bounds=None,
    wave_precomp=True,
    padded=True,
    z_nodes=None,
    rank1=None,
) -> BandTableContext:
    """A stub context whose gates all pass, except the one a test changes."""
    if bounds is None:
        bounds = {name: (8.0, 14.0) for name in free}
    if fixed is None:
        fixed = {"agn_ir_frac": 0.0}
    return BandTableContext(
        spec=_AGNSpec(free, fixed, bounds),
        wave=_WAVE,
        filters=None,
        padded_filters=(np.ones((2, 3)), np.ones((2, 3))) if padded else (None, None),
        z_nodes=z_nodes,
        wave_precomp=WavePrecomp() if wave_precomp else None,
        chain=[_agn_stub(model, recipe or _AGN_RECIPE)],
        dust_ir_diffuse_screen=False,
        atten_free_ok=frozenset(),
        agn_axis_grids=None,
        ask=lambda name, slot: rank1 if (name, slot) == ("agn", "term_band_response") else None,
    )


def _agn_decline(ctx: BandTableContext) -> str:
    from tengri.components.agn.blocks import composable_precompute

    result = composable_precompute.band_table(ctx, component="agn", slot="band_table")
    assert isinstance(result, Decline)
    assert result.reason is not None
    return result.reason


def test_agn_declines_a_self_contained_model_as_not_composable():
    assert _agn_decline(_agn_context(model="grahsp")) == "the AGN is not a composable recipe"


def test_agn_declines_without_wave_precomp():
    assert _agn_decline(_agn_context(wave_precomp=False)) == "WavePrecomp is off"


def test_agn_declines_without_padded_filter_curves():
    assert _agn_decline(_agn_context(padded=False)) == "no padded filter curves"


def test_agn_declines_on_a_free_redshift():
    ctx = _agn_context(z_nodes=np.array([0.0, 1.0]))
    assert _agn_decline(ctx) == (
        "the registry adapter is built at one redshift and the redshift is not Fixed"
    )


def test_agn_declines_a_normalization_other_than_cigale_joint():
    recipe = {**_AGN_RECIPE, "norm": "independent"}
    assert _agn_decline(_agn_context(recipe=recipe)) == (
        "the table is evaluated under agn_norm='cigale_joint' and polar_law='smc' "
        "(this AGN resolves to norm='independent', polar_law='smc')"
    )


def test_agn_declines_a_torus_fraction_that_is_not_fixed_at_zero():
    assert _agn_decline(_agn_context(fixed={"agn_ir_frac": 0.1})) == (
        "agn_ir_frac is not Fixed at 0, so the torus fraction follows the stellar dust"
    )


def test_agn_declines_more_free_axes_than_the_node_table_takes():
    free = ("agn_log_lbol", "agn_p1", "agn_p2", "agn_p3", "agn_p4")
    assert _agn_decline(_agn_context(free=free)) == ("5 free AGN parameters exceed the 4-axis cap")


def test_agn_declines_a_prior_without_finite_bounds():
    ctx = _agn_context(bounds={"agn_log_lbol": (-np.inf, np.inf)})
    assert _agn_decline(ctx) == (
        "the priors of ['agn_log_lbol'] have no finite bounds to span with nodes"
    )


def test_agn_declines_when_the_rank_one_term_response_serves_it_exactly():
    rank1 = BandTable({"values": np.zeros((1, 2))})
    assert _agn_decline(_agn_context(rank1=rank1)) == (
        "the rank-1 term band response serves this AGN exactly"
    )


def test_agn_table_is_accurate_and_matches_the_direct_band_flux_at_a_node(monkeypatch):
    """The table passes its cell-center check, and at a node it is the component's own flux."""
    from tengri.components.agn.blocks import composable_precompute as adapter
    from tengri.components.agn.component import AGNSEDComponent

    model = cases.build_composable_agn()
    chain = model._build_component_chain()
    state = next(c for c in chain if getattr(c, "name", "") == "stellar")._state
    ctx = BandTableContext(
        spec=model.spec,
        wave=model._rest_wavelength,
        filters=(model.filter_waves, model.filter_trans),
        padded_filters=(state.phot_fw_padded, state.phot_ft_padded),
        z_nodes=None,
        wave_precomp=WavePrecomp(),
        chain=chain,
        dust_ir_diffuse_screen=False,
        atten_free_ok=SEDModel._BAND_RESPONSE_ATTEN_FREE_OK,
        agn_axis_grids=None,
        ask=lambda name, slot: None,
    )
    reports = []
    worst_of = adapter._agn_table_cell_center_error

    def spy(*args):
        worst = worst_of(*args)
        reports.append(worst)
        return worst

    monkeypatch.setattr(adapter, "_agn_table_cell_center_error", spy)
    result = adapter.band_table(ctx, component="agn", slot="band_table")
    assert isinstance(result, BandTable)
    assert len(reports) == 1
    assert reports[0] < adapter._AGN_TABLE_RTOL

    table = result.arrays
    grid = np.asarray(table["axes"]["agn_log_lbol"])
    params = {"agn_log_lbol": float(grid[len(grid) // 2])}
    full = {**model.spec.get_fixed_values(), **params}
    table_row = np.asarray(AGNSEDComponent._band_table_phot(full, table)[0])
    model._band_tables["agn"]["band_table"] = Decline("test")
    direct = np.asarray(model.predict_state(params).derived["agn_phot_lnu_precomp"])
    np.testing.assert_allclose(table_row, direct, rtol=1e-6)


# -- the generic term path: radio and X-ray (#3290, step 5) -------------------


@pytest.fixture(scope="module")
def _free_z_radio_xray():
    """A free-redshift model whose radio and X-ray emitters have every parameter fixed.

    The free X-ray slope of ``_band_table_cases.build_free_z_radio_xray`` makes its term
    response decline by design, so the rows are checked on this fixed-shape variant.
    """
    return SEDModel.build(
        ssp_data=cases._bare_ssp(),
        observation=Observation(photometry=Photometry.from_names(list(cases.FILTERS))),
        approx=WavePrecomp(),
        redshift=Uniform(0.05, 0.3),
        sfh=cases._dpl_sfh(),
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": "modified_blackbody", "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        xray={"type": "simple", "all_params": Fixed(DEFAULT)},
    )


@pytest.mark.parametrize("name", ["radio", "xray"])
def test_term_band_response_row_is_the_dense_filter_integral_of_its_shape(
    _free_z_radio_xray, name
):
    """A term's row at one redshift node is the dense filter integral of that term, bit for bit.

    The adapter builds the table through the generic path; this recomputes one live term's
    integral with ``lnu_filter_integral_batch`` at the same node and compares every filter.
    """
    from tengri.forward.precompute.band_table import arrays_of

    model = _free_z_radio_xray
    table = arrays_of(model.band_table(name, "term_band_response"))
    assert table is not None, f"the {name} term band response is not engaged in the fixture"

    chain = model._cached_component_chain
    comp = next(c for c in chain if getattr(c, "name", "") == name)
    ctx = model._band_table_context(chain)
    fw_pad, ft_pad = ctx.padded_filters
    fixed = {k: jnp.asarray(v) for k, v in ctx.spec.get_fixed_values().items()}

    node = table["ln1pz"].shape[0] // 2
    z = jnp.expm1(table["ln1pz"][node])
    terms = comp.emission_terms({**fixed, "redshift": z}, ctx.wave, **comp.EMITTER_PROBE_INPUTS[0])
    live = [k for k, shape in terms.items() if float(jnp.max(jnp.abs(shape))) > 0.0]
    assert live, f"the {name} fixture has no live term at the node"
    nonzero = 0
    for key in live:
        dense = np.asarray(lnu_filter_integral_batch(terms[key], ctx.wave, fw_pad, ft_pad, z))
        row = np.asarray(table["R"][node, list(terms).index(key)])
        assert np.array_equal(row, dense), (name, key, np.max(np.abs(row - dense)))
        nonzero += int(np.any(dense != 0.0))
    if name == "radio":
        assert nonzero, "the radio terms integrate to zero against the fixture's filters"


def test_every_component_with_emission_terms_or_precompute_key_has_a_band_table_provider():
    """Starts #2981: each emitter and each dust-emission template has a band-table owner.

    An additive emitter (one declaring ``emission_terms``) is owned by its own name. The
    dust-emission templates (one declaring ``precompute_key``) are built by the
    ``dust_emission`` provider. A new component that declares either attribute with no owner
    in ``BAND_TABLE_PROVIDERS`` fails here rather than falling back to the dense integral.
    """
    import importlib
    import inspect
    import pkgutil

    import tengri.components as components
    from tengri.forward.precompute.registry import BAND_TABLE_PROVIDERS

    owners: dict[str, str | None] = {}
    for info in pkgutil.walk_packages(components.__path__, components.__name__ + "."):
        module = importlib.import_module(info.name)
        for cls in vars(module).values():
            if not (inspect.isclass(cls) and cls.__module__ == module.__name__):
                continue
            if hasattr(cls, "emission_terms"):
                owners[f"{cls.__module__}.{cls.__qualname__}"] = getattr(cls, "name", None)
            elif hasattr(cls, "precompute_key"):
                owners[f"{cls.__module__}.{cls.__qualname__}"] = "dust_emission"

    assert any(v == "radio" for v in owners.values()), "the census found no radio emitter"
    missing = sorted(
        f"{qualname} -> {owner!r}"
        for qualname, owner in owners.items()
        if owner not in BAND_TABLE_PROVIDERS
    )
    assert missing == [], missing
