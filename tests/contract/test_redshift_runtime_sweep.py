# SPDX-License-Identifier: BSD-3-Clause
"""Runtime redshift class sweep (#3505).

Redshift is a runtime input of the forward model for every representation (exact
and tabulated) and every way of supplying it: a promoted fixed-z model, a free
redshift, a catalog range, a call-time override on a fixed-z model, and a vmapped
per-galaxy traced z. One compiled program must serve several redshifts, and its
output at each z must equal a model built fixed at that z.
"""

from __future__ import annotations

import warnings
from functools import cache, lru_cache

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

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
from tengri.inference.fitter import _model_at_override_redshift

SSP_PATH = "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"
ZS = (0.1, 0.3, 0.8)
Z_PLACEHOLDER = 0.1
Z_RANGE = (0.05, 1.0)
RTOL = {"exact": 1e-10, "wp": 3e-3}
REPRESENTATIONS = ("exact", "wp")
STRUCTURES = ("plain", "rich")
MODES = ("promoted", "free", "catalog", "override", "promoted_vmap")


@lru_cache(maxsize=1)
def _ssp():
    return load_ssp_data(SSP_PATH)


@lru_cache(maxsize=1)
def _obs():
    return Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"]))


def _approx(representation, catalog=False):
    if representation == "exact":
        return None
    if catalog:
        return WavePrecomp(catalog_z_range=Z_RANGE)
    return WavePrecomp()


def _build(representation, structure, redshift, *, catalog=False):
    kw = {}
    if structure == "rich":
        kw["dust_emission"] = {"type": "dale2014", "all_params": Fixed(DEFAULT)}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=_obs(),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=redshift,
            approx=_approx(representation, catalog=catalog),
            **kw,
        )


@cache
def _reference(representation, structure, z):
    """Photometry of a model built fixed at ``z`` (the ground truth for the sweep)."""
    model = _build(representation, structure, Fixed(z))
    return np.asarray(model.predict_photometry({}))


def _runtime_model(representation, structure, mode):
    if mode == "promoted":
        base = _build(representation, structure, Fixed(Z_PLACEHOLDER))
        return base.with_runtime_redshift(Z_RANGE if representation == "wp" else None)
    if mode == "promoted_vmap":
        return _runtime_model(representation, structure, "promoted")
    if mode == "free":
        return _build(representation, structure, Uniform(*Z_RANGE))
    if mode == "catalog":
        return _build(representation, structure, Fixed(Z_PLACEHOLDER), catalog=True)
    if mode == "override":
        # Exact only: a tabulated override needs a table spanning its own z, so it
        # gets one table per override value (see test_tabulated_override_spans_its_z).
        base = _build(representation, structure, Fixed(Z_PLACEHOLDER))
        models = [_model_at_override_redshift(base, {"redshift": z}) for z in ZS]
        assert all(m is models[0] for m in models), (
            "one promotion must serve every override redshift; got a distinct model per z"
        )
        return models[0]
    raise ValueError(mode)


def _cases():
    for representation in REPRESENTATIONS:
        for structure in STRUCTURES:
            for mode in MODES:
                if mode == "catalog" and representation == "exact":
                    continue
                if mode == "override" and representation == "wp":
                    continue
                if mode == "promoted_vmap":
                    continue
                yield representation, structure, mode
    yield "wp", "plain", "promoted_vmap"
    yield "exact", "plain", "promoted_vmap"


@pytest.mark.parametrize(("representation", "structure", "mode"), list(_cases()))
def test_one_trace_serves_every_redshift(representation, structure, mode):
    """One jit trace serves three redshifts; each output equals a fixed-z build."""
    model = _runtime_model(representation, structure, mode)
    traces = []

    if mode == "promoted_vmap":

        def program(zs):
            traces.append(1)
            return jax.vmap(lambda z: model.predict_photometry({"redshift": z}))(zs)

        got = np.asarray(jax.jit(program)(jnp.asarray(ZS)))
        outs = list(got)
    else:

        def program(z):
            traces.append(1)
            return model.predict_photometry({"redshift": z})

        jitted = jax.jit(program)
        outs = [np.asarray(jitted(jnp.asarray(z))) for z in ZS]

    assert len(traces) == 1, f"{mode}: {len(traces)} traces for {len(ZS)} redshifts"
    for z, got in zip(ZS, outs, strict=True):
        np.testing.assert_allclose(
            got,
            _reference(representation, structure, z),
            rtol=RTOL[representation],
            atol=0.0,
            err_msg=f"{representation}/{structure}/{mode} at z={z}",
        )


def test_override_promotes_once_and_serves_every_redshift():
    """A call-time override on a fixed-z model no longer raises and reuses one model."""
    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    promoted = _model_at_override_redshift(base, {"redshift": ZS[0]})
    assert promoted is not base
    assert _model_at_override_redshift(base, {"redshift": ZS[2]}) is promoted
    for z in ZS:
        got = np.asarray(promoted.predict_photometry({"redshift": z}))
        np.testing.assert_allclose(got, _reference("exact", "plain", z), rtol=RTOL["exact"])


def test_tabulated_override_spans_its_z():
    """A tabulated override promotion's table brackets the override redshift."""
    base = _build("wp", "plain", Fixed(Z_PLACEHOLDER))
    promoted = _model_at_override_redshift(base, {"redshift": 0.3})
    got = np.asarray(promoted.predict_photometry({"redshift": 0.3}))
    np.testing.assert_allclose(got, _reference("wp", "plain", 0.3), rtol=RTOL["wp"])


def test_runtime_z_defaults_to_declared_redshift():
    """A runtime model with no z in params uses its declared redshift (#3505 default)."""
    promoted = _build("exact", "plain", Fixed(Z_PLACEHOLDER)).with_runtime_redshift()
    got = np.asarray(promoted.predict_photometry({}))
    np.testing.assert_allclose(
        got, _reference("exact", "plain", Z_PLACEHOLDER), rtol=RTOL["exact"]
    )


def test_non_runtime_fixed_model_still_refuses_call_time_override():
    """The refusal stays for a Fixed redshift that is not promoted (#2296)."""
    from tengri.config.exceptions import ParameterError

    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ParameterError):
        base.predict_photometry({"redshift": 0.3})


def test_tabulated_promotion_needs_a_range():
    """A tabulated fixed-z model must name its redshift table span to be promoted."""
    base = _build("wp", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ValueError, match="z_range"):
        base.with_runtime_redshift()


def test_exact_promotion_rejects_a_table_range():
    """An exact model has no redshift table, so a range is refused, not ignored."""
    base = _build("exact", "plain", Fixed(Z_PLACEHOLDER))
    with pytest.raises(ValueError, match="z_range=None"):
        base.with_runtime_redshift(Z_RANGE)


# Bit-identity pin (#3505): a Fixed-z model that is NOT promoted keeps the fast path.
# Values are photometry [erg/s/cm2/Hz] computed on origin/main before this change,
# as float64 hex, at redshifts 0.1 (plain and rich structures) and 0.1 with WavePrecomp.
_PIN = {
    ("exact", "plain", 0.1): [
        "0x1.eeaad1813c405p-92",
        "0x1.a602837635024p-91",
        "0x1.28addff427fe6p-90",
    ],
    ("exact", "rich", 0.1): [
        "0x1.eec012a2d4d8bp-92",
        "0x1.a621c19a74dc0p-91",
        "0x1.28cdbd620f124p-90",
    ],
    ("wp", "rich", 0.1): [
        "0x1.eec677c37d44bp-92",
        "0x1.a6250caadc550p-91",
        "0x1.28c989d37249ap-90",
    ],
}


@pytest.mark.parametrize(("representation", "structure", "z"), list(_PIN))
def test_fixed_unpromoted_model_is_bit_identical_to_main(representation, structure, z):
    """A Fixed model that is not promoted is bit-identical to origin/main (#3505)."""
    got = np.asarray(_build(representation, structure, Fixed(z)).predict_photometry({}))
    expected = np.array([float.fromhex(v) for v in _PIN[(representation, structure, z)]])
    np.testing.assert_array_equal(got, expected)


# ── Component cells: every table a component bakes follows the runtime z ────────
#
# Each cell builds one model with a single redshift-dependent component, with every
# other parameter Fixed, so its output at z is fully determined by z. The runtime
# model (promoted from a Fixed placeholder) evaluated at three redshifts under ONE
# jit trace must equal a model built Fixed at each z, in both representations.

import tempfile
from pathlib import Path

from tengri import LineList

_FSPS_PATH = Path(__file__).resolve().parents[2] / "data" / "fsps_prsc_miles_chabrier.h5"
_Z_HIGH = (3.2, 3.8, 4.5)
_Z_RANGE_HIGH = (3.0, 5.0)
_LINE_NAMES = ("Hbeta", "OIII_5007", "Halpha")
_BANDS_OPT = ["sdss_g", "sdss_r", "sdss_i"]
_BANDS_IR = ["sdss_r", "wise_w1", "wise_w3", "wise_w4"]
_BANDS_RADIO = ["sdss_r", "wise_w3", "wise_w4", "herschel_250"]
_TAU = {"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)}
_AGN = {
    "type": "composable",
    "disc": {"type": "powerlaw", "all_params": Fixed(DEFAULT)},
    "torus": {"type": "simple", "all_params": Fixed(DEFAULT)},
    "blr": {"type": "none"},
    "nlr": {"type": "none"},
    "feii": {"type": "none"},
    "atten": {"type": "none"},
}
#: cell -> (bands, build kwargs, redshifts, table range)
_COMPONENT_CELLS = {
    "nebular_cue": (
        _BANDS_OPT,
        {"neb": {"type": "cue", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "dust_emission": (
        _BANDS_IR,
        {
            "dust_attenuation": {
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            "dust_emission": {"type": "dale2014", "all_params": Fixed(DEFAULT)},
        },
        ZS,
        Z_RANGE,
    ),
    "agn": (_BANDS_IR, {"agn": _AGN, "dust_attenuation": {"type": "none"}}, ZS, Z_RANGE),
    "radio": (
        _BANDS_RADIO,
        {
            "radio": {
                "sf": {"type": "bell2003"},
                "agn": {"type": "powerlaw"},
                "all_params": Fixed(DEFAULT),
            },
            "dust_attenuation": _TAU,
        },
        ZS,
        Z_RANGE,
    ),
    "xray": (
        _BANDS_RADIO,
        {"xray": {"type": "simple", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "shock_mappings": (
        _BANDS_OPT,
        {"shock": {"type": "mappings", "all_params": Fixed(DEFAULT)}, "dust_attenuation": _TAU},
        ZS,
        Z_RANGE,
    ),
    "igm_high_z": (
        ["sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        {"igm": {"type": "inoue14"}, "dust_attenuation": {"type": "none"}},
        _Z_HIGH,
        _Z_RANGE_HIGH,
    ),
}
_COMPONENT_CASES = [(cell, rep) for cell in _COMPONENT_CELLS for rep in REPRESENTATIONS]


@cache
def _fsps():
    if not _FSPS_PATH.is_file():
        pytest.skip(f"needs {_FSPS_PATH}")
    return load_ssp_data(str(_FSPS_PATH))


def _component_model(cell, representation, redshift, z_range=None):
    bands, kw, _, _ = _COMPONENT_CELLS[cell]
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_fsps(),
            observation=Observation(photometry=Photometry.from_names(bands)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(redshift),
            approx=approx,
            **kw,
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


def _runtime_outputs(model, zs, observable):
    traces = []

    def program(z):
        traces.append(1)
        return observable(model, {"redshift": z})

    jitted = jax.jit(program)
    outs = [np.asarray(jitted(jnp.asarray(z))) for z in zs]
    assert len(traces) == 1, f"{len(traces)} traces for {len(zs)} redshifts"
    return outs


def _photometry(model, params):
    return model.predict_photometry(params)


@pytest.mark.parametrize(("cell", "representation"), _COMPONENT_CASES)
def test_component_photometry_follows_runtime_z(cell, representation):
    """Runtime-z photometry of a component cell equals a Fixed-z build at each z."""
    _, _, zs, z_range = _COMPONENT_CELLS[cell]
    placeholder = zs[0]
    model = _component_model(
        cell, representation, placeholder, z_range if representation == "wp" else None
    )
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    for z, got in zip(zs, _runtime_outputs(model, zs, _photometry), strict=True):
        ref = np.asarray(_component_model(cell, representation, z).predict_photometry({}))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"{cell}/{representation} z={z}"
        )


@cache
def _line_waves():
    cat = LineList.default_optical()
    return jnp.asarray([float(w) for n, w in zip(cat.names, cat.wavelengths) if n in _LINE_NAMES])


def _line_fluxes(model, params):
    return model.predict_line_fluxes(params, target_wavelengths=_line_waves())


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_nebular_line_fluxes_follow_runtime_z(representation):
    """Cue line fluxes at a runtime z equal a Fixed-z build's, at three redshifts."""
    cell = "nebular_cue"
    placeholder = ZS[0]
    _line_waves()  # built outside the jit trace
    model = _component_model(
        cell, representation, placeholder, Z_RANGE if representation == "wp" else None
    )
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    for z, got in zip(ZS, _runtime_outputs(model, ZS, _line_fluxes), strict=True):
        ref = np.asarray(_line_fluxes(_component_model(cell, representation, z), {}))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"lines/{representation} z={z}"
        )


def _grid_model(redshift, z_range=None):
    from tengri import FeaturePrecomp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=_fsps(),
            observation=Observation(photometry=Photometry.from_names(_BANDS_OPT)),
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            met={"logzsol": Fixed(0.0)},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Uniform(-4.0, -1.0),
                "logZ_gas": Uniform(-1.5, 0.3),
            },
            dust_attenuation={"type": "none"},
            dust_emission={"type": "none"},
            redshift=Fixed(redshift),
            approx=(WavePrecomp(), FeaturePrecomp(n_grid=4)),
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


def test_nebular_grid_photometry_follows_runtime_z():
    """The per-Q_H nebular grid's band photometry follows a runtime z (tabulated bound)."""
    free = {"neb_logU": -2.5, "neb_logZ_gas": 0.0}
    model = _grid_model(ZS[0], Z_RANGE)
    assert model._nebular_grid_table is not None, "no nebular grid: the cell tests nothing"
    traces = []

    def program(z):
        traces.append(1)
        return model.predict_photometry({**free, "redshift": z})

    jitted = jax.jit(program)
    for z in ZS:
        got = np.asarray(jitted(jnp.asarray(z)))
        ref = np.asarray(_grid_model(z).predict_photometry(free))
        np.testing.assert_allclose(got, ref, rtol=RTOL["wp"], atol=0.0, err_msg=f"grid z={z}")
    assert len(traces) == 1


def test_nebular_grid_line_fluxes_follow_runtime_z():
    """Line fluxes served from the per-Q_H grid follow a runtime z (tabulated bound)."""
    free = {"neb_logU": -2.5, "neb_logZ_gas": 0.0}
    _line_waves()
    model = _grid_model(ZS[0], Z_RANGE)
    outs = _runtime_outputs(model, ZS, lambda m, p: _line_fluxes(m, {**free, **p}))
    for z, got in zip(ZS, outs, strict=True):
        ref = np.asarray(_line_fluxes(_grid_model(z), free))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL["wp"], atol=0.0, err_msg=f"grid lines z={z}"
        )


# ── Catalog entry points: N galaxies at N redshifts, one promoted model ────────
#
# Every catalog-level entry point derives one table span from the catalog's redshifts
# and promotes the model once. A second run on the same catalog adds no lowering, and
# the model the engine fits predicts at each galaxy's z what a Fixed-z build does.

import jax.monitoring as _jax_monitoring

from tengri.forward import convenience
from tengri.forward.sed_model import SEDModel as _SEDModel

CATALOG_ZS = (0.12, 0.31, 0.58, 0.77)
_TRUTH_LOGM = 9.5
_LOWERING = "/jax/core/compile/jaxpr_to_mlir_module_duration"
#: Programs a fit builds afresh around its result, whatever the redshift: the MAP
#: stationarity derivatives (one per fit, keyed on that fit's loss object), the posterior
#: summary maps and the result conversion. They are not the forward or loss programs the
#: promotion governs, so the second-run bar counts everything else.
_POST_FIT_PROGRAMS = frozenset({"jit(live)", "jit(mapped)", "jit(_convert_one)"})


def _mass_model(representation, redshift):
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=_ssp(),
            observation=_obs(),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(redshift),
            approx=approx,
        )


def _catalog_rows(representation):
    rows = []
    for z in CATALOG_ZS:
        phot = np.asarray(
            _mass_model(representation, z).predict_photometry(
                {"sfh_dpl_log_total_mass": _TRUTH_LOGM}
            )
        )
        row = {"z": z}
        row.update({f"f{j}": float(v) for j, v in enumerate(phot)})
        row.update({f"e{j}": float(0.05 * v) for j, v in enumerate(phot)})
        rows.append(row)
    return rows


def _run_catalog(entry, representation, rows, base, tmp_path):
    approx = None if representation == "exact" else "auto"
    if entry == "catalog_fitter":
        from tengri.inference.catalog_fitter import _CatalogFitterOriginal as CatalogFitter

        n_b = len(_obs().photometry.filters)
        galaxies = [
            {
                "flux_obs": jnp.asarray([r[f"f{j}"] for j in range(n_b)]),
                "noise": jnp.asarray([r[f"e{j}"] for j in range(n_b)]),
                "redshift": jnp.asarray(r["z"]),
            }
            for r in rows
        ]
        return (
            CatalogFitter(base, galaxies, approx=approx)
            .run("mcmc_nuts", key=jax.random.PRNGKey(0), n_warmup=8, n_samples=8, verbose=False)
            .posteriors
        )
    n_b = len(_obs().photometry.filters)
    kw = {"output_dir": str(tmp_path)} if entry == "fit_batch_checkpoint" else {}
    return convenience.fit_batch(
        base,
        rows,
        flux_cols=[f"f{j}" for j in range(n_b)],
        err_cols=[f"e{j}" for j in range(n_b)],
        redshift_col="z",
        method="map",
        n_steps=20,
        verbose=False,
        approx=approx,
        **kw,
    )


@pytest.mark.parametrize("representation", REPRESENTATIONS)
@pytest.mark.parametrize("entry", ("catalog", "catalog_fitter", "fit_batch_checkpoint"))
def test_catalog_promotes_once_and_a_rerun_adds_no_lowering(entry, representation, tmp_path):
    """One promotion for the whole catalog; a second run compiles nothing new."""
    base = _mass_model(representation, Z_PLACEHOLDER)
    rows = _catalog_rows(representation)
    promotions = []
    real = _SEDModel._promote_to_runtime_redshift

    def _count(self, z_range):
        promotions.append(z_range)
        return real(self, z_range)

    lowerings = []

    def _listener(name, _duration, **kw):
        if name == _LOWERING and kw.get("fun_name") not in _POST_FIT_PROGRAMS:
            lowerings.append(kw.get("fun_name"))

    _SEDModel._promote_to_runtime_redshift = _count
    _jax_monitoring.register_event_duration_secs_listener(_listener)
    try:
        first = _run_catalog(entry, representation, rows, base, tmp_path / "a")
        after_first = len(lowerings)
        n_promotions = len(promotions)
        second = _run_catalog(entry, representation, rows, base, tmp_path / "b")
    finally:
        _SEDModel._promote_to_runtime_redshift = real
        _jax_monitoring.unregister_event_duration_listener(_listener)

    assert n_promotions == 1, f"{entry}/{representation}: promotions {promotions}"
    assert len(promotions) == 1, "the second run promoted again instead of reusing the first"
    assert len(lowerings) == after_first, (
        f"{entry}/{representation}: the second run lowered {lowerings[after_first:]}"
    )
    assert len(first) == len(second) == len(CATALOG_ZS)
    if (promoted := getattr(first[0], "model", None)) is not None:
        inner = promoted.populations[0].sed if hasattr(promoted, "populations") else promoted
        params = {"sfh_dpl_log_total_mass": _TRUTH_LOGM}
        for z in CATALOG_ZS:
            got = np.asarray(inner.predict_photometry({**params, "redshift": z}))
            ref = np.asarray(_mass_model(representation, z).predict_photometry(params))
            np.testing.assert_allclose(
                got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"{entry}/{representation}"
            )


@cache
def _cloudy_grid_path():
    """The synthetic CLOUDY grid, written once per process (xdist workers do not share it)."""
    from tests._cloudy_grid import write_synthetic_cloudy_grid

    return write_synthetic_cloudy_grid(Path(tempfile.mkdtemp(prefix="tengri_sweep_")) / "grid.h5")


def _cloudy_model(representation, redshift, z_range=None):
    from tests._cloudy_grid import synthetic_ionizing_ssp

    grid = _cloudy_grid_path()
    approx = None if representation == "exact" else WavePrecomp()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=synthetic_ionizing_ssp(),
            observation=Observation(
                photometry=Photometry.from_names(["sdss_g", "sdss_r", "galex_fuv", "2mass_ks"])
            ),
            sfh={
                "type": "dpl",
                "all_params": Fixed(DEFAULT),
                "log_total_mass": Uniform(8.0, 12.0),
            },
            dust_attenuation={"type": "none"},
            neb={"type": "cloudy", "all_params": Fixed(DEFAULT), "grid": str(grid)},
            redshift=Fixed(redshift),
            approx=approx,
        )
        return model if z_range is None else model.with_runtime_redshift(z_range)


@pytest.mark.parametrize("representation", REPRESENTATIONS)
def test_cloudy_nebular_photometry_follows_runtime_z(representation):
    """The CLOUDY-grid band tables follow a runtime z (synthetic grid, tabulated bound)."""
    free = {"sfh_dpl_log_total_mass": 10.0}
    model = _cloudy_model(representation, ZS[0], Z_RANGE if representation == "wp" else None)
    if representation == "exact":
        model = model.with_runtime_redshift(None)
    outs = _runtime_outputs(model, ZS, lambda m, p: m.predict_photometry({**free, **p}))
    for z, got in zip(ZS, outs, strict=True):
        ref = np.asarray(_cloudy_model(representation, z).predict_photometry(free))
        np.testing.assert_allclose(
            got, ref, rtol=RTOL[representation], atol=0.0, err_msg=f"cloudy/{representation} z={z}"
        )
