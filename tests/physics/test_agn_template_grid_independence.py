# SPDX-License-Identifier: BSD-3-Clause
"""AGN template and analytic components do not depend on the caller's wavelength grid.

A component's absolute scale is ``L_bol x shape / (integral of shape)``. When that
integral is a trapezoid over the CALLER's wavelength array, the SED at a fixed
wavelength changes with how the caller samples wavelength, and is grossly wrong
(factors of order unity to 10^3) on a grid that does not cover the component's
support (a 912 A - 3 um SSP-like grid cuts the torus, the X-ray tail, ...).

Invariant tested, per component, at fixed probe wavelengths inside its support:

* a coarse grid that starts at 10 A (SSP-like, ~450 log nodes) and a fine grid that
  starts at 0.01 A (5000 log nodes) give the same L_nu to 1e-6;
* the fine grid and a truncated grid that does not cover the support give the same
  L_nu to 1e-6 inside the truncated grid's range.

The probe wavelengths are members of every grid, so the comparison is node-for-node:
what could differ is only the component's normalization, never an interpolation of
the output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.limit

_PROBES_DISC = np.geomspace(1.0e3, 2.5e4, 40)  # A
_PROBES_IR = np.geomspace(2.0e4, 3.5e5, 40)
_PROBES_WIDE = np.geomspace(1.0e3, 2.0e5, 40)

# (probes, truncated-grid range [A]); the truncated grid does not cover the support.
_DISC = (_PROBES_DISC, (912.0, 3.0e4))
_IR = (_PROBES_IR, (1.0e4, 4.0e5))
_WIDE = (_PROBES_WIDE, (912.0, 3.0e5))

_RTOL = 1.0e-6


def _grid(lo, hi, n, probes):
    wave = np.unique(np.concatenate([np.geomspace(lo, hi, n), probes]))
    return wave, np.searchsorted(wave, probes)


def _eval(fn, wave, idx):
    return np.asarray(fn(jnp.asarray(wave)), dtype=float)[idx]


def _max_rel(a, ref):
    """Max |a/ref - 1| over probes carrying more than 1e-6 of the peak."""
    mask = np.abs(ref) > 1.0e-6 * np.max(np.abs(ref))
    assert mask.sum() >= 10, "probe set misses the component's support"
    return float(np.max(np.abs(a[mask] / ref[mask] - 1.0)))


# -- component constructors (imports inside so one missing data file skips one case) --


def _relagn():
    from tengri.components.agn.disc import load_relagn_default_grid, relagn_disc_from_grid

    grid = load_relagn_default_grid()
    return lambda w: relagn_disc_from_grid(
        grid,
        w,
        agn_log_lbol=11.5,
        agn_log_mbh=8.0,
        agn_log_mdot=-1.0,
        agn_astar=0.0,
        agn_cos_inc=0.8,
    )


def _kd18():
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_sed

    return lambda w: kd18_agnfitter_sed(w, agn_log_lbol=11.5, agn_log_mbh=8.0, agn_log_ledd=-1.0)


def _kd18_warm():
    from tengri.components.agn.kd18_agnfitter import kd18_agnfitter_warmindex_sed

    return lambda w: kd18_agnfitter_warmindex_sed(
        w, agn_log_lbol=11.5, agn_log_mbh=8.0, agn_log_ledd=-1.0, agn_gamma_warm=2.5
    )


def _slone_netzer():
    from tengri.components.agn.slone_netzer import slone_netzer_sed

    return lambda w: slone_netzer_sed(w, agn_log_lbol=11.5, agn_log_mbh=8.6, agn_log_ledd=-1.5)


def _qsogen():
    from tengri.components.agn.qsogen import compute_qsogen_sed

    return lambda w: compute_qsogen_sed(w, agn_log_lbol=11.5)


def _simple_torus():
    from tengri.components.agn.torus import simple_torus

    return lambda w: simple_torus(w, 11.5)


def _two_temperature_torus():
    from tengri.components.agn.torus import two_temperature_torus

    return lambda w: two_temperature_torus(w, 11.5)


def _nenkova_torus():
    from tengri.components.agn.torus import nenkova_torus

    return lambda w: nenkova_torus(w, agn_log_lbol=11.5, agn_tau=30.0)


def _silva04():
    from tengri.components.agn.silva04 import silva04_sed

    return lambda w: silva04_sed(w, agn_log_lbol=11.5, agn_log_nh_silva=23.0, agn_torus_frac=0.5)


def _fritz():
    from tengri.components.agn.fritz import fritz_sed

    return lambda w: fritz_sed(w, agn_log_lbol=11.5)


def _by_name(module, name):
    import importlib

    fn = getattr(importlib.import_module(f"tengri.components.agn.{module}"), name)
    return lambda w: fn(w, agn_log_lbol=11.5)


def _adaf():
    from tengri.components.agn.adaf import adaf_spectrum

    return lambda w: adaf_spectrum(w, 11.5, agn_log_mbh=8.0)


def _relagn_agn():
    from tengri.components.agn.unified import relagn_agn

    return lambda w: relagn_agn(w, agn_log_mbh=8.0, agn_log_mdot=-1.0, agn_astar=0.0)


_CASES = {
    "relagn_disc_from_grid": (_relagn, _DISC),
    "kd18_agnfitter": (_kd18, _DISC),
    "kd18_agnfitter_warmindex": (_kd18_warm, _DISC),
    "slone_netzer": (_slone_netzer, _DISC),
    "qsogen": (_qsogen, _DISC),
    "simple_torus": (_simple_torus, _IR),
    "two_temperature_torus": (_two_temperature_torus, _IR),
    "nenkova_torus": (_nenkova_torus, _IR),
    "silva04": (_silva04, _IR),
    "fritz": (_fritz, _IR),
    "skirtor_agnfitter_3p": (lambda: _by_name("skirtor_agnfitter", "skirtor_agnfitter_sed"), _IR),
    "skirtor_agnfitter_1p": (
        lambda: _by_name("skirtor_agnfitter_1p", "skirtor_agnfitter_1p_sed"),
        _IR,
    ),
    "skirtor_agnfitter_2p": (
        lambda: _by_name("skirtor_agnfitter_2p", "skirtor_agnfitter_2p_sed"),
        _IR,
    ),
    "nenkova_agnfitter_3p": (lambda: _by_name("nenkova_agnfitter", "nenkova_agnfitter_sed"), _IR),
    "nenkova_agnfitter_2p": (
        lambda: _by_name("nenkova_agnfitter_2p", "nenkova_agnfitter_2p_sed"),
        _IR,
    ),
    "nenkova_agnfitter_3p_alias": (
        lambda: _by_name("nenkova_agnfitter_3p", "nenkova_agnfitter_3p_sed"),
        _IR,
    ),
    "cat3d_wind": (lambda: _by_name("cat3d_wind", "cat3d_wind_sed"), _IR),
    "cat3d_wind_lowfwd": (lambda: _by_name("cat3d_wind_lowfwd", "cat3d_wind_lowfwd_sed"), _IR),
    "adaf_spectrum": (_adaf, _WIDE),
    "relagn_agn": (_relagn_agn, _WIDE),
}


@pytest.fixture(scope="module", params=sorted(_CASES))
def case(request):
    make, (probes, trunc) = _CASES[request.param]
    try:
        fn = make()
    except (FileNotFoundError, ImportError, OSError) as err:
        pytest.skip(f"{request.param}: template data unavailable ({err})")
    fine_w, fine_i = _grid(0.01, 1.0e8, 5000, probes)
    return {
        "name": request.param,
        "fn": fn,
        "probes": probes,
        "trunc": trunc,
        "fine": _eval(fn, fine_w, fine_i),
    }


def test_coarse_grid_matches_fine_grid(case):
    """A 10 A-start SSP-like grid reproduces the 0.01 A-start fine grid to 1e-6."""
    wave, idx = _grid(10.0, 1.0e8, 450, case["probes"])
    err = _max_rel(_eval(case["fn"], wave, idx), case["fine"])
    assert err < _RTOL, f"{case['name']}: coarse-vs-fine max rel diff {err:.3e}"


def test_truncated_grid_matches_fine_grid(case):
    """A grid that does not cover the support reproduces the fine grid to 1e-6."""
    lo, hi = case["trunc"]
    wave, idx = _grid(lo, hi, 450, case["probes"])
    err = _max_rel(_eval(case["fn"], wave, idx), case["fine"])
    assert err < _RTOL, f"{case['name']}: truncated-vs-fine max rel diff {err:.3e}"


# -- published L_* diagnostics (#2745) -------------------------------------------------
#
# A published luminosity is the integral of the SED the component EMITS, taken on the
# component's own publication grid, under the key's documented definition. Three checks:
#
# * identity: the value equals an independent wide-grid quadrature of the emitted SED, at
#   two parameter points (the quadrature is written here, not taken from the component);
# * grid independence: the value is the same on a dense, a 40-point and a 12-point caller
#   grid, to the file's 1e-6;
# * the SKIRTOR emitted SED does not depend on the caller grid, and at delta=0 it is the
#   tabulated disc (bit-identical to the un-retilted template on the same grid).

_C_AA = 2.99792458e18
# SKIRTOR's analytic disc is zero below 8 nm and above 1e6 nm: the edges are nodes, so the
# quadrature does not straddle the jump at either edge.
_BAND_WIDE = np.geomspace(10.0, 1.0e8, 200_000)
_DISC_EDGES_AA = [x for e in (80.0, 1.0e7) for x in (e * (1 - 1e-9), e, e * (1 + 1e-9))]
_WIDE = np.unique(np.concatenate([np.geomspace(1.0e-3, 1.0e10, 200_000), _DISC_EDGES_AA]))
_PUB_DENSE = np.geomspace(10.0, 1.0e7, 4000)
_PUB_COARSE = np.geomspace(10.0, 1.0e7, 40)
_PUB_FILTERS = np.array(
    [
        1528.0,
        2271.0,
        3551.0,
        4686.0,
        6166.0,
        7480.0,
        8932.0,
        12350.0,
        16620.0,
        21590.0,
        33526.0,
        46028.0,
    ]
)
_PUB_GRIDS = (_PUB_DENSE, _PUB_COARSE, _PUB_FILTERS)


def _data_file(name):
    from tengri._data_setup import find_data

    found = find_data(name)
    if found is None:
        pytest.skip(f"template data unavailable ({name})")
    return str(found)


def _ln_nu_integral(sed, wave):
    """Bolometric integral of L_nu on any grid: trapezoid in ln(nu) of L_nu * nu [erg/s]."""
    nu = _C_AA / np.asarray(wave, dtype=float)
    order = np.argsort(nu)
    y = np.asarray(sed, dtype=float)[order] * nu[order]
    return float(np.trapezoid(y, np.log(nu[order])))


def _load_component(module, cls_name, cfg_name, grid_file=None, overrides=None):
    """Build one SEDModelComponent with its templates loaded and its declared defaults."""
    import importlib

    mod = importlib.import_module(f"tengri.components.agn.{module}")
    kwargs = {} if grid_file is None else {"grid_path": _data_file(grid_file)}
    comp = getattr(mod, cls_name)(config=getattr(mod, cfg_name)(**kwargs))
    # Frozen dataclasses refuse plain assignment; the pipeline stores ``data`` the same way.
    object.__setattr__(comp, "data", comp.load(jnp.asarray(_PUB_DENSE)))
    prefix = comp.parameter_prefix
    params = {
        decl.name[len(prefix) :]: jnp.asarray(decl.prior.default, dtype=jnp.float64)
        for decl in comp.declared_parameters()
    }
    for name, value in (overrides or {}).items():
        params[name] = jnp.asarray(value, dtype=jnp.float64)
    return comp, params


def _publisher(comp, params, key):
    """``fn(wave) -> published[key]``, through the component's real ``predict``."""

    def fn(wave):
        w = jnp.asarray(wave)
        _, published = comp.predict(params, jnp.zeros_like(w), w)
        return float(published[key])

    return fn


def _emitter(comp, params):
    """``fn(wave) -> emitted SED`` for a single-component class (sed_in = 0)."""

    def fn(wave):
        w = jnp.asarray(wave)
        sed, _ = comp.predict(params, jnp.zeros_like(w), w)
        return np.asarray(sed, dtype=float)

    return fn


# Single-component classes: (module, class, config, grid file, key, parameter overrides).
_SINGLE = {
    "silva04_torus.L_agn_torus": (
        "silva04_model",
        "Silva04Torus",
        "Silva04TorusConfig",
        "silva04_torus_grid.h5",
        "L_agn_torus",
        ({}, {"log_nh_silva": 22.5, "torus_frac": 0.3, "log_lbol": 11.2}),
    ),
    "cat3d_torus.L_agn_torus": (
        "cat3d_torus_model",
        "CAT3DTorus",
        "CAT3DTorusConfig",
        "cat3d_wind_torus_grid.h5",
        "L_agn_torus",
        ({}, {"cos_inc": 0.3, "torus_frac": 0.3}),
    ),
    "skirtor_agnfitter_torus.L_agn_torus": (
        "skirtor_agnfitter_model",
        "SKIRTORAgnfitterTorus",
        "SKIRTORAgnfitterTorusConfig",
        "skirtor_mean3p_torus_grid.h5",
        "L_agn_torus",
        ({}, {"torus_frac": 0.3}),
    ),
    "powerlaw_disc.L_agn_disc": (
        "powerlaw_disc_model",
        "PowerLawDisc",
        "PowerLawDiscConfig",
        None,
        "L_agn_disc",
        ({}, {"lum_ratio": 0.3}),
    ),
    "kd18_disc.L_agn_disc": (
        "kd18_disc_model",
        "KD18Disc",
        "KD18DiscConfig",
        None,
        "L_agn_disc",
        ({}, {"cos_inc": 0.5}),
    ),
}


@pytest.fixture(scope="module", params=sorted(_SINGLE))
def single_case(request):
    module, cls_name, cfg_name, grid_file, key, variants = _SINGLE[request.param]
    try:
        built = [_load_component(module, cls_name, cfg_name, grid_file, ov) for ov in variants]
    except (FileNotFoundError, ImportError, OSError) as err:
        pytest.skip(f"{request.param}: template data unavailable ({err})")
    return {"name": request.param, "key": key, "built": built}


def test_single_published_matches_wide_quadrature_of_emitted_sed(single_case):
    """Published L equals the wide-grid integral of the SED the component emits (1e-6)."""
    for comp, params in single_case["built"]:
        published = _publisher(comp, params, single_case["key"])(_PUB_DENSE)
        # PowerLaw's emitted tail is not integrable beyond its 10 A - 1e8 A normalization band,
        # so its identity is taken over that band; the other classes over the full range.
        grid = _BAND_WIDE if single_case["name"].startswith("powerlaw") else _WIDE
        emitted = _ln_nu_integral(_emitter(comp, params)(grid), grid)
        err = abs(published / emitted - 1.0)
        assert err < _RTOL, f"{single_case['name']}: published-vs-emitted rel {err:.3e}"


def test_single_published_independent_of_caller_grid(single_case):
    """The same L comes out on dense, 40-point and filter-wavelength caller grids."""
    for comp, params in single_case["built"]:
        values = [_publisher(comp, params, single_case["key"])(g) for g in _PUB_GRIDS]
        for label, val in zip(("coarse-40", "filter-12"), values[1:], strict=True):
            err = abs(val / values[0] - 1.0)
            assert err < _RTOL, f"{single_case['name']} {label}: rel diff {err:.3e}"


def test_kd18_float32_frac_zero_is_zero_not_nan():
    """KD18 at lum_ratio = 0 publishes exactly 0.0 in float32; no corner gives NaN."""
    with jax.enable_x64(False):
        comp, params = _load_component("kd18_disc_model", "KD18Disc", "KD18DiscConfig")
        for log_lbol, lum_ratio in ((8.0, 0.0), (14.0, 0.0), (8.0, 0.2), (14.0, 1.0)):
            p = {k: jnp.asarray(v, dtype=jnp.float32) for k, v in params.items()}
            p["log_lbol"] = jnp.asarray(log_lbol, dtype=jnp.float32)
            p["lum_ratio"] = jnp.asarray(lum_ratio, dtype=jnp.float32)
            w = jnp.asarray(_PUB_COARSE, dtype=jnp.float32)
            sed, pub = comp.predict(p, jnp.zeros_like(w), w)
            value = float(pub["L_agn_disc"])
            assert not np.isnan(value), f"NaN at log_lbol={log_lbol}, lum_ratio={lum_ratio}"
            assert not np.any(np.isnan(np.asarray(sed))), "NaN in the emitted SED"
            if lum_ratio == 0.0:
                assert value == 0.0, f"L_agn_disc={value} at lum_ratio=0 (log_lbol={log_lbol})"


def test_skirtor_published_match_wide_quadrature_of_emitted_pieces():
    """SKIRTOR: each published L equals the wide-grid integral of its emitted piece."""
    for overrides in ({}, {"delta": 0.3, "polar_ebv": 0.2, "cos_inc": 0.3}):
        try:
            comp, params = _load_component(
                "skirtor_model",
                "SKIRTORTorus",
                "SKIRTORTorusConfig",
                "skirtor_templates_v3.h5",
                overrides,
            )
        except (FileNotFoundError, ImportError, OSError) as err:
            pytest.skip(f"SKIRTOR templates unavailable ({err})")
        pieces = _skirtor_emitted_pieces(comp, params, _WIDE)
        pub = {
            key: _publisher(comp, params, key)(_PUB_DENSE)
            for key in ("L_agn_disc", "L_agn_torus", "L_agn_polar_dust")
        }
        for key, wide in (
            ("L_agn_disc", pieces["disc"]),
            ("L_agn_torus", pieces["dust"]),
            ("L_agn_polar_dust", pieces["polar"]),
        ):
            emitted = _ln_nu_integral(wide, _WIDE)
            err = abs(pub[key] / emitted - 1.0)
            assert err < _RTOL, f"SKIRTOR {key} {overrides}: published-vs-emitted rel {err:.3e}"


def _skirtor_emitted_pieces(comp, params, wave):
    """Emitted SKIRTOR disc, dust and polar-reemission SEDs, from the component's own functions.

    The disc is re-tilted by the CIGALE shape ratio and renormalized by the ratio of two
    integrals over the analytic disc's support, on a wide grid, so it matches the component's
    publication-grid factor to the quadrature error of that grid.
    """
    from tengri.components.agn.disc_cigale import skirtor_disk_spectrum
    from tengri.components.agn.polar_dust import polar_dust_emission, polar_dust_extinction

    p = {k: float(v) for k, v in params.items()}
    w = jnp.asarray(wave)
    comps = comp.data(
        wavelength=w,
        agn_log_lbol=p["log_lbol"],
        agn_tau_skirtor=p["tau_skirtor"],
        agn_p_skirtor=p["p_skirtor"],
        agn_q_skirtor=p["q_skirtor"],
        agn_oa_skirtor=p["oa_skirtor"],
        agn_radius_ratio=p["radius_ratio"],
        agn_cos_inc=p["cos_inc"],
        frac_agn=p["torus_frac"],
    )
    wave_nm = w / 10.0
    shape_sel = skirtor_disk_spectrum(wave_nm, delta=p["delta"])
    shape_ref = skirtor_disk_spectrum(wave_nm, delta=0.0)
    retilt = shape_sel / jnp.maximum(shape_ref, 1e-100)
    support = (shape_ref > 0.0).astype(w.dtype)
    raw = comps.disk
    k = _ln_nu_integral(np.asarray(raw * support), wave) / _ln_nu_integral(
        np.asarray(raw * retilt * support), wave
    )
    disc = raw * retilt * k
    _, l_abs = polar_dust_extinction(
        disc, w, p["cos_inc"], p["oa_skirtor"], p["polar_ebv"], law="smc"
    )
    l_abs_total = _ln_nu_integral(np.asarray(l_abs), wave)
    polar = polar_dust_emission(
        l_abs_total, w, temperature=p["polar_T"], beta=p["polar_beta"], lambda_0=2e6
    )
    return {"disc": np.asarray(disc), "dust": np.asarray(comps.dust), "polar": np.asarray(polar)}


def test_skirtor_delta0_sed_is_the_tabulated_disc_and_torus():
    """At delta=0 with no polar dust the emitted SED is the tabulated disc plus torus dust."""
    comp, params = _load_component(
        "skirtor_model",
        "SKIRTORTorus",
        "SKIRTORTorusConfig",
        "skirtor_templates_v3.h5",
        {"delta": 0.0, "polar_ebv": 0.0},
    )
    wave = jnp.asarray(np.geomspace(1.0e2, 1.0e6, 400))
    sed, _ = comp.predict(params, jnp.zeros_like(wave), wave)
    comps = comp.data(
        wavelength=wave,
        agn_log_lbol=params["log_lbol"],
        agn_tau_skirtor=params["tau_skirtor"],
        agn_p_skirtor=params["p_skirtor"],
        agn_q_skirtor=params["q_skirtor"],
        agn_oa_skirtor=params["oa_skirtor"],
        agn_radius_ratio=params["radius_ratio"],
        agn_cos_inc=params["cos_inc"],
        frac_agn=params["torus_frac"],
    )
    expected = np.asarray(comps.disk + comps.dust)
    np.testing.assert_allclose(np.asarray(sed), expected, rtol=1e-12, atol=0.0)


def test_skirtor_emitted_sed_independent_of_caller_grid():
    """At delta != 0 the emitted SED at shared probe wavelengths is the same on any caller grid."""
    comp, params = _load_component(
        "skirtor_model",
        "SKIRTORTorus",
        "SKIRTORTorusConfig",
        "skirtor_templates_v3.h5",
        {"delta": 0.3, "polar_ebv": 0.2},
    )
    probes = np.geomspace(1.0e3, 1.0e5, 20)
    dense = np.union1d(probes, _PUB_DENSE)
    coarse = np.union1d(probes, _PUB_COARSE)
    seds = []
    for grid in (dense, coarse):
        sed, _ = comp.predict(params, jnp.zeros(grid.shape), jnp.asarray(grid))
        seds.append(np.asarray(sed)[np.searchsorted(grid, probes)])
    err = float(np.max(np.abs(seds[0] / seds[1] - 1.0)))
    assert err < _RTOL, f"SKIRTOR emitted SED rel diff across caller grids {err:.3e}"
