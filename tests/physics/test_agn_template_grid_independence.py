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
        "L_agn_disc_los",
        ({}, {"cos_inc": 0.2}),
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


@pytest.mark.parametrize("name", ["powerlaw_disc.L_agn_disc", "kd18_disc.L_agn_disc"])
def test_disc_apply_with_declared_parameters_publishes_the_integral(name):
    """``apply`` on the real pipeline path, given only declared parameters, publishes L_agn_disc.

    The disc's luminosity fraction is ``lum_ratio``; a read of an undeclared ``frac`` raised
    ``KeyError`` here, which the contract tests (they pass only ``lum_ratio`` to ``predict``)
    could not see.
    """
    from tengri.protocols import ForwardState

    module, cls_name, cfg_name, grid_file, key, _ = _SINGLE[name]
    comp, params = _load_component(module, cls_name, cfg_name, grid_file)
    prefix = comp.parameter_prefix
    wave = jnp.asarray(_PUB_DENSE)
    state = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave), derived={})
    out = comp.apply(state, {prefix + k: v for k, v in params.items()})
    published = float(out.derived[key])
    direct = _publisher(comp, params, key)(_PUB_DENSE)
    assert np.isfinite(published) and published > 0.0
    assert published == pytest.approx(direct, rel=1e-12)


def _kd18_with(**overrides):
    return _load_component("kd18_disc_model", "KD18Disc", "KD18DiscConfig", None, overrides)


def _kd18_published(comp, params, wave):
    _, pub = comp.predict(params, jnp.zeros_like(wave), wave)
    return float(pub["L_agn_disc"]), float(pub["L_agn_disc_los"])


def test_kd18_angle_integrated_disc_power_is_independent_of_inclination():
    """L_agn_disc is the angle-integrated power: the same at cos i = 0.1, 0.5 and 0.95."""
    wave = jnp.asarray(_PUB_COARSE)
    values = [_kd18_published(*_kd18_with(cos_inc=c), wave)[0] for c in (0.1, 0.5, 0.95)]
    for v in values[1:]:
        assert abs(v / values[0] - 1.0) < 1e-10, f"L_agn_disc moved with inclination: {values}"


def test_kd18_line_of_sight_power_is_the_integral_of_the_emitted_sed():
    """L_agn_disc_los equals the dense integral of the emitted, projected SED (1e-6)."""
    for cos_inc in (0.1, 0.5, 0.95):
        comp, params = _kd18_with(cos_inc=cos_inc)
        _, los = _kd18_published(comp, params, jnp.asarray(_PUB_COARSE))
        emitted = _ln_nu_integral(_emitter(comp, params)(_WIDE), _WIDE)
        assert abs(los / emitted - 1.0) < _RTOL, f"cos_inc={cos_inc}: rel {los / emitted - 1:.3e}"


def test_kd18_los_to_angle_integrated_ratio_is_two_cos_i_for_the_disc():
    """With the corona removed, L_los / L_angle-integrated = 2 cos i (independent expectation)."""
    wave = jnp.asarray(_PUB_COARSE)
    for cos_inc in (0.1, 0.3, 0.95):
        comp, params = _kd18_with(cos_inc=cos_inc, f_hard=0.0)
        total, los = _kd18_published(comp, params, wave)
        # f_hard = 0 leaves a corona only through its (tiny) floor; the ratio is exact to it.
        assert los / total == pytest.approx(2.0 * cos_inc, rel=2e-3), (cos_inc, los / total)


def test_kd18_angle_integrated_power_matches_the_accretion_power():
    """L_agn_disc is lum_ratio times the accretion power 10**log_lbol L_sun, less the cut tail."""
    l_sun = 3.828e33
    wave = jnp.asarray(_PUB_COARSE)
    for lum_ratio in (1.0, 0.4):
        comp, params = _kd18_with(lum_ratio=lum_ratio)
        total, _ = _kd18_published(comp, params, wave)
        expected = lum_ratio * 10.0 ** float(params["log_lbol"]) * l_sun
        assert total / expected == pytest.approx(1.0, abs=2e-3), (lum_ratio, total / expected)


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


def _skirtor_with_disk_type(disk_type, overrides):
    """SKIRTOR component with the given static ``disk_type`` and its declared defaults."""
    from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

    try:
        grid = _data_file("skirtor_templates_v3.h5")
        comp = SKIRTORTorus(config=SKIRTORTorusConfig(grid_path=grid, disk_type=disk_type))
        object.__setattr__(comp, "data", comp.load(jnp.asarray(_PUB_DENSE)))
    except (FileNotFoundError, ImportError, OSError) as err:
        pytest.skip(f"SKIRTOR templates unavailable ({err})")
    params = {
        d.name[len(comp.parameter_prefix) :]: jnp.asarray(d.prior.default, dtype=jnp.float64)
        for d in comp.declared_parameters()
    }
    params.update({k: jnp.asarray(v, dtype=jnp.float64) for k, v in overrides.items()})
    return comp, params


@pytest.mark.parametrize("disk_type", [1, 2])
@pytest.mark.parametrize(
    "overrides",
    [{"delta": 0.3, "polar_ebv": 0.0}, {"delta": 0.3, "polar_ebv": 0.2, "cos_inc": 0.3}],
)
def test_skirtor_other_disc_types_published_match_emitted(disk_type, overrides):
    """Schartmann (1) and ADAF (2) discs: published L equal integrals of the emitted SED.

    These discs have breakpoints of their own (disc_cigale), so the publication nodes and the
    polar segments must contain them, or the closed form and the quadrature miss the kinks.
    The emitted SED itself is integrated on a dense grid: with no polar dust it carries
    exactly the published disc plus torus, with polar dust the published polar power is the
    absorbed disc power that the emitted re-emission carries.
    """
    comp, params = _skirtor_with_disk_type(disk_type, overrides)
    pub = {
        key: _publisher(comp, params, key)(_PUB_COARSE)
        for key in ("L_agn_disc", "L_agn_torus", "L_agn_polar_dust")
    }
    pieces = _skirtor_emitted_pieces(comp, params, _WIDE, disk_type=disk_type)
    for key, piece in (("L_agn_disc", "disc"), ("L_agn_torus", "dust")):
        err = abs(pub[key] / _ln_nu_integral(pieces[piece], _WIDE) - 1.0)
        assert err < _RTOL, f"disk_type={disk_type} {overrides} {key}: rel {err:.3e}"
    sed = _emitter(comp, params)(_WIDE)
    total = _ln_nu_integral(sed, _WIDE)
    if float(params["polar_ebv"]) == 0.0:
        err = abs(total / (pub["L_agn_disc"] + pub["L_agn_torus"]) - 1.0)
        assert err < _RTOL, f"disk_type={disk_type} {overrides} SED total: rel {err:.3e}"
        assert pub["L_agn_polar_dust"] == 0.0
    else:
        err = abs(pub["L_agn_polar_dust"] / pieces["polar_absorbed"] - 1.0)
        assert err < _RTOL, f"disk_type={disk_type} {overrides} polar absorbed: rel {err:.3e}"


def _skirtor_emitted_pieces(comp, params, wave, disk_type=0):
    """Emitted SKIRTOR disc, dust and polar-reemission SEDs, from the component's own functions.

    The disc is re-tilted by the CIGALE shape ratio and renormalized by the ratio of two
    integrals over the analytic disc's support, on a wide grid, so it matches the component's
    publication-grid factor to the quadrature error of that grid.
    """
    from tengri.components.agn.disc_cigale import (
        adaf_disk_spectrum,
        schartmann2005_disk_spectrum,
        skirtor_disk_spectrum,
    )
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
    shape_fn = (skirtor_disk_spectrum, schartmann2005_disk_spectrum, adaf_disk_spectrum)[disk_type]
    shape_sel = shape_fn(wave_nm, delta=p["delta"])
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
    return {
        "disc": np.asarray(disc),
        "dust": np.asarray(comps.dust),
        "polar": np.asarray(polar),
        "polar_absorbed": l_abs_total,
    }


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


def test_skirtor_adaf_disc_published_is_differentiable_in_delta():
    """The ADAF disc's limits move with delta; the published disc and polar power stay finite."""
    comp, params = _skirtor_with_disk_type(2, {"delta": 0.3, "polar_ebv": 0.2, "cos_inc": 0.3})
    wave = jnp.asarray(_PUB_COARSE)

    def published(delta, key):
        p = {**params, "delta": delta}
        _, pub = comp.predict(p, jnp.zeros_like(wave), wave)
        return pub[key]

    for key in ("L_agn_disc", "L_agn_polar_dust"):
        grad = jax.grad(published)(jnp.asarray(0.3), key)
        assert bool(jnp.isfinite(grad)), f"d{key}/d delta = {grad}"
        assert float(grad) != 0.0, f"d{key}/d delta is identically zero"


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


# -- GRAHSP published L_* (#2745 follow-up) --------------------------------------------
#
# GRAHSP publishes three luminosities. Each is the integral of the SED the component emits,
# under the key's documented definition (Buchner+ 2024 section 2.1.4-2.1.5, arXiv:2405.19297):
#
# * L_agn_bol: intrinsic BBB + lines + FeII + Balmer continuum above the Lyman limit (91.2 nm),
#   before the bi-attenuation;
# * L_agn_torus: intrinsic torus continuum + Si feature over its full support;
# * L_agn_absorbed: intrinsic minus attenuated AGN-side power, over the full support.
#
# The identity reference is a wide, dense trapezoid in ln(nu) of the emitted L_nu, built from
# GRAHSP variants that switch the other pieces off (the component's own sums, not a copy of them).

_GRAHSP_KEYS = ("L_agn_bol", "L_agn_torus", "L_agn_absorbed")
_GRAHSP_EDGES_AA = [
    x for e in (124.0, 912.0, 2500.0, 3646.0, 11000.0) for x in (e * (1 - 1e-9), e, e * (1 + 1e-9))
]


def _template_nodes_aa(*names):
    """Native nodes [Angstrom] of GRAHSP template axes, plus a pair either side of each end.

    A tabulated piece is piecewise linear between its nodes and zero outside, so the
    reference trapezoid needs every node, and a point a hair outside each end: where the
    template does not vanish at its edge (the FeII forest starts at 24 % of its peak) the
    emitted SED jumps there, and a trapezoid that straddles the jump is first order in the grid
    spacing.
    """
    from tengri.components.agn.grahsp.templates import load_grahsp_templates

    templates = load_grahsp_templates()
    out = []
    for name in names:
        axis = 10.0 * np.asarray(getattr(templates, name), dtype=float)
        out.extend([axis, [axis.min() * (1 - 1e-9), axis.max() * (1 + 1e-9)]])
    return np.concatenate(out)


def _grahsp_wide(*template_axes):
    """Dense reference grid [Angstrom]: log grid, hard edges and the selected template nodes."""
    return np.unique(
        np.concatenate(
            [
                np.geomspace(100.0, 1.0e9, 1_200_000),
                _GRAHSP_EDGES_AA,
                _template_nodes_aa(*template_axes),
            ]
        )
    )


_GRAHSP_WIDE = _grahsp_wide("feii_wave_nm")
_GRAHSP_RTOL = 1.0e-6


def _grahsp_comp(**overrides):
    import dataclasses

    from tengri.components.agn.grahsp.component import (
        GRAHSPSEDComponent,
        GRAHSPSEDComponentConfig,
    )

    cfg = dataclasses.replace(GRAHSPSEDComponentConfig(), **overrides)
    return GRAHSPSEDComponent(config=cfg)


def _grahsp_value(prior):
    """Declared default, else the bounds' midpoint (geometric for a log-uniform prior)."""
    if prior.default is not None:
        return prior.default
    lo, hi = prior.bounds
    return float(np.sqrt(lo * hi)) if type(prior).__name__ == "LogUniform" else 0.5 * (lo + hi)


def _grahsp_params(comp):
    return {
        d.name: jnp.asarray(_grahsp_value(d.prior), dtype=jnp.float64)
        for d in comp.declared_parameters()
    }


def _grahsp_apply(comp, params, wave_aa, tpl):
    """One ``apply`` on a caller grid [Angstrom] -> (derived dict, emitted L_nu [erg/s/Hz])."""
    from tengri.protocols import ForwardState

    w = jnp.asarray(np.asarray(wave_aa, dtype=float))
    state = ForwardState(wave=w, sed_intrinsic=jnp.zeros_like(w), derived={})
    out = comp.apply(state, params, templates_state=tpl)
    return out.derived, np.asarray(out.sed_intrinsic, dtype=float)


def _grahsp_sed(comp, params, wave_aa, tpl):
    """Emitted L_nu on a long grid, evaluated in chunks (the published values are not used)."""
    chunks = np.array_split(np.asarray(wave_aa, dtype=float), 8)
    return np.concatenate([_grahsp_apply(comp, params, c, tpl)[1] for c in chunks])


def _grahsp_published(comp, params, wave_aa, tpl):
    derived, _ = _grahsp_apply(comp, params, wave_aa, tpl)
    return {k: float(derived[k]) for k in _GRAHSP_KEYS}


def test_grahsp_published_L_independent_of_caller_grid():
    """Each GRAHSP L_* is the same on a dense, a 40-point and a 12-point caller grid (1e-6)."""
    comp = _grahsp_comp()
    tpl = comp.precompute()
    params = _grahsp_params(comp)
    dense = _grahsp_published(comp, params, _PUB_DENSE, tpl)
    for label, grid in (("coarse-40", _PUB_COARSE), ("filter-12", _PUB_FILTERS)):
        other = _grahsp_published(comp, params, grid, tpl)
        for key in _GRAHSP_KEYS:
            err = abs(other[key] / dense[key] - 1.0)
            assert err < _GRAHSP_RTOL, f"GRAHSP {key} {label}: rel diff {err:.3e}"


def _grahsp_identity_errors(over, wide):
    """Relative error of each published GRAHSP L_* against the dense integral of its SED.

    ``over`` are the configuration overrides (the non-default option under test); the
    reference is built from the component's own variants that switch the other pieces off.
    """
    full = _grahsp_comp(**over)
    tpl = full.precompute()
    params = _grahsp_params(full)
    pub = _grahsp_published(full, params, _PUB_FILTERS, tpl)

    bbb_side = _grahsp_comp(**{**over, "include_torus": False, "apply_attenuation": False})
    torus_side = _grahsp_comp(
        **{
            **over,
            "include_bbb": False,
            "include_lines": False,
            "include_feii": False,
            "include_balmer": False,
            "apply_attenuation": False,
        }
    )
    intrinsic = _grahsp_comp(**{**over, "apply_attenuation": False})

    _, sed_bbb = _grahsp_apply(bbb_side, params, wide, tpl)
    _, sed_torus = _grahsp_apply(torus_side, params, wide, tpl)
    _, sed_int = _grahsp_apply(intrinsic, params, wide, tpl)
    _, sed_att = _grahsp_apply(full, params, wide, tpl)

    lyman = wide >= 912.0  # the Lyman-limit cut is a node: the trapezoid starts on it
    ref = {
        "L_agn_bol": _ln_nu_integral(sed_bbb[lyman], wide[lyman]),
        "L_agn_torus": _ln_nu_integral(sed_torus, wide),
        "L_agn_absorbed": _ln_nu_integral(sed_int - sed_att, wide),
    }
    for key in _GRAHSP_KEYS:
        assert ref[key] > 0.0, f"{key}: reference is not positive"
    return {key: pub[key] / ref[key] - 1.0 for key in _GRAHSP_KEYS}


def test_grahsp_published_L_equals_dense_integral_of_emitted_sed():
    """Each GRAHSP L_* equals the wide dense integral of the SED it defines (1e-6)."""
    errors = _grahsp_identity_errors({}, _GRAHSP_WIDE)
    for key, err in errors.items():
        assert abs(err) < _GRAHSP_RTOL, f"GRAHSP {key}: published-vs-emitted rel {err:.3e}"


# The non-default options. Each one swaps a tabulated piece, so the reference grid carries that
# template's nodes (and a point outside each end).
_GRAHSP_VARIANTS = {
    "mn12_torus": (
        {"torus_model": "mn12"},
        ("feii_wave_nm", "torus_mn12_wave_nm", "torus_mn12_si_wave_nm"),
    ),
    "netzer_disc": ({"disc_model": "netzer"}, ("feii_wave_nm", "disc_wave_nm")),
    "veroncetty_feii": ({"feii_template": "veroncetty2004"}, ("feii_vc04_wave_nm",)),
}


@pytest.fixture(params=sorted(_GRAHSP_VARIANTS))
def grahsp_variant(request):
    over, axes = _GRAHSP_VARIANTS[request.param]
    return request.param, over, axes


def test_grahsp_variant_published_L_independent_of_caller_grid(grahsp_variant):
    """MN12 torus, Netzer disc and Veron-Cetty FeII: L_* agree on dense, 40- and 12-point grids."""
    label, over, _ = grahsp_variant
    comp = _grahsp_comp(**over)
    tpl = comp.precompute()
    params = _grahsp_params(comp)
    dense = _grahsp_published(comp, params, _PUB_DENSE, tpl)
    for grid_label, grid in (("coarse-40", _PUB_COARSE), ("filter-12", _PUB_FILTERS)):
        other = _grahsp_published(comp, params, grid, tpl)
        for key in _GRAHSP_KEYS:
            err = abs(other[key] / dense[key] - 1.0)
            assert err < _GRAHSP_RTOL, f"GRAHSP[{label}] {key} {grid_label}: rel diff {err:.3e}"


def test_grahsp_variant_published_L_equals_dense_integral(grahsp_variant):
    """MN12 torus, Netzer disc, Veron-Cetty FeII: L_* equal the dense SED integral (1e-6)."""
    label, over, axes = grahsp_variant
    errors = _grahsp_identity_errors(over, _grahsp_wide(*axes))
    for key, err in errors.items():
        assert abs(err) < _GRAHSP_RTOL, (
            f"GRAHSP[{label}] {key}: published-vs-emitted rel {err:.3e}"
        )


_CORNER_RTOL = 1.0e-7


def test_grahsp_published_L_identity_at_the_attenuation_and_amplitude_corners():
    """L_agn_bol and L_agn_absorbed match the dense integral at every box corner (1e-7).

    The corners are those of the parameters that enter the attenuation (E(B-V), E(B-V)_AGN)
    and the line and FeII amplitudes (A_lines, A_FeII); the rest sit at their declared
    defaults, and L5100 scales every piece alike. The lines, FeII forest and continua are
    linear in the amplitudes, so the pieces are evaluated once at unit amplitude on the dense
    grid and combined per corner with the attenuation of that corner.
    """
    import itertools

    full = _grahsp_comp()
    tpl = full.precompute()
    W = _GRAHSP_WIDE
    only = dict(
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_torus=False,
        include_si=False,
        include_balmer=False,
        apply_attenuation=False,
    )
    unit = _grahsp_params(full)
    unit["agn_grahsp_a_lines"] = jnp.asarray(1.0, dtype=jnp.float64)
    unit["agn_grahsp_a_feii"] = jnp.asarray(1.0, dtype=jnp.float64)
    piece = {
        name: _grahsp_sed(_grahsp_comp(**{**only, **on}), unit, W, tpl)
        for name, on in (
            ("bbb", {"include_bbb": True}),
            ("lines", {"include_lines": True}),
            ("feii", {"include_feii": True}),
            ("torus", {"include_torus": True, "include_si": True}),
        )
    }
    bounds = {d.name: d.prior.bounds for d in full.declared_parameters()}
    names = ("agn_grahsp_ebv", "agn_grahsp_ebv_agn", "agn_grahsp_a_feii", "agn_grahsp_a_lines")
    lyman = W >= 912.0
    factor_cache = {}
    worst = {"L_agn_bol": 0.0, "L_agn_absorbed": 0.0}
    for corner in itertools.product(*(bounds[n] for n in names)):
        params = dict(unit)
        params.update({n: jnp.asarray(v, dtype=jnp.float64) for n, v in zip(names, corner)})
        a_lines, a_feii = float(params["agn_grahsp_a_lines"]), float(params["agn_grahsp_a_feii"])
        key = corner[:2]
        if key not in factor_cache:
            factor_cache[key] = np.asarray(full._attenuation(jnp.asarray(W) * 0.1, params))
        bbb_side = piece["bbb"] + a_lines * (piece["lines"] + a_feii * piece["feii"])
        intrinsic = bbb_side + piece["torus"]
        ref_bol = _ln_nu_integral(bbb_side[lyman], W[lyman])
        ref_abs = _ln_nu_integral(intrinsic * (1.0 - factor_cache[key]), W)
        pub = _grahsp_published(full, params, _PUB_FILTERS, tpl)
        worst["L_agn_bol"] = max(worst["L_agn_bol"], abs(pub["L_agn_bol"] / ref_bol - 1.0))
        worst["L_agn_absorbed"] = max(
            worst["L_agn_absorbed"], abs(pub["L_agn_absorbed"] / ref_abs - 1.0)
        )
    for key, err in worst.items():
        assert err < _CORNER_RTOL, f"GRAHSP {key}: worst corner rel {err:.3e}"


def _grahsp_binding_params(comp):
    """Declared defaults with a corner where the emitted max(si, -torus) clip binds."""
    params = _grahsp_params(comp)
    params.update(
        {
            "agn_grahsp_si": jnp.asarray(-4.0, dtype=jnp.float64),
            "agn_grahsp_cool_lam_um": jnp.asarray(30.0, dtype=jnp.float64),
            "agn_grahsp_cool_width": jnp.asarray(0.2, dtype=jnp.float64),
            "agn_grahsp_hot_lam_um": jnp.asarray(1.0, dtype=jnp.float64),
            "agn_grahsp_hot_width": jnp.asarray(0.2, dtype=jnp.float64),
            "agn_grahsp_hot_fcov": jnp.asarray(0.04, dtype=jnp.float64),
        }
    )
    return params


def test_grahsp_published_L_identity_where_si_clip_binds():
    """At a corner where the emitted max(si, -torus) binds, published L_* match the dense integral.

    The clip is a kink of the emitted torus term; the published values carry its correction
    over the Si windows. This test fails if that correction is dropped, and the corner is one
    where the emitted torus+Si reaches zero (the clip is active).
    """
    full = _grahsp_comp()
    tpl = full.precompute()
    params = _grahsp_binding_params(full)
    W = _GRAHSP_WIDE

    torus_att = _grahsp_comp(
        include_bbb=False, include_lines=False, include_feii=False, include_balmer=False
    )
    torus_int = _grahsp_comp(
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_balmer=False,
        apply_attenuation=False,
    )
    pub_torus = _grahsp_published(torus_att, params, _PUB_FILTERS, tpl)["L_agn_torus"]
    ref_torus = _ln_nu_integral(_grahsp_sed(torus_int, params, W, tpl), W)
    assert abs(pub_torus / ref_torus - 1.0) < _GRAHSP_RTOL, (
        f"L_agn_torus at the binding corner: rel {pub_torus / ref_torus - 1:.3e}"
    )

    full_int = _grahsp_comp(apply_attenuation=False)
    pub_abs = _grahsp_published(full, params, _PUB_FILTERS, tpl)["L_agn_absorbed"]
    ref_abs = _ln_nu_integral(
        _grahsp_sed(full_int, params, W, tpl) - _grahsp_sed(full, params, W, tpl), W
    )
    assert abs(pub_abs / ref_abs - 1.0) < _GRAHSP_RTOL, (
        f"L_agn_absorbed at the binding corner: rel {pub_abs / ref_abs - 1:.3e}"
    )


def test_grahsp_published_L_torus_clip_gate_never_skips_a_binding_clip():
    """Across random draws from the declared box, L_agn_torus equals the dense integral.

    The Si clip correction is skipped when a rigorous bound shows ``max(si, -torus)`` cannot
    bind. The draws cover both classes (bound and unbound), so a gate that skipped a binding
    clip would show as a miss against the dense integral of the emitted torus term, here built
    from the component's own torus and Si functions on a 200 000-point grid over the Si windows.
    """
    from tengri.components.agn.grahsp.torus import si_feature, torus_dust_continuum

    comp = _grahsp_comp(
        include_bbb=False,
        include_lines=False,
        include_feii=False,
        include_balmer=False,
        apply_attenuation=False,
    )
    tpl = comp.precompute()
    bounds = {d.name: d.prior.bounds for d in comp.declared_parameters()}
    rng = np.random.default_rng(20261012)
    lam_nm = np.linspace(3600.0, 21500.0, 200_001)
    n_bound = 0
    n_unbound = 0
    for _ in range(80):
        params = _grahsp_params(comp)
        for name in (
            "agn_grahsp_si",
            "agn_grahsp_cool_lam_um",
            "agn_grahsp_cool_width",
            "agn_grahsp_hot_lam_um",
            "agn_grahsp_hot_width",
        ):
            params[name] = jnp.asarray(rng.uniform(*bounds[name]), dtype=jnp.float64)
        lo, hi = np.log(bounds["agn_grahsp_hot_fcov"])
        params["agn_grahsp_hot_fcov"] = jnp.asarray(np.exp(rng.uniform(lo, hi)), dtype=jnp.float64)
        torus_kw = dict(
            wave_nm=jnp.asarray(lam_nm),
            l5100=params["agn_grahsp_l5100"],
            fcov=params["agn_grahsp_fcov"],
        )
        cont = torus_dust_continuum(
            **torus_kw,
            cool_lam_um=params["agn_grahsp_cool_lam_um"],
            cool_width=params["agn_grahsp_cool_width"],
            hot_lam_um=params["agn_grahsp_hot_lam_um"],
            hot_width=params["agn_grahsp_hot_width"],
            hot_fcov=params["agn_grahsp_hot_fcov"],
        )
        si_term = si_feature(**torus_kw, si=params["agn_grahsp_si"])
        clipped = np.asarray(jnp.maximum(si_term, -cont))
        correction = np.trapezoid(clipped - np.asarray(si_term), lam_nm)  # >= 0, erg/s
        n_bound += int(correction > 0.0)
        n_unbound += int(correction == 0.0)
        # The correction enters L_agn_torus on top of the closed forms, so compare the whole
        # published value with the same closed forms plus the dense correction.
        amp_cool, amp_hot, _, _ = comp._torus_terms(params)
        expected = float(amp_cool + amp_hot + comp._si_terms(params)[1]) + correction
        pub = float(_grahsp_published(comp, params, _PUB_FILTERS, tpl)["L_agn_torus"])
        assert abs(pub / expected - 1.0) < 1.0e-8, (
            f"L_agn_torus {pub / expected - 1:+.2e} (dense correction {correction:.3e})"
        )
    assert n_bound >= 5 and n_unbound >= 5, (
        f"vacuous draw set: bound {n_bound}, unbound {n_unbound}"
    )


def test_grahsp_feii_absorbed_piece_matches_independent_quadrature_at_the_corners():
    """The FeII x (1 - attenuation) piece equals 24-point Gauss-Legendre per segment (1e-10).

    The reference is a different rule at a different resolution from the one it checks (two
    points per linear segment), so it does not share that rule's error. The corners are those
    of the parameters that enter the attenuation and the FeII amplitude.
    """
    import itertools

    comp = _grahsp_comp(
        include_bbb=False,
        include_lines=False,
        include_torus=False,
        include_si=False,
        include_balmer=False,
    )
    tpl = comp.precompute()
    wave = np.asarray(tpl.feii_wave_nm, dtype=float)
    lumin = np.asarray(tpl.feii_lumin, dtype=float)
    x, wt = np.polynomial.legendre.leggauss(24)
    mid, half = 0.5 * (wave[:-1] + wave[1:]), 0.5 * (wave[1:] - wave[:-1])
    lam = mid[:, None] + half[:, None] * x[None, :]
    lum = (
        lumin[:-1, None]
        + (lumin[1:] - lumin[:-1])[:, None]
        * (lam - wave[:-1, None])
        / (wave[1:] - wave[:-1])[:, None]
    )
    bounds = {d.name: d.prior.bounds for d in comp.declared_parameters()}
    names = ("agn_grahsp_ebv", "agn_grahsp_ebv_agn", "agn_grahsp_a_feii", "agn_grahsp_a_lines")
    for corner in itertools.product(*(bounds[n] for n in names)):
        params = _grahsp_params(comp)
        params.update({n: jnp.asarray(v, dtype=jnp.float64) for n, v in zip(names, corner)})
        one_minus_f = 1.0 - np.asarray(comp._attenuation(jnp.asarray(lam.ravel()), params))
        ref = float(comp._feii_amplitude(params)) * float(
            np.sum(half[:, None] * wt[None, :] * lum * one_minus_f.reshape(lam.shape))
        )
        pub = _grahsp_published(comp, params, _PUB_FILTERS, tpl)["L_agn_absorbed"]
        assert abs(pub / ref - 1.0) < 1.0e-10, f"FeII piece at {corner}: rel {pub / ref - 1:+.2e}"
