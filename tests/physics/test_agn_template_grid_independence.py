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
# The ``outputs`` a component publishes are part of its physics: a published
# bolometric luminosity is an integral over the component's own support, not over
# whatever wavelength array the caller passed. Each published key is evaluated on
# three grids and must agree to the same 1e-6 as the SED tests above.

_PUB_DENSE = np.geomspace(10.0, 1.0e7, 4000)  # 10 A - 1 mm, dense log grid
_PUB_COARSE = np.geomspace(10.0, 1.0e7, 40)  # 40-point grid
# Filter-effective-wavelength-like grid, 12 points [A].
_PUB_FILTERS = np.array(
    [
        *(1528.0, 2271.0, 3551.0, 4686.0, 6166.0, 7480.0, 8932.0, 12350.0, 16620.0, 21590.0),
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


def _component_publisher(module, cls_name, cfg_name, key, grid_file=None):
    """Return ``fn(wave) -> published[key]`` for one SEDModelComponent class."""
    import importlib

    mod = importlib.import_module(f"tengri.components.agn.{module}")
    cls = getattr(mod, cls_name)
    cfg_cls = getattr(mod, cfg_name)
    kwargs = {} if grid_file is None else {"grid_path": _data_file(grid_file)}
    comp = cls(config=cfg_cls(**kwargs))
    # Frozen dataclasses refuse plain assignment; the pipeline stores ``data`` the same way.
    object.__setattr__(comp, "data", comp.load(jnp.asarray(_PUB_DENSE)))
    prefix = comp.parameter_prefix
    params = {
        (decl.name[len(prefix) :] if decl.name.startswith(prefix) else decl.name): jnp.asarray(
            decl.prior.default, dtype=jnp.float64
        )
        for decl in comp.declared_parameters()
    }

    # The pipeline hands the disc block its luminosity fraction as ``frac``.
    if "lum_ratio" in params:
        params["frac"] = params.pop("lum_ratio")

    def fn(wave):
        wave = jnp.asarray(wave)
        _, published = comp.predict(params, jnp.zeros_like(wave), wave)
        return float(published[key])

    return fn


_SKIRTOR = ("skirtor_model", "SKIRTORTorus", "SKIRTORTorusConfig")
_PUBLISHED_CASES = {
    "silva04_torus.L_agn_torus": (
        "silva04_model",
        "Silva04Torus",
        "Silva04TorusConfig",
        "L_agn_torus",
        "silva04_torus_grid.h5",
    ),
    "cat3d_torus.L_agn_torus": (
        "cat3d_torus_model",
        "CAT3DTorus",
        "CAT3DTorusConfig",
        "L_agn_torus",
        "cat3d_wind_torus_grid.h5",
    ),
    "skirtor_agnfitter_torus.L_agn_torus": (
        "skirtor_agnfitter_model",
        "SKIRTORAgnfitterTorus",
        "SKIRTORAgnfitterTorusConfig",
        "L_agn_torus",
        "skirtor_mean3p_torus_grid.h5",
    ),
    "powerlaw_disc.L_agn_disc": (
        "powerlaw_disc_model",
        "PowerLawDisc",
        "PowerLawDiscConfig",
        "L_agn_disc",
        None,
    ),
    "kd18_disc.L_agn_disc": ("kd18_disc_model", "KD18Disc", "KD18DiscConfig", "L_agn_disc", None),
    "skirtor.L_agn_disc": (*_SKIRTOR, "L_agn_disc", "skirtor_templates_v3.h5"),
    "skirtor.L_agn_torus": (*_SKIRTOR, "L_agn_torus", "skirtor_templates_v3.h5"),
    "skirtor.L_agn_polar_dust": (*_SKIRTOR, "L_agn_polar_dust", "skirtor_templates_v3.h5"),
    "skirtor.L_2500_30deg": (*_SKIRTOR, "L_2500_30deg", "skirtor_templates_v3.h5"),
    "skirtor.L_6um": (*_SKIRTOR, "L_6um", "skirtor_templates_v3.h5"),
    "skirtor.L_12um": (*_SKIRTOR, "L_12um", "skirtor_templates_v3.h5"),
}


@pytest.fixture(scope="module", params=sorted(_PUBLISHED_CASES))
def published_case(request):
    try:
        fn = _component_publisher(*_PUBLISHED_CASES[request.param])
    except (FileNotFoundError, ImportError, OSError) as err:
        pytest.skip(f"{request.param}: template data unavailable ({err})")
    return {"name": request.param, "values": [fn(grid) for grid in _PUB_GRIDS]}


def test_published_luminosity_independent_of_caller_grid(published_case):
    """Dense, 40-point and filter-wavelength grids publish the same L_* to 1e-6."""
    ref, *others = published_case["values"]
    name = published_case["name"]
    assert np.isfinite(ref) and ref > 0.0, f"{name}: dense-grid value {ref}"
    for label, val in zip(("coarse-40", "filter-12"), others, strict=True):
        err = abs(val / ref - 1.0)
        assert err < _RTOL, f"{name} {label}: rel diff {err:.3e} (dense {ref:.6e})"
