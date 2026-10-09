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

from pathlib import Path

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


# -- published L_* diagnostics of the SEDModelComponent classes -----------------
#
# Each component publishes its bolometric luminosities from a fixed budget grid, so the
# value must not move with how the caller samples wavelength. A coarse caller grid (60
# log points over 0.1-1000 um, the span a catalog fit uses) and a fine one (6000 points)
# must publish the same L_* to 1e-6.

_L_COARSE = np.geomspace(1.0e3, 1.0e7, 60)  # 0.1-1000 um [A]
_L_FINE = np.geomspace(1.0e3, 1.0e7, 6000)

_DATA = Path(__file__).resolve().parents[2] / "data"


def _make_published_component(name):
    """Build one SEDModelComponent that reads its template library from ``data/``."""
    from tengri.components.agn.cat3d_torus_model import CAT3DTorus, CAT3DTorusConfig
    from tengri.components.agn.kd18_disc_model import KD18Disc
    from tengri.components.agn.powerlaw_disc_model import PowerLawDisc
    from tengri.components.agn.silva04_model import Silva04Torus, Silva04TorusConfig
    from tengri.components.agn.skirtor_agnfitter_model import (
        SKIRTORAgnfitterTorus,
        SKIRTORAgnfitterTorusConfig,
    )
    from tengri.components.agn.skirtor_model import SKIRTORTorus, SKIRTORTorusConfig

    makers = {
        "cat3d_wind": lambda: CAT3DTorus(
            config=CAT3DTorusConfig(grid_path=str(_DATA / "cat3d_wind_torus_grid.h5"))
        ),
        "kd18_disc": KD18Disc,
        "powerlaw_disc": PowerLawDisc,
        "silva04": lambda: Silva04Torus(
            config=Silva04TorusConfig(grid_path=str(_DATA / "silva04_torus_grid.h5"))
        ),
        "skirtor_agnfitter": lambda: SKIRTORAgnfitterTorus(
            config=SKIRTORAgnfitterTorusConfig(
                grid_path=str(_DATA / "skirtor_mean3p_torus_grid.h5")
            )
        ),
        "skirtor": lambda: SKIRTORTorus(
            config=SKIRTORTorusConfig(grid_path=str(_DATA / "skirtor_templates_v3.h5"))
        ),
    }
    return makers[name]()


def _grahsp_published(wave):
    """GRAHSP publishes through the ForwardState adapter, at its fiducial parameters."""
    from tengri.components.agn.grahsp import GRAHSPSEDComponent
    from tengri.protocols.component import ForwardState

    comp = GRAHSPSEDComponent()
    params = {
        "agn_grahsp_l5100": jnp.array(1.0e44),
        "agn_grahsp_uvslope": jnp.array(0.0),
        "agn_grahsp_plslope": jnp.array(-1.7),
        "agn_grahsp_plbendloc_nm": jnp.array(100.0),
        "agn_grahsp_plbendwidth": jnp.array(1.0),
        "agn_grahsp_cutoff_nm": jnp.array(10000.0),
        "agn_grahsp_a_lines": jnp.array(1.0),
        "agn_grahsp_a_feii": jnp.array(5.0),
        "agn_grahsp_linewidth_kms": jnp.array(5000.0),
        "agn_grahsp_fcov": jnp.array(0.4),
        "agn_grahsp_si": jnp.array(0.0),
        "agn_grahsp_cool_lam_um": jnp.array(17.0),
        "agn_grahsp_cool_width": jnp.array(0.45),
        "agn_grahsp_hot_lam_um": jnp.array(2.0),
        "agn_grahsp_hot_width": jnp.array(0.5),
        "agn_grahsp_hot_fcov": jnp.array(1.0),
        "agn_grahsp_ebv": jnp.array(0.05),
        "agn_grahsp_ebv_agn": jnp.array(0.05),
    }
    out = comp.apply(ForwardState(wave=jnp.asarray(wave)), params)
    return {k: float(np.asarray(v)) for k, v in out.derived.items() if k.startswith("L_")}


def _published_luminosities(name, wave):
    """The L_* dict a component publishes on ``wave``, at its declared default parameters."""
    from tengri.forward.orchestrator import default_params_dict

    if name == "grahsp":
        return _grahsp_published(wave)
    comp = _make_published_component(name)
    wave = jnp.asarray(wave)
    try:
        data = comp.load(wave)
    except (FileNotFoundError, OSError) as err:
        pytest.skip(f"{name}: template data unavailable ({err})")
    object.__setattr__(comp, "data", data)
    prefix = comp.parameter_prefix
    p = {k.removeprefix(prefix): v for k, v in default_params_dict([comp]).items()}
    _, published = comp.predict(p, jnp.zeros_like(wave), wave)
    return {k: float(np.asarray(v)) for k, v in published.items() if k.startswith("L_")}


@pytest.mark.parametrize(
    "name",
    [
        "cat3d_wind",
        "kd18_disc",
        "powerlaw_disc",
        "silva04",
        "skirtor_agnfitter",
        "skirtor",
        "grahsp",
    ],
)
def test_published_L_star_is_independent_of_caller_grid(name):
    """Published L_* on a coarse caller grid equals the value on a fine grid to 1e-6."""
    coarse = _published_luminosities(name, _L_COARSE)
    fine = _published_luminosities(name, _L_FINE)
    assert fine, f"{name}: publishes no L_* luminosity"
    for key, ref in fine.items():
        assert ref > 0.0, f"{name}.{key}: non-positive reference {ref:.3e}"
        rel = abs(coarse[key] / ref - 1.0)
        assert rel < _RTOL, f"{name}.{key}: coarse-vs-fine L_* rel diff {rel:.3e}"


# -- SKIRTOR torus carries its normalized power after resampling (#2319) ----------

#: Caller grid of the reproducer: 0.1-1000 um, 2000 log points, resampled off the
#: 136-node SKIRTOR template axis.
_SKIRTOR_CALLER = np.geomspace(1.0e3, 1.0e7, 2000)


@pytest.mark.parametrize("cos_inc", [1.0, float(np.cos(np.pi / 4.0)), 0.0])
def test_skirtor_torus_integral_equals_its_target_on_caller_grid(cos_inc):
    """The resampled torus integrates to its enforced bolometric power to 1e-4.

    The target is ``10**log_lbol * L_sun * frac_agn``. The integral is taken on the
    caller grid, the way a downstream consumer would integrate the SED.
    """
    from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu
    from tengri.components.agn.skirtor import create_skirtor_components_from_grid
    from tengri.utils.physics_constants import L_SUN

    grid = _DATA / "skirtor_templates_v3.h5"
    if not grid.is_file():
        pytest.skip(f"SKIRTOR v3 template grid unavailable ({grid})")
    components = create_skirtor_components_from_grid(str(grid))
    wave = jnp.asarray(_SKIRTOR_CALLER)
    out = components(wave, agn_log_lbol=10.0, agn_cos_inc=cos_inc, frac_agn=0.5)
    target = 10.0**10.0 * L_SUN * 0.5
    nu = wavelength_to_nu(wave)
    # The disc is hot and extends below this grid's 1000 A start, so only the torus
    # dust, which the caller grid covers, is checked here.
    power = float(np.asarray(jnp.abs(bolometric_integral_nu(out.dust, nu))))
    rel = abs(power / target - 1.0)
    assert rel < 1.0e-4, f"cos_inc={cos_inc:.3f} torus dust: rel error {rel:.3e}"


# -- SKIRTOR polar-dust absorbed reference is the converged integral (#2322) -------


def _fiducial_polar_templates():
    """The face-on disc, the i=30 dust and the norm ratio at the fiducial node, from the file."""
    import h5py

    from tengri.components.agn.skirtor import _find_skirtor_grid

    with h5py.File(_find_skirtor_grid(), "r") as f:
        wl = np.asarray(f["wavelength"][:])
        axes = {
            "tau": np.asarray(f["grid/tau_97"][:]),
            "p": np.asarray(f["grid/p"][:]),
            "q": np.asarray(f["grid/q"][:]),
            "oa": np.asarray(f["grid/opening_angle"][:]),
            "R": np.asarray(f["grid/radius_ratio"][:]),
            "cos_inc": np.asarray(f["grid/cos_inclination"][:]),
        }
        idx = (
            int(np.argmin(np.abs(axes["tau"] - 7.0))),
            int(np.argmin(np.abs(axes["p"] - 1.0))),
            int(np.argmin(np.abs(axes["q"] - 1.0))),
            int(np.argmin(np.abs(axes["oa"] - 40.0))),
            int(np.argmin(np.abs(axes["R"] - 20.0))),
        )
        j_0 = int(np.argmin(np.abs(axes["cos_inc"] - 1.0)))
        j_i = int(np.argmin(np.abs(axes["cos_inc"] - np.cos(np.deg2rad(30.0)))))
        disk0 = np.asarray(f["spectra/disk_emission"][(*idx, j_0)])
        dust_i = np.asarray(f["spectra/dust_emission"][(*idx, j_i)])
        norm = np.asarray(f["spectra/norm"][idx])
    return wl, disk0, dust_i, float(norm[j_0] / norm[j_i])


def test_skirtor_polar_reference_matches_converged_integral():
    """R_faceon equals a 40000-node integral of the same templates to 1e-6.

    The reference resamples the file's own templates onto 40000 log nodes and
    integrates them with the trapezoid rule, the converged value of the integrand the
    polar reference is defined by. The 136-node trapezoid it replaces is 6e-5 off.
    """
    from tengri.components.agn.skirtor import skirtor_disc_dust_ratio
    from tengri.utils.grid_interp import resample_template

    wl, disk0, dust_i, ratio = _fiducial_polar_templates()
    fine = np.geomspace(wl[0], wl[-1], 40000)

    def _on_fine(y):
        return np.asarray(
            resample_template(
                jnp.asarray(fine), jnp.asarray(wl), jnp.asarray(y), left=0.0, right=0.0
            )
        )

    ref = np.trapezoid(_on_fine(disk0), fine) * ratio / np.trapezoid(_on_fine(dust_i), fine)
    wave = jnp.asarray(np.geomspace(1.0e2, 1.0e8, 3000))
    got = float(
        skirtor_disc_dust_ratio(
            wave, jnp.ones_like(wave), jnp.ones_like(wave), agn_cos_inc=np.cos(np.deg2rad(30.0))
        ).R_faceon
    )
    rel = abs(got / ref - 1.0)
    assert rel < 1.0e-6, f"R_faceon(i=30) rel error vs 40000-node reference {rel:.3e}"
