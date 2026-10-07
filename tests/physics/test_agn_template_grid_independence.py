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
