# SPDX-License-Identifier: BSD-3-Clause
"""#2564: SEDModel-built torus / dust-emission SEDs match the block's native-grid evaluation.

Before the fix the AGNfitter-lineage torus blocks (``nenkova_agnfitter*``,
``skirtor_agnfitter*``, ``cat3d_wind_lowfwd``) and the tabulated dust-emission
models ``schreiber2018`` / ``dh02_ce01`` declared no native wavelength grid, so
``SEDModel.build`` evaluated them on the SSP grid (9 points above 10 um at
20 um spacing, ending at 160 um). Values at grid points were right, but the
sampling missed the IR peak and the 160-1000 um tail.

Conventions
-----------
* The "native" reference is the registered block / closure evaluated directly
  on its own template wavelength axis (``native_wave_*``), i.e. the exact
  function the model calls, with the same parameters.
* SED shape quantity is ``L_nu`` (the ``sed_agn_torus`` / ``sed_dust_ir``
  components), wavelengths in Angstrom, band edges in micron.
* A band mean is the mean of ``L_nu`` over ``ln(lambda)`` between the band
  edges (trapezoid in ``ln(lambda)`` on the grid's own points, band edges by
  log-log interpolation). Dust sub-bands are compared as fractions of the
  3-1000 um total.
* Three parity statements are checked per model: (i) the peak wavelength equals the
  dense-grid peak of the block to within one declared-grid step; (ii) at the native template nodes inside the
  band the SEDModel SED has the native shape (log-log interpolation of the
  SEDModel grid onto the nodes; exact when the master grid contains the nodes;
  see ``node_parity`` for the scale convention); (iii) the SEDModel band mean equals the band mean of the block
  evaluated on a dense reference grid (the quadrature-converged truth of the
  block's own interpolant). The native grids of the coarse blocks (105-136
  points) have a ~0.4 per cent trapezoid error against that truth, so the
  native-grid band mean itself is not the reference for (iii).
* Peak wavelength: argmax of ``L_nu`` over 1-1000 um.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, load_ssp_data
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.components.dust.emission.emission import DUST_EMISSION_MODELS
from tengri.forward.wavelength_extension import (
    native_wave_agn_torus,
    native_wave_dust_emission,
)

pytestmark = pytest.mark.crossval

_LSUN = 3.828e33  # erg/s, converts the model's l_* properties [Lsun] to erg/s
_UM = 1.0e4  # Angstrom per micron
_TORUS_BAND_UM = (8.0, 500.0)
_DUST_BAND_UM = (3.0, 1000.0)
_PEAK_BAND_UM = (1.0, 1000.0)

# Tolerances, measured after the fix (see the #2564 commit message).
_NODE_RTOL = 1.0e-8  # (ii) shape at the declared nodes; measured worst 1.4e-14 (float64)
_TORUS_BAND_RTOL = 5.0e-3  # (iii) measured worst 2.4e-3 (the 105-136 node grids: quadrature limit)
_DUST_BAND_RTOL = 2.0e-3  # (iii) measured worst 8e-4
_N_DENSE = 20000

_COS30 = float(np.cos(np.deg2rad(30.0)))
# (block, block kwargs). Kwargs use the block-function names; the SEDModel
# torus dict takes the same names without the ``agn_`` prefix.
_TORUS_CASES = [
    ("nenkova_agnfitter", {"agn_cos_inc": _COS30}),
    ("nenkova_agnfitter_2p", {"agn_cos_inc": _COS30, "agn_oa_nenkova": 40.0}),
    ("nenkova_agnfitter_3p", {"agn_cos_inc": _COS30, "agn_oa_nenkova": 40.0, "agn_tv_nenkova": 60.0}),
    ("skirtor_agnfitter", {"agn_oa_skirtor": 40.0, "agn_incl_skirtor": 30.0, "agn_tv_skirtor": 7.0}),
    ("skirtor_agnfitter_1p", {"agn_incl_skirtor": 30.0}),
    ("skirtor_agnfitter_2p", {"agn_oa_skirtor": 40.0, "agn_incl_skirtor": 30.0}),
    ("cat3d_wind_lowfwd", {"agn_cos_inc": _COS30}),
]
_DUST_CASES = [
    ("schreiber2018", {"dust_T": 30.0, "dust_f_pah": 0.03}),
    ("schreiber2018", {"dust_T": 45.0, "dust_f_pah": 0.05}),
    ("dh02_ce01", {}),
]

_SFH = {
    "type": "declining_exp",
    "sfh_declining_exp_tau_gyr": Fixed(0.3),
    "sfh_declining_exp_age_gyr": Fixed(1.0),
    "sfh_declining_exp_log_total_mass": Fixed(10.5),
    "all_params": Fixed(DEFAULT),
}


@pytest.fixture(scope="module")
def ssp():
    return load_ssp_data("data/bc03_pdva_stelib_chabrier.h5")


def band_mean(wave_aa, lnu, lo_um, hi_um):
    """Mean of ``lnu`` over ln(lambda) in [lo_um, hi_um], own grid points plus interpolated edges."""
    w = np.asarray(wave_aa, float)
    y = np.asarray(lnu, float)
    order = np.argsort(w)
    w, y = w[order], y[order]
    lo, hi = lo_um * _UM, hi_um * _UM
    inside = (w > lo) & (w < hi)
    edges = np.array([lo, hi])
    y_edges = np.exp(np.interp(np.log(edges), np.log(w), np.log(np.clip(y, 1e-300, None))))
    ww = np.concatenate([edges[:1], w[inside], edges[1:]])
    yy = np.concatenate([y_edges[:1], y[inside], y_edges[1:]])
    lw = np.log(ww)
    return float(np.trapezoid(yy, lw) / (lw[-1] - lw[0]))


def peak_um(wave_aa, lnu):
    """(peak wavelength [um], grid step in ln(lambda) at the peak) over 1-1000 um."""
    w = np.asarray(wave_aa, float)
    y = np.asarray(lnu, float)
    m = (w >= _PEAK_BAND_UM[0] * _UM) & (w <= _PEAK_BAND_UM[1] * _UM)
    ws, ys = w[m], y[m]
    i = int(np.argmax(ys))
    hi = min(i + 1, len(ws) - 1)
    lo = max(i - 1, 0)
    step = max(np.log(ws[hi] / ws[i]), np.log(ws[i] / ws[lo]))
    return float(ws[i] / _UM), float(step)


def _predict_torus(ssp, block, kw):
    torus = {"type": block, "all_params": Fixed(DEFAULT)}
    torus.update({k.removeprefix("agn_"): Fixed(v) for k, v in kw.items()})
    model = SEDModel.build(
        ssp_data=ssp,
        sfh=_SFH,
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "disc": {"type": "none"},
            "torus": torus,
            "nlr": {"type": "none"},
            "blr": {"type": "none"},
            "atten": {"type": "none"},
            "agn_log_lbol": Fixed(11.0),
            "all_params": Fixed(DEFAULT),
            "norm": "independent",
        },
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict({})
    return (
        np.asarray(pred.sed.components["wavelength"], float),
        np.asarray(pred.sed.components["sed_agn_torus"], float),
    )


def _eval_torus(block, kw, w):
    """Block evaluated on wavelength grid ``w`` [A]; returns ``L_nu`` (the runner converts L_lambda with lambda^2 / c)."""
    w = np.asarray(w, float)
    l_lambda = AGN_BLOCKS["torus"][block](jnp.asarray(w), 11.0, 0.0, **kw)
    return np.asarray(l_lambda, float) * w**2 / 2.99792458e18


def node_parity(w_m, l_m, w_n, l_n, lo_um, hi_um):
    """Shape deviation at the native nodes inside the band.

    ``r = L_model(node) / L_native(node)``; returns ``max |r / median(r) - 1|``.
    The blocks normalise their SED to the AGN / absorbed-energy budget by a
    trapezoid over the grid they are handed, so a coarse native grid carries a
    constant quadrature offset in scale (~0.4 per cent for the 105-136 point
    grids); the *shape* at the nodes must match exactly. The absolute scale is
    tested against the dense-grid reference by the band-mean checks.
    """
    w_n = np.asarray(w_n, float)
    sel = (w_n >= lo_um * _UM) & (w_n <= hi_um * _UM) & (np.asarray(l_n) > 0)
    o = np.argsort(w_m)
    lm_at = np.exp(
        np.interp(np.log(w_n[sel]), np.log(np.asarray(w_m)[o]), np.log(np.clip(np.asarray(l_m)[o], 1e-300, None)))
    )
    r = lm_at / np.asarray(l_n)[sel]
    return float(np.max(np.abs(r / np.median(r) - 1.0)))


def _dense(block_wave):
    return np.geomspace(block_wave.min(), block_wave.max(), _N_DENSE)


@pytest.mark.parametrize(("block", "kw"), _TORUS_CASES, ids=[c[0] for c in _TORUS_CASES])
def test_torus_sedmodel_matches_native_grid(ssp, block, kw):
    w_m, l_m = _predict_torus(ssp, block, kw)
    w_n = native_wave_agn_torus(block)
    assert w_n is not None, f"{block}: no native grid declared"
    w_n = np.asarray(w_n, float)
    l_n = _eval_torus(block, kw, w_n)

    w_d = _dense(w_n)
    l_d = _eval_torus(block, kw, w_d)
    pk_m, _ = peak_um(w_m, l_m)
    pk_d, _ = peak_um(w_d, l_d)
    _, step_n = peak_um(w_n, l_n)
    assert abs(np.log(pk_m / pk_d)) <= step_n, (
        f"{block}: SEDModel peak {pk_m:.3f} um vs dense {pk_d:.3f} um (declared step {step_n:.4f} in ln lambda)"
    )

    dev = node_parity(w_m, l_m, w_n, l_n, *_TORUS_BAND_UM)
    assert dev <= _NODE_RTOL, f"{block}: max node deviation in 8-500 um = {dev:.3e}"

    bm = band_mean(w_m, l_m, *_TORUS_BAND_UM)
    bd = band_mean(w_d, l_d, *_TORUS_BAND_UM)
    assert bm == pytest.approx(bd, rel=_TORUS_BAND_RTOL), f"{block}: 8-500 um band mean SEDModel/dense = {bm / bd:.5f}"


def _predict_dust(ssp, name, kw):
    model = SEDModel.build(
        ssp_data=ssp,
        sfh=_SFH,
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "dust_tau_v": Fixed(1.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_emission={"type": name, **{k: Fixed(v) for k, v in kw.items()}, "all_params": Fixed(DEFAULT)},
        neb={"type": "none"},
        redshift=Fixed(0.0),
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pred = model.predict({})
    return (
        np.asarray(pred.sed.components["wavelength"], float),
        np.asarray(pred.sed.components["sed_dust_ir"], float),
        pred,
    )


def _eval_dust(name, kw, pred, w):
    """Dust closure on grid ``w`` [A] with the model's own absorbed / IR budgets (erg/s); returns ``L_nu``."""
    l_abs = float(pred.properties["l_dust_absorbed"]) * _LSUN
    extra = {"log_L_ir": float(np.log10(float(pred.properties["l_dust_absorbed"]) * _LSUN))} if name == "dh02_ce01" else {}
    return np.asarray(DUST_EMISSION_MODELS[name](jnp.asarray(w), l_abs, **kw, **extra), float)


_DUST_SUBBANDS_UM = [(3.0, 8.0), (8.0, 30.0), (30.0, 100.0), (100.0, 160.0), (160.0, 1000.0)]


@pytest.mark.parametrize(
    ("name", "kw"),
    _DUST_CASES,
    ids=[f"{n}-{'-'.join(f'{k}{v:g}' for k, v in kw.items()) or 'default'}" for n, kw in _DUST_CASES],
)
def test_dust_sedmodel_matches_native_grid(ssp, name, kw):
    w_m, l_m, pred = _predict_dust(ssp, name, kw)
    w_n = native_wave_dust_emission(name)
    assert w_n is not None, f"{name}: no native grid declared"
    w_n = np.asarray(w_n, float)
    l_n = _eval_dust(name, kw, pred, w_n)

    w_d = _dense(w_n)
    l_d = _eval_dust(name, kw, pred, w_d)
    pk_m, _ = peak_um(w_m, l_m)
    pk_d, _ = peak_um(w_d, l_d)
    _, step_n = peak_um(w_n, l_n)
    assert abs(np.log(pk_m / pk_d)) <= step_n, (
        f"{name}: SEDModel peak {pk_m:.3f} um vs dense {pk_d:.3f} um (declared step {step_n:.4f})"
    )

    dev = node_parity(w_m, l_m, w_n, l_n, *_DUST_BAND_UM)
    assert dev <= _NODE_RTOL, f"{name}: max node deviation in 3-1000 um = {dev:.3e}"

    tot_m = band_mean(w_m, l_m, *_DUST_BAND_UM)
    tot_d = band_mean(w_d, l_d, *_DUST_BAND_UM)
    assert tot_m == pytest.approx(tot_d, rel=_DUST_BAND_RTOL), f"{name}: 3-1000 um mean SEDModel/dense = {tot_m / tot_d:.5f}"
    for lo, hi in _DUST_SUBBANDS_UM:
        frac_m = band_mean(w_m, l_m, lo, hi) / tot_m
        frac_d = band_mean(w_d, l_d, lo, hi) / tot_d
        assert frac_m == pytest.approx(frac_d, rel=_DUST_BAND_RTOL), (
            f"{name}: {lo:g}-{hi:g} um band fraction SEDModel/dense = {frac_m / frac_d:.5f}"
        )
