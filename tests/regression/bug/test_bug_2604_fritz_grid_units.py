# SPDX-License-Identifier: BSD-3-Clause
r"""The Fritz et al. (2006) torus template is a luminosity per unit wavelength.

The shipped ``data/fritz2006_torus_grid.h5`` holds CIGALE's ``model.dust`` and
``model.disk`` arrays [1]_: :math:`L_\lambda` [W/nm]. ``dust`` is normalized to
unit integral over wavelength; ``disk`` is in the same units relative to that
dust. The loader converts them with

.. math::

    L_\nu = L_\lambda \, \lambda^2 / c

and normalizes :math:`\int L_\nu \, d\nu` over the template's own wavelength grid
to the requested luminosity (never over the caller's grid), so the
torus power is the template's own and the energy sits at the library's
wavelengths. Reading the array as :math:`L_\nu` instead multiplies the SED by
:math:`\lambda^{-2}`: the 1-3 micron band then holds 76 % of the power in place
of 28 %, and the median power wavelength is 1.8 micron in place of 5.8 micron
at the CIGALE default node (Fritz et al. 2006 [2]_ tabulate the torus as a
luminosity spectrum).

Four groups of cells:

* ``test_raw_*``: exact conversion. The raw template of a node, read with h5py,
  goes through the loader's normalization function on a grid whose nodes all
  carry that one template, so no parameter-space interpolation enters, and is
  compared with an independent numpy ``raw * lambda^2 / c``.
* ``*_pcigale``: shape against CIGALE's own library read directly, at four
  nodes. The bounds (12 % per band holding at least 10 % of the torus power,
  10 % in the median power wavelength) are measured values at those four nodes
  and are not a bound on the lookup residual of the six-dimensional triweight
  kernel, which reaches 28 % per band and 18 % in the median elsewhere on the
  grid (``test_lookup_residual_*``).
* ``*_pinned``: the same checks against five band fractions per node written
  down from CIGALE, so they run without it.
* ``test_model_*`` and the power, dtype and gradient cells: the public
  ``SEDModel`` path.

References
----------
.. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission,"
   A&A, 622, A103 (2019). arXiv:1811.03094.
   https://doi.org/10.1051/0004-6361/201834156
.. [2] J. Fritz, A. Franceschini and E. Hatziminaoglou, "Revisiting the
   infrared spectra of active galactic nuclei with a new torus emission
   model," MNRAS, 366, 767 (2006). arXiv:astro-ph/0511428.
   https://doi.org/10.1111/j.1365-2966.2006.09866.x
"""

from __future__ import annotations

import warnings
from pathlib import Path

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn import fritz as FR
from tengri.utils.interpolation import edges_for_grid
from tengri.utils.physics_constants import C_AA, L_SUN
from tests._grad_parity import assert_grad_matches_fd

pytestmark = pytest.mark.regression_bug

_GRID = Path(__file__).resolve().parents[3] / "data" / "fritz2006_torus_grid.h5"

#: Template tables: (r_ratio, tau, beta, gamma, Theta [pcigale full opening
#: angle, deg], psy [deg]). Theta = 60/100/140 are the three opening angles of
#: CIGALE's module; the library and the loader index them by the half-angle
#: (180 - Theta) / 2 = 60/40/20.
_NODES = (
    (60.0, 1.0, -0.5, 4.0, 100.0, 50.1),
    (60.0, 1.0, -0.5, 4.0, 60.0, 89.99),
    (150.0, 10.0, -1.0, 6.0, 60.0, 80.1),
    (10.0, 0.1, 0.0, 0.0, 140.0, 30.1),
)
_NODE_IDS = tuple(f"r{n[0]:g}-tau{n[1]:g}-Th{n[4]:g}-psy{n[5]:g}" for n in _NODES)

#: Band edges [micron] for the band fractions of the torus power.
_BANDS = ((1.0, 3.0), (3.0, 8.0), (8.0, 20.0), (20.0, 50.0), (50.0, 1000.0))

#: Fractions of the torus power per band and median power wavelength [micron]
#: of CIGALE's ``model.dust`` at each node of ``_NODES`` (log-log resampled onto
#: 0.1 - 1000 micron, 4000 log-spaced points).
_PCIGALE_LITERALS = (
    ((0.28259, 0.28245, 0.27827, 0.13509, 0.01248), 5.8135),
    ((0.34151, 0.28920, 0.23413, 0.11115, 0.01159), 4.4505),
    ((0.20162, 0.34157, 0.24418, 0.16408, 0.04406), 6.6905),
    ((0.23359, 0.41999, 0.31758, 0.02452, 0.00059), 5.0749),
)

_BAND_RTOL = 0.12
_MEDIAN_RTOL = 0.10
_SIGNIFICANT_BAND = 0.10

_WAVE_NM = np.logspace(2.0, 6.0, 4000)  # 0.1 - 1000 micron


def _half_angle(theta: float) -> float:
    return (180.0 - theta) / 2.0


def _require_grid() -> None:
    if not _GRID.is_file():
        pytest.skip(f"Fritz grid not found at {_GRID}")


def _resample(wave_out: np.ndarray, wave_in: np.ndarray, flux: np.ndarray) -> np.ndarray:
    """Power-law interpolation of a tabulated spectrum, zero outside its range.

    Linear in ``log(wave)`` and ``log(flux)`` where both neighboring nodes are
    positive, linear in ``flux`` otherwise (the template holds exact zeros).
    """
    lx = np.log(wave_in)
    lq = np.log(wave_out)
    i = np.clip(np.searchsorted(lx, lq) - 1, 0, wave_in.size - 2)
    t = np.clip((lq - lx[i]) / (lx[i + 1] - lx[i]), 0.0, 1.0)
    y0, y1 = flux[i], flux[i + 1]
    both = (y0 > 0.0) & (y1 > 0.0)
    ly0 = np.log(np.where(both, y0, 1.0))
    ly1 = np.log(np.where(both, y1, 1.0))
    out = np.where(both, np.exp(ly0 + t * (ly1 - ly0)), y0 + t * (y1 - y0))
    return np.where((wave_out < wave_in[0]) | (wave_out > wave_in[-1]), 0.0, out)


def _power_nu(lnu: np.ndarray, wave_aa: np.ndarray) -> float:
    """The loader's integral: trapezoid of L_nu over nu, ascending wavelength in."""
    nu = C_AA / wave_aa
    return float(-np.trapezoid(lnu, nu))


def _loglog_integral(x: np.ndarray, y: np.ndarray) -> float:
    """Integral of ``y`` over ``x`` with a power law between nodes (linear where a node is 0)."""
    x0, x1, y0, y1 = x[:-1], x[1:], y[:-1], y[1:]
    pos = (y0 > 0.0) & (y1 > 0.0)
    y0s, y1s = np.where(pos, y0, 1.0), np.where(pos, y1, 1.0)
    a = np.log(x1 / x0)
    u = a + np.log(y1s) - np.log(y0s)
    ratio = np.where(np.abs(u) < 1e-7, 1.0 + 0.5 * u, np.expm1(u) / np.where(u == 0.0, 1.0, u))
    return float(np.sum(np.where(pos, x0 * y0s * a * ratio, 0.5 * (y0 + y1) * (x1 - x0))))


def _native_power_nu(lnu: np.ndarray, wave_aa: np.ndarray) -> float:
    """Power of ``lnu`` read at the template nodes: L_nu dnu = L_nu (c / lambda^2) dlambda."""
    return _loglog_integral(wave_aa, lnu * C_AA / wave_aa**2)


def _native_wave_aa() -> np.ndarray:
    """The template's own wavelength nodes [Angstrom]."""
    with h5py.File(_GRID, "r") as f:
        return np.asarray(f["fritz2006/wavelength_aa"][:])


def _band_fractions(l_lam: np.ndarray, wave_nm: np.ndarray) -> np.ndarray:
    total = np.trapezoid(l_lam, wave_nm)
    out = []
    for lo, hi in _BANDS:
        sel = (wave_nm >= lo * 1e3) & (wave_nm <= hi * 1e3)
        out.append(np.trapezoid(l_lam[sel], wave_nm[sel]))
    return np.asarray(out) / total


def _median_power_um(l_lam: np.ndarray, wave_nm: np.ndarray) -> float:
    cum = np.cumsum(l_lam * np.gradient(wave_nm))
    return float(wave_nm[np.searchsorted(cum / cum[-1], 0.5)] / 1e3)


def _tengri_l_lambda(node: tuple, wave_nm: np.ndarray = _WAVE_NM) -> np.ndarray:
    """Public ``fritz_sed`` at ``node`` (unit torus power), as L_lambda."""
    r, tau, beta, gamma, theta, psy = node
    wave_aa = wave_nm * 10.0
    lnu = np.asarray(
        FR.fritz_sed(
            jnp.asarray(wave_aa),
            agn_log_lbol=0.0,
            agn_torus_frac=1.0,
            agn_fritz_r_ratio=r,
            agn_fritz_tau=tau,
            agn_fritz_beta=beta,
            agn_fritz_gamma=gamma,
            agn_fritz_oa=_half_angle(theta),
            agn_fritz_psy=psy,
        )
    )
    return lnu * C_AA / wave_aa**2


def _raw_template(node: tuple, key: str) -> tuple[np.ndarray, np.ndarray, tuple]:
    """Raw ``key`` ('dust' or 'disk') template of ``node`` from the h5 file."""
    r, tau, beta, gamma, theta, psy = node
    with h5py.File(_GRID, "r") as f:
        g = f["fritz2006"]
        axes = tuple(
            np.asarray(g[f"{n}_axis"][:])
            for n in ("r_ratio", "tau", "beta", "gamma", "opening_angle", "psy")
        )
        values = (r, tau, beta, gamma, _half_angle(theta), psy)
        idx = tuple(int(np.argmin(np.abs(ax - v))) for ax, v in zip(axes, values, strict=True))
        for ax, v, i in zip(axes, values, idx, strict=True):
            assert ax[i] == pytest.approx(v, rel=1e-9), "node is not on the grid axes"
        return np.asarray(g[key][idx]), np.asarray(g["wavelength_aa"][:]), axes


def _loader_on_one_template(raw, wave_grid, axes, wave_out):
    """``_interpolate_and_normalize`` at unit power on a grid of identical templates."""
    sub_axes = tuple(jnp.asarray(ax[:3]) for ax in axes)
    grid = jnp.asarray(np.broadcast_to(raw, (3,) * 6 + raw.shape).copy())
    point = tuple(float(ax[1]) for ax in sub_axes)
    return np.asarray(
        FR._interpolate_and_normalize(
            grid,
            jnp.asarray(wave_grid),
            sub_axes,
            tuple(edges_for_grid(ax) for ax in sub_axes),
            jnp.asarray(wave_out),
            point,
            1.0,
        )
    )


def _expected_lnu(raw, wave_grid, wave_out):
    """numpy ``raw * lambda^2 / c`` on ``wave_out``, unit power over the template's native grid."""
    l_lam = _resample(wave_out, wave_grid, raw)
    lnu = l_lam * wave_out**2 / C_AA
    return lnu / _loglog_integral(wave_grid, raw)


# -- a. exact conversion ---------------------------------------------------


@pytest.mark.parametrize("key", ["dust", "disk"])
@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_raw_template_converted_as_l_lambda_exactly(node, key):
    """``_interpolate_and_normalize`` returns ``raw * lambda^2 / c`` at unit native power.

    ``_interpolate_and_normalize`` converts the dust array (the torus, used by
    ``fritz_sed``) and the disc array (used by ``fritz_components``, which calls
    the same function for both); both are CIGALE L_lambda tables.
    """
    _require_grid()
    raw, wave_grid, axes = _raw_template(node, key)
    wave_out = np.logspace(2.0, 6.9, 3000)  # Angstrom, inside the template range
    got = _loader_on_one_template(raw, wave_grid, axes, wave_out)
    want = _expected_lnu(raw, wave_grid, wave_out)
    peak = np.max(want)
    np.testing.assert_allclose(got, want, rtol=1e-6, atol=1e-9 * peak)
    # unit power is the template's own: read it off at the template's nodes
    at_nodes = _loader_on_one_template(raw, wave_grid, axes, wave_grid)
    assert _native_power_nu(at_nodes, wave_grid) == pytest.approx(1.0, rel=1e-6)


def test_fritz_components_disc_and_dust_share_the_conversion():
    """Both ``fritz_components`` arrays carry unit power and ``dust`` is ``fritz_sed``."""
    _require_grid()
    wave = _native_wave_aa()
    kw = {
        "agn_log_lbol": 0.0,
        "agn_torus_frac": 1.0,
        "agn_fritz_r_ratio": 60.0,
        "agn_fritz_tau": 1.0,
        "agn_fritz_beta": -0.5,
        "agn_fritz_gamma": 4.0,
        "agn_fritz_oa": 40.0,
        "agn_fritz_psy": 50.1,
    }
    comp = FR.fritz_components(jnp.asarray(wave), **kw)
    for arr in (comp.disk, comp.dust):
        assert _native_power_nu(np.asarray(arr), wave) == pytest.approx(L_SUN, rel=1e-6)
    np.testing.assert_allclose(
        np.asarray(comp.dust), np.asarray(FR.fritz_sed(jnp.asarray(wave), **kw)), rtol=1e-12
    )


# -- b. shape against CIGALE ---------------------------------------------


def _assert_shape(tengri_l_lam, fractions, median_um, label):
    """Bands holding >= 10 % of the power within 12 %, median power wavelength within 10 %."""
    got = _band_fractions(tengri_l_lam, _WAVE_NM)
    want = np.asarray(fractions)
    sig = want >= _SIGNIFICANT_BAND
    np.testing.assert_allclose(
        got[sig], want[sig], rtol=_BAND_RTOL, err_msg=f"{label}: band fractions {got} vs {want}"
    )
    med = _median_power_um(tengri_l_lam, _WAVE_NM)
    assert med == pytest.approx(median_um, rel=_MEDIAN_RTOL), f"{label}: median {med} um"


@pytest.mark.parametrize(("node", "literal"), zip(_NODES, _PCIGALE_LITERALS), ids=_NODE_IDS)
def test_shape_matches_pcigale_pinned(node, literal):
    """The public torus shape against the band fractions of CIGALE's library.

    The bounds are the values measured at these four nodes, where the triweight
    kernel mixes neighboring tables; they are not a bound on the lookup
    residual (see ``test_lookup_residual_*``).
    Without the ``lambda^2 / c`` factor the 1-3 micron band is 2.2-3.5 times too
    high and the 8-20 micron band 8-16 times too low, far outside it.
    """
    _require_grid()
    _assert_shape(_tengri_l_lambda(node), literal[0], literal[1], str(node))


@pytest.mark.parametrize(("node", "literal"), zip(_NODES, _PCIGALE_LITERALS), ids=_NODE_IDS)
def test_shape_matches_pcigale_library(node, literal):
    """As the pinned twin, against ``SimpleDatabase('fritz2006')`` read here.

    Also checks the angle mapping on the node axes (CIGALE Theta against the
    half-angle of the library) and that the pinned literals are CIGALE's.
    """
    _require_grid()
    pytest.importorskip("pcigale")
    from pcigale.data import SimpleDatabase

    r, tau, beta, gamma, theta, psy = node
    half = _half_angle(theta)
    with SimpleDatabase("fritz2006") as db:
        entry = db.get(r_ratio=r, tau=tau, beta=beta, gamma=gamma, opening_angle=half, psy=psy)
    assert entry.opening_angle == pytest.approx(half)
    # Mapping on the node axes: the library table at (Theta -> half) is the h5 table.
    raw, wave_grid, axes = _raw_template(node, "dust")
    np.testing.assert_allclose(entry.wl * 10.0, wave_grid, rtol=1e-9)
    np.testing.assert_allclose(entry.dust, raw, rtol=1e-4, atol=1e-9 * np.max(raw))
    assert {_half_angle(t) for t in (60.0, 100.0, 140.0)} == set(axes[4])

    pc = _resample(_WAVE_NM, entry.wl, entry.dust)
    fracs = _band_fractions(pc, _WAVE_NM)
    med = _median_power_um(pc, _WAVE_NM)
    np.testing.assert_allclose(fracs, literal[0], atol=6e-6)
    assert med == pytest.approx(literal[1], abs=6e-5)
    _assert_shape(_tengri_l_lambda(node), fracs, med, str(node))


# -- c. public SEDModel path ------------------------------------------------

_MODEL_AGN_LOG_LBOL = 11.0


def _build_fritz_model(ssp, obs, node):
    r, tau, beta, gamma, theta, psy = node
    agn = {
        "type": "composable",
        "norm": "independent",
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {
            "type": "fritz",
            "agn_torus_frac": Fixed(1.0),
            "agn_fritz_r_ratio": Fixed(r),
            "agn_fritz_tau": Fixed(tau),
            "agn_fritz_beta": Fixed(beta),
            "agn_fritz_gamma": Fixed(gamma),
            "agn_fritz_oa": Fixed(_half_angle(theta)),
            "agn_fritz_psy": Fixed(psy),
        },
        "agn_log_lbol": Fixed(_MODEL_AGN_LOG_LBOL),
        "all_params": Fixed(DEFAULT),
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT), "log_total_mass": Fixed(10.0)},
            agn=agn,
            redshift=Fixed(0.1),
        )


@pytest.fixture(scope="module")
def _torus_states(synthetic_ssp_wide, synthetic_tophat_obs):
    """``derived`` of the built model at each node (built once, shared by cells)."""
    _require_grid()
    out = {}
    for node in _NODES:
        model = _build_fritz_model(synthetic_ssp_wide, synthetic_tophat_obs, node)
        state = model.predict_state({})
        out[node] = (np.asarray(state.wave), np.asarray(state.derived["sed_agn_torus"]))
    return out


@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_model_torus_median_power_wavelength_matches_pcigale(node, _torus_states):
    """``sed_agn_torus`` of the built model peaks its power where CIGALE's does."""
    pytest.importorskip("pcigale")
    from pcigale.data import SimpleDatabase

    r, tau, beta, gamma, theta, psy = node
    wave_aa, lnu = _torus_states[node]
    wave_nm = wave_aa / 10.0
    with SimpleDatabase("fritz2006") as db:
        entry = db.get(
            r_ratio=r, tau=tau, beta=beta, gamma=gamma, opening_angle=_half_angle(theta), psy=psy
        )
    pc = _resample(wave_nm, entry.wl, entry.dust)
    got = _median_power_um(lnu * C_AA / wave_aa**2, wave_nm)
    assert got == pytest.approx(_median_power_um(pc, wave_nm), rel=_MEDIAN_RTOL)


@pytest.mark.parametrize(("node", "literal"), zip(_NODES, _PCIGALE_LITERALS), ids=_NODE_IDS)
def test_model_torus_median_power_wavelength_pinned(node, literal, _torus_states):
    """As above against the pinned CIGALE median (on its 0.1 - 1000 micron grid)."""
    wave_aa, lnu = _torus_states[node]
    wave_nm = wave_aa / 10.0
    sel = (wave_nm >= _WAVE_NM[0]) & (wave_nm <= _WAVE_NM[-1])
    got = _median_power_um((lnu * C_AA / wave_aa**2)[sel], wave_nm[sel])
    assert got == pytest.approx(literal[1], rel=_MEDIAN_RTOL)


#: Torus power of the built model, ``integral of sed_agn_torus d(nu)`` [erg/s], at
#: ``agn_log_lbol = 11``, ``agn_torus_frac = 1``: 1e11 L_sun. The conversion
#: redistributes the power over wavelength and leaves its total unchanged; the
#: literal is the total the loader returns when the template is read as L_nu.
_MODEL_TORUS_POWER_ERG_S = 3.828e44


@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_model_torus_power_unchanged(node, _torus_states):
    wave_aa, lnu = _torus_states[node]
    power = _power_nu(lnu, wave_aa)
    assert power == pytest.approx(_MODEL_TORUS_POWER_ERG_S, rel=1e-5)


# -- e. the lookup residual ---------------------------------------------------

#: On-node points where the triweight lookup departs most from the library's
#: node table: (r_ratio, tau, beta, gamma, half-angle, psy), the recorded
#: largest departure of a band holding >= 10 % of the power, and the recorded
#: departure of the median power wavelength (both relative).
_LOOKUP_POINTS = (
    ((60.0, 1.0, -0.75, 4.0, 40.0, 50.1), 0.276, 0.183),
    ((60.0, 0.1, -0.5, 4.0, 40.0, 50.1), 0.273, -0.043),
)
_LOOKUP_IDS = ("beta-0.75", "tau0.1")


def _lookup_expected_lnu(point, wave_aa):
    """lambda^2 / c times the triweight-weighted sum of the raw tables, unit power."""
    from tengri.utils.interpolation import compute_grid_weights

    names = ("r_ratio", "tau", "beta", "gamma", "opening_angle", "psy")
    with h5py.File(_GRID, "r") as f:
        g = f["fritz2006"]
        axes = [np.asarray(g[f"{n}_axis"][:]) for n in names]
        template = np.asarray(g["dust"][:], dtype=float)
        wave_grid = np.asarray(g["wavelength_aa"][:])
    for ax, v in zip(axes, point, strict=True):
        template = np.tensordot(
            np.asarray(
                compute_grid_weights(
                    v,
                    jnp.asarray(ax),
                    scatter=0.5 * (ax[1] - ax[0]),
                    edges=edges_for_grid(jnp.asarray(ax)),
                    index_space_interp=True,
                )
            ),
            template,
            axes=([0], [0]),
        )
    return _expected_lnu(template, wave_grid, wave_aa)


@pytest.mark.parametrize(("point", "band_dev", "median_dev"), _LOOKUP_POINTS, ids=_LOOKUP_IDS)
def test_lookup_residual_is_the_triweight_weighting(point, band_dev, median_dev):
    """The public path is the triweight-weighted raw tables; the recorded departure
    from the library's own node table is the lookup residual.

    The first assertion is the definition of the lookup (1e-6). The second and
    third record how far that lookup sits from the library table at the node
    itself (largest band holding >= 10 % of the power, and the median power
    wavelength, each +-2 percentage points). They are the residual tracked in
    #2606 and are removed when that issue closes.
    """
    _require_grid()
    r, tau, beta, gamma, half, psy = point
    wave_aa = _WAVE_NM * 10.0
    lnu = np.asarray(
        FR.fritz_sed(
            jnp.asarray(wave_aa),
            agn_log_lbol=0.0,
            agn_torus_frac=1.0,
            agn_fritz_r_ratio=r,
            agn_fritz_tau=tau,
            agn_fritz_beta=beta,
            agn_fritz_gamma=gamma,
            agn_fritz_oa=half,
            agn_fritz_psy=psy,
        )
    )
    want = _lookup_expected_lnu(point, wave_aa) * L_SUN
    np.testing.assert_allclose(lnu, want, rtol=1e-6, atol=1e-9 * np.max(want))

    node = (r, tau, beta, gamma, 180.0 - 2.0 * half, psy)
    raw, wave_grid, _ = _raw_template(node, "dust")
    library = _resample(_WAVE_NM, wave_grid / 10.0, raw)
    public = lnu * C_AA / wave_aa**2
    f_lib, f_pub = _band_fractions(library, _WAVE_NM), _band_fractions(public, _WAVE_NM)
    sig = f_lib >= _SIGNIFICANT_BAND
    got_band = float(np.max(np.abs(f_pub[sig] / f_lib[sig] - 1.0)))
    got_median = _median_power_um(public, _WAVE_NM) / _median_power_um(library, _WAVE_NM) - 1.0
    assert got_band == pytest.approx(band_dev, abs=0.02)
    assert got_median == pytest.approx(median_dev, abs=0.02)


# -- d. power ---------------------------------------------------------------


@pytest.mark.parametrize("node", _NODES, ids=_NODE_IDS)
def test_torus_power_equals_l_scale(node):
    """``integral of L_nu d(nu)`` equals ``10**log_lbol * L_sun * frac`` to 1e-6.

    The loader normalizes on the template's own native grid, so the integral
    read at those nodes is the requested luminosity to rounding, whatever grid
    the caller evaluates on.
    """
    _require_grid()
    r, tau, beta, gamma, theta, psy = node
    wave = _native_wave_aa()
    lnu = FR.fritz_sed(
        jnp.asarray(wave),
        agn_log_lbol=1.5,
        agn_torus_frac=0.4,
        agn_fritz_r_ratio=r,
        agn_fritz_tau=tau,
        agn_fritz_beta=beta,
        agn_fritz_gamma=gamma,
        agn_fritz_oa=_half_angle(theta),
        agn_fritz_psy=psy,
    )
    assert _native_power_nu(np.asarray(lnu), wave) == pytest.approx(
        10.0**1.5 * L_SUN * 0.4, rel=1e-6
    )


# -- float32 and gradient ------------------------------------------------


def test_fritz_float32_agrees_with_float64():
    """Single precision reproduces the float64 torus to 1e-4 of its peak.

    ``agn_log_lbol = 0`` keeps the luminosity scale (1 L_sun = 3.8e33 erg/s)
    inside float32's range; the model path carries larger luminosities through
    its reference-L_bol factoring.
    """
    _require_grid()
    wave = np.logspace(2.0, 6.9, 600)
    kw = {
        "agn_log_lbol": 0.0,
        "agn_torus_frac": 0.5,
        "agn_fritz_r_ratio": 60.0,
        "agn_fritz_tau": 1.5,
        "agn_fritz_beta": -0.5,
        "agn_fritz_gamma": 4.0,
        "agn_fritz_oa": 40.0,
        "agn_fritz_psy": 50.1,
    }
    with jax.enable_x64(True):
        ref = np.asarray(
            FR.fritz_sed_from_grid(
                FR.load_fritz_grid.__wrapped__(str(_GRID)), jnp.asarray(wave), **kw
            )
        )
    with jax.enable_x64(False):
        out = FR.fritz_sed_from_grid(
            FR.load_fritz_grid.__wrapped__(str(_GRID)), jnp.asarray(wave, dtype=jnp.float32), **kw
        )
        assert out.dtype == jnp.float32
        out = np.asarray(out, dtype=np.float64)
    assert np.all(np.isfinite(out))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4 * np.max(ref))


def test_far_ir_flux_gradient_wrt_tau_is_finite_and_nonzero():
    """``d(far-IR flux)/d tau`` is finite, non-zero and matches finite differences."""
    _require_grid()
    wave = np.logspace(2.0, 6.9, 1500)
    nu = C_AA / wave
    band = jnp.asarray((wave >= 5e5) & (wave <= 1e7), dtype=jnp.float64)

    def far_ir_flux(tau):
        lnu = FR.fritz_sed(
            jnp.asarray(wave),
            agn_log_lbol=0.0,
            agn_torus_frac=1.0,
            agn_fritz_r_ratio=60.0,
            agn_fritz_tau=tau,
            agn_fritz_beta=-0.5,
            agn_fritz_gamma=4.0,
            agn_fritz_oa=40.0,
            agn_fritz_psy=50.1,
        )
        return -jnp.trapezoid(lnu * band, jnp.asarray(nu)) / L_SUN

    tau0 = jnp.asarray(1.5)
    g = jax.grad(far_ir_flux)(tau0)
    assert jnp.isfinite(g)
    assert abs(float(g)) > 1e-6
    assert_grad_matches_fd(far_ir_flux, tau0, rtol=1e-3)
