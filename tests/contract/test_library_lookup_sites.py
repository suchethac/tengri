# SPDX-License-Identifier: BSD-3-Clause
"""Every template-library lookup reproduces its stored nodes exactly (#2861).

Owner ruling: a library template lookup is node-exact PCHIP through
:func:`tengri.utils.grid_interp.library_lookup`. The triweight smoother averages
across neighboring nodes, so at a node it returns something other than the
stored value. Each test queries a site at its own grid nodes and compares with
the stored table. The gradient test checks that a fitted axis still moves the
output across the declared prior box.

Sites covered: the WG00 attenuation curve, the Synthesizer AGN NLR lines, the
Feltre AGN NLR backend, and the Feltre and CB19 nebular precompute lookups.
"""

from __future__ import annotations

import importlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular import agn_nebular as an
from tengri.utils.interpolation import edges_for_grid

pytestmark = pytest.mark.contract


def _ascending_axis(n: int, lo: float, hi: float) -> np.ndarray:
    return np.linspace(lo, hi, n)


def _wg00_module():
    return importlib.import_module("tengri.components.dust.wg00")


def test_wg00_lookup_reproduces_table_nodes():
    wg00 = _wg00_module()
    path = wg00._find_wg00_grid()
    raw = wg00._load_wg00_arrays(path)
    is_, id_, ig_ = wg00._select_indices("mw", "shell", "homogeneous")
    table = np.asarray(raw["a_lambda"][is_, id_, ig_])
    wave = jnp.asarray(raw["wavelength"])
    tau = np.asarray(raw["tau_v_axis"])
    curve = wg00.create_wg00_from_grid(path)
    for k in range(0, tau.size, max(1, tau.size // 6)):
        got = np.asarray(curve(wave, float(tau[k])))
        np.testing.assert_allclose(got, table[k], rtol=1e-12, atol=0.0)


def test_wg00_lookup_gradient_nonzero_across_prior_box():
    wg00 = _wg00_module()
    path = wg00._find_wg00_grid()
    raw = wg00._load_wg00_arrays(path)
    wave = jnp.asarray(raw["wavelength"])
    curve = wg00.create_wg00_from_grid(path)

    def total(tau_v):
        return jnp.sum(curve(wave, tau_v))

    for tau_v in (0.3, 1.0, 7.0):
        grad = float(jax.grad(total)(tau_v))
        assert np.isfinite(grad)
        assert grad != 0.0


def test_feltre_backend_reproduces_grid_nodes():
    backend = an.FeltreNLRBackend(an._DEFAULT_FELTRE_GRID_PATH)
    axes = [np.asarray(a) for a in backend._axes]
    log_hb = np.asarray(backend._logHB_sorted)
    ratios = np.asarray(backend._ratios_sorted)
    rng = np.random.default_rng(0)
    for _ in range(3):
        idx = tuple(int(rng.integers(0, a.shape[0])) for a in axes)
        point = tuple(float(a[i]) for a, i in zip(axes, idx, strict=True))
        _, lum = backend.predict_agn_nlr_lines(
            alpha_pl=point[0],
            neb_logU=point[1],
            neb_logn=point[2],
            neb_logZ_gas=point[3],
            xi_d=point[4],
            log_qh=0.0,
            neb_fesc=0.0,
        )
        expected = ratios[idx] * 10.0 ** log_hb[idx] / an._LSUN_ERG
        np.testing.assert_allclose(np.asarray(lum), expected, rtol=1e-10, atol=0.0)


def _synthetic_nlr_backend() -> tuple[an.SynthesizerNLRBackend, np.ndarray]:
    rng = np.random.default_rng(1)
    axes = [
        _ascending_axis(3, lo, hi)
        for lo, hi in [(7, 10), (-2, 0), (0, 1), (-1, 0.5), (-4, -1), (2, 4)]
    ]
    shape = (*tuple(a.shape[0] for a in axes), 2)
    log_line = rng.uniform(-3.0, -0.5, size=shape)

    class _Grid:
        pass

    grid = _Grid()
    grid.log_line_per_qh = jnp.asarray(log_line)
    grid.line_wavelengths_aa = np.array([5007.0, 6563.0])
    grid.log_qh_specific = jnp.asarray(rng.uniform(52.0, 54.0, size=shape[:6]))
    grid.spectra_wavelengths_aa = np.linspace(900.0, 9000.0, 37)
    grid.nebular_per_lbol = jnp.asarray(rng.uniform(0.0, 1.0, size=(*shape[:6], 37)))
    backend = an.SynthesizerNLRBackend.__new__(an.SynthesizerNLRBackend)
    backend.grid = grid
    names = ("mass", "edd", "cos", "met", "ionU", "nH")
    for name, ax in zip(names, axes, strict=True):
        setattr(backend, f"_{name}_sorted", jnp.asarray(ax))
        setattr(backend, f"_{name}_descending", False)
    backend._edges_mass = edges_for_grid(jnp.asarray(axes[0]))
    backend._edges_edd = edges_for_grid(jnp.asarray(axes[1]))
    backend._edges_cos = edges_for_grid(jnp.asarray(axes[2]))
    backend._edges_met = edges_for_grid(jnp.asarray(axes[3]))
    backend._edges_ionU = edges_for_grid(jnp.asarray(axes[4]))
    backend._edges_nH = edges_for_grid(jnp.asarray(axes[5]))
    return backend, log_line, axes


def test_synthesizer_nlr_lookup_reproduces_grid_nodes():
    backend, log_line, axes = _synthetic_nlr_backend()
    for idx in np.ndindex(*(a.shape[0] for a in axes)):
        _, lum = backend.predict_agn_nlr_lines(
            log_bh_mass=float(axes[0][idx[0]]),
            log_eddington=float(axes[1][idx[1]]),
            cosine_inclination=float(axes[2][idx[2]]),
            log_metallicity=float(axes[3][idx[3]]),
            log_ionU=float(axes[4][idx[4]]),
            log_nH=float(axes[5][idx[5]]),
            log_qh=0.0,
            neb_fesc=0.0,
        )
        np.testing.assert_allclose(np.asarray(lum), 10.0 ** log_line[idx], rtol=1e-10, atol=0.0)


def _synthetic_feltre_preint() -> dict:
    rng = np.random.default_rng(2)
    axes = tuple(
        jnp.asarray(_ascending_axis(3, lo, hi))
        for lo, hi in [(-2.0, -1.2), (-4.0, -1.0), (2.0, 4.0), (-2.0, 0.2), (0.1, 0.5)]
    )
    shape = tuple(a.shape[0] for a in axes)
    return {
        "line_wavelengths": np.array([5007.0, 6563.0]),
        "logHB_per_logq": jnp.asarray(rng.uniform(-14.0, -12.0, size=shape)),
        "line_ratios": jnp.asarray(rng.uniform(-1.0, 1.0, size=(*shape, 2))),
        "line_weight_matrix": np.eye(2),
        "grid_axes": axes,
        "_collapsed_axes": {},
    }


def test_feltre_precompute_lookup_reproduces_grid_nodes():
    from tengri.components.nebular import feltre_precompute as fp

    preint = _synthetic_feltre_preint()
    lookup = fp.build_lookup(preint)
    axes = preint["grid_axes"]
    for idx in np.ndindex(*(a.shape[0] for a in axes)):
        point = [float(a[i]) for a, i in zip(axes, idx, strict=True)]
        _, lum = lookup["predict_lines"](0.0, *point)
        expected = (
            np.asarray(preint["line_ratios"])[idx]
            * 10.0 ** np.asarray(preint["logHB_per_logq"])[idx]
            / an._LSUN_ERG
        )
        np.testing.assert_allclose(np.asarray(lum), expected, rtol=1e-10, atol=0.0)


def test_cb19_precompute_lookup_reproduces_grid_nodes():
    from tengri.components.nebular import cb19_precompute as cp

    rng = np.random.default_rng(3)
    axes = tuple(
        jnp.asarray(_ascending_axis(3, lo, hi))
        for lo, hi in [(-4.0, -1.0), (-2.0, 0.2), (0.1, 0.5)]
    )
    shape = tuple(a.shape[0] for a in axes)
    preint = {
        "line_wavelengths": np.array([5007.0, 6563.0]),
        "log_line_ratios": jnp.asarray(rng.uniform(-3.0, 0.0, size=(*shape, 2))),
        "log_hb_per_qh": -12.0,
        "grid_axes": axes,
        "_collapsed_axes": {},
    }
    lookup = cp.build_lookup(preint)
    for idx in np.ndindex(*shape):
        point = [float(a[i]) for a, i in zip(axes, idx, strict=True)]
        _, lum = lookup["predict_lines"](0.0, *point)
        expected = 10.0 ** np.asarray(preint["log_line_ratios"])[idx] * 10.0**-12.0 / an._LSUN_ERG
        np.testing.assert_allclose(np.asarray(lum), expected, rtol=1e-10, atol=0.0)


def _interior_point(axes: list[np.ndarray], seed: int) -> list[float]:
    rng = np.random.default_rng(seed)
    return [float(a[0] + (a[-1] - a[0]) * rng.uniform(0.2, 0.8)) for a in axes]


def _assert_each_axis_moves(f, point: list[float]) -> None:
    grads = jax.grad(f, argnums=tuple(range(len(point))))(*point)
    for k, g in enumerate(grads):
        assert np.isfinite(float(g)), f"axis {k} gradient not finite"
        assert float(g) != 0.0, f"axis {k} has an exactly zero gradient inside the prior box"


def test_feltre_backend_gradient_nonzero_on_every_axis():
    backend = an.FeltreNLRBackend(an._DEFAULT_FELTRE_GRID_PATH)
    axes = [np.asarray(a) for a in backend._axes]
    keys = ("alpha_pl", "neb_logU", "neb_logn", "neb_logZ_gas", "xi_d")

    def total(*pt):
        kwargs = dict(zip(keys, pt, strict=True))
        _, lum = backend.predict_agn_nlr_lines(**kwargs, log_qh=0.0, neb_fesc=0.0)
        return jnp.sum(lum)

    _assert_each_axis_moves(total, _interior_point(axes, seed=4))


def test_synthesizer_nlr_gradient_nonzero_on_every_axis():
    backend, _, axes = _synthetic_nlr_backend()
    keys = (
        "log_bh_mass",
        "log_eddington",
        "cosine_inclination",
        "log_metallicity",
        "log_ionU",
        "log_nH",
    )

    def total(*pt):
        kwargs = dict(zip(keys, pt, strict=True))
        _, lum = backend.predict_agn_nlr_lines(**kwargs, log_qh=0.0, neb_fesc=0.0)
        return jnp.sum(lum)

    _assert_each_axis_moves(total, _interior_point(axes, seed=5))


def test_feltre_precompute_gradient_nonzero_on_every_axis():
    from tengri.components.nebular import feltre_precompute as fp

    preint = _synthetic_feltre_preint()
    lookup = fp.build_lookup(preint)
    axes = [np.asarray(a) for a in preint["grid_axes"]]

    def total(*pt):
        _, lum = lookup["predict_lines"](0.0, *pt)
        return jnp.sum(lum)

    _assert_each_axis_moves(total, _interior_point(axes, seed=6))


def test_synthesizer_log_qh_specific_lookup_reproduces_grid_nodes():
    backend, _, axes = _synthetic_nlr_backend()
    table = np.asarray(backend.grid.log_qh_specific)
    for idx in np.ndindex(*(a.shape[0] for a in axes)):
        got = backend.interp_log_qh_specific(
            log_bh_mass=float(axes[0][idx[0]]),
            log_eddington=float(axes[1][idx[1]]),
            cosine_inclination=float(axes[2][idx[2]]),
            log_metallicity=float(axes[3][idx[3]]),
            log_ionU=float(axes[4][idx[4]]),
            log_nH=float(axes[5][idx[5]]),
        )
        np.testing.assert_allclose(np.asarray(got), table[idx], rtol=1e-10, atol=0.0)


def test_synthesizer_blr_spectrum_lookup_reproduces_grid_nodes():
    backend, _, axes = _synthetic_nlr_backend()
    table = np.asarray(backend.grid.nebular_per_lbol)
    wave = backend.grid.spectra_wavelengths_aa
    l_bol, cov = 1.0, 1.0
    for idx in np.ndindex(*(a.shape[0] for a in axes)):
        got = backend.predict_agn_nebular_spectrum(
            jnp.asarray(wave),
            l_bol,
            covering_fraction=cov,
            log_bh_mass=float(axes[0][idx[0]]),
            log_eddington=float(axes[1][idx[1]]),
            log_metallicity=float(axes[3][idx[3]]),
            log_ionU=float(axes[4][idx[4]]),
            log_nH=float(axes[5][idx[5]]),
        )
        # The lookup holds cosine_inclination at 0.5 (isotropic line regions), so the
        # node is compared at that inclination.
        cos_idx = int(np.argmin(np.abs(np.asarray(axes[2]) - 0.5)))
        if abs(float(axes[2][cos_idx]) - 0.5) > 1e-12:
            pytest.skip("the synthetic axis has no node at cos_inclination = 0.5")
        np.testing.assert_allclose(
            np.asarray(got), table[(*idx[:2], cos_idx, *idx[3:])] * l_bol * cov, rtol=1e-10
        )


def test_template_photometry_lookup_reproduces_grid_nodes():
    from tengri.forward.precompute.templates import build_template_photometry_lookup

    rng = np.random.default_rng(8)
    axes = tuple(jnp.asarray(_ascending_axis(3, lo, hi)) for lo, hi in [(0.5, 1.5), (-1.0, 1.0)])
    phot = jnp.asarray(rng.uniform(0.1, 2.0, size=(3, 3, 4)))

    class _Preint:
        pass

    preint = _Preint()
    preint.phot = phot
    preint.axes = axes
    preint.edges = tuple(edges_for_grid(a) for a in axes)
    lookup = build_template_photometry_lookup(preint)
    for i in range(3):
        for j in range(3):
            got = lookup(1.0, float(axes[0][i]), float(axes[1][j]))
            np.testing.assert_allclose(np.asarray(got), np.asarray(phot[i, j]), rtol=1e-10)


def test_nenkova_torus_lookup_reproduces_template_nodes():
    """The torus template at a tabulated tau is the stored row, normalized to L_bol."""
    from tengri._data_setup import package_or_env_data_path
    from tengri.components.agn import torus
    from tengri.components.agn._template_grid import native_bolometric_nu
    from tengri.utils.physics_constants import L_SUN

    path = str(package_or_env_data_path("nenkova08_torus_grid.h5"))
    raw = torus._load_nenkova_arrays(path)
    wave = jnp.asarray(raw["wavelength"])
    fn = torus.create_nenkova_from_grid(path)
    l_bol_erg = 10.0**45.0 * L_SUN
    tau = raw["tau_axis"]
    for k in range(0, tau.size, max(1, tau.size // 4)):
        row = jnp.asarray(raw["template"][k])
        expected = l_bol_erg * row / native_bolometric_nu(row, wave)
        got = fn(wave, agn_log_lbol=45.0, agn_tau=float(tau[k]), agn_torus_frac=1.0)
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-8)


def test_mappings_shock_lookup_reproduces_grid_nodes():
    from tengri.components.nebular import mappings_shock_precompute as ms

    rng = np.random.default_rng(9)
    axes = tuple(
        jnp.asarray(_ascending_axis(3, lo, hi)) for lo, hi in [(5.0, 7.0), (0.0, 1.0), (-2.0, 0.0)]
    )
    ratios = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 3, 2)))
    preint = {
        "line_wavelengths": np.array([5007.0, 6563.0]),
        "line_ratios_grid": ratios,
        "grid_axes": axes,
        "_collapsed_axes": {},
    }
    lookup = ms.build_lookup(preint)
    for idx in np.ndindex(3, 3, 3):
        point = [float(a[i]) for a, i in zip(axes, idx, strict=True)]
        _, lum = lookup["predict_lines"](1.0, *point)
        row = np.asarray(ratios[idx])
        expected = (row / row[-1]) * 1.0 / an._LSUN_ERG
        np.testing.assert_allclose(np.asarray(lum), expected, rtol=1e-10)


def test_skirtor_photometry_lookup_reproduces_grid_nodes():
    from tengri.components.agn import skirtor_precompute as sp

    rng = np.random.default_rng(10)
    axes = tuple(
        jnp.asarray(_ascending_axis(3, lo, hi))
        for lo, hi in [(3.0, 11.0), (0.0, 1.5), (0.0, 2.0), (20.0, 60.0), (0.0, 1.0)]
    )
    grid_phot = jnp.asarray(rng.uniform(0.1, 3.0, size=(3, 3, 3, 3, 3, 4)))
    fn = sp.build_skirtor_photometry_lookup({"grid_phot": grid_phot, "axes": axes})
    for idx in np.ndindex(3, 3, 3, 3, 3):
        point = [float(a[i]) for a, i in zip(axes, idx, strict=True)]
        got = fn(45.0, *point, 0.5)
        expected = 10.0**45.0 * 0.5 * np.asarray(grid_phot[idx])
        np.testing.assert_allclose(np.asarray(got), expected, rtol=1e-10)
