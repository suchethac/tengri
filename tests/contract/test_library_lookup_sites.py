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
