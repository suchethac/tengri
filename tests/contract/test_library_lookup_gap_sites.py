# SPDX-License-Identifier: BSD-3-Clause
"""Node-exact and gradient checks for the remaining template-lookup sites (#2861).

Each site must return the stored node value at every grid node, and its output
must move with each fitted axis across the declared box. Synthetic grids are
built with smooth dependence on every axis, so a zero gradient can only come from
a lookup that does not interpolate. Run with the triweight kernel swapped back
into ``library_lookup`` to see the node checks fail.

Sites: SKIRTOR total spectrum and disc-inclination ratio, the relagn disc, the
composable collapsed lookup, the DL07 and Dale2014 emission builders, and the
GRAHSP and QSOgen collapsed closures.
"""

from __future__ import annotations

import itertools

import h5py
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_NODES3 = (0.0, 0.5, 1.0)


def _check_nodes_and_grads(fn, axes, table_at, *, wave=None):
    """Node values equal ``table_at(idx)``; the sum over the output moves with every axis."""
    for idx in itertools.product(*(range(a.shape[0]) for a in axes)):
        point = tuple(float(a[i]) for a, i in zip(axes, idx, strict=True))
        got = np.asarray(fn(*point))
        np.testing.assert_allclose(got, table_at(idx), rtol=1e-9, atol=0.0)
    inside = tuple(float(a[0] + 0.4 * (a[-1] - a[0])) for a in axes)
    grads = jax.grad(lambda *p: jnp.sum(fn(*p)), argnums=tuple(range(len(axes))))(*inside)
    for k, g in enumerate(grads):
        assert np.isfinite(float(g)), f"axis {k} gradient not finite"
        assert float(g) != 0.0, f"axis {k} has an exactly zero gradient inside the box"


# ── SKIRTOR total spectrum (v4 layout, synthetic file) ────────────────────────


def _skirtor_v4_file(tmp_path):
    from tengri.components.agn.skirtor import _C_AA_PER_S  # noqa: F401  (import check)

    tau = np.array([3.0, 7.0, 11.0])
    p = np.array([0.0, 1.0, 2.0])
    q = np.array([0.5, 1.0, 1.5])
    oa = np.array([20.0, 40.0, 60.0])
    rr = np.array([5.0, 20.0, 50.0])
    ideg = np.array([0.0, 45.0, 90.0])
    wave = np.geomspace(1.0e3, 2.0e4, 40)
    grids = np.meshgrid(tau, p, q, oa, rr, ideg, indexing="ij")
    # Spectral slope depends on every axis, so the normalized shape moves with each.
    slope = 0.5 + 0.02 * grids[0] + 0.1 * grids[1] + 0.05 * grids[2] + 0.002 * grids[3]
    slope = slope + 0.001 * grids[4] + 0.004 * grids[5]
    total = (wave[None, None, None, None, None, None, :] / 5000.0) ** (-slope[..., None])
    total = total * (1.0 + 0.1 * np.sin(grids[1][..., None]))
    path = tmp_path / "skirtor_raw_v4.h5"
    with h5py.File(path, "w") as f:
        g = f.create_group("grid")
        g["tau_97"], g["p"], g["q"] = tau, p, q
        g["opening_angle"], g["radius_ratio"], g["inclination_deg"] = oa, rr, ideg
        f["spectra/total_emission"] = total
        f["wavelength"] = wave
    return path, tau, p, q, oa, rr, ideg, wave, total


def test_skirtor_total_spectrum_reproduces_nodes_and_moves(tmp_path):
    from tengri.components.agn import skirtor as sk

    path, tau, p, q, oa, rr, ideg, wave, total = _skirtor_v4_file(tmp_path)
    fn = sk.create_skirtor_raw_total_from_grid(str(path))
    nu = sk._C_AA_PER_S / wave
    order = np.argsort(nu)
    l_bol = 10.0**45.0 * sk._L_SUN

    def fn_point(tau_v, p_v, q_v, oa_v, rr_v, cos_v):
        return fn(
            jnp.asarray(wave),
            45.0,
            tau_v,
            p_v,
            q_v,
            oa_v,
            rr_v,
            cos_v,
            1.0,
        )

    def table_at(idx):
        spec = total[idx[0], idx[1], idx[2], idx[3], idx[4], idx[5]]
        bolo = np.trapezoid(spec[order], nu[order])
        return l_bol * spec / abs(bolo)

    cos_nodes = np.cos(np.deg2rad(ideg))
    axes = (tau, p, q, oa, rr, cos_nodes)
    _check_nodes_and_grads(fn_point, [np.asarray(a) for a in axes], table_at)


# ── SKIRTOR disc inclination ratio (synthetic disk cube) ──────────────────────


def test_skirtor_disc_ratio_reproduces_nodes_and_moves():
    from tengri.components.agn import skirtor as sk

    tau = np.array([3.0, 7.0, 11.0])
    p = np.array([0.0, 1.0, 2.0])
    q = np.array([0.5, 1.0, 1.5])
    oa = np.array([20.0, 40.0, 60.0])
    rr = np.array([5.0, 20.0, 50.0])
    cos = np.array(_NODES3)
    wave = np.geomspace(1.0e3, 2.0e4, 30)
    g = np.meshgrid(tau, p, q, oa, rr, cos, indexing="ij")
    k = 0.3 * (1.0 + 0.1 * g[0] + 0.05 * g[1] + 0.02 * g[2] + 0.005 * g[3] + 0.001 * g[4])
    disk = np.exp(k[..., None] * g[5][..., None] * np.ones_like(wave)) * (1.0 + 0.01 * wave / 1e4)
    grid = sk._disc_atten_bundle({"axes": (tau, p, q, oa, rr, cos), "disk": disk, "wave": wave})

    def fn_point(tau_v, p_v, q_v, oa_v, rr_v, cos_v):
        return sk.skirtor_disc_attenuation_from_grid(
            grid,
            jnp.asarray(wave),
            agn_tau_skirtor=tau_v,
            agn_p_skirtor=p_v,
            agn_q_skirtor=q_v,
            agn_oa_skirtor=oa_v,
            agn_radius_ratio=rr_v,
            agn_cos_inc=cos_v,
        )

    def table_at(idx):
        face = disk[idx[0], idx[1], idx[2], idx[3], idx[4], 2]
        return np.clip(disk[idx] / face, 0.0, 1.5)

    _check_nodes_and_grads(fn_point, [tau, p, q, oa, rr, cos], table_at)


# ── relagn disc (real grid on disk) ───────────────────────────────────────────


def test_relagn_disc_reproduces_nodes_and_moves():
    from tengri._data_setup import package_or_env_data_path
    from tengri.components.agn import disc
    from tengri.components.agn._template_grid import native_bolometric_nu
    from tengri.utils.physics_constants import L_SUN

    path = str(package_or_env_data_path("relagn_disc_grid.h5"))
    with h5py.File(path, "r") as f:
        log_mbh = np.asarray(f["log_mbh"][:])
        log_mdot = np.asarray(f["log_mdot"][:])
        astar = np.asarray(f["astar"][:])
        lnu = np.asarray(f["lnu_disc"][:])
        wave = np.asarray(f["wavelength_aa"][:])
    fn = disc.create_relagn_disc_from_grid(path)
    l_bol = 10.0**45.0 * L_SUN
    # Keep the sweep small: a corner, a middle node and the far corner of the box.
    picks = [(0, 0, 0), (len(log_mbh) // 2, len(log_mdot) // 2, len(astar) // 2)]
    picks.append((len(log_mbh) - 1, len(log_mdot) - 1, len(astar) - 1))
    for i, j, k in picks:
        row = jnp.asarray(lnu[i, j, k])
        expected = 2.0 * 0.5 * l_bol * row / native_bolometric_nu(row, jnp.asarray(wave))
        got = fn(
            wavelength=jnp.asarray(wave),
            agn_log_mbh=float(log_mbh[i]),
            agn_log_mdot=float(log_mdot[j]),
            agn_astar=float(astar[k]),
            agn_cos_inc=0.5,
            agn_log_lbol=45.0,
        )
        np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-8)
    inside = (
        float(log_mbh[0] + 0.4 * (log_mbh[-1] - log_mbh[0])),
        float(log_mdot[0] + 0.4 * (log_mdot[-1] - log_mdot[0])),
        float(astar[0] + 0.4 * (astar[-1] - astar[0])),
        0.5,
    )

    def total(a, b, c, d):
        return jnp.sum(
            fn(
                wavelength=jnp.asarray(wave),
                agn_log_mbh=a,
                agn_log_mdot=b,
                agn_astar=c,
                agn_cos_inc=d,
                agn_log_lbol=45.0,
            )
        )

    grads = jax.grad(total, argnums=(0, 1, 2, 3))(*inside)
    for n, gr in enumerate(grads):
        assert np.isfinite(float(gr))
        assert float(gr) != 0.0, f"relagn axis {n} gradient is zero"


# ── composable collapsed lookup ───────────────────────────────────────────────


def test_composable_collapsed_lookup_reproduces_nodes_and_moves():
    from tengri.components.agn.blocks import composable_precompute as cp

    rng = np.random.default_rng(11)
    axes = tuple(jnp.asarray(np.linspace(lo, hi, 3)) for lo, hi in [(0.0, 1.0), (-1.0, 1.0)])
    grid_phot = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 4)))
    preint = {
        "grid_phot": grid_phot,
        "axes": axes,
        "_axis_names": ("agn_a", "agn_b", "agn_c"),
        "_collapsed_axes": {2: 0.3},
        "_interp_axis_transform": (),
        "_preint": None,
    }
    lookup = cp.build_lookup(preint)
    _check_nodes_and_grads(
        lambda a, b: lookup(1.0, a, b),
        [np.asarray(x) for x in axes],
        lambda idx: np.asarray(grid_phot[idx]),
    )


# ── DL07 and Dale2014 emission builders ───────────────────────────────────────


def test_dl07_emission_builder_reproduces_nodes_and_moves():
    from tengri.components.dust import dust_emission_precompute as dep

    rng = np.random.default_rng(12)
    qpah = jnp.asarray(np.linspace(0.01, 0.05, 3))
    umin = jnp.asarray(np.linspace(0.1, 10.0, 3))
    single = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 5)))
    power = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 5)))
    fn = dep.build_dl07_photometry_lookup(
        {
            "single_u_phot": single,
            "powerlaw_phot": power,
            "umin_grid": umin,
            "qpah_grid": qpah,
        }
    )
    _check_nodes_and_grads(
        lambda q, u: fn(1.0, u, 0.0, q),
        [np.asarray(qpah), np.asarray(umin)],
        lambda idx: np.asarray(single[idx]),
    )
    _check_nodes_and_grads(
        lambda q, u: fn(1.0, u, 1.0, q),
        [np.asarray(qpah), np.asarray(umin)],
        lambda idx: np.asarray(power[idx]),
    )


def test_dale2014_emission_builder_reproduces_nodes_and_moves():
    from tengri.components.dust import dust_emission_precompute as dep

    rng = np.random.default_rng(13)
    alpha = jnp.asarray(np.linspace(1.0, 3.0, 4))
    phot = jnp.asarray(rng.uniform(0.5, 2.0, size=(4, 5)))
    fn = dep.build_dale2014_photometry_lookup({"alpha_grid": alpha, "phot": phot})
    _check_nodes_and_grads(
        lambda a: fn(1.0, a),
        [np.asarray(alpha)],
        lambda idx: np.asarray(phot[idx]),
    )


# ── GRAHSP and QSOgen collapsed closures ──────────────────────────────────────


def test_grahsp_collapsed_closure_reproduces_nodes_and_moves():
    from tengri.components.agn.grahsp import precompute as gp

    rng = np.random.default_rng(14)
    axes = tuple(jnp.asarray(np.linspace(lo, hi, 3)) for lo, hi in [(0.0, 1.0), (-1.0, 1.0)])
    grid_phot = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 4)))
    fn = gp.build_lookup({"grid_phot": grid_phot, "axes": axes, "_collapsed_axes": {2: 0.0}})
    _check_nodes_and_grads(
        lambda a, b: fn(a, b),
        [np.asarray(x) for x in axes],
        lambda idx: np.asarray(grid_phot[idx]),
    )


def test_qsogen_collapsed_closure_reproduces_nodes_and_moves():
    from tengri.components.agn import qsogen_precompute as qp

    rng = np.random.default_rng(15)
    axes = tuple(jnp.asarray(np.linspace(lo, hi, 3)) for lo, hi in [(0.0, 1.0), (-1.0, 1.0)])
    grid_phot = jnp.asarray(rng.uniform(0.5, 2.0, size=(3, 3, 4)))
    fn = qp.build_lookup({"grid_phot": grid_phot, "axes": axes, "_collapsed_axes": {2: 0.0}})
    _check_nodes_and_grads(
        lambda a, b: fn(0.0, a, b),
        [np.asarray(x) for x in axes],
        lambda idx: np.asarray(grid_phot[idx]),
    )
