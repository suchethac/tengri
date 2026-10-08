# SPDX-License-Identifier: BSD-3-Clause
r"""The torus template lookups return the tabulated SED at an exact grid node.

SKIRTOR and Fritz are tabulated radiative-transfer libraries, so the SED at a node
is the model. The lookup used to be the triweight smoother, a kernel-weighted
average of the node and its neighbours; at a node the band shares were off by up
to 65 % (#2606). The lookup is now the node-exact PCHIP interpolant, so at every
node the normalized template equals the stored node to floating-point precision.

The expected value is independent of the code under test: the stored node is
normalized by its own trapezoid integral over the template grid and converted
with lambda^2 / c, the same convention the lookup applies to its result.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp

from tengri.utils.physics_constants import C_AA

RTOL = 1e-6

SKIRTOR_NODES = (
    (2, 2, 2, 3, 1, 3),
    (2, 2, 2, 3, 1, 0),
    (2, 2, 2, 3, 1, 1),
    (0, 0, 0, 0, 0, 3),
    (4, 3, 3, 2, 1, 5),
    (1, 2, 1, 4, 2, 6),
)
FRITZ_NODES = (
    (3, 2, 2, 2, 1, 3),
    (0, 0, 1, 0, 2, 1),
    (1, 3, 2, 4, 0, 2),
    (3, 5, 3, 3, 2, 5),
    (2, 1, 0, 1, 1, 0),
    (4, 4, 1, 2, 0, 4),
)


def _expected_lambda_norm(wave: np.ndarray, node: np.ndarray) -> np.ndarray:
    """SKIRTOR: unit wavelength integral of the node, then L_nu = L_lam lam^2 / c."""
    return node / np.trapezoid(node, wave) * wave**2 / C_AA


def _expected_nu_norm(wave: np.ndarray, node: np.ndarray) -> np.ndarray:
    """Fritz: unit frequency integral of L_nu = L_lam lam^2 / c over the template grid."""
    from tengri.components.agn._template_grid import native_bolometric_nu_np

    l_nu = node * wave**2 / C_AA
    return l_nu / native_bolometric_nu_np(l_nu, wave)


def _lookup_at_node(fn, cube, wave, axes, idx):
    point = tuple(float(np.asarray(a)[i]) for a, i in zip(axes, idx, strict=True))
    return np.asarray(
        fn(
            jnp.asarray(cube),
            jnp.asarray(wave),
            tuple(jnp.asarray(a) for a in axes),
            jnp.asarray(wave),
            point,
            1.0,
        )
    )


@pytest.fixture(scope="module")
def skirtor_grid():
    from tengri.components.agn import skirtor

    try:
        return skirtor.load_skirtor_grid()
    except Exception as exc:  # missing template data
        pytest.skip(f"SKIRTOR template grid unavailable: {exc}")


@pytest.fixture(scope="module")
def fritz_grid():
    from tengri.components.agn import fritz

    try:
        return fritz.load_fritz_grid(fritz._find_fritz_grid())
    except Exception as exc:  # missing template data
        pytest.skip(f"Fritz template grid unavailable: {exc}")


@pytest.mark.parametrize("idx", SKIRTOR_NODES)
def test_skirtor_lookup_equals_node_at_exact_node(skirtor_grid, idx):
    from tengri.components.agn import skirtor

    g = skirtor_grid
    idx = tuple(min(i, len(a) - 1) for i, a in zip(idx, g.axes, strict=True))
    wave = np.asarray(g.wave_grid)
    node = np.asarray(g.grid)[idx]
    got = _lookup_at_node(skirtor._interpolate_and_normalize, g.grid, wave, g.axes, idx)
    exp = _expected_lambda_norm(wave, node)
    assert np.max(np.abs(got - exp)) <= RTOL * np.max(np.abs(exp))


@pytest.mark.parametrize("idx", FRITZ_NODES)
def test_fritz_lookup_equals_node_at_exact_node(fritz_grid, idx):
    from tengri.components.agn import fritz

    g = fritz_grid
    idx = tuple(min(i, len(a) - 1) for i, a in zip(idx, g.axes, strict=True))
    wave = np.asarray(g.wave_grid)
    node = np.asarray(g.dust)[idx]
    got = _lookup_at_node(fritz._interpolate_and_normalize, g.dust, wave, g.axes, idx)
    exp = _expected_nu_norm(wave, node)
    assert np.max(np.abs(got - exp)) <= RTOL * np.max(np.abs(exp))


def test_skirtor_lookup_off_node_is_finite_with_finite_gradient(skirtor_grid):
    from tengri.components.agn import skirtor

    g = skirtor_grid
    wave = jnp.asarray(g.wave_grid)
    axes = tuple(jnp.asarray(a) for a in g.axes)
    base = np.array([float(np.asarray(a)[len(a) // 2]) for a in g.axes])
    base = base + 0.37 * np.array([np.diff(np.asarray(a))[0] for a in g.axes])

    def total(x):
        pt = tuple(x[i] for i in range(len(axes)))
        return jnp.sum(
            skirtor._interpolate_and_normalize(jnp.asarray(g.grid), wave, axes, wave, pt, 1.0)
        )

    val = float(total(jnp.asarray(base)))
    grad = np.asarray(jax.grad(total)(jnp.asarray(base)))
    assert np.isfinite(val)
    assert np.all(np.isfinite(grad))
