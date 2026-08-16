# SPDX-License-Identifier: BSD-3-Clause
"""Smoke tests for the alternative AGN torus precompute adapters (PR 2).

Covers ``silva04`` and ``cat3d_wind`` — the SKIRTOR alternatives.
"""

from __future__ import annotations

from pathlib import Path

import chex
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

_DATA = Path(__file__).resolve().parents[4] / "data"
_SILVA04_GRID = _DATA / "silva04_torus_grid.h5"
_CAT3D_GRID = _DATA / "cat3d_wind_torus_grid.h5"


def _declared_param_names(registry_key: str) -> set[str]:
    """Fully-prefixed parameter names a registered component actually declares.

    Used to check an adapter's ``AXIS_PARAMS`` against the component it serves,
    rather than against a copy of itself. Pinning ``AXIS_PARAMS`` to a literal
    only proves nobody edited the literal: both adapters here asserted their own
    declaration and stayed green while naming parameters no ``Parameters`` can
    hold, so the axes could never collapse (#1738).
    """
    import tengri  # noqa: F401  -- populates the component registry
    from tengri.components.sed_model_component import _REGISTRY

    cls = _REGISTRY[registry_key]
    prefix = getattr(cls, "parameter_prefix", "") or ""
    return {
        f"{prefix}{attr}"
        for attr, value in vars(cls).items()
        if not attr.startswith("_")
        and (hasattr(value, "sample") or hasattr(value, "unstandardize"))
    }


@pytest.fixture(scope="module")
def filter_set():
    centers = np.array([3.5e4, 1.0e5, 3.0e5, 1.0e6])  # MIR–FIR (Angstrom)
    widths = np.array([0.5e4, 1.5e4, 5.0e4, 2.0e5])
    waves: list[np.ndarray] = []
    trans: list[np.ndarray] = []
    for c, w in zip(centers, widths):
        wv = np.linspace(c - 3 * w, c + 3 * w, 64)
        tr = np.exp(-0.5 * ((wv - c) / w) ** 2)
        waves.append(wv)
        trans.append(tr)
    return waves, trans


class TestSilva04Adapter:
    def test_imports_and_axes(self):
        from tengri.components.agn import silva04_precompute

        axes = silva04_precompute.AXIS_PARAMS
        assert len(axes) == 1, f"silva04 has one grid axis, declares {len(axes)}"
        assert set(axes) <= _declared_param_names("silva04"), (
            f"silva04_precompute declares axis names {sorted(axes)} that the "
            f"silva04 component does not declare; the axis can never be "
            f"collapsed and nothing raises (#1738)"
        )

    @pytest.mark.skipif(not _SILVA04_GRID.exists(), reason="silva04 grid not present")
    def test_precompute_and_lookup(self, filter_set):
        from tengri.components.agn import silva04_precompute

        waves, trans = filter_set
        result = silva04_precompute.precompute(
            waves, trans, redshift=0.5, parameters=None, grid_path=str(_SILVA04_GRID)
        )
        lookup = silva04_precompute.build_lookup(result)
        # Probe arity defensively (signature may include torus_frac, etc.).
        import inspect

        sig = inspect.signature(lookup)
        n_args = len(sig.parameters)
        args = tuple(jnp.float64(0.5) for _ in range(n_args))
        out = jax.jit(lookup)(*args)
        chex.assert_tree_all_finite(np.asarray(out))

    def test_registered(self):
        from tengri.forward.precompute.registry import _REGISTRY

        assert _REGISTRY["silva04"].endswith("silva04_precompute")


class TestCAT3DWindAdapter:
    def test_imports_and_axes(self):
        from tengri.components.agn import cat3d_precompute

        axes = cat3d_precompute.AXIS_PARAMS
        assert len(axes) == 3, f"cat3d_wind has three grid axes, declares {len(axes)}"
        assert set(axes) <= _declared_param_names("cat3d_wind"), (
            f"cat3d_precompute declares axis names {sorted(axes)} that the "
            f"cat3d_wind component does not declare; the axes can never be "
            f"collapsed and nothing raises (#1738)"
        )

    @pytest.mark.skipif(not _CAT3D_GRID.exists(), reason="cat3d_wind grid not present")
    def test_precompute_and_lookup(self, filter_set):
        from tengri.components.agn import cat3d_precompute

        waves, trans = filter_set
        result = cat3d_precompute.precompute(
            waves, trans, redshift=0.5, parameters=None, grid_path=str(_CAT3D_GRID)
        )
        lookup = cat3d_precompute.build_lookup(result)
        import inspect

        sig = inspect.signature(lookup)
        n_args = len(sig.parameters)
        args = tuple(jnp.float64(0.5) for _ in range(n_args))
        out = jax.jit(lookup)(*args)
        chex.assert_tree_all_finite(np.asarray(out))

    def test_registered(self):
        from tengri.forward.precompute.registry import _REGISTRY

        assert _REGISTRY["cat3d_wind"].endswith("cat3d_precompute")
