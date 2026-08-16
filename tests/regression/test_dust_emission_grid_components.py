# SPDX-License-Identifier: BSD-3-Clause
"""Bit-exact regression tests for dust emission grid SEDModelComponents.

These tests verify that the grid/template dust-emission components (dale2014,
draine_li2007, draine_li2014, astrodust, bosa, themis) are bit-exact with
the original closures they wrap.

Markers
-------
regression_bug
    These are regression tests tracking known bugs or parity fixes.
"""

from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug


@pytest.fixture
def golden_dir() -> Path:
    """Return the golden regression data directory."""
    return Path(__file__).parent / "data" / "dust_emission_golden"


@pytest.fixture
def golden_data(golden_dir: Path) -> dict:
    """Load params and metadata from golden params.json."""
    params_file = golden_dir / "params.json"
    if not params_file.exists():
        pytest.skip(f"Golden params not found: {params_file}")
    with open(params_file) as f:
        return json.load(f)


@pytest.fixture
def wave_grid(golden_data: dict) -> jnp.ndarray:
    """Reconstruct wavelength grid from golden metadata."""
    spec = golden_data["wave_spec"]
    wave = np.linspace(
        spec["min_aa"],
        spec["max_aa"],
        spec["n_wave"],
        dtype=np.float64,
    )
    return jnp.array(wave, dtype=jnp.float64)


@pytest.fixture
def L_ir(golden_data: dict) -> float:
    """Extract absorbed luminosity from golden metadata."""
    return golden_data["L_ir_erg_s"]


class TestDale2014GridPort:
    """Bit-exact regression for dale2014 SEDModelComponent."""

    def test_dale2014_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """Dale2014IRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "dale2014" not in templates:
            pytest.skip("dale2014 golden data not available")

        golden_npy = golden_dir / "dale2014.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "dale2014" in _REGISTRY
        comp_cls = _REGISTRY["dale2014"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["dale2014"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)


class TestDale2014CigaleGridPort:
    """Bit-exact regression for dale2014_cigale SEDModelComponent."""

    def test_dale2014_cigale_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """Dale2014CigaleIRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "dale2014_cigale" not in templates:
            pytest.skip("dale2014_cigale golden data not available")

        golden_npy = golden_dir / "dale2014_cigale.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "dale2014_cigale" in _REGISTRY
        comp_cls = _REGISTRY["dale2014_cigale"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["dale2014_cigale"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)


class TestDraineLi2007GridPort:
    """Bit-exact regression for draine_li2007 SEDModelComponent."""

    def test_draine_li2007_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """DraineLi2007IRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "draine_li2007" not in templates:
            pytest.skip("draine_li2007 golden data not available")

        golden_npy = golden_dir / "draine_li2007.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "draine_li2007" in _REGISTRY
        comp_cls = _REGISTRY["draine_li2007"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["draine_li2007"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)


class TestDraineLi2014GridPort:
    """Bit-exact regression for draine_li2014 SEDModelComponent."""

    def test_draine_li2014_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """DraineLi2014IRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "draine_li2014" not in templates:
            pytest.skip("draine_li2014 golden data not available")

        golden_npy = golden_dir / "draine_li2014.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "draine_li2014" in _REGISTRY
        comp_cls = _REGISTRY["draine_li2014"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["draine_li2014"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)


# NOTE (#871): the astrodust component was re-parameterized from the DL07-costume
# (umin/gamma/qpah) to the faithful Hensley & Draine 2023 native lgU model. Its
# bit-exact regression now lives in tests/regression/test_dust_goldens_852.py
# (component-native golden, energy-balance, no-op guard) — the lazy_loader golden path
# used here no longer applies.


class TestBosaGridPort:
    """Bit-exact regression for bosa SEDModelComponent."""

    def test_bosa_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """BosaIRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "bosa" not in templates:
            pytest.skip("bosa golden data not available")

        golden_npy = golden_dir / "bosa.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "bosa" in _REGISTRY
        comp_cls = _REGISTRY["bosa"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["bosa"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)


class TestThemisGridPort:
    """Bit-exact regression for themis SEDModelComponent."""

    def test_themis_bit_exact(
        self,
        golden_data: dict,
        golden_dir: Path,
        wave_grid: jnp.ndarray,
        L_ir: float,
    ):
        """ThemisIRSEDComponent matches closure exactly."""
        from tengri.components.sed_model_component import _REGISTRY

        # Skip if golden is missing or was skipped
        templates = golden_data.get("templates", {})
        if "themis" not in templates:
            pytest.skip("themis golden data not available")

        golden_npy = golden_dir / "themis.npy"
        if not golden_npy.exists():
            pytest.skip(f"Golden file not found: {golden_npy}")

        # Load component and golden
        assert "themis" in _REGISTRY
        comp_cls = _REGISTRY["themis"]
        comp = comp_cls()
        golden = np.load(golden_npy)

        # Extract params from golden
        params = templates["themis"]["params"]
        p_stripped = {k.replace("dust_", ""): v for k, v in params.items()}

        # Predict
        sed_in = jnp.zeros_like(wave_grid)
        sed_out, _ = comp.predict(p_stripped, sed_in, wave_grid, L_ir=L_ir)

        # Assert exact match (same closure)
        np.testing.assert_allclose(sed_out, golden, rtol=1e-14, atol=1e-15)
