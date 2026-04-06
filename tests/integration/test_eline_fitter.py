"""Integration test: Fitter with emission line marginalization (skips if no SSP data)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)

from tengri.models.observation.line_list import LineCatalog
from tengri.models.observation.spectroscopy_config import SpectroscopyConfig


@pytest.fixture
def spec_config():
    wave_obs = jnp.linspace(4000, 7500, 500)
    return SpectroscopyConfig(
        wave_obs=wave_obs,
        resolution=2000.0,
        eline_mode="marginalized",
        eline_catalog=LineCatalog.default_13(),
        eline_prior_type="flat",
        eline_fix_doublets=True,
    )


class TestSpectroscopyConfig:
    def test_has_eline_fitting_marginalized(self, spec_config):
        assert spec_config.has_eline_fitting is True

    def test_has_eline_fitting_off(self):
        wave_obs = jnp.linspace(4000, 7500, 500)
        cfg = SpectroscopyConfig(wave_obs=wave_obs, resolution=2000.0)
        assert cfg.has_eline_fitting is False

    def test_effective_catalog_default(self):
        wave_obs = jnp.linspace(4000, 7500, 500)
        cfg = SpectroscopyConfig(wave_obs=wave_obs)
        cat = cfg.effective_catalog
        assert cat.n_lines == 13  # default_13

    def test_effective_catalog_custom(self, spec_config):
        cat = spec_config.effective_catalog
        assert cat.n_lines == 13

    def test_desi_like_factory(self):
        wave_obs = jnp.linspace(3600, 9800, 1000)
        cfg = SpectroscopyConfig.desi_like(wave_obs)
        assert cfg.eline_mode == "marginalized"
        assert cfg.eline_prior_type == "cloudy"
        assert cfg.calibration_order == 3
        assert cfg.eline_fix_doublets is True
        assert cfg.effective_catalog.n_lines >= 35
