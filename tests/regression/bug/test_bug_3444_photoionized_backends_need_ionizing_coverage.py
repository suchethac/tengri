# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #3444: every photoionized backend refuses an SSP with no ionizing coverage.

An SSP wavelength axis that starts above the Lyman limit (912 A) has no ionizing spectrum,
so no photoionized nebular backend can predict emission from it. Gutkin16 used to build
silently with zero ionizing flux, and Cloudy refused for the wrong reason. Each backend now
refuses such a grid by name, through one shared check (#3417 gave Cue the same refusal).
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.nebular.ionizing_spectrum import require_ionizing_coverage
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import ConfigError
from tests._cloudy_grid import write_synthetic_cloudy_grid

pytestmark = pytest.mark.regression_bug


def _ssp_on(wave) -> SSPData:
    n_met, n_age = 8, 15
    n_wave = wave.shape[0]
    rng = np.random.default_rng(0)
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.asarray(rng.uniform(0.5, 1.5, (n_met, n_age, n_wave)), dtype=jnp.float64),
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.1, n_age),
        ssp_lgmet=jnp.linspace(-2.0, 0.3, n_met),
    )


def _build_gutkin16(ssp: SSPData):
    from tengri.components.nebular.gutkin16 import Gutkin16Backend

    return Gutkin16Backend(ssp_data=ssp)


def _build_cloudy(ssp: SSPData, tmp_path):
    from tengri.components.nebular.cloudy_grid import CloudyGridBackend

    grid = write_synthetic_cloudy_grid(tmp_path / "grid.h5")
    return CloudyGridBackend(str(grid), ssp_data=ssp)


def _build_mappings(ssp: SSPData):
    from tengri.components.nebular.mappings_photo import MappingsPhotoStellarBackend

    # The SB99 ionizing source is unrelated to the SSP and raises by default; this test is
    # about coverage, so that warning is silenced here.
    return MappingsPhotoStellarBackend(ssp_data=ssp, ionizing_source_warning="suppress")


_TRUNCATED_BACKENDS = [
    pytest.param("gutkin16", _build_gutkin16, False, id="gutkin16"),
    pytest.param("cloudy", _build_cloudy, True, id="cloudy"),
    pytest.param("mappings", _build_mappings, False, id="mappings"),
    pytest.param(
        "cb19",
        None,
        False,
        id="cb19",
        marks=pytest.mark.skip(
            reason="CB19 cannot be built on this machine: its default grid is a placeholder"
        ),
    ),
]


class TestBug3444TruncatedSSPRefused:
    """Each photoionized backend refuses an SSP whose axis starts above the Lyman limit."""

    @pytest.mark.parametrize(("backend", "build", "needs_tmp"), _TRUNCATED_BACKENDS)
    def test_axis_above_lyman_limit_is_refused_by_name(self, backend, build, needs_tmp, tmp_path):
        ssp = _ssp_on(jnp.logspace(3.0, 4.5, 200))  # starts at 1000 A
        with pytest.raises(ConfigError) as info:
            build(ssp, tmp_path) if needs_tmp else build(ssp)
        message = str(info.value)
        assert "Lyman limit" in message
        assert backend in message


class TestBug3444ControlBuilds:
    """A grid that reaches below the Lyman limit still builds, so the check is not over-broad."""

    def test_gutkin16_axis_below_lyman_limit_builds(self):
        ssp = _ssp_on(jnp.logspace(2.0, 4.5, 300))  # reaches 100 A
        assert _build_gutkin16(ssp) is not None


class TestBug3444RequireIonizingCoverage:
    """The shared check on its own, with no model."""

    def test_raises_when_axis_starts_above_lyman_limit(self):
        with pytest.raises(ConfigError, match="Lyman limit"):
            require_ionizing_coverage(jnp.logspace(3.0, 4.5, 200), backend="test")

    def test_passes_when_axis_reaches_below_lyman_limit(self):
        require_ionizing_coverage(jnp.logspace(2.0, 4.5, 300), backend="test")

    def test_passes_when_exactly_one_point_sits_on_the_lyman_limit(self):
        from tengri.components.nebular.ionizing_spectrum import HI_LIMIT

        wave = np.array([HI_LIMIT, 1000.0, 2000.0, 5000.0])
        require_ionizing_coverage(wave, backend="test")
