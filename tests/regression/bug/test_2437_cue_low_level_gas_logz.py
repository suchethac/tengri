# SPDX-License-Identifier: BSD-3-Clause
"""#2437: Cue low-level path threading of neb_logZ_gas parameter.

The low-level path (_resolve_cue_params with ssp_weights=None) was dropping
neb_logZ_gas and using 0.0 always, while the high-level path honored it. This
test verifies the low-level path now responds to neb_logZ_gas, matching the
high-level behavior.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri.components.nebular import (
    _DEFAULT_CUE_WEIGHTS_PATH,
    CueBackend,
)

pytestmark = pytest.mark.regression_bug


def test_cue_low_level_gas_logz_varies_sed():
    """Low-level Cue backend responds to neb_logZ_gas: max rel diff > 1e-3."""
    jax.config.update("jax_enable_x64", True)

    # Check weights file exists
    if not Path(_DEFAULT_CUE_WEIGHTS_PATH).exists():
        pytest.skip(f"Cue weights not found at {_DEFAULT_CUE_WEIGHTS_PATH}")

    be = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))

    wave = np.logspace(2.5, 5.0, 3000)
    wave_jnp = jnp.asarray(wave)

    # Reference at neb_logZ_gas = 0.0
    ref = np.asarray(
        be.predict_nebular_sed(
            ssp_wave=wave_jnp,
            ssp_weights=None,
            ssp_log_ages_yr=None,
            log_z=jnp.asarray(0.0),
            neb_logU=jnp.asarray(-3.0),
            neb_logZ_gas=jnp.asarray(0.0),
            gas_logqion=jnp.asarray(53.0),
        )
    )

    # Vary neb_logZ_gas; each should produce a different SED
    for z_val in [-1.0, 0.3]:
        sed = np.asarray(
            be.predict_nebular_sed(
                ssp_wave=wave_jnp,
                ssp_weights=None,
                ssp_log_ages_yr=None,
                log_z=jnp.asarray(0.0),
                neb_logU=jnp.asarray(-3.0),
                neb_logZ_gas=jnp.asarray(z_val),
                gas_logqion=jnp.asarray(53.0),
            )
        )
        max_rel_diff = np.max(np.abs(sed - ref)) / np.max(np.abs(ref))
        assert max_rel_diff > 1e-3, f"neb_logZ_gas={z_val} should move SED, got {max_rel_diff:.2e}"


def test_cue_low_level_gas_logz_positive_control():
    """Verify neb_logU (positive control) also moves the SED on low-level path."""
    jax.config.update("jax_enable_x64", True)

    # Check weights file exists
    if not Path(_DEFAULT_CUE_WEIGHTS_PATH).exists():
        pytest.skip(f"Cue weights not found at {_DEFAULT_CUE_WEIGHTS_PATH}")

    be = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))

    wave = np.logspace(2.5, 5.0, 3000)
    wave_jnp = jnp.asarray(wave)

    # Reference at neb_logU = -3.0, neb_logZ_gas = 0.0
    ref = np.asarray(
        be.predict_nebular_sed(
            ssp_wave=wave_jnp,
            ssp_weights=None,
            ssp_log_ages_yr=None,
            log_z=jnp.asarray(0.0),
            neb_logU=jnp.asarray(-3.0),
            neb_logZ_gas=jnp.asarray(0.0),
            gas_logqion=jnp.asarray(53.0),
        )
    )

    # Positive control: neb_logU should move SED (was already working)
    sed_logu = np.asarray(
        be.predict_nebular_sed(
            ssp_wave=wave_jnp,
            ssp_weights=None,
            ssp_log_ages_yr=None,
            log_z=jnp.asarray(0.0),
            neb_logU=jnp.asarray(-2.0),
            neb_logZ_gas=jnp.asarray(0.0),
            gas_logqion=jnp.asarray(53.0),
        )
    )
    max_rel_logu = np.max(np.abs(sed_logu - ref)) / np.max(np.abs(ref))
    assert max_rel_logu > 1e-3, f"neb_logU positive control failed: {max_rel_logu:.2e}"


def test_cue_unit_conversion_solar_metallicity():
    """Solar neb_logZ_gas (absolute) resolves to gas_logz=0.0 (relative)."""
    from tengri.components.nebular._constants import _LOG10_ZSUN

    jax.config.update("jax_enable_x64", True)

    # Check weights file exists
    if not Path(_DEFAULT_CUE_WEIGHTS_PATH).exists():
        pytest.skip(f"Cue weights not found at {_DEFAULT_CUE_WEIGHTS_PATH}")

    be = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH))

    # Solar metallicity in absolute log10(Z): -1.848
    solar_z_abs = _LOG10_ZSUN

    # Resolve with the low-level path
    resolved = be._resolve_cue_params(
        ssp_weights=None,
        neb_logZ_gas=solar_z_abs,
    )

    # Should resolve to relative log10(Z/Zsun) = 0.0
    assert abs(resolved["gas_logz"] - 0.0) < 1e-10, (
        f"Solar Z should resolve to 0.0, got {resolved['gas_logz']}"
    )


def test_cue_low_level_matches_high_level_gas_logz_conversion():
    """Low-level gas_logz conversion agrees with the high-level one (cue.py ~1606).

    Both paths are supposed to apply the SAME ``neb_logZ_gas - LOG10_ZSUN``
    conversion (``_resolve_cue_params``'s ``_default_gas_logz`` for the
    low-level branch; ``_compute_weighted_cue_params``'s ``gas_logz_rel`` for
    the high-level one), so for the same explicit ``neb_logZ_gas`` they must
    agree exactly, not merely each independently look reasonable.
    """
    jax.config.update("jax_enable_x64", True)

    _bare_ssp = "data/fsps_prsc_miles_chabrier.h5"
    if not Path(_bare_ssp).is_file():
        pytest.skip(f"missing bare SSP {_bare_ssp}")
    if not Path(_DEFAULT_CUE_WEIGHTS_PATH).exists():
        pytest.skip(f"Cue weights not found at {_DEFAULT_CUE_WEIGHTS_PATH}")

    ssp = tengri.load_ssp_data(_bare_ssp)
    be = CueBackend(str(_DEFAULT_CUE_WEIGHTS_PATH), ssp_data=ssp)

    n_age = ssp.ssp_lg_age_gyr.shape[0]
    ssp_log_ages_yr = jnp.asarray(ssp.ssp_lg_age_gyr) + 9.0  # log10(Gyr) -> log10(yr)
    # Concentrate weight on the youngest bins so Q_H is non-degenerate;
    # the exact SFH shape is irrelevant here -- only gas_logz is compared.
    ssp_weights = jnp.where(jnp.arange(n_age) < 5, 1.0 / 5.0, 0.0)

    for x in (-1.0, 0.0, 0.3):
        low = be._resolve_cue_params(ssp_weights=None, neb_logZ_gas=x)
        high = be._resolve_cue_params(
            ssp_weights=ssp_weights,
            ssp_log_ages_yr=ssp_log_ages_yr,
            log_z=-2.0,  # irrelevant: neb_logZ_gas is explicit on both paths
            neb_logZ_gas=x,
        )
        assert abs(float(low["gas_logz"]) - float(high["gas_logz"])) < 1e-12, (
            f"neb_logZ_gas={x}: low-level gas_logz={low['gas_logz']} != "
            f"high-level gas_logz={high['gas_logz']}"
        )
