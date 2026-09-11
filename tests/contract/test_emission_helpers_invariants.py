# SPDX-License-Identifier: BSD-3-Clause
"""Invariant tests for emission_helpers.py.

Bug classes covered:
- IGM: z=0 → transmission=1, no NaN for short-wavelength photons.

``attenuate_emission``, this module's dust-attenuation helper for emission
components, was removed in #2223 (no public callers; the no-state line-screen
fallback now dispatches to the dust component's own
``attenuate_line_catalog``, so its old ``{n_slope, dust_bump_strength}`` kwarg
dict -- which could not thread ``dust_delta``/``dust_Rv``/``redshift`` -- is
gone). The classes this file used to run through that function covered:

- shape (output shape matches input wave shape) and mode completeness (all
  four modes return finite arrays): now exercised on the live dust components
  in ``tests/components/dust/test_dust.py`` (two-component) and
  ``tests/components/dust/test_single_component_dust.py`` (single-component);
- zero-dust identity (tau=0 -> attenuation ~1): same two files;
- absorbed luminosity / energy conservation (L_transmitted + L_absorbed =
  L_incident): ``tests/contract/test_dust_energy_balance.py``;
- the line-vs-continuum parity ``attenuate_emission``'s missing kwargs broke
  (#2223 itself): ``tests/regression/bug/test_bug_2223_line_screen_kwargs.py``.

No SSP data needed. The remaining test uses synthetic jnp arrays.
"""

from __future__ import annotations

import chex
import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tests._bounds import assert_non_negative

# ── IGM absorption ────────────────────────────────────────────────


class TestIGMAbsorption:
    def test_igm_at_z0_is_all_ones(self) -> None:
        """At z=0, IGM is transparent: transmission should be ≈ 1 everywhere."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(900.0, 10000.0, 100)
        trans = igm_transmission(wave_obs, 0.0)
        chex.assert_tree_all_finite(trans)
        # At z=0, transmission must be 1 everywhere (no absorbers along sightline)
        assert jnp.allclose(trans, 1.0, atol=1e-6), (
            f"IGM transmission at z=0 deviates from 1. min={float(jnp.min(trans)):.4f}"
        )

    def test_igm_no_nan_short_wavelength(self) -> None:
        """No NaN at short wavelengths for moderate z (negative base fractional-power bug)."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(100.0, 1000.0, 80)
        trans = igm_transmission(wave_obs, 2.0)
        assert jnp.all(jnp.isfinite(trans)), (
            "IGM transmission contains NaN/Inf at short wavelengths "
            "(possible negative-base fractional-power bug)"
        )

    def test_igm_transmission_between_0_and_1(self) -> None:
        """IGM transmission is a fraction in [0, 1]."""
        from tengri.components.igm import igm_transmission

        wave_obs = jnp.linspace(1000.0, 10000.0, 120)
        trans = igm_transmission(wave_obs, 3.0)
        assert_non_negative(trans, name="trans", msg="IGM transmission has negative values")
        assert jnp.all(trans <= 1.0 + 1e-6), "IGM transmission exceeds 1.0"

    def test_igm_increases_opacity_with_redshift(self) -> None:
        """Higher redshift → more IGM absorption → lower average transmission at Lya."""
        from tengri.components.igm import igm_transmission

        # Short wavelengths (Lyman-series) — absorbed more at higher z
        wave_obs = jnp.linspace(1000.0, 2000.0, 50)
        trans_z1 = igm_transmission(wave_obs, 1.0)
        trans_z4 = igm_transmission(wave_obs, 4.0)
        assert float(jnp.mean(trans_z4)) <= float(jnp.mean(trans_z1)) + 1e-3, (
            "Higher z should give less IGM transmission, not more"
        )
