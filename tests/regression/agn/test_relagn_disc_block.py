# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for RELAGN relativistic disc composable block.

Verifies that the ``("disc", "relagn")`` composable AGN block is registered,
works with the composable runner, and that disc SEDs respond to parameter
variations (spin, accretion rate).

Grid-gated: requires ``data/relagn_disc_grid.h5`` (gitignored).

Marker: contract, regression_paper

References
----------
.. [1] Hagen, S. & Done, C. (2023). MNRAS, 521, 251.
       RELAGN: A relativistic accretion disc model for high spin and high
       inclination AGN. https://doi.org/10.1093/mnras/stad478
.. [2] Dovciak, M., Karas, V., & Yaqoob, T. (2004). ApJS, 153, 205.
       KYCONV: Emission from the accretion disk of a Kerr black hole.
       https://doi.org/10.1086/421115
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import pytest

pytestmark = [
    pytest.mark.contract,
    pytest.mark.regression_paper,
    pytest.mark.skipif(
        not Path("data/relagn_disc_grid.h5").exists(),
        reason="RELAGN grid not available (data/relagn_disc_grid.h5)",
    ),
]


class TestRelagnDiscBlockRegistration:
    """Tests: relagn disc block is registered and discoverable."""

    def test_relagn_disc_block_registered(self):
        """relagn disc block is in the AGN_BLOCKS registry.

        Notes
        -----
        **Marker:** contract

        The block must be discoverable via
        :data:`tengri.components.agn.blocks._protocol.AGN_BLOCKS`.
        """
        from tengri.components.agn.blocks._protocol import AGN_BLOCKS

        assert "relagn" in AGN_BLOCKS["disc"], (
            "relagn disc block not registered. Available disc blocks: "
            f"{sorted(AGN_BLOCKS['disc'])}"
        )

    def test_relagn_in_valid_agn_disc_types(self):
        """relagn is in _VALID_AGN_DISC_TYPES for builder validation.

        Notes
        -----
        **Marker:** contract

        The builder's group grammar must recognize the block selector.
        """
        from tengri.parameters.groups import _VALID_AGN_DISC_TYPES

        assert "relagn" in _VALID_AGN_DISC_TYPES, (
            f"relagn not in _VALID_AGN_DISC_TYPES. Available: {sorted(_VALID_AGN_DISC_TYPES)}"
        )

    def test_relagn_consumes_expected_params(self):
        """relagn disc block consumes the expected set of ``agn_*`` params.

        Notes
        -----
        **Marker:** contract

        Parameters consumed: agn_log_mbh, agn_log_mdot, agn_astar, agn_cos_inc.
        """
        from tengri.components.agn.blocks._consumes import AGN_BLOCK_CONSUMES

        expected = frozenset({"agn_log_mbh", "agn_log_mdot", "agn_astar", "agn_cos_inc"})
        actual = AGN_BLOCK_CONSUMES.get(("disc", "relagn"))

        assert actual is not None, "relagn disc block not in AGN_BLOCK_CONSUMES"
        assert actual == expected, (
            f"relagn consumes unexpected params. Expected {expected}, got {actual}"
        )


class TestRelagnDiscBlockPhysics:
    """Tests: relagn disc block produces valid SEDs and responds to parameters."""

    @pytest.fixture
    def wavelength(self):
        """Standard test wavelength grid [Å]."""
        return jnp.logspace(2, 5, 256)

    def test_relagn_composable_produces_finite_sed(self, wavelength):
        """Composable relagn (disc=relagn, torus=none) produces finite SED.

        Notes
        -----
        **Marker:** contract

        A minimal recipe with relagn disc and no other components should
        emit a finite, non-zero L_ν.
        """
        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        sed = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-1.0,
            agn_astar=0.0,
            agn_cos_inc=0.86602540378443864,
        )

        # All values must be finite and positive
        assert jnp.all(jnp.isfinite(sed)), "SED contains NaN or Inf"
        assert jnp.all(sed > 0.0), "SED has non-positive values"

        # Peak flux should be reasonable (> 1e19 erg/s/Hz)
        assert float(jnp.max(sed)) > 1e19, (
            f"SED peak suspiciously low: {jnp.max(sed):.2e} erg/s/Hz"
        )

    def test_relagn_spin_variation_changes_sed(self, wavelength):
        """Different spin parameters (agn_astar) produce different SEDs.

        Notes
        -----
        **Marker:** regression_paper

        RELAGN disc spectrum depends sensitively on black hole spin; low-spin
        (a*=0) and high-spin (a*=0.9) should differ significantly.
        """
        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        # Low spin
        sed_low_spin = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-1.0,
            agn_astar=0.0,
            agn_cos_inc=0.86602540378443864,
        )

        # High spin
        sed_high_spin = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-1.0,
            agn_astar=0.9,
            agn_cos_inc=0.86602540378443864,
        )

        # SEDs must be different (not just floating-point rounding)
        relative_diff = jnp.abs(sed_high_spin - sed_low_spin) / jnp.maximum(sed_low_spin, 1e-30)
        mean_diff = float(jnp.mean(relative_diff))

        assert mean_diff > 0.01, (
            f"Spin variation too small: mean relative diff = {mean_diff:.2e}. "
            "Check that agn_astar is actually used."
        )

    def test_relagn_accretion_rate_variation_changes_sed(self, wavelength):
        """Different accretion rates (agn_log_mdot) produce different SEDs.

        Notes
        -----
        **Marker:** regression_paper

        RELAGN disc luminosity is self-consistent with accretion rate; sub-
        and super-Eddington rates should produce markedly different SEDs.
        """
        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        # Sub-Eddington
        sed_sub_edd = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-1.5,
            agn_astar=0.0,
            agn_cos_inc=0.86602540378443864,
        )

        # Super-Eddington
        sed_super_edd = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=0.3,
            agn_astar=0.0,
            agn_cos_inc=0.86602540378443864,
        )

        # SEDs must be different
        relative_diff = jnp.abs(sed_super_edd - sed_sub_edd) / jnp.maximum(sed_sub_edd, 1e-30)
        mean_diff = float(jnp.mean(relative_diff))

        assert mean_diff > 0.01, (
            f"Accretion-rate variation too small: mean relative diff = "
            f"{mean_diff:.2e}. Check that agn_log_mdot is actually used."
        )


class TestRelagnIndexSpaceInterp:
    """Golden values for relagn with index-space interpolation (#2061).

    Grid: wavelength 100-100000 Angstrom (256 points, logspace). Parameters:
    log_mbh=8.0, log_mdot=-0.5, agn_cos_inc=0.5.

    A/B sweep (main vs fix): 40-point astar in [0, 0.998], max change 19% at
    a*=0.998 (concentrates above a*~0.93). Physical-space (unfixed main) gave
    1.88% error at a*=0.9675 and 3.86% at a*=0.998 due to non-uniform node
    over-smoothing. Index-space correction (fix) reduces these to reference
    values below.
    """

    def test_relagn_astar_golden_0p9675(self):
        """Golden: sum(L_nu) at a*=0.9675 (#2061 index-space fix).

        Notes
        -----
        **Marker:** regression_bug

        Old (unfixed main, physical-space): 4.357619789700e+31
        New (worktree, index-space):        4.275819488080e+31
        Relative change:                   -1.88% (physical smooths over-much)
        """
        wavelength = jnp.logspace(2, 5, 256)

        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        sed = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-0.5,
            agn_astar=0.9675,
            agn_cos_inc=0.5,
        )
        obj = float(jnp.sum(sed))

        assert jnp.isclose(obj, 4.275819488080e31, rtol=1e-6), (
            f"a*=0.9675: expected 4.275819488080e+31, got {obj:.12e}. "
            f"(FAILS on unfixed main with ~1.88% error)"
        )

    def test_relagn_astar_golden_0p998(self):
        """Golden: sum(L_nu) at a*=0.998 (#2061 index-space fix).

        Notes
        -----
        **Marker:** regression_bug

        Old (unfixed main, physical-space): 4.303561013957e+31
        New (worktree, index-space):        4.137520596764e+31
        Relative change:                   -3.86% (physical severely smooths)
        """
        wavelength = jnp.logspace(2, 5, 256)

        from tengri.components.agn.blocks.runner import composable_agn_l_nu

        sed = composable_agn_l_nu(
            wavelength,
            agn_log_lbol=12.0,
            agn_disc_block="relagn",
            agn_nlr_block="none",
            agn_blr_block="none",
            agn_feii_block="none",
            agn_torus_block="none",
            agn_attenuation_block="none",
            agn_log_mbh=8.0,
            agn_log_mdot=-0.5,
            agn_astar=0.998,
            agn_cos_inc=0.5,
        )
        obj = float(jnp.sum(sed))

        assert jnp.isclose(obj, 4.137520596764e31, rtol=1e-6), (
            f"a*=0.998: expected 4.137520596764e+31, got {obj:.12e}. "
            f"(FAILS on unfixed main with ~3.86% error)"
        )
