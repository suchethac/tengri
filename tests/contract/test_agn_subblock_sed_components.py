# SPDX-License-Identifier: BSD-3-Clause
"""Contract: AGN per-sub-block public SED components (task13, NAMING_CONTRACT §4b.5).

The composable AGN assembly (:func:`tengri.components.agn.blocks.runner.compose_l_nu`)
sums a disc, a torus, NLR+BLR+FeII lines, and (when the ``polar_dust``
attenuation block is selected) a polar-dust re-emission graybody into one
``sed_agn`` array. Before this task, only the summed total was published;
this file pins that each addend is ALSO reachable as its own named
component -- ``sed_agn_disc``, ``sed_agn_torus``, ``sed_agn_lines``
(nlr + blr + feii), ``sed_agn_polar`` -- via
``model.predict(params).sed.components[...]``, and that the four sum
exactly (to floating-point reassociation) back to ``sed_agn``.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn.component import AGNSEDComponent, AGNSEDComponentConfig
from tengri.protocols.component import ForwardState

_WAVE = jnp.logspace(jnp.log10(500.0), jnp.log10(1e8), 400)

_SUBBLOCK_KEYS = ("sed_agn_disc", "sed_agn_torus", "sed_agn_lines", "sed_agn_polar")

#: A fully-populated composable recipe: every sub-block category active
#: (disc, torus, nlr, blr, feii) plus the standalone polar_dust attenuation
#: block, so all four public components are simultaneously non-trivial.
#: 'independent' norm per the AGN norm-policy rule (no cross-block energy
#: coupling to confound the sum-contract check).
_COMPOSABLE_CONFIG = AGNSEDComponentConfig(
    model="composable",
    agn_disc_block="multicolor",
    agn_nlr_block="analytic",
    agn_blr_block="analytic",
    agn_feii_block="boroson_green",
    agn_torus_block="two_temperature",
    agn_attenuation_block="polar_dust",
    agn_norm="independent",
)

_PARAMS = {
    "agn_log_lbol": jnp.asarray(12.0),
    "agn_lum_ratio": jnp.asarray(1.0),
    "agn_cos_inc": jnp.asarray(0.6),
    "agn_theta_torus": jnp.asarray(30.0),
    "agn_polar_ebv": jnp.asarray(0.3),
    "agn_polar_oa": jnp.asarray(45.0),
    "agn_polar_T": jnp.asarray(120.0),
    "agn_polar_beta": jnp.asarray(1.6),
}


def _apply(config: AGNSEDComponentConfig, params: dict) -> ForwardState:
    state0 = ForwardState(wave=_WAVE)
    return AGNSEDComponent(config=config).apply(state0, params)


class TestSubblockComponentsPresentForComposable:
    def test_all_four_keys_present_and_finite(self):
        state = _apply(_COMPOSABLE_CONFIG, _PARAMS)
        for key in ("sed_agn_disc", "sed_agn_torus", "sed_agn_lines", "sed_agn_polar"):
            assert key in state.derived, f"{key} missing from AGN derived publications"
            arr = np.asarray(state.derived[key])
            assert arr.shape == (_WAVE.shape[0],)
            assert np.all(np.isfinite(arr))

    def test_components_sum_to_sed_agn(self):
        state = _apply(_COMPOSABLE_CONFIG, _PARAMS)
        sed_agn = np.asarray(state.derived["sed_agn"])
        total = (
            np.asarray(state.derived["sed_agn_disc"])
            + np.asarray(state.derived["sed_agn_torus"])
            + np.asarray(state.derived["sed_agn_lines"])
            + np.asarray(state.derived["sed_agn_polar"])
        )
        denom = np.max(np.abs(sed_agn)) + 1e-300
        max_rel_diff = np.max(np.abs(total - sed_agn)) / denom
        assert max_rel_diff < 1e-12, (
            f"sed_agn_disc + sed_agn_torus + sed_agn_lines + sed_agn_polar differs "
            f"from sed_agn by {max_rel_diff:.3e} relative (expected < 1e-12)."
        )

    def test_polar_component_is_zero_when_atten_is_not_polar_dust(self):
        """sed_agn_polar is present but all-zero when a different atten block
        is selected -- "configured but inactive" (acceptable per the task
        design), never silently omitted, and the sum contract still holds."""
        cfg = AGNSEDComponentConfig(
            **{**_COMPOSABLE_CONFIG.__dict__, "agn_attenuation_block": "smc_prevot"}
        )
        state = _apply(cfg, {**_PARAMS, "agn_attenuation_ebv": jnp.asarray(0.2)})
        polar = np.asarray(state.derived["sed_agn_polar"])
        assert np.all(polar == 0.0)
        sed_agn = np.asarray(state.derived["sed_agn"])
        total = (
            np.asarray(state.derived["sed_agn_disc"])
            + np.asarray(state.derived["sed_agn_torus"])
            + np.asarray(state.derived["sed_agn_lines"])
            + polar
        )
        denom = np.max(np.abs(sed_agn)) + 1e-300
        assert np.max(np.abs(total - sed_agn)) / denom < 1e-12


class TestSubblockComponentsAbsentForMonolithic:
    @pytest.mark.parametrize("agn_model", ["multicolor_agn", "kubota_done", "qsogen"])
    def test_absent_for_non_composable_models(self, agn_model):
        config = AGNSEDComponentConfig(model=agn_model)
        state = _apply(config, {"agn_log_lbol": jnp.asarray(11.0)})
        assert "sed_agn" in state.derived
        for key in ("sed_agn_disc", "sed_agn_torus", "sed_agn_lines", "sed_agn_polar"):
            assert key not in state.derived, (
                f"{key} should be absent for the monolithic model {agn_model!r} "
                "(only the composable runner decomposes sub-blocks)."
            )


class TestSubblockComponentsZeroForMonolithicViaPredictSed:
    """task13 fix-round-1 item 5: the public model.predict(...).sed.components
    surface (not just the lower-level AGNSEDComponent.apply() state.derived
    dict tested above) returns zeros-shaped arrays for the four keys when the
    AGN model is a monolithic (non-composable) one, going through a full
    SEDModel.build -- state_to_sed_components' "absent reads as zero"
    convention shared by every other component key in that dict."""

    def test_monolithic_model_gives_zero_subblock_components(self, synthetic_ssp_wide):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "multicolor_agn",
                "all_params": Fixed(DEFAULT),
                "agn_log_lbol": Fixed(12.0),
            },
            redshift=Fixed(0.05),
        )
        pred = model.predict(model.spec.get_fixed_values())
        comp = pred.sed.components
        # sed_agn itself is genuinely nonzero -- this is not a "no AGN at
        # all" degenerate case, just a monolithic one with no sub-blocks.
        assert float(np.max(np.abs(np.asarray(comp["sed_agn"])))) > 0.0
        for key in _SUBBLOCK_KEYS:
            arr = np.asarray(comp[key])
            assert arr.shape == comp["sed_agn"].shape
            assert np.all(arr == 0.0), f"{key} is not all-zero for a monolithic AGN model."


class TestSubblockComponentsReachableViaPredictSed:
    """End-to-end: model.predict(params).sed.components[...] (NAMING_CONTRACT §4b.5)."""

    def test_reachable_via_predict_sed_components(self, synthetic_ssp_wide):
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "law": "power_law",
                "type": "two_component",
                "all_params": Fixed(DEFAULT),
            },
            agn={
                "type": "composable",
                "disc": {"type": "multicolor"},
                "torus": {"type": "two_temperature"},
                "nlr": {"type": "analytic"},
                "blr": {"type": "none"},
                "atten": {"type": "polar_dust", "polar_ebv": 0.3},
                "agn_log_lbol": Fixed(12.0),
                "norm": "independent",
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.05),
        )
        pred = model.predict(model.spec.get_fixed_values())
        comp = pred.sed.components
        for key in ("sed_agn_disc", "sed_agn_torus", "sed_agn_lines", "sed_agn_polar"):
            assert key in comp
        sed_agn = np.asarray(comp["sed_agn"])
        total = sum(np.asarray(comp[k]) for k in _SUBBLOCK_KEYS)
        denom = np.max(np.abs(sed_agn)) + 1e-300
        assert np.max(np.abs(total - sed_agn)) / denom < 1e-12


class TestPolarDustSharesTheAgnDustBudget:
    """The AGN dust budget includes the polar re-emission (R59).

    Under ``agn_norm='cigale_joint'`` and ``'conserving'``, torus + polar
    **are** the AGN dust budget: the polar screen reprocesses disc light into
    the FIR and takes its share out of the budget the torus block already
    normalized itself to, so the AGN dust total does not move when
    ``agn_polar_ebv`` does. That is CIGALE's accounting -- ``skirtor2016.py``
    adds the polar blackbody to ``dust`` BEFORE ``norm = 1/int dust``, then
    reports ``lumin_dust = agn_power`` and
    ``lumin_torus = agn_power - lumin_polar_dust``.

    Before this fix the graybody was added on top, so the AGN dust total grew
    with reddening: measured at the SKIRTOR fiducial, 1.000 / 1.409 / 1.915 /
    2.332 over ``agn_polar_ebv`` = 0 / 0.03 / 0.10 / 0.30. It is now invariant
    to floating point.

    Under ``'independent'`` each component stays on its own luminosity scale
    -- that is what the policy means -- so there the re-emission is additive
    and the total DOES grow. Both halves are pinned: an invariance test that
    passed under every policy would not be measuring the budget.
    """

    _EBVS = (0.0, 0.03, 0.10, 0.30)

    def _totals(self, norm):
        """AGN dust total (polar + torus) at each E(B-V), one policy."""
        from tengri.components.agn.blocks.runner import compose_l_nu
        from tengri.utils.physics_constants import C_AA

        nu = C_AA / _WAVE
        order = jnp.argsort(nu)
        out = []
        for ebv in self._EBVS:
            _sed, comps = compose_l_nu(
                _WAVE,
                12.0,
                agn_disc_block="schartmann2005",
                agn_nlr_block="none",
                agn_blr_block="none",
                agn_feii_block="none",
                agn_torus_block="skirtor",
                agn_attenuation_block="polar_dust",
                agn_norm=norm,
                agn_ir_frac=0.3,
                agn_polar_ebv=ebv,
                agn_polar_oa=40.0,
                agn_polar_T=100.0,
                agn_polar_beta=1.6,
                agn_oa_skirtor=40.0,
                return_components=True,
            )
            total = 0.0
            for key in ("polar", "torus"):
                arr = jnp.asarray(comps[key])
                total += float(jnp.abs(jnp.trapezoid(arr[order], nu[order])))
            out.append(total)
        return out

    def test_cigale_joint_total_is_invariant_in_ebv(self):
        """torus + polar = the budget, so reddening only re-partitions it."""
        totals = self._totals("cigale_joint")
        assert totals[0] > 0.0, "probe setup failed: no AGN dust at all"
        for ebv, total in zip(self._EBVS[1:], totals[1:], strict=True):
            rel = abs(total / totals[0] - 1.0)
            assert rel < 1e-6, (
                f"agn_polar_ebv={ebv}: AGN dust total moved by {rel:.3e} relative "
                f"({total:.6e} vs {totals[0]:.6e}). Under cigale_joint the polar "
                "re-emission must come OUT of the AGN dust budget, not be added "
                "on top of it (R59)."
            )

    def test_conserving_total_is_invariant_in_ebv(self):
        """The conserving ledger shares the budget the same way."""
        totals = self._totals("conserving")
        assert totals[0] > 0.0
        for total in totals[1:]:
            assert abs(total / totals[0] - 1.0) < 1e-6

    def test_independent_total_grows_with_ebv(self):
        """The negative control: 'independent' keeps the additive contract.

        Without this, the invariance tests above would pass just as well on a
        build where polar dust did nothing at all.
        """
        totals = self._totals("independent")
        assert totals[0] > 0.0
        assert totals[-1] > 1.05 * totals[0], (
            "under agn_norm='independent' the polar re-emission is additive by "
            f"contract, so the AGN dust total must grow with E(B-V); got "
            f"{totals[-1]:.6e} vs {totals[0]:.6e}"
        )

    def test_polar_share_is_positive_and_below_one(self):
        """The share is a partition, so no E(B-V) can zero or invert the torus."""
        from tengri.components.agn.blocks.runner import compose_l_nu
        from tengri.utils.physics_constants import C_AA

        nu = C_AA / _WAVE
        order = jnp.argsort(nu)
        for ebv in (0.03, 0.3, 1.0):
            _sed, comps = compose_l_nu(
                _WAVE,
                12.0,
                agn_disc_block="schartmann2005",
                agn_nlr_block="none",
                agn_blr_block="none",
                agn_feii_block="none",
                agn_torus_block="skirtor",
                agn_attenuation_block="polar_dust",
                agn_norm="cigale_joint",
                agn_ir_frac=0.3,
                agn_polar_ebv=ebv,
                agn_polar_oa=40.0,
                return_components=True,
            )
            integ = {
                k: float(jnp.abs(jnp.trapezoid(jnp.asarray(comps[k])[order], nu[order])))
                for k in ("polar", "torus")
            }
            share = integ["polar"] / (integ["polar"] + integ["torus"])
            assert 0.0 < share < 1.0, f"agn_polar_ebv={ebv}: share={share}"
            assert integ["torus"] > 0.0, f"agn_polar_ebv={ebv}: torus driven to zero"
