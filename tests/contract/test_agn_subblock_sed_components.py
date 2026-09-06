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
