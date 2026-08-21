# SPDX-License-Identifier: BSD-3-Clause
"""Test AGN attenuation law-key alignment: smc_prevot uses 'law', others use 'type'."""

import pytest

from tengri import FREE, Fixed, SEDModel, Uniform

pytestmark = pytest.mark.contract


class TestAgNAttenLawKey:
    """Test agn.atten law-key refactoring: smc_prevot via law, genuine models via type."""

    def test_atten_law_key_smc_prevot(self, synthetic_ssp_wide, simple_observation):
        """Test that smc_prevot can be selected via law key."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"law": "prevot_smc", "attenuation_ebv": Uniform(0.0, 0.5)}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "smc_prevot"
        assert "agn_attenuation_ebv" in model.spec.free_params

    def test_atten_law_key_invalid_dust_law_raises(self, synthetic_ssp_wide, simple_observation):
        """Test that invalid law name raises with suggestion."""
        with pytest.raises(ValueError) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=simple_observation,
                agn={"atten": {"law": "nonexistent_law"}},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        assert "law" in error_msg
        assert "prevot_smc" in error_msg or "dust law" in error_msg

    def test_atten_type_polar_dust_unchanged(self, synthetic_ssp_wide, simple_observation):
        """Test that polar_dust still uses type key (genuine model)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "polar_dust", "polar_ebv": 0.1}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "polar_dust"

    def test_atten_type_qsogen_unchanged(self, synthetic_ssp_wide, simple_observation):
        """Test that qsogen still uses type key (genuine model)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "qsogen", "attenuation_ebv": 0.1}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "qsogen"

    def test_atten_type_qsogen_smc_unchanged(self, synthetic_ssp_wide, simple_observation):
        """Test that qsogen_smc still uses type key (genuine model)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "qsogen_smc"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "qsogen_smc"

    def test_atten_type_grahsp_biatten_unchanged(self, synthetic_ssp_wide, simple_observation):
        """Test that grahsp_biatten still uses type key (genuine model)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "grahsp_biatten"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "grahsp_biatten"

    def test_atten_type_none_unchanged(self, synthetic_ssp_wide, simple_observation):
        """Test that none still uses type key (genuine model)."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "none"}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "none"

    def test_old_smc_prevot_type_key_raises(self, synthetic_ssp_wide, simple_observation):
        """Test that old type: smc_prevot raises with new law: prevot_smc example."""
        with pytest.raises(ValueError) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=simple_observation,
                agn={"atten": {"type": "smc_prevot"}},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Error should mention the new form with law key
        assert "law" in error_msg
        assert "prevot_smc" in error_msg
        # Error message should have parseable example
        assert "agn=" in error_msg  # Should suggest valid config syntax

    def test_atten_params_preserved_with_law_key(self, synthetic_ssp_wide, simple_observation):
        """Test that parameters are preserved when using law key."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={
                "atten": {
                    "law": "prevot_smc",
                    "attenuation_ebv": FREE,
                }
            },
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "smc_prevot"
        assert "agn_attenuation_ebv" in model.spec.free_params

    def test_atten_params_fixed_with_law_key(self, synthetic_ssp_wide, simple_observation):
        """Test that Fixed params work with law key."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={
                "atten": {
                    "law": "prevot_smc",
                    "attenuation_ebv": Fixed(0.2),
                }
            },
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_attenuation_block == "smc_prevot"
        # Fixed params should not be in free_params
        assert "agn_attenuation_ebv" not in model.spec.free_params

    def test_roundtrip_emits_law_key_for_smc_prevot(self, synthetic_ssp_wide, simple_observation):
        """Test that roundtrip emits law key for smc_prevot."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"law": "prevot_smc", "attenuation_ebv": 0.1}},
            redshift=Fixed(0.1),
        )
        groups = model.spec.to_groups()
        # Should emit law key, not type
        assert "atten" in groups["agn"]
        atten_dict = groups["agn"]["atten"]
        assert "law" in atten_dict
        assert atten_dict["law"] == "prevot_smc"
        assert "type" not in atten_dict

    def test_roundtrip_emits_type_for_genuine_models(self, synthetic_ssp_wide, simple_observation):
        """Test that roundtrip emits type key for genuine attenuation models."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            agn={"atten": {"type": "polar_dust", "polar_ebv": 0.1}},
            redshift=Fixed(0.1),
        )
        groups = model.spec.to_groups()
        assert "atten" in groups["agn"]
        atten_dict = groups["agn"]["atten"]
        assert "type" in atten_dict
        assert atten_dict["type"] == "polar_dust"
        assert "law" not in atten_dict

    def test_roundtrip_preserves_params(self, synthetic_ssp_wide, simple_observation):
        """Test that roundtrip preserves parameters through grammar."""
        original_groups = {
            "redshift": Fixed(0.1),
            "agn": {"atten": {"law": "prevot_smc", "attenuation_ebv": 0.15}},
        }
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            **original_groups,
        )
        groups = model.spec.to_groups()

        # Re-parse emitted groups
        model2 = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=simple_observation,
            **groups,
        )

        # Both models should have same attenuation block
        assert model.spec.agn_attenuation_block == model2.spec.agn_attenuation_block
        # Both should emit the same law key in groups
        groups1 = model.spec.to_groups()
        groups2 = model2.spec.to_groups()
        assert groups1["agn"]["atten"] == groups2["agn"]["atten"]

    def test_atten_law_and_type_together_raises(self, synthetic_ssp_wide, simple_observation):
        """Test that specifying both law and type keys raises ValueError.

        The grammar enforces an XOR: law='prevot_smc' for smc_prevot, or
        type=<other_model> for all other attenuation models. Specifying
        both is ambiguous and must raise with a clear error naming both keys.
        """
        with pytest.raises(ValueError) as exc_info:
            SEDModel.build(
                ssp_data=synthetic_ssp_wide,
                observation=simple_observation,
                agn={"atten": {"law": "prevot_smc", "type": "polar_dust"}},
                redshift=Fixed(0.1),
            )
        error_msg = str(exc_info.value)
        # Error must name both conflicting keys
        assert "law" in error_msg.lower()
        assert "type" in error_msg.lower()
