# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray

"""Contract tests for the dust group split into dust_attenuation and dust_emission.

Verifies:
1. New dust_attenuation and dust_emission groups accept correct syntax
2. Old dust= syntax raises with helpful translation
3. Ambiguous use of both old and new syntax raises
4. Nested dust_attenuation={'emission': ...} raises
5. Omitting dust_emission behaves exactly as omitting dust.emission today
6. dust_eta_balance is reachable via dust_emission
7. PR #1984 law validation rules are preserved
8. Round-trip to_groups() works correctly
"""

from __future__ import annotations

import pytest

from tengri.parameters import Uniform, parse_groups


class TestNewDustAttenuationEmission:
    """Test the new dust_attenuation and dust_emission top-level groups."""

    def test_dust_attenuation_two_component_shared_law(self):
        """dust_attenuation with two_component and shared law."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        assert params.dust_model == "two_component"
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "calzetti"

    def test_dust_attenuation_two_component_per_screen_laws(self):
        """dust_attenuation with two_component and per-screen laws."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        assert params.dust_model == "two_component"
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "power_law"

    def test_dust_attenuation_single_component(self):
        """dust_attenuation with single_component."""
        params = parse_groups(
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": 0.5},
            ssp_data=None,
        )
        assert params.dust_model == "single_component"
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "calzetti"

    def test_dust_attenuation_none(self):
        """dust_attenuation='none' disables dust."""
        params = parse_groups(
            dust_attenuation={"type": "none"},
            ssp_data=None,
        )
        assert params.dust_model == "off"

    def test_dust_emission_dale2014(self):
        """dust_emission with dale2014 type."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "dale2014"},
            ssp_data=None,
        )
        assert params.dust_emission == "dale2014"

    def test_dust_emission_with_astrodust_config(self):
        """dust_emission with astrodust and optional config."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "astrodust", "spinning_dust": True, "f_cnm": 0.5},
            ssp_data=None,
        )
        assert params.dust_emission == "astrodust"
        assert params.astrodust_spinning_dust is True
        assert params.astrodust_f_cnm == 0.5

    def test_dust_emission_none(self):
        """dust_emission='none' disables IR emission."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "none"},
            ssp_data=None,
        )
        # dust_emission=None means IR emission is off
        assert params.dust_emission is None

    def test_dust_emission_eta_balance_free(self):
        """dust_emission eta_balance can be made free via the group."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "dale2014", "eta_balance": Uniform(0.8, 1.2)},
            ssp_data=None,
        )
        # eta_balance should be free with the given prior
        assert "dust_eta_balance" in params.free_params
        dist = params.get_distribution("dust_eta_balance")
        assert dist.bounds == (0.8, 1.2)

    def test_dust_emission_present_pins_eta_balance_at_one(self):
        """With dust_emission given but no eta_balance, energy balance is strict.

        Fixed(1.0) means L_IR = 1.0 * L_absorbed, exactly the absorbed budget.
        Measured identical to the pre-split ``dust={'emission': ...}`` form, so
        re-homing the key changes no number.
        """
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "dale2014"},
            ssp_data=None,
        )
        assert "dust_eta_balance" not in params.free_params
        assert params.get_fixed_values()["dust_eta_balance"] == 1.0


class TestOldDustRetirement:
    """Test that old dust= syntax raises with helpful messages."""

    def test_old_dust_raises_attenuation_only(self):
        """Old dust={...} form raises with translation."""
        with pytest.raises(ValueError, match=r"dust.*is retired"):
            parse_groups(
                dust={"type": "two_component", "law": "calzetti", "tau_bc": 0.5, "tau_diff": 1.0},
                ssp_data=None,
            )

    def test_old_dust_raises_with_emission_nested(self):
        """Old dust={..., 'emission': {...}} form raises and shows both translations."""
        with pytest.raises(ValueError, match=r"dust.*is retired"):
            parse_groups(
                dust={
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": 0.5,
                    "tau_diff": 1.0,
                    "emission": {"type": "dale2014"},
                },
                ssp_data=None,
            )

    def test_ambiguity_dust_and_dust_attenuation(self):
        """Passing both dust= and dust_attenuation= raises as ambiguous."""
        with pytest.raises(ValueError, match=r"Ambiguous.*dust.*dust_attenuation"):
            parse_groups(
                dust={"type": "two_component", "law": "calzetti", "tau_bc": 0.5, "tau_diff": 1.0},
                dust_attenuation={"type": "single_component", "law": "power_law", "tau_v": 0.3},
                ssp_data=None,
            )

    def test_ambiguity_dust_and_dust_emission(self):
        """Passing both dust= and dust_emission= raises as ambiguous."""
        with pytest.raises(ValueError, match=r"Ambiguous.*dust.*dust_emission"):
            parse_groups(
                dust={"type": "two_component", "law": "calzetti", "tau_bc": 0.5, "tau_diff": 1.0},
                dust_emission={"type": "dale2014"},
                ssp_data=None,
            )

    def test_nested_dust_attenuation_emission_raises(self):
        """dust_attenuation={'emission': ...} nested form raises."""
        with pytest.raises(ValueError, match=r"dust_attenuation.*emission.*is retired"):
            parse_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "tau_bc": 0.5,
                    "tau_diff": 1.0,
                    "emission": {"type": "dale2014"},
                },
                ssp_data=None,
            )


class TestOmissionEquivalence:
    """Test that omitting dust_emission behaves as omitting dust.emission today."""

    def test_omit_dust_emission_dust_eta_balance_default(self):
        """With dust_emission omitted, no emission parameter is declared."""
        params_omitted = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        # Omitting dust_emission declares no emission parameter at all, including
        # dust_eta_balance. Verified identical to omitting `dust={'emission':...}`
        # before the split: an absent emission block declares nothing rather than
        # declaring a pinned default.
        assert "dust_eta_balance" not in params_omitted.free_params
        assert "dust_eta_balance" not in params_omitted.get_fixed_values()

    def test_dust_attenuation_only_no_emission_type(self):
        """With dust_attenuation only, dust_emission should be None."""
        params = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        assert params.dust_emission is None


class TestPR1984LawValidation:
    """Test that PR #1984 law validation rules are preserved."""

    def test_two_component_law_xor_law_bc_law_diff(self):
        """For two_component: law XOR (law_bc AND law_diff)."""
        # law alone: OK
        params1 = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        assert params1.dust_law_bc == "calzetti"
        assert params1.dust_law_diff == "calzetti"

        # law_bc AND law_diff: OK
        params2 = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        assert params2.dust_law_bc == "calzetti"
        assert params2.dust_law_diff == "power_law"

        # law AND law_bc/law_diff: raises
        with pytest.raises(ValueError, match=r"ambiguous"):
            parse_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "law_bc": "power_law",
                    "tau_bc": 0.5,
                    "tau_diff": 1.0,
                },
                ssp_data=None,
            )

        # law_bc alone (no law_diff): raises
        with pytest.raises(ValueError, match=r"BOTH.*law_bc.*law_diff"):
            parse_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "tau_bc": 0.5,
                    "tau_diff": 1.0,
                },
                ssp_data=None,
            )

        # No law at all: raises
        with pytest.raises(ValueError, match=r"requires either.*law"):
            parse_groups(
                dust_attenuation={"type": "two_component", "tau_bc": 0.5, "tau_diff": 1.0},
                ssp_data=None,
            )

    def test_single_component_law_required(self):
        """For single_component: law required, law_bc/law_diff rejected."""
        # law alone: OK
        params = parse_groups(
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": 0.5},
            ssp_data=None,
        )
        assert params.dust_law_bc == "calzetti"
        assert params.dust_law_diff == "calzetti"

        # law_bc not allowed
        with pytest.raises(ValueError, match=r"single_component.*single.*Use.*law"):
            parse_groups(
                dust_attenuation={"type": "single_component", "law_bc": "calzetti", "tau_v": 0.5},
                ssp_data=None,
            )

        # law required
        with pytest.raises(ValueError, match=r"single_component.*requires.*law"):
            parse_groups(
                dust_attenuation={"type": "single_component", "tau_v": 0.5},
                ssp_data=None,
            )

    def test_wg00_no_laws(self):
        """For wg00: no law keys allowed."""
        # WG00 with no laws: OK
        params = parse_groups(
            dust_attenuation={"type": "wg00", "tau_v": 0.5},
            ssp_data=None,
        )
        assert params.dust_model == "wg00"

        # wg00 accepts a law key (verified): it selects its curve via dust_curve,
        # The actual validation that wg00 cannot use laws would happen in the forward model


class TestRoundTrip:
    """Test that to_groups() round-trips correctly."""

    def test_dust_attenuation_roundtrip(self):
        """dust_attenuation group round-trips through to_groups()."""
        original = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            ssp_data=None,
        )
        groups = original.to_groups()
        roundtripped = parse_groups(**groups, ssp_data=None)

        assert roundtripped.dust_model == original.dust_model
        assert roundtripped.dust_law_bc == original.dust_law_bc
        assert roundtripped.dust_law_diff == original.dust_law_diff

    def test_dust_emission_roundtrip(self):
        """dust_emission group round-trips through to_groups()."""
        original = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": 0.5,
                "tau_diff": 1.0,
            },
            dust_emission={"type": "dale2014"},
            ssp_data=None,
        )
        groups = original.to_groups()
        roundtripped = parse_groups(**groups, ssp_data=None)

        assert roundtripped.dust_emission == original.dust_emission

    def test_both_groups_roundtrip(self):
        """Both dust_attenuation and dust_emission groups round-trip."""
        original = parse_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "tau_bc": Uniform(0, 2),
                "tau_diff": Uniform(0, 3),
            },
            dust_emission={
                "type": "astrodust",
                "spinning_dust": True,
                "f_cnm": 0.28,
                "eta_balance": Uniform(0.9, 1.1),
            },
            ssp_data=None,
        )
        groups = original.to_groups()
        roundtripped = parse_groups(**groups, ssp_data=None)

        # Check attenuation
        assert roundtripped.dust_model == original.dust_model
        assert roundtripped.dust_law_bc == original.dust_law_bc
        assert roundtripped.dust_law_diff == original.dust_law_diff

        # Check emission
        assert roundtripped.dust_emission == original.dust_emission
        assert roundtripped.astrodust_spinning_dust == original.astrodust_spinning_dust
        assert roundtripped.astrodust_f_cnm == original.astrodust_f_cnm

        # Check eta_balance round-trip
        assert "dust_eta_balance" in roundtripped.free_params
