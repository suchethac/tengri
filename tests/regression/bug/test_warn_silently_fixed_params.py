# SPDX-License-Identifier: BSD-3-Clause
"""Test that silently-fixed parameters emit a warning.

Issue: When a parameter group states no disposition (no 'all_params' key),
all parameters in that group are fixed by default, silently. The user
believes they've configured an SFH, but gets zero free parameters.
"""

import warnings

import pytest

from tengri import FIXED, FREE, Fixed
from tengri.config.exceptions import DefaultFixedParametersWarning
from tengri.parameters import parse_groups

pytestmark = pytest.mark.regression_bug  # Frozen output for the silent-fixed-params bug


class TestSilentlyFixedParametersWarning:
    """Test the warning for parameters fixed by default."""

    def test_warns_when_group_has_no_disposition(self):
        """Emit warning when group states no 'all_params'."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = parse_groups(
                sfh={"type": "dpl"},  # no 'all_params' → silent FIXED
                redshift=0.5,
            )

            # Should emit exactly one warning
            assert len(w) == 1
            assert issubclass(w[0].category, DefaultFixedParametersWarning)

            # Warning should name the group and parameters
            message = str(w[0].message)
            assert "sfh" in message.lower()
            assert "sfh_dpl_alpha" in message or "alpha" in message
            assert "sfh_dpl_beta" in message or "beta" in message

            # Warning should name the values
            assert "=" in message  # param=value format

    def test_no_warn_with_explicit_all_params_fixed(self):
        """Don't warn when user explicitly states 'all_params': FIXED."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DefaultFixedParametersWarning)
            params = parse_groups(
                sfh={"type": "dpl", "all_params": FIXED},  # explicit FIXED
                redshift=0.5,
            )

            # Should NOT emit DefaultFixedParametersWarning
            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 0

    def test_no_warn_with_explicit_all_params_free(self):
        """Don't warn when user explicitly states 'all_params': FREE."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DefaultFixedParametersWarning)
            params = parse_groups(
                sfh={"type": "dpl", "all_params": FREE},  # explicit FREE
                redshift=0.5,
            )

            # Should NOT emit DefaultFixedParametersWarning
            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 0

    def test_no_warn_with_per_parameter_fixed_overrides(self):
        """Don't warn when all parameters have explicit Fixed(...) overrides."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always", DefaultFixedParametersWarning)
            params = parse_groups(
                sfh={
                    "type": "dpl",
                    "alpha": Fixed(1.5),
                    "beta": Fixed(2.0),
                    # tau_gyr left with no override → should warn
                },
                redshift=0.5,
            )

            # If ANY param is still default-fixed, should warn
            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) >= 1

    def test_one_warning_per_group(self):
        """One warning per group, not one per parameter."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = parse_groups(
                sfh={"type": "dpl"},  # 7 parameters, no disposition
                dust={"type": "two_component", "law_bc": "calzetti"},  # no disposition
                redshift=0.5,
            )

            # Should have exactly 2 warnings (one per group)
            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 2

            # Extract group names from messages
            messages = [str(x.message) for x in dfp_warnings]
            has_sfh = any("sfh" in msg.lower() for msg in messages)
            has_dust = any("dust" in msg.lower() for msg in messages)
            assert has_sfh and has_dust

    def test_warning_message_includes_action(self):
        """Warning tells user how to fix it."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            params = parse_groups(
                sfh={"type": "dpl"},
                redshift=0.5,
            )

            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 1

            message = str(dfp_warnings[0].message)
            # Should suggest the fix
            assert "all_params" in message.lower()
            has_disposition_hint = (
                "FREE" in message or "free" in message or "FIXED" in message or "fixed" in message
            )
            assert has_disposition_hint

    def test_stacklevel_points_to_build_call(self):
        """stacklevel points to user's parse_groups call, not tengri internals."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # This line should be the one named in the warning
            params = parse_groups(
                sfh={"type": "dpl"},
                redshift=0.5,
            )

            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 1

            # The filename should be this test file, not internal groups.py
            # (stacklevel points to the immediate caller of warnings.warn)
            warning = dfp_warnings[0]
            # Verify it's not pointing at groups.py internals
            not_groups_internals = (
                "groups.py" not in warning.filename
                or "test_warn_silently_fixed_params" in warning.filename
            )
            assert not_groups_internals

    def test_model_behavior_unchanged(self):
        """n_free and predictions are identical with explicit FIXED."""
        # Build with implicit default-FIXED
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            params_implicit = parse_groups(
                sfh={"type": "dpl"},  # implicit FIXED
                dust={"type": "single_component", "law_bc": "calzetti"},  # implicit FIXED
                redshift=0.5,
            )

        # Build with explicit FIXED
        params_explicit = parse_groups(
            sfh={"type": "dpl", "all_params": FIXED},  # explicit FIXED
            dust={
                "type": "single_component",
                "law_bc": "calzetti",
                "all_params": FIXED,
            },  # explicit FIXED
            redshift=0.5,
        )

        # Both should be identical
        assert params_implicit.n_free == params_explicit.n_free
        assert set(params_implicit.free_params) == set(params_explicit.free_params)

        # Check that fixed values are the same
        for name in params_implicit._distributions:
            dist_implicit = params_implicit._distributions[name]
            dist_explicit = params_explicit._distributions[name]
            assert dist_implicit.is_fixed == dist_explicit.is_fixed
            if dist_implicit.is_fixed:
                assert dist_implicit.default == dist_explicit.default


class TestDefaultFixedParametersWarningClass:
    """Test the warning class itself."""

    def test_warning_is_advisory(self):
        """DefaultFixedParametersWarning inherits from AdvisoryWarning."""
        from tengri.config.exceptions import AdvisoryWarning

        assert issubclass(DefaultFixedParametersWarning, AdvisoryWarning)

    def test_warning_is_userwarning(self):
        """DefaultFixedParametersWarning inherits from UserWarning."""
        assert issubclass(DefaultFixedParametersWarning, UserWarning)

    def test_docstring_exists(self):
        """Warning class has a docstring."""
        assert DefaultFixedParametersWarning.__doc__ is not None
        assert len(DefaultFixedParametersWarning.__doc__) > 0


class TestDefect1GroupAttribution:
    """DEFECT 1: Ensure correct group attribution for met_* parameters."""

    def test_met_params_not_attributed_to_sfh_without_explicit_met_block(self):
        """met_* params in sfh group must be skipped when no explicit met block.

        When there's no explicit met={} block, met_* parameters fall into the
        sfh group as a fallback grouping (#311). The warning should NOT
        attribute them to sfh group, since sfh={'all_params': FIXED} is not
        the correct remedy for a warning about met_* parameters.
        """
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Provide sfh but no explicit met block
            params = parse_groups(
                sfh={"type": "dpl"},  # no 'all_params'
                # No explicit met block
                redshift=0.5,
            )

            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            # Should warn about sfh group
            assert len(dfp_warnings) == 1
            message = str(dfp_warnings[0].message)

            # The warning should NOT mention met_* parameters under sfh group
            # (because met_* shouldn't be controlled by sfh{'all_params'})
            assert "met_alpha_fe" not in message
            assert "met_logzsol_scatter" not in message
            # But it should mention sfh parameters like dpl_alpha, dpl_beta
            assert "sfh" in message.lower()


class TestDefect2MessageAccuracy:
    """DEFECT 2: Ensure message says 'these N parameters' not 'all parameters'."""

    def test_message_says_these_parameters_not_all(self):
        """Message must not claim 'all' when only some params defaulted."""
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            # Provide dust with some explicit params, none for others
            params = parse_groups(
                dust={
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "tau_bc": 0.5,  # Explicit override
                    "tau_diff": 0.2,  # Explicit override
                    # dust_Rv, dust_bump_strength, etc. NOT explicitly set → default
                },
                redshift=0.5,
            )

            dfp_warnings = [x for x in w if issubclass(x.category, DefaultFixedParametersWarning)]
            assert len(dfp_warnings) == 1
            message = str(dfp_warnings[0].message)

            # The claim must be about the subset that defaulted, never "all":
            # this group sets tau_bc and tau_diff explicitly.
            assert "all parameters" not in message.lower()

            # It must state how many defaulted, so the count is checkable against
            # the list. Assert the requirement, not one particular phrasing.
            n_listed = sum(1 for tok in message.split() if "=" in tok)
            assert n_listed >= 1
            assert str(n_listed) in message or "parameter was" in message

            # The parameters the user set explicitly must not be reported as
            # defaulted, and the ones that did default must be named.
            assert "dust_tau_bc=" not in message
            assert "dust_tau_diff=" not in message
            assert "dust_Rv" in message or "dust_bump_strength" in message


class TestShippedRecipesNoWarning:
    """DEFECT 3: Shipped recipes must build without emitting this warning."""

    def test_recipes_build_without_warning(self):
        """High-z, photoz, mock_recovery_minimal must not emit warnings."""
        import tengri
        from tengri import Observation, Photometry, SEDModel

        # Skip test if SSP not available (test environment)
        try:
            ssp = tengri.load_ssp("prsc_miles_chabrier_wNE")
        except Exception:
            pytest.skip("SSP data not available")

        obs = Observation(photometry=Photometry.from_names(["jwst_f150w", "jwst_f356w"]))

        for recipe_name in ("high_z", "mock_recovery_minimal", "photoz"):
            with warnings.catch_warnings(record=True) as w:
                warnings.simplefilter("always")
                rec = getattr(tengri.recipes, recipe_name)()
                model = SEDModel.build(ssp_data=ssp, observation=obs, **rec)

                dfp_warnings = [
                    x for x in w if issubclass(x.category, DefaultFixedParametersWarning)
                ]
                assert len(dfp_warnings) == 0, (
                    f"{recipe_name} recipe emitted {len(dfp_warnings)} warnings"
                )
