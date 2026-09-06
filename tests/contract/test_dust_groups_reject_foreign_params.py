# SPDX-License-Identifier: BSD-3-Clause
# Copyright (c) 2025 Suchetha Cooray
"""Contract: a dust group rejects per-parameter keys its selected variant never reads.

``sfh`` and ``igm`` have always raised on a foreign per-parameter key
(``sfh={'type': 'delayed', 'umin': ...}`` is an "Unknown key"). The two dust
groups did not, because their parameter partition is the *union* over every
variant they can dispatch to: ``dust_emission`` owns 22 names across nine IR
engines, ``dust_attenuation`` the curve-shape modifiers of 22 attenuation laws.
So ``dust_emission={'type': 'casey2012', 'umin': Fixed(2.0)}`` — ``umin`` is
Draine & Li's — built without a word, and the resolver wrote a parameter no
component consults.

Measured before the guard: all four of the reproductions below built cleanly
with ``free_params == []``. A silently ignored knob is either a flat direction
explored at full sampler cost, or a value the author believes is pinning
something.

The accepted set is derived from :func:`~tengri.parameters.groups._wildcard_scopes`,
the same variant scoping the ``all_params`` wildcard already applies, so the
validator and the wildcard cannot disagree about which parameters a variant reads.
"""

import pytest

from tengri import DEFAULT, FREE, Fixed, SEDModel, Uniform
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract

WILDCARD = Fixed(DEFAULT)

#: Attenuation group that reads nothing beyond the two optical depths. Calzetti
#: bakes R_V = 4.05 into the curve and takes no shape argument at all, which is
#: why ``slope`` and ``Rv`` are foreign to it.
CALZETTI = {"type": "two_component", "law": "calzetti", "all_params": WILDCARD}


def build_groups(**overrides):
    """Parse a minimal spec, with ``overrides`` replacing the dust defaults."""
    groups = {
        "sfh": {"type": "dpl", "all_params": WILDCARD},
        "dust_attenuation": dict(CALZETTI),
        "dust_emission": {"type": "dale2014", "all_params": WILDCARD},
        "redshift": Fixed(0.05),
    }
    groups.update(overrides)
    return parse_groups(**groups)


class TestForeignKeysRaise:
    """The four reproductions that built silently before the guard."""

    def test_casey2012_rejects_draine_li_umin(self):
        """``umin`` belongs to the Draine & Li grids, not the Casey MBB."""
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_emission={
                    "type": "casey2012",
                    "all_params": WILDCARD,
                    "umin": Fixed(2.0),
                }
            )
        message = str(exc_info.value)
        assert "'umin'" in message
        assert "casey2012" in message
        assert "dust_emission" in message

    def test_draine_li2007_rejects_the_2014_grid_alpha(self):
        """``alpha_dl14`` is the 2014 grid's extra axis; DL07 has no such knob."""
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_emission={
                    "type": "draine_li2007",
                    "all_params": WILDCARD,
                    "alpha_dl14": Fixed(2.5),
                }
            )
        message = str(exc_info.value)
        assert "'alpha_dl14'" in message
        assert "draine_li2007" in message

    def test_calzetti_rejects_slope(self):
        """``slope`` is read by ``power_law`` / ``conroy2010``, never by Calzetti."""
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": WILDCARD,
                    "slope": Fixed(-0.5),
                }
            )
        message = str(exc_info.value)
        assert "'slope'" in message
        assert "calzetti" in message
        assert "dust_attenuation" in message

    def test_calzetti_rejects_rv(self):
        """Calzetti bakes R_V = 4.05 in; ``Rv`` is Cardelli's / Conroy's."""
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": WILDCARD,
                    "Rv": Fixed(4.0),
                }
            )
        message = str(exc_info.value)
        assert "'Rv'" in message
        assert "calzetti" in message


class TestErrorMessageContent:
    """The message names the variant, the key, the alternatives, and the wildcard."""

    @pytest.fixture
    def message(self):
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_emission={
                    "type": "casey2012",
                    "all_params": WILDCARD,
                    "umin": Fixed(2.0),
                }
            )
        return str(exc_info.value)

    def test_names_the_group(self, message):
        assert "dust_emission" in message

    def test_names_the_selected_type(self, message):
        assert "casey2012" in message

    def test_names_the_offending_key(self, message):
        assert "'umin'" in message

    def test_lists_what_the_type_does_accept(self, message):
        # casey2012 declares T / beta_ir / alpha_mir; eta_balance is group-shared.
        for accepted in ("T", "beta_ir", "alpha_mir", "eta_balance"):
            assert accepted in message

    def test_mentions_both_wildcard_spellings(self, message):
        assert "all_params" in message
        assert "other_params" in message

    def test_is_a_value_error(self, message):
        """``ParameterError`` subclasses ``ValueError``, so ``except ValueError`` catches it."""
        assert issubclass(ParameterError, ValueError)

    def test_attenuation_message_names_the_law(self):
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": WILDCARD,
                    "Rv": Fixed(4.0),
                }
            )
        message = str(exc_info.value)
        assert "two_component" in message
        assert "calzetti" in message

    def test_attenuation_message_names_both_screens_when_they_differ(self):
        with pytest.raises(ParameterError) as exc_info:
            build_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law_bc": "calzetti",
                    "law_diff": "smc",
                    "all_params": WILDCARD,
                    "Rv": Fixed(4.0),
                }
            )
        message = str(exc_info.value)
        assert "law_bc='calzetti'" in message
        assert "law_diff='smc'" in message


class TestOwnParametersAccepted:
    """A variant's own declared parameters stay accepted."""

    @pytest.mark.parametrize(
        ("emission_type", "params"),
        [
            ("casey2012", {"alpha_mir": Uniform(1.0, 3.0), "T": Fixed(40.0)}),
            ("casey2012", {"beta_ir": Fixed(1.6)}),
            ("modified_blackbody", {"epsilon_mbb": Fixed(1e-2)}),
            ("draine_li2014", {"alpha_dl14": Fixed(2.5), "umin": Fixed(2.0)}),
            ("draine_li2007", {"umin": Fixed(2.0), "qpah": Fixed(3.0)}),
            ("dale2014", {"alpha_dale": Fixed(2.0), "frac_agn": Fixed(0.0)}),
            ("themis", {"qhac": Fixed(0.1), "gamma_dl": Fixed(0.05)}),
        ],
    )
    def test_emission_type_accepts_its_own(self, emission_type, params):
        spec = build_groups(
            dust_emission={"type": emission_type, "all_params": WILDCARD, **params}
        )
        assert spec is not None

    def test_calzetti_accepts_the_optical_depths(self):
        spec = build_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": WILDCARD,
                "tau_bc": Uniform(0.0, 2.0),
                "tau_diff": Uniform(0.0, 4.0),
            }
        )
        assert "dust_tau_bc" in spec.free_params
        assert "dust_tau_diff" in spec.free_params

    def test_two_component_per_screen_laws_accept_the_pair(self):
        """``law_bc`` + ``law_diff`` with both optical depths, the documented form."""
        spec = build_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "all_params": WILDCARD,
                "tau_bc": Uniform(0.0, 2.0),
                "tau_diff": Uniform(0.0, 4.0),
            }
        )
        assert "dust_tau_bc" in spec.free_params
        assert "dust_tau_diff" in spec.free_params

    def test_a_shape_param_read_by_either_screen_is_accepted(self):
        """``slope`` is live as soon as one screen names a law that reads it."""
        spec = build_groups(
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "power_law",
                "all_params": WILDCARD,
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.3),
                "slope": Fixed(-0.7),
            }
        )
        assert spec is not None

    @pytest.mark.parametrize(
        ("law", "param"),
        [
            ("cardelli", "Rv"),
            ("power_law", "slope"),
            ("noll09", "delta"),
            ("noll09", "bump_strength"),
            ("kriek_conroy", "delta"),
        ],
    )
    def test_a_law_accepts_the_shape_param_it_names(self, law, param):
        spec = build_groups(
            dust_attenuation={
                "type": "two_component",
                "law": law,
                "all_params": WILDCARD,
                param: Fixed(1.0),
            }
        )
        assert spec is not None


class TestGroupSharedKnobs:
    """``eta_balance`` is the emission group's own knob, not any engine's."""

    @pytest.mark.parametrize(
        "emission_type", ["casey2012", "dale2014", "draine_li2007", "themis", "pah_drude"]
    )
    def test_eta_balance_accepted_on_every_engine(self, emission_type):
        spec = build_groups(
            dust_emission={
                "type": emission_type,
                "all_params": WILDCARD,
                "eta_balance": Fixed(1.0),
            }
        )
        assert spec is not None

    def test_full_name_spelling_of_the_shared_knob(self):
        spec = build_groups(
            dust_emission={
                "type": "casey2012",
                "all_params": WILDCARD,
                "dust_eta_balance": Fixed(1.0),
            }
        )
        assert spec is not None

    def test_a_declaration_free_engine_still_rejects_a_foreign_key(self):
        """``pah_drude`` is a pure template shape; every per-parameter key is foreign."""
        with pytest.raises(ParameterError, match="pah_drude"):
            build_groups(
                dust_emission={
                    "type": "pah_drude",
                    "all_params": WILDCARD,
                    "umin": Fixed(2.0),
                }
            )


class TestAliasTypes:
    """Grammar aliases resolve to their canonical class before the check."""

    def test_mbb_accepts_the_canonical_parameters(self):
        spec = build_groups(
            dust_emission={
                "type": "mbb",
                "all_params": WILDCARD,
                "epsilon_mbb": Fixed(1e-2),
                "beta_ir": Fixed(1.8),
            }
        )
        assert spec is not None

    def test_dl14_accepts_the_canonical_parameters(self):
        spec = build_groups(
            dust_emission={
                "type": "dl14",
                "all_params": WILDCARD,
                "alpha_dl14": Fixed(2.5),
                "umin": Fixed(2.0),
            }
        )
        assert spec is not None

    def test_dl07_alias_rejects_the_2014_axis(self):
        """The alias must narrow like its canonical name, not fall back to the union."""
        with pytest.raises(ParameterError, match="alpha_dl14"):
            build_groups(
                dust_emission={
                    "type": "dl07",
                    "all_params": WILDCARD,
                    "alpha_dl14": Fixed(2.5),
                }
            )

    def test_mbb_alias_rejects_a_draine_li_key(self):
        with pytest.raises(ParameterError, match="'umin'"):
            build_groups(dust_emission={"type": "mbb", "all_params": WILDCARD, "umin": Fixed(2.0)})


class TestFullNameSpelling:
    """Short and fully prefixed spellings follow the same rule."""

    def test_full_name_of_an_own_parameter_is_accepted(self):
        spec = build_groups(
            dust_emission={
                "type": "casey2012",
                "all_params": WILDCARD,
                "dust_alpha_mir": Uniform(1.0, 3.0),
            }
        )
        assert "dust_alpha_mir" in spec.free_params

    def test_full_name_of_a_foreign_emission_parameter_is_rejected(self):
        with pytest.raises(ParameterError, match="'dust_umin'"):
            build_groups(
                dust_emission={
                    "type": "casey2012",
                    "all_params": WILDCARD,
                    "dust_umin": Fixed(2.0),
                }
            )

    def test_full_name_of_a_foreign_attenuation_parameter_is_rejected(self):
        with pytest.raises(ParameterError, match="'dust_Rv'"):
            build_groups(
                dust_attenuation={
                    "type": "two_component",
                    "law": "calzetti",
                    "all_params": WILDCARD,
                    "dust_Rv": Fixed(4.0),
                }
            )

    def test_full_name_of_an_own_attenuation_parameter_is_accepted(self):
        spec = build_groups(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": WILDCARD,
                "dust_tau_bc": Uniform(0.0, 2.0),
                "dust_tau_diff": Uniform(0.0, 4.0),
            }
        )
        assert "dust_tau_bc" in spec.free_params


class TestWildcardOnlyBuildsUnaffected:
    """A group with no per-parameter key at all is untouched by the guard."""

    @pytest.mark.parametrize(
        "emission_type",
        ["casey2012", "dale2014", "draine_li2007", "draine_li2014", "themis", "mbb"],
    )
    def test_emission_wildcard_only(self, emission_type):
        spec = build_groups(dust_emission={"type": emission_type, "all_params": WILDCARD})
        assert spec is not None

    @pytest.mark.parametrize("law", ["calzetti", "power_law", "cardelli", "noll09", "smc"])
    def test_attenuation_wildcard_only(self, law):
        spec = build_groups(
            dust_attenuation={"type": "two_component", "law": law, "all_params": WILDCARD}
        )
        assert spec is not None

    def test_wg00_wildcard_only(self):
        spec = build_groups(dust_attenuation={"type": "wg00", "all_params": WILDCARD})
        assert spec is not None

    def test_no_emission_group_at_all(self):
        spec = parse_groups(
            sfh={"type": "dpl", "all_params": WILDCARD},
            dust_attenuation=dict(CALZETTI),
            redshift=Fixed(0.05),
        )
        assert spec is not None

    def test_other_params_synonym_only(self):
        spec = build_groups(dust_emission={"type": "casey2012", "other_params": WILDCARD})
        assert spec is not None

    def test_free_wildcard_only(self):
        spec = build_groups(dust_emission={"type": "themis", "all_params": FREE})
        assert spec is not None


class TestUnknownKeysKeepTheirOwnMessage:
    """A key no variant of the group declares is a typo, not a foreign parameter."""

    def test_typo_gets_the_did_you_mean_message(self):
        with pytest.raises(ValueError, match="Unknown key 'tau_qpah'") as exc_info:
            build_groups(
                dust_emission={
                    "type": "casey2012",
                    "all_params": WILDCARD,
                    "tau_qpah": Fixed(1.0),
                }
            )
        assert "Did you mean" in str(exc_info.value)


class TestUserRegisteredComponent:
    """A user-registered subclass's own priors are accepted (#391 escape hatch)."""

    @pytest.fixture
    def custom_engine(self):
        import jax.numpy as jnp

        from tengri.components.dust.emission._component_base import EmissionComponent
        from tengri.components.sed_model_component import _REGISTRY

        class _TestOnlyIRDust(EmissionComponent):
            """Minimal user-registered IR emission engine."""

            name = "_test_only_ir_dust"
            T_custom = Fixed(35.0)

            def predict(self, p, sed_in, wave, **inputs):
                sed = jnp.zeros_like(wave)
                return sed_in + sed, {"sed_dust_ir": sed}

        try:
            yield _TestOnlyIRDust.name
        finally:
            _REGISTRY.pop(_TestOnlyIRDust.name, None)

    def test_own_prior_is_accepted(self, custom_engine):
        spec = build_groups(
            dust_emission={
                "type": custom_engine,
                "all_params": WILDCARD,
                "T_custom": Fixed(35.0),
            }
        )
        assert spec is not None

    def test_foreign_key_is_still_rejected(self, custom_engine):
        with pytest.raises(ParameterError, match="'umin'"):
            build_groups(
                dust_emission={
                    "type": custom_engine,
                    "all_params": WILDCARD,
                    "umin": Fixed(2.0),
                }
            )

    @pytest.fixture
    def odd_prefix_engine(self):
        """A subclass whose ``parameter_prefix`` is not one ``_extract_short_name`` knows.

        The narrowed set is built from fully prefixed names, and the short
        spelling is recovered by stripping a *known* prefix. ``oddir_`` is not
        one, so ``_name_spellings`` alone yields only ``oddir_T_custom`` and the
        bare ``T_custom`` a user would naturally write is rejected. The
        ``_short_names_for_registered_type`` union in the validator is what keeps
        it accepted; without it this component has no usable short spelling.
        """
        from typing import ClassVar

        import jax.numpy as jnp

        from tengri.components.sed_model_component import _REGISTRY, SEDModelComponent

        class _TestOnlyOddPrefixIRDust(SEDModelComponent):
            """User-registered IR engine with a non-dust parameter prefix."""

            name = "_test_only_odd_prefix_ir_dust"
            parameter_prefix = "oddir_"

            T_custom = Fixed(35.0)

            optional_inputs: ClassVar[dict[str, str]] = {"L_ir": "erg/s"}
            outputs: ClassVar[dict[str, str]] = {"sed_dust_ir": "erg/s/Hz"}

            def predict(self, p, sed_in, wave, **inputs):
                sed = jnp.zeros_like(wave)
                return sed_in + sed, {"sed_dust_ir": sed}

        try:
            yield _TestOnlyOddPrefixIRDust.name
        finally:
            _REGISTRY.pop(_TestOnlyOddPrefixIRDust.name, None)

    @pytest.mark.parametrize("spelling", ["T_custom", "oddir_T_custom"])
    def test_odd_prefix_component_keeps_both_spellings(self, odd_prefix_engine, spelling):
        spec = build_groups(
            dust_emission={
                "type": odd_prefix_engine,
                "all_params": WILDCARD,
                spelling: Fixed(35.0),
            }
        )
        assert spec is not None


class TestReachableThroughTheBuilder:
    """The guard fires on the public surface, not only on ``parse_groups``."""

    def test_sed_model_build_rejects_a_foreign_emission_key(self, ssp_data_wne):
        with pytest.raises(ParameterError, match="'umin'"):
            SEDModel.build(
                ssp_data=ssp_data_wne,
                sfh={"type": "dpl", "all_params": WILDCARD},
                dust_attenuation=dict(CALZETTI),
                dust_emission={
                    "type": "casey2012",
                    "all_params": WILDCARD,
                    "umin": Fixed(2.0),
                },
                redshift=Fixed(0.05),
            )

    def test_sed_model_build_accepts_the_engine_s_own_key(self, ssp_data_wne):
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            sfh={"type": "dpl", "all_params": WILDCARD},
            dust_attenuation=dict(CALZETTI),
            dust_emission={
                "type": "casey2012",
                "all_params": WILDCARD,
                "alpha_mir": Uniform(1.0, 3.0),
            },
            redshift=Fixed(0.05),
        )
        assert "dust_alpha_mir" in model.spec.free_params
