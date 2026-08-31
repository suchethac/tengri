# SPDX-License-Identifier: BSD-3-Clause
"""Contract for homogenized component activation.

Three changes, one rule: a block is on because its dict is there, and a value
that matters is one the caller wrote.

* ``redshift`` is required. It used to default to ``Fixed(0.1)``, so a model
  that omitted it sat at z=0.1 without saying so.
* ``apply_igm`` is retired. IGM activation is derived from the ``igm`` group,
  removing the secondary switch that could disagree with it.
* The typeless ``igm`` default is ``inoue14``, matching what the flat
  ``Parameters`` path already used.

The requiredness tests are about *mechanism*, not just the error. Requiredness
is a question about the call, and the first implementation encoded the answer as
a value -- ``Uniform(0.0, 10.0, "SENTINEL: ...")`` in the declaration, rejected
by ``is`` identity. That value is the most natural photo-z prior in this
package's target science, compares equal to a user's own ``Uniform(0, 10)``, and
has an identical repr, so it was silently wrong whichever comparison was used:
``is`` lets any rebuilt default (a copy, a serializer, a ``to_groups``
round-trip) through as a free z in [0, 10] fit, and ``==`` refuses legitimate
photo-z fits over that range. ``test_photo_z_over_the_full_range_is_accepted``
is the test that fails under either.
"""

from __future__ import annotations

import pytest

from tengri import DEFAULT, Fixed, Gaussian, LogUniform, Uniform
from tengri.parameters.groups import parse_groups
from tengri.parameters.parameters import Parameters

pytestmark = pytest.mark.contract


class TestRedshiftIsRequired:
    def test_omitted_redshift_raises(self):
        with pytest.raises(ValueError, match="redshift is required"):
            parse_groups(sfh={"type": "dpl"})

    def test_the_error_names_all_three_spellings(self):
        """A required argument's error has to say what to write instead."""
        with pytest.raises(ValueError) as excinfo:
            parse_groups(sfh={"type": "dpl"})
        msg = str(excinfo.value)
        assert "Fixed(z)" in msg
        assert "Uniform(lo, hi)" in msg
        assert "Distribution" in msg

    @pytest.mark.parametrize(
        "prior",
        [Fixed(0.05), Fixed(3.0), Uniform(0.0, 2.0), LogUniform(0.01, 5.0)],
        ids=["fixed-low", "fixed-high", "uniform", "loguniform"],
    )
    def test_every_distribution_spelling_is_accepted(self, prior):
        spec = parse_groups(sfh={"type": "dpl"}, redshift=prior)
        assert "redshift" in spec.valid_param_names

    def test_an_unbounded_prior_is_refused(self):
        """A redshift prior must not admit negative z.

        ``Gaussian`` has support (-inf, inf), and redshift declares ``lo >= 0``,
        so this is refused on its bounds rather than on its presence. Included
        because "any Distribution" in the required-argument message is not quite
        true, and the boundary belongs in a test rather than in the prose.
        """
        with pytest.raises(ValueError, match="bounds"):
            parse_groups(sfh={"type": "dpl"}, redshift=Gaussian(1.0, 0.1))

    def test_photo_z_over_the_full_range_is_accepted(self):
        """``Uniform(0, 10)`` is a real photo-z prior, not a missing value.

        The sentinel this replaced *was* ``Uniform(0.0, 10.0)``. Any
        implementation that decides requiredness by comparing values fails here
        or fails its mirror below -- there is no value that is both "absent" and
        not a legal prior.
        """
        spec = parse_groups(sfh={"type": "dpl"}, redshift=Uniform(0.0, 10.0))
        assert "redshift" in spec.free_params

    def test_a_fixed_redshift_is_not_free(self):
        spec = parse_groups(sfh={"type": "dpl"}, redshift=Fixed(0.5))
        assert "redshift" not in spec.free_params

    def test_introspection_mode_needs_no_redshift(self):
        """Registry introspection has no call site to require it of.

        The redshift is deliberately absent: supplying one here would satisfy
        the requirement by hand and the test would pass whether or not
        introspection mode is exempt, which is the whole claim. The sweep that
        made every caller state its redshift briefly added one, because a
        caller whose subject is the *absence* of an argument looks identical
        to a caller that merely forgot it.
        """
        spec = parse_groups(sfh={"type": "dpl"}, _allow_empty_wildcard=True)
        assert "redshift" in spec.valid_param_names

    def test_a_group_wildcard_does_not_free_redshift(self):
        """#887: redshift is a top-level argument, not a component parameter.

        A wildcard reaching it would let ``all_params: FREE`` elsewhere in the
        model turn a fixed-redshift fit into a photo-z one, which is the largest
        behavioral change the package can make silently.
        """
        from tengri import FREE

        spec = parse_groups(sfh={"all_params": FREE}, redshift=Fixed(0.5))
        assert "redshift" not in spec.free_params


class TestApplyIgmIsRetired:
    def test_apply_igm_kwarg_raises(self):
        with pytest.raises(ValueError, match="apply_igm is retired"):
            parse_groups(sfh={"type": "dpl"}, redshift=Fixed(0.1), apply_igm=True)

    def test_the_error_teaches_the_igm_dict(self):
        with pytest.raises(ValueError) as excinfo:
            parse_groups(sfh={"type": "dpl"}, redshift=Fixed(0.1), apply_igm=True)
        msg = str(excinfo.value)
        assert "igm={'type':" in msg or 'igm={"type":' in msg

    def test_apply_igm_false_also_raises(self):
        """Both values raise: the argument is gone, not merely defaulted.

        Accepting ``apply_igm=False`` while rejecting ``True`` would leave the
        switch half-alive and able to disagree with the igm dict again.
        """
        with pytest.raises(ValueError, match="apply_igm is retired"):
            parse_groups(sfh={"type": "dpl"}, redshift=Fixed(0.1), apply_igm=False)


class TestIgmActivationFollowsTheGroup:
    def test_omitted_igm_group_is_off(self):
        spec = parse_groups(sfh={"type": "dpl"}, redshift=Fixed(3.0))
        assert spec.apply_igm is False

    def test_a_typed_igm_group_is_on(self):
        spec = parse_groups(sfh={"type": "dpl"}, igm={"type": "inoue14"}, redshift=Fixed(3.0))
        assert spec.apply_igm is True

    def test_type_none_is_explicit_off(self):
        spec = parse_groups(sfh={"type": "dpl"}, igm={"type": "none"}, redshift=Fixed(3.0))
        assert spec.apply_igm is False

    def test_a_typeless_igm_group_is_still_on(self):
        """Presence activates; ``type`` only selects which model."""
        spec = parse_groups(
            sfh={"type": "dpl"}, igm={"all_params": Fixed(DEFAULT)}, redshift=Fixed(3.0)
        )
        assert spec.apply_igm is True


class TestTheTwoEntryPointsAgreeOnTheDefaultModel:
    def test_grammar_and_flat_form_pick_the_same_igm_model(self):
        """madau vs inoue14 is a different curve, not a different name.

        The grammar defaulted to madau while ``Parameters.__init__`` defaulted
        to ``inoue`` (an alias of inoue14), so one model written two ways got
        different IGM physics. Asserted as equality between the paths rather
        than against a literal, so the test pins the agreement rather than
        today's choice.
        """
        from_grammar = parse_groups(
            sfh={"type": "dpl"}, igm={"all_params": Fixed(DEFAULT)}, redshift=Fixed(3.0)
        )
        from_flat = Parameters(redshift=Fixed(3.0), apply_igm=True)
        assert from_grammar.igm_model == from_flat.igm_model
