# SPDX-License-Identifier: BSD-3-Clause
"""DefaultFixedParametersWarning must key on zero free parameters (#1995).

The warning added in #1982 fires when a group states no ``all_params``
disposition. That is a proxy for the bug it exists to catch, and the proxy
misfires: ``met={"logzsol": Uniform(...)}`` states no disposition but plainly
configured the group, and it warned. 115 warning instances were baked into 9
published notebook renders, with ``met`` warning in all nine.

The bug the warning exists to catch is "I configured a group and got **zero**
free parameters out of it"::

    SEDModel.build(..., sfh={"type": "dpl"})  # n_free == 0, silently

A group that produced at least one free parameter cannot have hit that
failure mode, so it must stay silent.
"""

import warnings

import pytest

from tengri import DEFAULT, FREE, Fixed, Uniform
from tengri.config.exceptions import DefaultFixedParametersWarning
from tengri.parameters import parse_groups

pytestmark = pytest.mark.regression_bug


def _warnings_for(**kwargs):
    """Return the DefaultFixedParametersWarning messages raised by a build."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DefaultFixedParametersWarning)
        params = parse_groups(**kwargs)
    msgs = [
        str(w.message) for w in caught if issubclass(w.category, DefaultFixedParametersWarning)
    ]
    return params, msgs


class TestGroupWithFreeParametersIsSilent:
    """A group that yielded a free parameter cannot have hit the footgun."""

    def test_met_with_a_free_prior_does_not_warn(self):
        """The ``Uniform`` metallicity idiom must be silent.

        Measured across the affected renders, this spelling appears in
        ``05_fitting_photometry``, ``07_joint_photo_spec`` and
        ``10_fastspecfit_joint_fit``. The ``Fixed(...)`` and ``type: table``
        spellings yield nothing free in the group and keep warning, which is
        the rule working rather than a gap in it.
        """
        params, msgs = _warnings_for(met={"logzsol": Uniform(-1.5, 0.3)}, redshift=0.5)

        assert "met_logzsol" in params.free_params
        assert msgs == [], f"met yielded a free parameter but still warned: {msgs[:1]}"

    def test_dust_attenuation_with_a_free_prior_does_not_warn(self):
        """Same rule on a second group, so the fix is not met-specific."""
        # Both screens named: a two-component model refuses one without the
        # other, so naming only tau_diff would fail on that guard rather than
        # on the behavior under test.
        params, msgs = _warnings_for(
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Uniform(0.0, 2.0),
                "tau_diff": Uniform(0.0, 2.0),
            },
            redshift=0.5,
        )

        assert "dust_tau_diff" in params.free_params
        assert [m for m in msgs if "dust_attenuation" in m] == []


class TestGroupWithoutFreeParametersStillWarns:
    """The original footgun must keep warning. This is the whole point."""

    def test_group_yielding_nothing_free_warns(self):
        """``sfh={'type': 'dpl'}`` gives n_free == 0 and must say so."""
        params, msgs = _warnings_for(sfh={"type": "dpl"}, redshift=0.5)

        assert not [p for p in params.free_params if p.startswith("sfh_")]
        assert len([m for m in msgs if "'sfh'" in m]) == 1

    def test_engaged_group_yielding_nothing_free_still_warns(self):
        """Engagement with one parameter is not engagement with the group.

        A weaker rule -- "the user supplied a parameter, so assume they meant
        it" -- goes quiet here. ``Fixed(1.5)`` proves the user reasoned about
        ``alpha``; it proves nothing about the other parameters the group
        declares, and those are exactly what the warning enumerates. Pinning
        one knob and assuming the group is now handled is the failure this
        warning catches, so the rule keys on the free count.
        """
        params, msgs = _warnings_for(
            sfh={"type": "dpl", "alpha": Fixed(1.5), "beta": Fixed(2.0)},
            redshift=0.5,
        )

        assert not [p for p in params.free_params if p.startswith("sfh_")]
        assert len([m for m in msgs if "'sfh'" in m]) == 1

    def test_tabulated_metallicity_keeps_warning(self):
        """Guard: ``met={'type': 'table'}`` must stay loud.

        Not a red-green test -- it pins behavior that already holds, because
        this particular warning is load-bearing. A tabulated stellar history
        sits beside ``neb_logZ_gas``'s declared ``Fixed(-0.3)``, so it enriches
        the stars and not the gas, and that shipped as a bug once already
        (#1677). It is in-grid, so no range check sees it. Whatever a future
        narrowing does to which parameters get enumerated, this group yields
        nothing free and must keep saying so.
        """
        params, msgs = _warnings_for(met={"type": "table"}, redshift=0.5)

        assert not [p for p in params.free_params if p.startswith("met_")]
        assert len([m for m in msgs if "'met'" in m]) == 1


class TestExistingBehaviorPreserved:
    """#1982's contract that is not being changed."""

    def test_explicit_fixed_is_silent(self):
        _, msgs = _warnings_for(sfh={"type": "dpl", "all_params": Fixed(DEFAULT)}, redshift=0.5)
        assert msgs == []

    def test_explicit_free_is_silent(self):
        _, msgs = _warnings_for(sfh={"type": "dpl", "all_params": FREE}, redshift=0.5)
        assert msgs == []

    def test_free_parameters_and_fixed_values_are_unchanged(self):
        """Warning-only change: the resolved model must be identical."""
        quiet, _ = _warnings_for(met={"logzsol": Uniform(-1.5, 0.3)}, redshift=0.5)
        stated, _ = _warnings_for(
            met={"logzsol": Uniform(-1.5, 0.3), "all_params": Fixed(DEFAULT)}, redshift=0.5
        )

        assert quiet.n_free == stated.n_free
        assert set(quiet.free_params) == set(stated.free_params)
        for name, dist in quiet._distributions.items():
            other = stated._distributions[name]
            assert dist.is_fixed == other.is_fixed
            if dist.is_fixed:
                assert dist.default == other.default
