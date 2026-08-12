# SPDX-License-Identifier: BSD-3-Clause
"""Regression: fit_interim must be able to follow its own advice (#1720).

When the MAP initialization diverges, ``fit_interim`` raised::

    all 8 MAP restarts diverged to a non-finite loss (NaN or inf) ... try a
    smaller learning_rate=, fewer n_steps=, or a larger n_restarts= so at least
    one init survives.

and accepted **none** of those, nor ``n_map_steps``. The message is inherited
verbatim from the inner ``Fitter`` and surfaced through a wrapper with a
narrower signature, so it described the callee's knobs rather than the caller's.
Same class as #1575, where the advice could not reach a mode outside the
support; here it could not be typed at all.

That made a real regime unreachable. The configuration recorded in
``psd_bank_conv/bank_meta.json`` — the one behind the handoff doc's §5 — is
``truth_sigma=0.75``, ``truth_tau_myr=150``, SNR 20/10. Run through
``fit_interim`` it does not give a different R-hat, it gives **no fit**: 8 of 8
MAP restarts diverge. The bank that produced §5's numbers never used
``fit_interim``; ``scripts/hierarchical_psd_fit_bank.py`` drives ``Fitter``
directly and escalates ``n_map_steps`` (tripling, hence ``psd_bank_map40k``),
which is §4i-ter's recorded remedy for #1537.

``map_options`` threads that through. Default ``None`` keeps today's behavior
exactly — the HMC backend does its own initialization and no separate MAP runs.
"""

from __future__ import annotations

import inspect
from typing import ClassVar

import pytest

from tengri.inference.population.interim import _validate_map_options, fit_interim

pytestmark = pytest.mark.regression_bug

#: The three the diverged-MAP message actually recommends.
_ADVISED = ("learning_rate", "n_steps", "n_restarts")


class _Mock:
    truth_params: ClassVar[list] = []
    table: ClassVar[list] = []


class TestTheAdviceIsNowFollowable:
    @pytest.mark.parametrize("name", _ADVISED)
    def test_each_advised_option_is_accepted(self, name):
        """The error names these three. Accepting them is the whole fix."""
        _validate_map_options({name: 1})

    @pytest.mark.parametrize("name", _ADVISED)
    def test_fit_interim_takes_them_via_map_options(self, name):
        """Reaching the bounds guard rather than TypeError proves the kwarg is
        routed; before #1720 every one of these raised
        ``unexpected keyword argument``."""
        with pytest.raises(ValueError, match="sigma_bounds"):
            fit_interim(
                None,
                _Mock(),
                key=None,
                # a bad bound, so we fail at validation and never touch a model
                interim_bounds={"sigma_bounds": (-1.0, 1.0), "tau_bounds_myr": (10.0, 500.0)},
                map_options={name: 1},
            )

    def test_map_options_is_in_the_signature(self):
        assert "map_options" in inspect.signature(fit_interim).parameters


class TestItRejectsWhatTheBackendWillNot:
    def test_an_unknown_option_raises_naming_the_valid_set(self):
        """Forwarding a bad key would surface deep inside the MAP backend with
        no hint that it came from map_options."""
        with pytest.raises(ValueError) as excinfo:
            _validate_map_options({"n_leapfrog_steps": 10})
        message = str(excinfo.value)
        assert "n_leapfrog_steps" in message
        assert "learning_rate" in message  # the valid set is listed

    def test_the_message_does_not_advise_what_it_rejects(self):
        """The defect being fixed was advice pointing at unusable arguments; the
        replacement must not repeat it."""
        with pytest.raises(ValueError) as excinfo:
            _validate_map_options({"nonsense": 1})
        assert "n_steps" in str(excinfo.value)


class TestItDoesNotOverreach:
    def test_none_is_allowed_and_is_the_default(self):
        """Default None must keep today's behavior: no separate MAP stage."""
        assert inspect.signature(fit_interim).parameters["map_options"].default is None
        _validate_map_options(None)

    def test_an_empty_dict_is_allowed(self):
        _validate_map_options({})

    def test_a_non_mapping_raises_rather_than_being_splatted(self):
        """``**map_options`` on a list fails obscurely inside the call."""
        with pytest.raises(TypeError, match="map_options"):
            _validate_map_options(["n_steps", 10])
