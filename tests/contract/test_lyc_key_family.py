# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the ``lyc_`` key family rename (owner ruling #2529, "one lyc_ key family").

``dust_attenuation``'s two retired boolean structural keys, ``lyc_absorb_all``
(replaced by the string ``lyc_reprocessed_by``, ``'young'``/``'all'``) and
``eb_include_lyc`` (replaced by ``lyc_in_energy_balance``, same bool), now
raise a loud rename hint at parse time (``parameters/groups.py``, the same
interception mechanism as ``_neb_fdust_retired_error``). This file is the
dedicated contract for the rename itself -- the renamed keys' physics is
already covered by ``test_energy_balance_lyc_toggle.py``,
``test_lyc_fesc_two_component.py``, ``test_bug_2439_precomp_lyc_mask.py`` and
``test_2539_fdust_energy_in_ir_budget.py``.

**Old->new equivalence, "internal config mapping" chosen over a bootstrap-head
snapshot**: a genuinely pre-rename reference would need a second checkout
(``git archive <bootstrap-head> | tar -x -C <dir>``) built in its own venv so
its JAX compile cache does not collide with this one -- slow and fragile for
a contract test, and this package's pre-1.0 policy is "no shims", so the old
boolean code path no longer exists to call directly either way. Instead,
:class:`TestOldToNewInternalConfigMapping` builds a reference model through
the grammar, shallow-copies its already-correctly-resolved ``Parameters``
spec, and sets ``spec.dust_lyc_reprocessed_by``/``spec.dust_lyc_in_energy_balance``
directly -- exactly the attributes ``groups.py``'s parse populates, and
exactly what a pre-rename ``spec.dust_lyc_absorb_all``/``spec.dust_eb_include_lyc``
boolean attribute would have been named. A model built from the mutated spec
must be bit-identical to one built through the grammar with the same value:
this is precisely the translation (``_translate_dust_attenuation`` ->
``Parameters`` attribute -> ``SEDModel``/component config) the rename edited,
so a mismatch here is exactly the class of bug a rename like this can
introduce silently.

Markers
-------
- ``@pytest.mark.contract`` -- Contract test
"""

from __future__ import annotations

import copy

import numpy as np
import pytest

import tengri
from tengri import DEFAULT, Fixed

pytestmark = [pytest.mark.contract]


def _dust_attenuation(*, lyc_reprocessed_by="young", lyc_in_energy_balance=False):
    return {
        "type": "two_component",
        "law_bc": "calzetti",
        "law_diff": "calzetti",
        "tau_bc": Fixed(0.3),
        "tau_diff": Fixed(0.3),
        "all_params": Fixed(DEFAULT),
        "lyc_reprocessed_by": lyc_reprocessed_by,
        "lyc_in_energy_balance": lyc_in_energy_balance,
    }


def _build_raw(ssp, obs, dust_attenuation: dict):
    """Build with a caller-supplied ``dust_attenuation`` dict, verbatim.

    Used by the rename-hint and invalid-value tests, which need to write
    the retired keys (or a deliberately bad value) directly rather than
    through :func:`_dust_attenuation`'s current-grammar spelling.
    """
    return tengri.SEDModel.build(
        ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(10.0),
            "all_params": Fixed(DEFAULT),
        },
        neb={
            "type": "cue",
            "neb_fesc": Fixed(0.3),
            "neb_fdust_frac": Fixed(0.2),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation=dust_attenuation,
        redshift=Fixed(0.0),
    )


def _build(ssp, obs, **dust_kwargs):
    return _build_raw(ssp, obs, _dust_attenuation(**dust_kwargs))


class TestOldToNewInternalConfigMapping:
    """Grammar-built vs. internal-config-mapping-built models must agree
    bit-for-bit, for every young/all x EB on/off combination. See module
    docstring for why this, not a bootstrap-head snapshot, is the chosen
    old->new equivalence check.
    """

    @pytest.mark.parametrize("lyc_reprocessed_by", ["young", "all"])
    @pytest.mark.parametrize("lyc_in_energy_balance", [False, True])
    def test_internal_attribute_matches_grammar(
        self,
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        lyc_reprocessed_by,
        lyc_in_energy_balance,
    ):
        m_grammar = _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            lyc_reprocessed_by=lyc_reprocessed_by,
            lyc_in_energy_balance=lyc_in_energy_balance,
        )

        # The "internal config mapping": start from a spec built via the
        # grammar's DEFAULT values, then set the renamed attributes
        # directly -- exactly the Parameters attributes groups.py's parse
        # populates -- bypassing the dust_attenuation dict entirely. A
        # shallow copy is enough (and the only option: Parameters carries a
        # mappingproxy provenance dict that copy.deepcopy cannot pickle) --
        # only the two top-level scalar attributes below are reassigned, on
        # the copy's own __dict__, never mutating anything shared with the
        # original instance.
        m_base = _build(synthetic_ssp_wide, synthetic_tophat_obs)
        spec = copy.copy(m_base.spec)
        spec.dust_lyc_reprocessed_by = lyc_reprocessed_by
        spec.dust_lyc_in_energy_balance = lyc_in_energy_balance
        m_mapped = tengri.SEDModel(spec, synthetic_ssp_wide, observation=synthetic_tophat_obs)

        phot_grammar = np.asarray(m_grammar.predict_photometry({}))
        phot_mapped = np.asarray(m_mapped.predict_photometry({}))
        np.testing.assert_array_equal(phot_grammar, phot_mapped)

        s_grammar = m_grammar.predict_state({})
        s_mapped = m_mapped.predict_state({})
        L_grammar = float(np.asarray(s_grammar.derived["log_L_absorbed"]))
        L_mapped = float(np.asarray(s_mapped.derived["log_L_absorbed"]))
        assert L_grammar == L_mapped, (
            f"log_L_absorbed differs between the grammar build ({L_grammar}) and "
            f"the internal-config-mapping build ({L_mapped}) at "
            f"lyc_reprocessed_by={lyc_reprocessed_by!r}, "
            f"lyc_in_energy_balance={lyc_in_energy_balance!r}"
        )


class TestRenameHints:
    """The two retired keys raise a loud, mapping-carrying error at parse
    time, intercepted before the generic did-you-mean resolver (same
    mechanism as ``_neb_fdust_retired_error``).
    """

    def test_lyc_absorb_all_false_raises_mentioning_young(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        dust = _dust_attenuation()
        del dust["lyc_reprocessed_by"]
        dust["lyc_absorb_all"] = False
        with pytest.raises(ValueError, match=r"lyc_reprocessed_by.*'young'"):
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)

    def test_lyc_absorb_all_true_raises_mentioning_all(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        dust = _dust_attenuation()
        del dust["lyc_reprocessed_by"]
        dust["lyc_absorb_all"] = True
        with pytest.raises(ValueError, match=r"lyc_reprocessed_by.*'all'"):
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)

    def test_eb_include_lyc_raises_mentioning_lyc_in_energy_balance(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        dust = _dust_attenuation()
        del dust["lyc_in_energy_balance"]
        dust["eb_include_lyc"] = True
        with pytest.raises(ValueError, match="lyc_in_energy_balance"):
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)

    def test_hint_intercepted_before_generic_resolver(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """The hint names the replacement, not a generic 'did you mean' --
        the retirement interception in ``_check_dict_keys`` must fire before
        the difflib-based suggestion resolver reaches these keys.
        """
        dust = _dust_attenuation()
        del dust["lyc_reprocessed_by"]
        dust["lyc_absorb_all"] = True
        with pytest.raises(ValueError) as excinfo:
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)
        assert "renamed" in str(excinfo.value)
        assert "Did you mean" not in str(excinfo.value)


class TestInvalidValues:
    """Parse-time validation: an unknown ``lyc_reprocessed_by`` value lists
    the allowed values; a non-bool ``lyc_in_energy_balance`` is rejected.
    """

    def test_unknown_lyc_reprocessed_by_value_lists_allowed(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        dust = _dust_attenuation()
        dust["lyc_reprocessed_by"] = "both"
        with pytest.raises(ValueError, match=r"'young'.*'all'|young.*all"):
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)

    def test_non_bool_lyc_in_energy_balance_raises(self, synthetic_ssp_wide, synthetic_tophat_obs):
        dust = _dust_attenuation()
        dust["lyc_in_energy_balance"] = 1
        with pytest.raises(ValueError, match="bool"):
            _build_raw(synthetic_ssp_wide, synthetic_tophat_obs, dust)


class TestRoundTrip:
    """``model.spec.to_groups()`` must round-trip both keys through a
    rebuild, at both non-default values, and emit neither key at the
    default (matching the existing retirement round-trip convention).
    """

    @pytest.mark.parametrize("lyc_reprocessed_by", ["young", "all"])
    @pytest.mark.parametrize("lyc_in_energy_balance", [False, True])
    def test_round_trip(
        self,
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        lyc_reprocessed_by,
        lyc_in_energy_balance,
    ):
        m = _build(
            synthetic_ssp_wide,
            synthetic_tophat_obs,
            lyc_reprocessed_by=lyc_reprocessed_by,
            lyc_in_energy_balance=lyc_in_energy_balance,
        )
        groups = m.spec.to_groups()
        atten = groups["dust_attenuation"]
        if lyc_reprocessed_by == "all":
            assert atten["lyc_reprocessed_by"] == "all"
        else:
            assert "lyc_reprocessed_by" not in atten
        if lyc_in_energy_balance:
            assert atten["lyc_in_energy_balance"] is True
        else:
            assert "lyc_in_energy_balance" not in atten

        m2 = tengri.SEDModel.build(synthetic_ssp_wide, observation=synthetic_tophat_obs, **groups)
        assert m2.spec.dust_lyc_reprocessed_by == lyc_reprocessed_by
        assert bool(m2.spec.dust_lyc_in_energy_balance) == lyc_in_energy_balance

        phot1 = np.asarray(m.predict_photometry({}))
        phot2 = np.asarray(m2.predict_photometry({}))
        np.testing.assert_array_equal(phot1, phot2)
