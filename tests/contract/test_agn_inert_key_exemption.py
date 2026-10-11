# SPDX-License-Identifier: BSD-3-Clause
"""Grid-free pins for the explicit-key refusal and its cross-block exemption.

The refusal is pure Python over the consumes table, so these tests need no
Synthesizer grid and no SSP. They pin three things: a key another selected
block reads is accepted, a key no selected block reads is refused and names its
readers, and the deprecated ``lines`` alias expands through the grammar's own
mapping before the refusal sees the selection.

Tier: contract.
"""

import pytest

from tengri.config.exceptions import ParameterError
from tengri.parameters.agn_ownership import _AGN_PARTITION, _agn_param_group
from tengri.parameters.groups import _agn_block_selection, _reject_inert_agn_subblock_keys

pytestmark = pytest.mark.contract

PARTITION = {name: _agn_param_group(name) for name in _AGN_PARTITION}
DISC_LOG_LEDD = {"type": "multicolor", "log_ledd": -1.0}


def test_key_read_by_another_selected_block_is_accepted():
    """The Synthesizer NLR reads agn_log_ledd (#2634), so disc log_ledd is live."""
    selection = {"disc": "multicolor", "nlr": "synthesizer"}
    _reject_inert_agn_subblock_keys("disc", dict(DISC_LOG_LEDD), selection, PARTITION)


@pytest.mark.parametrize("nlr", ["analytic", "grahsp"])
def test_key_read_by_no_selected_block_is_refused_and_names_its_readers(nlr):
    selection = {"disc": "multicolor", "nlr": nlr}
    with pytest.raises(ParameterError, match=r"does not read the key 'log_ledd'") as info:
        _reject_inert_agn_subblock_keys("disc", dict(DISC_LOG_LEDD), selection, PARTITION)
    assert "read by 'kd18_agnfitter'" in str(info.value)


def test_selection_without_a_nlr_or_blr_refuses_the_key():
    """Omitting the NLR and BLR leaves no other reader, so the key is refused."""
    selection = {"disc": "multicolor"}
    with pytest.raises(ParameterError, match=r"does not read the key 'log_ledd'"):
        _reject_inert_agn_subblock_keys("disc", dict(DISC_LOG_LEDD), selection, PARTITION)


def test_lines_alias_expands_to_the_synthesizer_nlr_and_blr():
    """The deprecated lines slot maps through the grammar's expand_lines_alias."""
    agn_top = {
        "disc": {"type": "multicolor"},
        "lines": {"type": "nlr_blr_synthesizer"},
    }
    assert _agn_block_selection(agn_top) == {
        "disc": "multicolor",
        "nlr": "synthesizer",
        "blr": "synthesizer",
    }


def test_disc_log_ledd_is_accepted_through_the_lines_alias():
    """Regression: a lines alias naming the Synthesizer pair must exempt disc log_ledd."""
    agn_top = {
        "disc": dict(DISC_LOG_LEDD),
        "lines": {"type": "nlr_blr_synthesizer"},
    }
    _reject_inert_agn_subblock_keys(
        "disc", dict(DISC_LOG_LEDD), _agn_block_selection(agn_top), PARTITION
    )


def test_explicit_nlr_takes_precedence_over_a_lines_alias():
    """The grammar refuses nlr together with lines; the selection still prefers nlr."""
    agn_top = {
        "nlr": {"type": "analytic"},
        "lines": {"type": "nlr_blr_synthesizer"},
    }
    selection = _agn_block_selection(agn_top)
    assert selection["nlr"] == "analytic"
    assert selection["blr"] == "synthesizer"


def test_unknown_lines_alias_leaves_the_selection_to_the_grammar():
    """An unknown alias is not expanded here; the grammar raises its own error."""
    agn_top = {"disc": {"type": "multicolor"}, "lines": {"type": "no_such_alias"}}
    assert _agn_block_selection(agn_top) == {"disc": "multicolor"}
