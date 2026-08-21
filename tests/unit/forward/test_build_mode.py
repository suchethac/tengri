# SPDX-License-Identifier: BSD-3-Clause
"""#1321: mode= is inferred from kwargs by default, assertable
explicitly, and validated with ONE mode-aware error."""

import pytest

from tengri import Fixed


def _sed(synthetic_ssp, simple_observation):
    from tengri import SEDModel

    return SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl"},
        redshift=Fixed(0.1),
    )


def test_mode_inferred_single(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    fwd = ForwardModel.build(sed=_sed(synthetic_ssp, simple_observation))
    assert fwd.mode == "single"


def test_mode_asserted_matches(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    fwd = ForwardModel.build(mode="single", sed=_sed(synthetic_ssp, simple_observation))
    assert fwd.mode == "single"


def test_mode_mismatch_is_one_clear_error(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    with pytest.raises(ValueError, match=r"mode='multi_population'.*populations="):
        ForwardModel.build(mode="multi_population", sed=_sed(synthetic_ssp, simple_observation))


def test_mode_hierarchical_reserved(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    with pytest.raises(NotImplementedError, match="1319"):
        ForwardModel.build(mode="hierarchical", sed=_sed(synthetic_ssp, simple_observation))


def test_unknown_mode_lists_valid_ones(synthetic_ssp, simple_observation):
    from tengri import ForwardModel

    with pytest.raises(ValueError, match=r"single.*multi_population.*hierarchical"):
        ForwardModel.build(mode="banana", sed=_sed(synthetic_ssp, simple_observation))
