# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for issue #1769: Catalog.from_histories rebuilds component chain.

When Catalog.from_histories validates a model before the fix, it called:
- _stellar_config(fwd) → calls _stellar_component → _build_component_chain
- _nebular_backend(fwd) → calls _build_component_chain
- _ssp_lgmet(fwd) → calls _stellar_component → _build_component_chain

That was 3 calls to the expensive chain build in one classmethod.
The fix memoizes the chain under a reserved key in _batched_cache_for (#1769).
This test verifies exactly 1 chain build occurs when calling from_histories.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

_Z_OBS = 0.05
_T_GYR = np.concatenate([np.array([0.0]), np.linspace(1.0, 13.0, 39)])


def _flat_histories(sfrs):
    """(N, n_t) histories that are one common shape scaled by each SFR level."""
    n, n_t = len(sfrs), _T_GYR.shape[0]
    shape = np.ones(n_t)
    shape[0] = 0.0  # anchor at the Big Bang
    t = np.broadcast_to(_T_GYR, (n, n_t)).copy()
    sfr = np.stack([shape * float(s) for s in sfrs])
    return t, sfr


@pytest.fixture
def fwd_for_chain_test(synthetic_ssp_wide, synthetic_tophat_obs):
    """ForwardModel with table SFH for testing component chain memoization."""
    from tengri import FIXED, ForwardModel, SEDModel
    from tengri.parameters.priors import Fixed, Uniform

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sed = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "table"},
            dust={
                "type": "two_component",
                "all_params": FIXED,
                "tau_bc": 0.5,
                "tau_diff": Uniform(0.0, 2.0),
            },
            neb={"type": "none"},
            redshift=Fixed(_Z_OBS),
        )
        return ForwardModel.build(sed=sed, observation=synthetic_tophat_obs)


@pytest.mark.regression_bug
def test_from_histories_chain_built_once(fwd_for_chain_test):
    """Assert Catalog.from_histories builds the component chain exactly once (#1769).

    Before the fix, from_histories called _build_component_chain 3 times via
    _stellar_config, _nebular_backend, and _ssp_lgmet. The fix memoizes the
    chain so it is built only once per ForwardModel validation.
    """
    from tengri import Catalog

    # Monkeypatch the sed's _build_component_chain to count calls
    original_build = fwd_for_chain_test.populations[0].sed._build_component_chain
    call_count = [0]

    def counting_build():
        call_count[0] += 1
        return original_build()

    fwd_for_chain_test.populations[0].sed._build_component_chain = counting_build

    # Call from_histories with simple histories
    t, sfr = _flat_histories([1.0, 5.0, 20.0])
    cat = Catalog.from_histories(
        fwd_for_chain_test,
        t_gyr=t,
        sfr=sfr,
        params={"dust_tau_diff": np.full(3, 0.2)},
    )

    # Verify the chain was built exactly once during from_histories validation
    assert call_count[0] == 1, (
        f"Expected 1 component chain build, got {call_count[0]}. "
        "The memoization in _component_chain_for may not be wired correctly."
    )
