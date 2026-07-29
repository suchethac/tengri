# SPDX-License-Identifier: BSD-3-Clause
"""Truth-placement guard for mock galaxy populations.

Rejects injected truths that could be mistaken for recovered parameters
when the estimator simply returns its prior unchanged.
"""

import pytest

pytestmark = pytest.mark.contract


def test_truth_at_the_prior_midpoint_is_rejected():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    # Uniform(0.1, 4.0) has arithmetic midpoint 2.05. A truth there cannot
    # distinguish recovery from the estimator returning its prior.
    with pytest.raises(ValueError, match="indistinguishable from the prior"):
        assert_truth_is_discriminating(2.05, (0.1, 4.0), name="sfh_field_psd_sigma")


def test_truth_at_the_geometric_mean_is_rejected():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    # Geometric mean of (0.1, 4.0) is 0.632.
    with pytest.raises(ValueError, match="indistinguishable from the prior"):
        assert_truth_is_discriminating(0.632, (0.1, 4.0), name="sfh_field_psd_sigma")


def test_a_well_separated_truth_is_accepted():
    from tengri.analysis.population_mocks import assert_truth_is_discriminating

    assert_truth_is_discriminating(1.30, (0.1, 4.0), name="sfh_field_psd_sigma")
