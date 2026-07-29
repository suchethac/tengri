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


def test_emission_line_wavelengths_match_canonical_catalog():
    """Verify that population_mocks uses vacuum wavelengths from canonical catalog.

    The mock population generator must use exactly the same emission-line
    wavelengths as the canonical ``LineList.default_optical()`` catalog.
    Mismatches (e.g. air vs vacuum, or hardcoded drifts) would shift the
    measurement windows and create systematic flux biases.

    This test protects against silent wavelength drift and ensures the mock
    is self-consistent with the likelihood's own line definitions.
    """
    import numpy as np

    from tengri.observation.line_list import LineList

    # Get the canonical wavelengths for the strong star-forming set
    line_names = [
        "OII_3726",
        "OII_3729",
        "OIII_4959",
        "OIII_5007",
        "Halpha",
        "Hbeta",
        "SII_6717",
        "NII_6584",
    ]
    canonical_lines = LineList.default_optical().select(names=line_names)
    # Reorder to match line_names order (select returns wavelength-sorted)
    canonical_dict = {
        name: wave for name, wave in zip(canonical_lines.names, canonical_lines.wavelengths)
    }
    wavelengths_in_order = np.array([canonical_dict[name] for name in line_names])

    # Expected vacuum wavelengths from the canonical catalog (in line_names order)
    expected = np.array(
        [3727.09, 3729.88, 4960.30, 5008.24, 6564.61, 4862.68, 6718.29, 6585.28]
    )

    # Verify all wavelengths match to high precision (1e-6 Å)
    np.testing.assert_allclose(
        wavelengths_in_order,
        expected,
        atol=1e-6,
        rtol=0,
        err_msg="Emission line wavelengths differ from canonical vacuum values",
    )
