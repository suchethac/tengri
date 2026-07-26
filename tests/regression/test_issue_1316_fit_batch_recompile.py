# SPDX-License-Identifier: BSD-3-Clause
"""#1316: fit_batch(redshift_col=...) recompiles per galaxy without catalog_z_range.

The zero-clone fix (skipping model clones when catalog_z_range is set) is
deferred pending a per-fit parameter-override API. This test verifies the
loud warning path: when redshift_col is set WITHOUT catalog_z_range, we
warn exactly once (not per row) that every galaxy triggers a full recompile.
"""

import warnings

import pytest

pytestmark = pytest.mark.regression_bug


def test_warns_without_catalog_z_range(synthetic_ssp, simple_observation):
    """Test that fit_batch warns exactly once when catalog_z_range is not set."""
    from tengri import Fixed, SEDModel
    from tengri.forward import convenience

    # Build a model without catalog_z_range
    model = SEDModel.build(
        ssp_data=synthetic_ssp,
        observation=simple_observation,
        sfh={"type": "dpl"},
        redshift=Fixed(0.0),
    )

    # Create a simple 3-row catalog (dict format)
    catalog = [
        {
            "z": 0.1,
            "flux_0": 1.0,
            "flux_1": 2.0,
            "flux_2": 3.0,
            "err_0": 0.1,
            "err_1": 0.2,
            "err_2": 0.3,
        },
        {
            "z": 0.5,
            "flux_0": 1.5,
            "flux_1": 2.5,
            "flux_2": 3.5,
            "err_0": 0.1,
            "err_1": 0.2,
            "err_2": 0.3,
        },
        {
            "z": 1.0,
            "flux_0": 2.0,
            "flux_1": 3.0,
            "flux_2": 4.0,
            "err_0": 0.1,
            "err_1": 0.2,
            "err_2": 0.3,
        },
    ]

    # Capture warnings and verify exactly one is emitted
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        convenience.fit_batch(
            model,
            catalog,
            flux_cols=["flux_0", "flux_1", "flux_2"],
            err_cols=["err_0", "err_1", "err_2"],
            redshift_col="z",
            method="map",
            n_steps=2,  # minimal
            verbose=False,
        )
        # Filter to only the catalog_z_range warning
        catalog_warnings = [
            x for x in w if "catalog_z_range" in str(x.message) and "recompiles" in str(x.message)
        ]
        assert len(catalog_warnings) == 1, (
            f"Expected exactly 1 warning about catalog_z_range; "
            f"got {len(catalog_warnings)}. All warnings: {[str(x.message) for x in w]}"
        )
