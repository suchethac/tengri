# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2368: age_kernel accuracy bound visibility + field=True advisory.

Issue #2368: the accuracy bound of age_kernel='dsps' was stated in a dev doc but
not on the discovery surface (registry, public docs), and field=True silently
forces 'dsps' without warning. This suite verifies:

1. Every age_kernel registry row's short_doc contains the bound statement.
2. field=True with the default kernel warns, naming the forced kernel and bound.
3. field=True with explicit age_kernel='dsps' does NOT warn (user chose it).
4. field=True with age_kernel='cic' still raises NotImplementedError.
"""

from __future__ import annotations

import re
import warnings

import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform
from tengri.components.stellar.component import (
    AGE_KERNEL_ACCURACY_BOUND,
    AgeKernelFieldWarning,
)
from tengri.observation import Observation, Photometry


pytestmark = [pytest.mark.regression_bug]


def test_age_kernel_registry_rows_include_bound_sentence() -> None:
    """Every age_kernel registry row must state the accuracy bound in short_doc.

    The bound is ~1e-3 at the sharpest SFH shapes. This ensures the bound is
    discoverable via the registry (not hidden in dev docs only).
    """
    kernels = tengri.list_age_kernels()

    # Enumerate rows; no hand-picked list.
    for row in kernels:
        kernel_name = row["name"]
        short_doc = row["short_doc"]

        if kernel_name == "cic":
            # 'cic' preserves mass-proportionality to roundoff.
            assert "roundoff" in short_doc.lower() or "float64" in short_doc.lower(), (
                f"age_kernel='{kernel_name}' short_doc must state it preserves "
                f"proportionality to roundoff. Got: {short_doc}"
            )
        elif kernel_name == "dsps":
            # 'dsps' has the bound ~1e-3. Must cite it (in some form).
            # Check for any representation of the bound value.
            has_bound = any(
                pattern in short_doc.lower()
                for pattern in ["1e-3", "1e-03", "1e-03", "1e-3", "~1e-3"]
            )
            assert has_bound, (
                f"age_kernel='{kernel_name}' short_doc must include the accuracy bound "
                f"(~1e-3). Got: {short_doc}"
            )


def test_field_true_with_default_kernel_warns_advisory(synthetic_ssp) -> None:
    """field=True with the default kernel warns naming the forced kernel and bound.

    When a user leaves age_kernel unset (None) and field=True forces 'dsps',
    an advisory warns what happened. Explicit age_kernel='dsps' does NOT warn
    because the user chose it.
    """
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_i"]),
    )

    # field=True with NO explicit age_kernel should warn.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            redshift=Fixed(0.1),
            sfh={
                "type": ["dpl", "field"],
                "all_params": Fixed(DEFAULT),
            },  # No age_kernel; triggers advisory.
            met={"type": "table"},
            neb={"type": "ssp"},
        )

        # One AgeKernelFieldWarning should have been raised.
        age_kernel_warns = [x for x in w if issubclass(x.category, AgeKernelFieldWarning)]
        assert len(age_kernel_warns) >= 1, (
            f"Expected at least one AgeKernelFieldWarning when field=True "
            f"with default kernel. Got {len(age_kernel_warns)} warnings. "
            f"All warnings: {[str(x.message) for x in w]}"
        )

        msg = str(age_kernel_warns[0].message)
        # Message must name the kernel and the bound.
        assert "dsps" in msg.lower(), f"Warning message must name the 'dsps' kernel. Got: {msg}"
        # Check for any representation of the bound value.
        has_bound = any(pattern in msg.lower() for pattern in ["1e-3", "1e-03", "~1e-3"])
        assert has_bound, f"Warning message must include the bound (~1e-3). Got: {msg}"


def test_field_true_with_explicit_dsps_does_not_warn(synthetic_ssp) -> None:
    """field=True with explicit age_kernel='dsps' does NOT warn.

    The user explicitly chose 'dsps', so there is nothing to warn about.
    """
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_i"]),
    )

    # field=True with explicit age_kernel='dsps' should NOT warn.
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")

        model = SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            redshift=Fixed(0.1),
            sfh={
                "type": ["dpl", "field"],
                "age_kernel": "dsps",
                "all_params": Fixed(DEFAULT),
            },  # Explicit 'dsps'.
            met={"type": "table"},
            neb={"type": "ssp"},
        )

        # No AgeKernelFieldWarning should be raised.
        age_kernel_warns = [x for x in w if issubclass(x.category, AgeKernelFieldWarning)]
        assert len(age_kernel_warns) == 0, (
            f"Expected NO AgeKernelFieldWarning when user explicitly chose 'dsps'. "
            f"Got {len(age_kernel_warns)} warning(s): "
            f"{[str(x.message) for x in age_kernel_warns]}"
        )


def test_field_true_with_cic_still_raises(synthetic_ssp) -> None:
    """field=True with age_kernel='cic' still raises NotImplementedError.

    The CIC kernel is not supported on the field path. This refusal is unchanged.
    """
    obs = Observation(
        photometry=Photometry.from_names(["sdss_g", "sdss_i"]),
    )

    # field=True + explicit age_kernel='cic' must raise.
    with pytest.raises(NotImplementedError, match="age_kernel.*cic"):
        SEDModel.build(
            ssp_data=synthetic_ssp,
            observation=obs,
            redshift=Fixed(0.1),
            sfh={
                "type": ["dpl", "field"],
                "age_kernel": "cic",
                "all_params": Fixed(DEFAULT),
            },  # Explicit 'cic': error.
            met={"type": "table"},
            neb={"type": "ssp"},
        )
