# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for #2368 / #2684: age_kernel accuracy bound visibility, field kernel default.

1. Every age_kernel registry row's short_doc contains the bound statement.
2. field=True resolves to the default 'cic' kernel (accurate for field and rough
   histories) without an advisory.
3. field=True with explicit 'dsps' or 'cic' builds.
"""

from __future__ import annotations

import warnings

import pytest

import tengri
from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.stellar.component import _resolve_age_kernel
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
            # 'dsps' has the bound ~1e-3. Must cite it on discovery surface.
            assert "1e-3" in short_doc, (
                f"age_kernel='{kernel_name}' short_doc must include the accuracy bound "
                f"(1e-3). Got: {short_doc}"
            )


def _build_field(ssp, kernel):
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_i"]))
    sfh = {"type": ["dpl", "field"], "all_params": Fixed(DEFAULT)}
    if kernel is not None:
        sfh["age_kernel"] = kernel
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            redshift=Fixed(0.1),
            sfh=sfh,
            met={"type": "table"},
            neb={"type": "ssp"},
        )
    return model, w


def test_field_default_kernel_is_cic_without_advisory(synthetic_ssp) -> None:
    model, w = _build_field(synthetic_ssp, None)
    assert not [x for x in w if "age_kernel" in str(x.message)], [str(x.message) for x in w]
    stellar = next(
        c for c in model._build_component_chain() if type(c).__name__.startswith("Stellar")
    )
    assert _resolve_age_kernel(stellar.config) == "cic"


@pytest.mark.parametrize("kernel", ["cic", "dsps"])
def test_field_true_builds_with_either_kernel(synthetic_ssp, kernel) -> None:
    model, w = _build_field(synthetic_ssp, kernel)
    assert not [x for x in w if "age_kernel" in str(x.message)]
    assert model is not None
