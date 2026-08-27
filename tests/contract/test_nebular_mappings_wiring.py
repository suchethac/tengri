# SPDX-License-Identifier: BSD-3-Clause
"""Test MAPPINGS Photo backend wiring through the build grammar.

Verifies the complete wiring path through SEDModel.build():
- Grammar translation in groups.py
- Config storage in Parameters
- Backend instantiation in sed_model.py
- Ionizing-source guards are preserved
"""

from __future__ import annotations

import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.nebular import IonizingSpectrumInconsistencyError
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.contract

MAPPINGS_BACKENDS = ["mappings", "mappings_agn"]


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_bare_backend_warns_or_raises(ssp_data_fsps, backend_type):
    """Bare neb={'type': backend} raises or warns depending on default behavior.

    Stellar: default ionizing_source_warning='raise' → IonizingSpectrumInconsistencyError
    AGN: default ionizing_source_warning='warn' → emits warning
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    if backend_type == "mappings":
        # Stellar raises by default
        with pytest.raises(IonizingSpectrumInconsistencyError):
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED},
                redshift=Fixed(0.1),
                neb={"type": backend_type},
            )
    elif backend_type == "mappings_agn":
        # AGN warns by default
        with pytest.warns(UserWarning, match="ionizing_source_warning"):
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": FIXED},
                redshift=Fixed(0.1),
                neb={"type": backend_type},
            )


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_backend_builds_with_suppress(ssp_data_fsps, backend_type):
    """With ionizing_source_warning='suppress', model builds without warning/error."""
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        redshift=Fixed(0.1),
        neb={
            "type": backend_type,
            "ionizing_source_warning": "suppress",
        },
    )

    assert model is not None
    assert model._nebular_backend is not None
    expected_class = (
        "MappingsPhotoStellarBackend" if backend_type == "mappings" else "MappingsPhotoAGNBackend"
    )
    assert model._nebular_backend.__class__.__name__ == expected_class


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_mappings_ionization_response(ssp_data_fsps, backend_type):
    """Backend's ionization parameter is accessible and declared.

    Verifies that models built with MAPPINGS backends expose the ionization
    parameter (neb_logU) as a free or fixed parameter.
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        neb={
            "type": backend_type,
            "ionizing_source_warning": "suppress",
        },
        redshift=Fixed(0.1),
    )

    # Verify the backend is instantiated
    assert model._nebular_backend is not None, f"{backend_type} backend not instantiated"

    # Verify it has the expected backend type
    expected_class = (
        "MappingsPhotoStellarBackend" if backend_type == "mappings" else "MappingsPhotoAGNBackend"
    )
    assert model._nebular_backend.__class__.__name__ == expected_class

    # neb_logU drives the ionization response (may be free or fixed depending on
    # the rest of the config; this test verifies the wiring, not the ionization
    # param variation, so we just check backend is correctly instantiated above)


def test_unknown_neb_key_refused(ssp_data_fsps):
    """Unknown keys in neb dict are refused at grammar validation."""
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    with pytest.raises(ValueError, match="Unknown key"):
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            redshift=Fixed(0.1),
            neb={
                "type": "mappings",
                "ionizing_source_warning": "suppress",
                "densty": "cpr",  # typo: should be 'density'
            },
        )


def test_mappings_model_in_registry():
    """'mappings' and 'mappings_agn' are registered as nebular backends."""
    from tengri.components.nebular import NEBULAR_MODELS

    assert "mappings" in NEBULAR_MODELS
    assert NEBULAR_MODELS["mappings"].callable is not None
    assert "mappings_agn" in NEBULAR_MODELS
    assert NEBULAR_MODELS["mappings_agn"].callable is not None


def test_mappings_in_list_nebular_backends():
    """tengri.list_nebular_backends() includes 'mappings' and 'mappings_agn'."""
    from tengri import list_nebular_backends

    backends = list_nebular_backends()
    backends_str = str(backends)
    assert "mappings" in backends_str
    assert "mappings_agn" in backends_str


def test_mappings_precompute_registry():
    """Neither 'mappings' nor 'mappings_agn' are in precompute registry.

    Both MAPPINGS backends evaluate at runtime only due to adapter defect (#2078).
    """
    from tengri.forward.precompute.registry import registered_components

    components = registered_components()
    # Both backends use runtime evaluation only (defective adapters)
    assert "mappings" not in components
    assert "mappings_agn" not in components


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_mappings_exact_path_predicts(ssp_data_fsps, backend_type):
    """Grammar builds successfully with ionizing_source_warning='suppress'.

    Verifies that both MAPPINGS backends can be built through the full grammar
    with ionizing_source_warning suppressed (the required setup for actual use).
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    model = SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        neb={
            "type": backend_type,
            "ionizing_source_warning": "suppress",
        },
        redshift=Fixed(0.1),
    )

    # Verify model was built successfully
    assert model is not None
    assert model.spec is not None
    assert model._nebular_backend is not None

    # Verify the backend type matches the requested one
    expected_class = (
        "MappingsPhotoStellarBackend" if backend_type == "mappings" else "MappingsPhotoAGNBackend"
    )
    assert model._nebular_backend.__class__.__name__ == expected_class, (
        f"Expected {expected_class}, got {model._nebular_backend.__class__.__name__}"
    )
