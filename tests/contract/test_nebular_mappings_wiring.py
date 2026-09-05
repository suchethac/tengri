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

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.nebular import IonizingSpectrumInconsistencyError
from tengri.config.exceptions import TengriIOError
from tengri.observation import Observation, Photometry

pytestmark = pytest.mark.contract

MAPPINGS_BACKENDS = ["mappings", "mappings_agn"]


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_bare_backend_warns_or_raises(ssp_data_fsps, backend_type):
    """Bare neb={'type': backend} raises depending on backend type.

    Both backends refuse at build time with clear errors:
    - Stellar: default ionizing_source_warning='raise' → IonizingSpectrumInconsistencyError
    - AGN: ValueError (model protocol surface incomplete)
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    if backend_type == "mappings":
        # Stellar raises by default (ionizing_source_warning='raise')
        with pytest.raises(IonizingSpectrumInconsistencyError):
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
                neb={"type": backend_type},
            )
    elif backend_type == "mappings_agn":
        # AGN raises at build time due to protocol incompleteness
        with pytest.raises(ValueError, match="predict_nebular_sed"):
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
                neb={"type": backend_type},
            )


@pytest.mark.parametrize("backend_type", MAPPINGS_BACKENDS)
def test_backend_builds_with_suppress(ssp_data_fsps, backend_type):
    """With ionizing_source_warning='suppress', building raises with a clear refusal message.

    Both backends are registered but refuse loudly at build time:
    - 'mappings': TengriIOError (grid data is incomplete, 51.2% NaN, 2656/5184 cells)
    - 'mappings_agn': ValueError (model protocol surface incomplete, missing predict_nebular_sed)
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    if backend_type == "mappings":
        # Stellar backend refuses at build time due to incomplete grid
        with pytest.raises(TengriIOError) as exc_info:
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
                neb={
                    "type": backend_type,
                    "ionizing_source_warning": "suppress",
                },
            )
        # Verify the error message names the exact problem
        msg = str(exc_info.value)
        assert "2656" in msg, "NaN cell count not in error message"
        assert "grid data is incomplete" in msg, "Grid completeness status not in error"
        assert "#2082" in msg, "Issue reference not in error message"

    elif backend_type == "mappings_agn":
        # AGN backend refuses at build time due to missing protocol surface
        with pytest.raises(ValueError) as exc_info:
            SEDModel.build(
                ssp_data=ssp_data_fsps,
                observation=obs,
                sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
                redshift=Fixed(0.1),
                neb={
                    "type": backend_type,
                    "ionizing_source_warning": "suppress",
                },
            )
        # Verify the error message names the missing protocol surface
        msg = str(exc_info.value)
        assert "predict_nebular_sed" in msg, "Protocol surface not in error message"
        assert "#2082" in msg, "Issue reference not in error message"


def test_mappings_ionization_response(ssp_data_fsps):
    """MappingsPhotoStellarBackend: building refuses with grid incompleteness guard.

    The backend's ionization parameter (neb_logU) is declared in the grammar,
    but the backend refuses at build time because the grid data is incomplete.
    (Issue #2082)
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    # Stellar backend refuses at build time due to grid incompleteness
    with pytest.raises(TengriIOError) as exc_info:
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={
                "type": "mappings",
                "ionizing_source_warning": "suppress",
            },
            redshift=Fixed(0.1),
        )

    # Verify the error message is clear about the grid issue
    msg = str(exc_info.value)
    assert "grid data is incomplete" in msg, "Grid status not in error message"
    assert "2656" in msg, "NaN cell count not in error message"
    assert "#2082" in msg, "Issue reference not in error message"


def test_unknown_neb_key_refused(ssp_data_fsps):
    """Unknown keys in neb dict are refused at grammar validation."""
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    with pytest.raises(ValueError, match="Unknown key"):
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
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


def test_mappings_exact_path_predicts(ssp_data_fsps):
    """MappingsPhotoStellarBackend: building refuses with clear grid incompleteness message.

    The stellar backend is registered but refuses at initialization because the
    grid file contains 51.2% NaN values in logHB_per_logq (2656/5184 cells),
    making it unsuitable for predictions. (Issue #2082)
    """
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r"]))

    # Building with the mappings backend should raise TengriIOError
    with pytest.raises(TengriIOError) as exc_info:
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={
                "type": "mappings",
                "ionizing_source_warning": "suppress",
            },
            redshift=Fixed(0.1),
        )

    # Verify the error message contains the exact problem details
    msg = str(exc_info.value)
    assert "grid data is incomplete" in msg, "Grid status not in error message"
    assert "51.2%" in msg or "2656" in msg, "NaN count or percentage not in error"
    assert "5184" in msg, "Total cell count not in error message"
    assert "scripts/build_flury2024_grids.py" in msg, "Resolution script not in error"
    assert "#2082" in msg, "Issue reference not in error message"


def test_mappings_ionization_response_varies(ssp_data_fsps):
    """MappingsPhotoStellarBackend: building refuses with any configuration.

    The backend refuses at build time due to grid incompleteness, regardless
    of observation configuration or other parameters. (Issue #2082)
    """
    from tengri.observation.filters import load_tophat_filter

    # Use a tophat filter that would capture ionization-sensitive lines (if it worked)
    filt = load_tophat_filter(6200 - 250, 6700 + 250, name="hbeta_oiii_z03")
    obs = Observation(photometry=Photometry(filters=[filt]))

    # The backend refuses at build time regardless of observation or parameters
    with pytest.raises(TengriIOError) as exc_info:
        SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={
                "type": "mappings",
                "ionizing_source_warning": "suppress",
            },
            redshift=Fixed(0.3),
        )

    # Verify the error is about grid incompleteness
    msg = str(exc_info.value)
    assert "grid data is incomplete" in msg, "Grid status not in error"
    assert "2656" in msg, "NaN cell count not in error"
    assert "#2082" in msg, "Issue reference not in error"
