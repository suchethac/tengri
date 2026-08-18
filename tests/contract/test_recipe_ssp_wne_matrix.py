# SPDX-License-Identifier: BSD-3-Clause
"""Recipe/SSP matrix contract: which recipes work with wNE grids, and error message quality.

This test validates that:
1. Recipes correctly reject wNE (with-Nebular-Emission) SSP grids when incompatible
2. Error messages include exact, copy-pasteable remedies (grid names, env vars, commands)
3. The remedy lines survive refactoring (revert mutation test)

Issue #1946 item 1: Six of ten recipes refuse to build against the shipped wNE SSP.
The physics refusals are correct. This test guards that users receive usable escape hatches.
"""

from typing import ClassVar

import numpy as np
import pytest

from tengri import Observation, Photometry, SEDModel, recipes
from tengri.components.nebular.cue import CueWNESSPError
from tengri.components.stellar.sps.dsps_wrapper import SSPData
from tengri.config.exceptions import TengriIOError
from tengri.observation.filters import load_filter_set

pytestmark = pytest.mark.contract


def _make_mock_ssp_data(nebular_flag="unknown"):
    """Create a minimal mock SSPData for testing.

    Parameters
    ----------
    nebular_flag : str
        One of "unknown", "bare", or "included" (wNE). Flags the SSP provenance
        without downloading or building real grids.

    Returns
    -------
    SSPData
        NamedTuple with synthetic minimal grids and the specified nebular flag.
    """
    n_met, n_age, n_wave = 5, 10, 100
    return SSPData(
        ssp_wave=np.linspace(100, 10000, n_wave),
        ssp_flux=np.ones((n_met, n_age, n_wave)) * 1e-20,  # arbitrary, not zero
        ssp_lg_age_gyr=np.linspace(-1, 1, n_age),
        ssp_lgmet=np.linspace(-2, 0, n_met),
        ssp_mass_remaining=np.ones((n_met, n_age)),
        ssp_alpha_fe=None,
        nebular=nebular_flag,
    )


class TestRecipeSspMatrixWithWNE:
    """Test behavior when recipes meet a wNE SSP grid."""

    @pytest.fixture
    def obs(self):
        """Minimal observation for all recipe tests."""
        return Observation(
            photometry=Photometry.from_filter_set(
                load_filter_set(["hst_f606w", "hst_f160w", "irac_36"])
            )
        )

    @pytest.fixture
    def wne_ssp(self):
        """Mock wNE-flagged SSP (nebular emission baked in)."""
        return _make_mock_ssp_data(nebular_flag="included")

    @pytest.fixture
    def bare_ssp(self):
        """Mock bare-stellar SSP (no nebular emission baked in)."""
        return _make_mock_ssp_data(nebular_flag="bare")

    # Recipes that should build with wNE SSP (no Cue nebular backend)
    _wne_compatible: ClassVar[list[str]] = [
        "dust_demo",
        "high_z",
        "mock_recovery_minimal",
        "photoz",
    ]

    # Recipes that should refuse wNE SSP (use Cue nebular backend)
    _wne_incompatible_cue: ClassVar[list[str]] = [
        "star_forming_photometry",
        "quiescent_z0",
        "stochastic_sfh_jwst",
        "agn_panchromatic",
        "composable_agn",
    ]

    # Recipes with external data dependencies
    _data_dependent: ClassVar[list[str]] = ["unified_agn"]

    def test_wne_compatible_recipes_build(self, obs, wne_ssp):
        """Recipes without Cue backend build successfully on wNE SSP."""
        for name in self._wne_compatible:
            recipe_fn = getattr(recipes, name)
            model = SEDModel.build(ssp_data=wne_ssp, observation=obs, **recipe_fn())
            assert model is not None, f"{name} should build on wNE SSP"
            assert hasattr(model, "spec"), f"{name} model should have spec"

    def test_cue_recipes_refuse_wne_ssp(self, obs, wne_ssp):
        """Recipes with Cue backend refuse to build on wNE SSP."""
        for name in self._wne_incompatible_cue:
            recipe_fn = getattr(recipes, name)
            with pytest.raises(CueWNESSPError):
                SEDModel.build(ssp_data=wne_ssp, observation=obs, **recipe_fn())

    def test_cue_error_names_exact_grid(self, obs, wne_ssp):
        """CueWNESSPError message must name the exact downloadable grid."""
        with pytest.raises(CueWNESSPError) as exc_info:
            SEDModel.build(
                ssp_data=wne_ssp,
                observation=obs,
                **recipes.star_forming_photometry(),
            )
        error_text = str(exc_info.value)

        # Must mention the exact downloadable grid name
        assert "fsps_prsc_miles_chabrier" in error_text, (
            "error must name the exact downloadable grid; got:\n" + error_text
        )

    def test_cue_error_includes_copy_pasteable_command(self, obs, wne_ssp):
        """CueWNESSPError message must include exact download command."""
        with pytest.raises(CueWNESSPError) as exc_info:
            SEDModel.build(
                ssp_data=wne_ssp,
                observation=obs,
                **recipes.star_forming_photometry(),
            )
        error_text = str(exc_info.value)

        # Must include the exact Python call users can copy
        assert "tengri.download_ssp('fsps_prsc_miles_chabrier')" in error_text, (
            "error must include exact copy-pasteable Python command; got:\n" + error_text
        )

    def test_cue_recipes_build_on_bare_ssp(self, obs, bare_ssp):
        """Recipes with Cue backend should build on bare-stellar SSP.

        This is the happy path: once the user downloads the right grid,
        the recipe works.
        """
        for name in self._wne_incompatible_cue:
            recipe_fn = getattr(recipes, name)
            # Note: some recipes may still fail for other reasons (e.g. free params
            # without values), but they should not fail due to Cue + wNE conflict.
            try:
                model = SEDModel.build(ssp_data=bare_ssp, observation=obs, **recipe_fn())
                assert model is not None, f"{name} should build on bare SSP"
            except CueWNESSPError:
                pytest.fail(
                    f"{name} raised CueWNESSPError on a bare-stellar SSP — "
                    "the wNE check is too broad"
                )

    def test_unified_agn_raises_tengrioerror_not_file_not_found(self, obs, bare_ssp):
        """unified_agn missing Synthesizer grid raises TengriIOError, not FileNotFoundError.

        This ensures the error message is user-friendly with clear remedies.
        """
        with pytest.raises(TengriIOError):
            SEDModel.build(ssp_data=bare_ssp, observation=obs, **recipes.unified_agn())

    def test_unified_agn_error_names_env_var(self, obs, bare_ssp):
        """unified_agn error must name the exact environment variable."""
        with pytest.raises(TengriIOError) as exc_info:
            SEDModel.build(ssp_data=bare_ssp, observation=obs, **recipes.unified_agn())
        error_text = str(exc_info.value)

        # Must name the exact env var users can set
        assert "TENGRI_SYNTHESIZER_AGN_GRID_DIR" in error_text, (
            "error must name the environment variable; got:\n" + error_text
        )

    def test_unified_agn_error_includes_download_command(self, obs, bare_ssp):
        """unified_agn error must include the exact download command."""
        with pytest.raises(TengriIOError) as exc_info:
            SEDModel.build(ssp_data=bare_ssp, observation=obs, **recipes.unified_agn())
        error_text = str(exc_info.value)

        # Must include the exact shell command
        assert "synthesizer-download --agn-test-grids" in error_text, (
            "error must include exact download command; got:\n" + error_text
        )


class TestRecipeDownloadDocstringPresence:
    """Guard that recipes document how to download required grids.

    The docstring is the user's first stop when they hit an error.
    """

    _cue_recipes: ClassVar[list[str]] = [
        "star_forming_photometry",
        "quiescent_z0",
        "stochastic_sfh_jwst",
        "agn_panchromatic",
        "composable_agn",
    ]

    def test_cue_recipes_have_download_docstring(self):
        """All Cue recipes must include download instructions in docstring."""
        for name in self._cue_recipes:
            recipe_fn = getattr(recipes, name)
            docstring = recipe_fn.__doc__ or ""
            assert "download_ssp" in docstring, (
                f"{name} docstring must mention download_ssp; got:\n{docstring}"
            )
            assert "fsps_prsc_miles_chabrier" in docstring, (
                f"{name} docstring must name exact grid; got:\n{docstring}"
            )
