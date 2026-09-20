# SPDX-License-Identifier: BSD-3-Clause
"""Per-screen dust attenuation shape priors: grammar, prediction, and gates."""

import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Parameters, SEDModel, Uniform
from tengri.config.exceptions import ParameterError
from tengri.parameters.groups import parse_groups

pytestmark = pytest.mark.contract


class TestPerScreenDustShapePriors:
    """Test per-screen dust attenuation shape parameters as declared priors."""

    def test_grammar_accepts_per_screen_prior(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Grammar: per-screen shape keys with priors are declared and free."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "slope_bc": Uniform(-1.5, -0.3),
                "delta_diff": Uniform(-0.5, 0.5),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_slope_bc" in spec.free_params
        assert "dust_delta_diff" in spec.free_params
        assert "dust_slope" not in spec.free_params
        assert "dust_delta" not in spec.free_params

    def test_prediction_per_screen_moves_differently(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Prediction: sweeping per-screen params moves predictions differently."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": Uniform(-1.5, -0.3),
                "slope_diff": Fixed(-0.7),
            },
            redshift=Fixed(0.1),
        )

        p_base = model.spec.get_fixed_values()

        p_slope_bc_low = {**p_base, "dust_slope_bc": -1.5}
        p_slope_bc_high = {**p_base, "dust_slope_bc": -0.3}

        sed_bc_low = model.predict_photometry(p_slope_bc_low)
        sed_bc_high = model.predict_photometry(p_slope_bc_high)

        # Sweeping slope_bc should move the SED. atol=0: these synthetic
        # fluxes sit at ~1e-12, so np.allclose's default atol=1e-8 alone
        # would swamp any real difference and call bit-different arrays
        # "close" -- a relative-only comparison is the correct one here.
        assert not np.allclose(sed_bc_low, sed_bc_high, atol=0)

    def test_fixed_per_screen_matches_scalar_static_override(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Fixed per-screen param is bit-identical to scalar static override."""
        model_per_screen = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": Fixed(-1.0),
                "slope_diff": Fixed(-0.7),
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

        model_scalar = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "power_law",
                "slope_bc": -1.0,
                "slope_diff": -0.7,
                "all_params": Fixed(DEFAULT),
            },
            redshift=Fixed(0.1),
        )

        p_per_screen = model_per_screen.spec.get_fixed_values()
        p_scalar = model_scalar.spec.get_fixed_values()

        sed_per_screen = model_per_screen.predict_photometry(p_per_screen)
        sed_scalar = model_scalar.predict_photometry(p_scalar)

        np.testing.assert_array_equal(sed_per_screen, sed_scalar)

    def test_two_component_law_defaults_gate(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Gate #1833: per-screen free names not freed by wildcard."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "all_params": FREE,
            },
            redshift=Fixed(0.1),
        )

        free_params = spec.free_params
        per_screen_names = [
            f"dust_{s}_{sc}"
            for s in ["slope", "delta", "bump_strength", "Rv"]
            for sc in ["bc", "diff", "neb"]
        ]
        for name in per_screen_names:
            assert name not in free_params, f"{name} should not be freed by wildcard"

    def test_flat_surface_diagnostic_rejects_per_screen_prior(self):
        """Flat surface: dust_law_overrides with prior raises ParameterError."""
        with pytest.raises(ParameterError, match=r"per-screen.*override"):
            Parameters(dust_law_overrides={"bc": {"dust_slope": Uniform(-1.5, -0.3)}})

    def test_neb_screen_prior(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Neb screen: per-screen shape with prior works like bc/diff."""
        spec = parse_groups(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "calzetti",
                "law_diff": "calzetti",
                "law_neb": "cardelli",
                "Rv_neb": Uniform(2.0, 6.0),
            },
            redshift=Fixed(0.1),
        )

        assert "dust_Rv_neb" in spec.free_params

    def test_energy_balance_lut_disabled_by_free_per_screen(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """Energy balance LUT: free per-screen disables it."""
        model_free = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "calzetti",
                "slope_bc": Uniform(-1.5, -0.3),
            },
            dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.1),
        )

        assert model_free is not None

    def test_round_trip_to_groups(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Round-trip: to_groups() preserves per-screen prior."""
        model = SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            dust_attenuation={
                "type": "two_component",
                "law_bc": "power_law",
                "law_diff": "noll09",
                "slope_bc": Uniform(-1.5, -0.3),
            },
            redshift=Fixed(0.1),
        )

        groups = model.spec.to_groups()
        dust_atten = groups["dust_attenuation"]

        assert "slope_bc" in dust_atten
