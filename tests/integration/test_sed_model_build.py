# SPDX-License-Identifier: BSD-3-Clause
"""Integration tests for SEDModel.build().

Verifies that the nested-dict form produces predictions identical to the
equivalent flat-kwarg Parameters(...) form. Requires the FSPS Chabrier
SSP file under data/; skips if absent.
"""

from pathlib import Path

import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, Parameters, SEDModel, Uniform, parse_groups
from tengri.components.stellar.sps.dsps_wrapper import load_ssp_data

# ── SSP fixture ───────────────────────────────────────────────────
_DATA_DIR = Path(__file__).resolve().parents[2] / "data"
_SSP_FILE = _DATA_DIR / "fsps_mist_c3k_a_chabrier.h5"

pytestmark = pytest.mark.skipif(
    not _SSP_FILE.is_file(),
    reason=f"SSP file {_SSP_FILE.name} not present in data/",
)


@pytest.fixture(scope="module")
def ssp():
    """Load the FSPS Chabrier SSP grid once per test module."""
    return load_ssp_data(str(_SSP_FILE))


# ── Construction tests ────────────────────────────────────────────


class TestFromGroupsConstruction:
    """Verify SEDModel.build returns a usable SEDModel."""

    def test_minimal_model_builds(self, ssp):
        """Smallest possible config builds without error."""
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert isinstance(model, SEDModel)
        assert isinstance(model.spec, Parameters)

    def test_spec_matches_parse_groups(self, ssp):
        """The internal spec is exactly what parse_groups would produce."""
        groups = dict(
            sfh={"type": "dpl", "*": FREE, "beta": Uniform(1, 3)},
            dust={
                "law_diff": "calzetti",
                "type": "two_component",
                "law_bc": "calzetti",
                "*": FIXED,
                "tau_bc": 0.5,
            },
            redshift=Fixed(0.05),
        )
        model = SEDModel.build(ssp_data=ssp, **groups)
        spec_via_from_groups = parse_groups(**groups)

        assert model.spec.free_params == spec_via_from_groups.free_params
        assert model.spec.fixed_params == spec_via_from_groups.fixed_params

    def test_none_groups_are_dropped(self, ssp):
        """Passing None for an optional group is a no-op."""
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FIXED},
            dust=None,
            neb=None,
            redshift=Fixed(0.1),
        )
        assert isinstance(model, SEDModel)


# ── Equivalence with flat-kwarg form ─────────────────────────────


class TestFlatEquivalence:
    """SEDModel.build must produce predictions identical to the flat form."""

    @pytest.fixture
    def equivalent_models(self, ssp):
        """A pair (grouped, flat) of SEDModels with identical physics."""
        grouped = SEDModel.build(
            ssp_data=ssp,
            sfh={
                "type": "dpl",
                "log_total_mass": Uniform(-1.0, 2.0),
                "alpha": Uniform(0.5, 3.0),
                "beta": Uniform(0.3, 2.0),
                "tau_gyr": Uniform(0.5, 10.0),
                "logzsol": Fixed(-0.1),
            },
            dust={
                "law_diff": "calzetti",
                "type": "two_component",
                "law_bc": "calzetti",
                "tau_bc": Fixed(0.5),
                "tau_diff": Fixed(0.3),
                "slope": Fixed(-0.7),
            },
            redshift=Fixed(0.05),
        )
        flat_spec = Parameters(
            mean_sfh_type="dpl",
            # Free by default in the flat form (it carries a registry prior), but never
            # varied here. Pin it at the registry default -- the value the forward model
            # silently substituted before #1015 made the omission a loud error (#1021).
            sfh_dpl_age_gyr=Fixed(13.81),
            sfh_dpl_log_total_mass=Uniform(7.0, 12.5),
            sfh_dpl_alpha=Uniform(0.5, 3.0),
            sfh_dpl_beta=Uniform(0.3, 2.0),
            sfh_dpl_tau_gyr=Uniform(0.5, 10.0),
            met_logzsol=Fixed(-0.1),
            dust_model="two_component",
            dust_law_bc="calzetti",
            dust_tau_bc=Fixed(0.5),
            dust_tau_diff=Fixed(0.3),
            dust_slope=Fixed(-0.7),
            redshift=Fixed(0.05),
        )
        flat = SEDModel(flat_spec, ssp)
        return grouped, flat

    def test_same_free_param_set(self, equivalent_models):
        """Both forms expose the same free + fixed parameter sets."""
        grouped, flat = equivalent_models
        assert set(grouped.spec.free_params) == set(flat.spec.free_params)
        assert set(grouped.spec.fixed_params) == set(flat.spec.fixed_params)

    def test_same_rest_sed_prediction(self, equivalent_models):
        """At identical parameter values, both forms predict the same rest-frame SED."""
        grouped, flat = equivalent_models

        truth = {
            "sfh_dpl_log_total_mass": float(np.log10(15.0)),
            "sfh_dpl_alpha": 2.0,
            "sfh_dpl_beta": 1.5,
            "sfh_dpl_tau_gyr": 2.0,
        }

        sed_grouped = grouped.predict_rest_sed(truth)
        sed_flat = flat.predict_rest_sed(truth)

        np.testing.assert_allclose(
            np.asarray(sed_grouped.sed),
            np.asarray(sed_flat.sed),
            rtol=0.0,
            atol=0.0,
            err_msg="Grouped and flat forms must produce bit-identical rest SEDs",
        )

    def test_same_sfh_prediction(self, equivalent_models):
        """At identical parameter values, both forms predict the same SFH."""
        grouped, flat = equivalent_models

        truth = {
            "sfh_dpl_log_total_mass": 1.0,
            "sfh_dpl_alpha": 1.8,
            "sfh_dpl_beta": 1.2,
            "sfh_dpl_tau_gyr": 3.0,
        }

        sfr_grouped = grouped.predict_sfh(truth)
        sfr_flat = flat.predict_sfh(truth)

        np.testing.assert_allclose(
            np.asarray(sfr_grouped["sfr_mean"]),
            np.asarray(sfr_flat["sfr_mean"]),
            rtol=0.0,
            atol=0.0,
            err_msg="Grouped and flat forms must produce bit-identical SFHs",
        )


# ── Validation propagates from parse_groups ───────────────────────


class TestValidation:
    """Errors from the parser must propagate out of SEDModel.build."""

    def test_agn_group_builds_composable_model(self, ssp):
        """AGN group activates composable AGN model (was deferred until PR4)."""
        model = SEDModel.build(
            ssp_data=ssp,
            sfh={"type": "dpl", "*": FIXED},
            agn={"disc": {"type": "powerlaw", "*": FIXED}},
            redshift=Fixed(0.1),
        )
        assert model.spec.agn_model == "composable"
        assert model.spec.agn_disc_block == "powerlaw"

    def test_unknown_type_raises_value_error(self, ssp):
        with pytest.raises(ValueError):
            SEDModel.build(
                ssp_data=ssp,
                sfh={"type": "banana", "*": FIXED},
                redshift=Fixed(0.1),
            )
