from __future__ import annotations

import pytest

# SPDX-License-Identifier: BSD-3-Clause
"""Tests for auto-inference of ``met_mode`` from prior keys.

The user can either set ``met_mode="..."`` explicitly or let
:class:`tengri.parameters.Parameters` infer it from the metallicity
parameter keys present in the kwargs. Inference is data-driven from
``_MET_MODE_DISCRIMINATORS`` in the metallicity registry.
"""


from tengri import Parameters
from tengri.components.stellar.sfh.met_registry import infer_met_mode
from tengri.parameters.priors import Fixed, Uniform

pytestmark = pytest.mark.bounds


class TestInferMetMode:
    """Direct tests of the pure inference function."""

    def test_empty_keys_default_to_delta(self):
        assert infer_met_mode(set()) == "delta"

    def test_bare_logzsol_is_delta(self):
        assert infer_met_mode({"met_logzsol"}) == "delta"

    def test_zero_and_final_imply_ramp(self):
        assert infer_met_mode({"met_logzsol_0", "met_logzsol_final"}) == "ramp"

    def test_step_age_implies_two_step(self):
        # The step_age key is unique to two_step.
        assert (
            infer_met_mode({"met_logzsol_old", "met_logzsol_young", "met_step_age_gyr"})
            == "two_step"
        )

    def test_burst_implies_psb_two_step(self):
        # met_logzsol_burst is the disambiguator vs two_step.
        assert infer_met_mode({"met_logzsol_old", "met_logzsol_burst"}) == "psb_two_step"

    def test_met_bin_implies_bins(self):
        assert infer_met_mode({"met_bin_0", "met_bin_1", "met_bin_2"}) == "bins"

    def test_logzsol_base_implies_bins_continuity(self):
        assert infer_met_mode({"met_logzsol_base", "met_d_log_z_0"}) == "bins_continuity"

    def test_chem_key_implies_chem_evol(self):
        assert infer_met_mode({"chem_yield"}) == "chem_evol"
        assert infer_met_mode({"chem_eta_outflow", "chem_f_gas_init"}) == "chem_evol"

    def test_irrelevant_keys_dont_change_inference(self):
        # SFH and dust keys must not perturb the metallicity inference.
        assert (
            infer_met_mode(
                {
                    "sfh_dpl_alpha",
                    "dust_tau_diff",
                    "redshift",
                    "met_logzsol_0",
                    "met_logzsol_final",
                }
            )
            == "ramp"
        )


class TestParametersAutoInfer:
    """End-to-end: passing the right priors should pick the right mode."""

    def test_default_is_delta(self):
        spec = Parameters(met_logzsol=Fixed(-0.3))
        assert spec.met_mode == "delta"

    def test_ramp_inferred_from_zero_and_final(self):
        spec = Parameters(
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
        )
        assert spec.met_mode == "ramp"
        assert spec.evolving_metallicity is True

    def test_chem_evol_inferred_from_chem_keys(self):
        # chem_evol derives Z(t) from SFH, so met_logzsol is intentionally
        # absent from the valid-param list under this mode.
        spec = Parameters(chem_yield=Fixed(0.03))
        assert spec.met_mode == "chem_evol"
        assert spec.chem_evol is True

    def test_explicit_overrides_inference(self):
        spec = Parameters(met_mode="delta", met_logzsol=Fixed(-0.3))
        assert spec.met_mode == "delta"

    def test_explicit_conflicting_with_keys_raises(self):
        with pytest.raises(ValueError, match="conflicts with parameter keys"):
            Parameters(
                met_mode="delta",
                met_logzsol_0=Uniform(-2.0, 0.2),
                met_logzsol_final=Uniform(-2.0, 0.2),
            )

    def test_legacy_evolving_metallicity_flag_still_works(self):
        spec = Parameters(
            evolving_metallicity=True,
            met_logzsol_0=Uniform(-2.0, 0.2),
            met_logzsol_final=Uniform(-2.0, 0.2),
        )
        assert spec.met_mode == "ramp"

    def test_legacy_chem_evol_flag_still_works(self):
        spec = Parameters(chem_evol=True)
        assert spec.met_mode == "chem_evol"
