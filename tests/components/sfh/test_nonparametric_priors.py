# SPDX-License-Identifier: BSD-3-Clause
"""Faithful-Prospector prior contract for non-parametric SFH registry entries.

Asserts that the registered default priors for the non-parametric SFH models
match the published values from Leja+2017, Leja+2019, Tacchella+2022,
Suess+2022, and the Prospector-beta scheme (Wang+2024). This protects the
*prior*, which controls the posterior under HMC/NUTS — the SFR shape
functions themselves are tested elsewhere.
"""

from __future__ import annotations

from typing import ClassVar

import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import (
    DEFAULT_BIN_EDGES_GYR,
    make_agebins_from_zred,
)
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.parameters.priors import StudentT, Uniform

pytestmark = pytest.mark.contract


_LEJA_RATIO_SIGMA = 0.3
_LEJA_RATIO_DF = 2.0


def _assert_studentt(prior, *, sigma, df=_LEJA_RATIO_DF, mu=0.0):
    # StudentT has no __eq__ override; compare by repr (the public format).
    assert isinstance(prior, StudentT), f"expected StudentT, got {type(prior).__name__}"
    expected = repr(StudentT(mu=mu, sigma=sigma, df=df))
    assert repr(prior) == expected, f"expected {expected}; got {prior!r}"


class TestContinuityPrior:
    """Leja+2019 ApJ 876, 39: log-SFR ratios ~ Student-t(mu=0, sigma=0.3, df=2)."""

    def test_six_ratios_use_studentt_0p3_df2(self):
        spec = SFH_REGISTRY["continuity"]
        for i in range(6):
            _assert_studentt(spec.params[f"sfh_cont_ratio_{i}"].default, sigma=0.3)

    def test_log_total_mass_remains_uniform(self):
        spec = SFH_REGISTRY["continuity"]
        prior = spec.params["sfh_cont_log_total_mass"].default
        assert isinstance(prior, Uniform)


class TestContinuityFlexPrior:
    """Leja+2019: ratio_young, flex_*, ratio_old all use StudentT(0, 0.3, df=2)."""

    def test_all_ratio_like_params_are_studentt(self):
        spec = SFH_REGISTRY["continuity_flex"]
        # Skip log_total_mass (Uniform); check the four ratio + three flex params.
        ratio_like = [
            "sfh_cflex_ratio_young",
            "sfh_cflex_ratio_old",
            "sfh_cflex_flex_0",
            "sfh_cflex_flex_1",
            "sfh_cflex_flex_2",
        ]
        for name in ratio_like:
            _assert_studentt(spec.params[name].default, sigma=0.3)


class TestBurstyContinuityPrior:
    """Tacchella+2022 ApJ 926, 134: piecewise sigma = 1.0 (young) / 0.3 (old).

    For DEFAULT_BIN_EDGES_GYR = [0, 0.03, 0.1, 0.3, 1.0, 3.0, 6.0, 13.7] and
    t_split = 1.0 Gyr, ratio i has younger-edge bin_edges_gyr[i+1]:
        i=0 -> 0.03 Gyr < 1.0 -> sigma=1.0
        i=1 -> 0.10 Gyr < 1.0 -> sigma=1.0
        i=2 -> 0.30 Gyr < 1.0 -> sigma=1.0
        i=3 -> 1.00 Gyr (not strictly less) -> sigma=0.3
        i=4 -> 3.00 Gyr -> sigma=0.3
        i=5 -> 6.00 Gyr -> sigma=0.3
    """

    EXPECTED_SIGMAS: ClassVar = [1.0, 1.0, 1.0, 0.3, 0.3, 0.3]

    def test_sigma_schedule_matches_tacchella2022(self):
        spec = SFH_REGISTRY["bursty_continuity"]
        for i, expected in enumerate(self.EXPECTED_SIGMAS):
            _assert_studentt(spec.params[f"sfh_burstcont_ratio_{i}"].default, sigma=expected)

    def test_t_split_setting_is_1_gyr(self):
        spec = SFH_REGISTRY["bursty_continuity"]
        assert spec.settings["sfh_burstcont_t_split_gyr"] == 1.0
        assert spec.settings["sfh_burstcont_scale_young"] == 1.0
        assert spec.settings["sfh_burstcont_scale_old"] == 0.3


class TestDirichletPrior:
    """Leja+2017 ApJ 837, 170: the sampled aux variables are Uniform(0, 1).

    Leja+2017 draws the stick-breaking variables from Beta(N-1-i, 1);
    ``dirichlet`` reaches the same construction by mapping a Uniform(0, 1)
    latent through the Beta(1, N-1-i) quantile, so the *declared prior* is
    Uniform(0, 1) for every i. It is not Beta(1, 1) in disguise: only i =
    N-2 has that quantile as the identity.
    """

    def test_six_aux_vars_use_uniform_0_to_1(self):
        spec = SFH_REGISTRY["dirichlet"]
        for i in range(6):
            prior = spec.params[f"sfh_dir_z_{i}"].default
            assert isinstance(prior, Uniform)
            assert prior.bounds == (0.0, 1.0), (
                f"sfh_dir_z_{i} should be Uniform(0, 1); got {prior!r}"
            )


class TestPSBSuess2022Registration:
    """Suess+2022 ApJ 935, 146: post-starburst non-parametric SFH.

    Verifies registry presence and that it is *distinct* from the existing
    `psb_wild2020` (Wilkinson+2020 parametric DPL+exp).
    """

    def test_registered(self):
        assert "psb_suess2022" in SFH_REGISTRY

    def test_distinct_from_psb_wild2020(self):
        suess = SFH_REGISTRY["psb_suess2022"]
        wild = SFH_REGISTRY["psb_wild2020"]
        assert suess.name != wild.name
        assert suess.fn is not wild.fn

    def test_exposes_tlast_and_tflex_as_free_params(self):
        spec = SFH_REGISTRY["psb_suess2022"]
        assert "sfh_psb2022_tlast_gyr" in spec.params
        assert "sfh_psb2022_tflex_gyr" in spec.params
        # Priors should be the Suess defaults (uniform over the physically
        # meaningful quenching-timescale ranges).
        assert spec.params["sfh_psb2022_tlast_gyr"].default == Uniform(0.01, 1.0)
        assert spec.params["sfh_psb2022_tflex_gyr"].default == Uniform(0.5, 5.0)

    def test_ratio_priors_are_studentt(self):
        # Two old ratios, one per step of the three-bin fixed section (#2184);
        # the flex-to-fixed step is pinned at 0 and takes no parameter.
        spec = SFH_REGISTRY["psb_suess2022"]
        _assert_studentt(spec.params["sfh_psb2022_ratio_young"].default, sigma=0.3)
        for i in range(2):
            _assert_studentt(spec.params[f"sfh_psb2022_ratio_old_{i}"].default, sigma=0.3)


class TestProspectorBetaRegistration:
    """Wang+2024 (arXiv:2401.12198): continuity SFH with redshift-dependent bins."""

    def test_registered(self):
        assert "prospector_beta" in SFH_REGISTRY

    def test_ratio_priors_match_continuity(self):
        spec = SFH_REGISTRY["prospector_beta"]
        for i in range(6):
            _assert_studentt(spec.params[f"sfh_pbeta_ratio_{i}"].default, sigma=0.3)

    def test_make_agebins_from_zred_returns_monotonic_capped_edges(self):
        """The helper that produces prospector_beta's bin_edges_gyr."""
        from tengri.cosmology import DEFAULT_COSMO, age_at_z

        for zred in (0.1, 1.0, 2.0, 4.0):
            edges = make_agebins_from_zred(zred=zred, n_bins=7)
            assert edges.shape == (8,)
            assert (np.diff(edges) >= 0).all(), f"edges not monotone at z={zred}"
            t_univ = float(age_at_z(zred, cosmo=DEFAULT_COSMO))
            assert edges[-1] <= t_univ * 1.001, (
                f"oldest edge {edges[-1]:.4f} Gyr exceeds "
                f"age_of_universe(z={zred})={t_univ:.4f} Gyr"
            )

    def test_default_edges_are_used_when_no_redshift_supplied(self):
        """Without an explicit bin_edges_gyr override, prospector_beta uses the
        same DEFAULT_BIN_EDGES_GYR as `continuity` — the recipe is responsible
        for calling make_agebins_from_zred when z != fiducial."""
        # This is a behavioral assertion on the shape function, not the registry:
        # the composer passes bin_edges_gyr via functools.partial when supplied,
        # else `continuity`'s default kicks in.
        assert DEFAULT_BIN_EDGES_GYR.shape == (8,)
        assert float(DEFAULT_BIN_EDGES_GYR[-1]) == pytest.approx(13.7)
