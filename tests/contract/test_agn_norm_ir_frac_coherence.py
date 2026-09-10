# SPDX-License-Identifier: BSD-3-Clause
"""R65: ``agn_norm='independent'`` beside an active fracAGN is refused.

``agn_ir_frac`` (fracAGN) is the CIGALE ``skirtor2016`` coupling: it derives
the AGN power from the **dust-absorbed stellar** luminosity,
``agn_power = L_absorbed x f/(1 - f)``, and every ``agn_norm`` policy routes
the torus through that derived power. Under ``'independent'`` the disc is
NOT: by that policy's contract it sits on ``10**agn_log_lbol`` and takes no
cross-block tie. So the two components' absolute scales are set by two
unrelated quantities -- the stellar mass and dust content on one side, the AGN
luminosity parameter on the other -- and their ratio is not modeled at all.

Measured on this branch (composable ``disc='schartmann2005'`` +
``torus='skirtor'``, ``sfh='delayed'``, ``dust_emission='dale2014_cigale'``,
``agn_log_lbol`` at its registry default), sweeping ONLY the stellar mass and
reading ``int(sed_agn_disc) / int(sed_agn_torus)`` over frequency:

============  ====================  ===================  ====================
``log M*``    independent, f=0.3    independent, no f    cigale_joint, f=0.3
============  ====================  ===================  ====================
0.0            5.004902e+10          2.001533            2.838156
4.0            5.004902e+06          2.001533            2.838156
7.0            5.004902e+03          2.001533            2.838156
10.0           5.004902e+00          2.001533            2.838156
12.0           5.004902e-02          2.001533            2.838156
============  ====================  ===================  ====================

Twelve orders of magnitude across the stellar-mass prior in the refused
column, and constant in both legal ones: the disc/torus ratio there is a
readout of ``M*``, not of any AGN parameter. Nothing raised and nothing
warned, and the four components still summed to ``sed_agn`` exactly
(1.000000), so the accounting stayed intact while the configuration meant
nothing.

References
----------
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE: the fracAGN coupling)
- Stalevski et al. 2016, MNRAS, 458, 2288 (SKIRTOR)
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed

pytestmark = pytest.mark.contract


def _build(ssp, *, norm, ir_frac=None, log_total_mass=10.0):
    """One composable AGN build: disc=schartmann2005 + torus=skirtor.

    ``ir_frac`` unset leaves ``agn_ir_frac`` at its registry default (0.0,
    inactive). ``norm`` is always explicit -- the policy is the subject here.
    """
    import tengri

    agn = {
        "type": "composable",
        "norm": norm,
        "disc": {"type": "schartmann2005", "all_params": Fixed(DEFAULT)},
        "torus": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
        "all_params": Fixed(DEFAULT),
    }
    if ir_frac is not None:
        agn["agn_ir_frac"] = ir_frac
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return tengri.SEDModel.build(
            ssp,
            sfh={
                "type": "delayed",
                "tau_gyr": Fixed(1.0),
                "age_gyr": Fixed(5.0),
                "log_total_mass": Fixed(log_total_mass),
                "all_params": Fixed(DEFAULT),
            },
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "tau_bc": Fixed(0.0),
                "tau_diff": Fixed(0.5),
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": "dale2014_cigale", "all_params": Fixed(DEFAULT)},
            agn=agn,
            redshift=Fixed(0.05),
        )


class TestIndependentNormRefusesActiveFracAgn:
    """The refusal, its message, and every neighboring build that must survive."""

    def test_fixed_positive_ir_frac_raises(self, synthetic_ssp_wide):
        """``Fixed(0.3)`` fracAGN beside ``'independent'``: refused."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_ir_frac"):
            _build(synthetic_ssp_wide, norm="independent", ir_frac=Fixed(0.3))

    def test_free_ir_frac_raises(self, synthetic_ssp_wide):
        """A FREE fracAGN is active by construction, so it is refused too."""
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError, match=r"agn_ir_frac"):
            _build(synthetic_ssp_wide, norm="independent", ir_frac=FREE)

    def test_refusal_names_both_joint_policies_and_the_zero_way_out(self, synthetic_ssp_wide):
        """The message must name what to switch to, and only legal things.

        A guard that recommends a configuration it also refuses is the #1364
        defect, so each of the three remedies named here is exercised by a
        test below.
        """
        from tengri.config.exceptions import ConfigError

        with pytest.raises(ConfigError) as exc:
            _build(synthetic_ssp_wide, norm="independent", ir_frac=Fixed(0.3))
        msg = str(exc.value)
        for token in ("agn_ir_frac", "independent", "cigale_joint", "conserving"):
            assert token in msg, f"the refusal does not name {token!r}: {msg}"
        assert "L_absorbed" in msg, (
            f"the refusal must say where the AGN power comes from under the coupling: {msg}"
        )

    # ── every neighboring configuration must keep building ─────────

    def test_independent_without_ir_frac_builds(self, synthetic_ssp_wide):
        """The policy itself is untouched: no fracAGN, no coupling, no refusal."""
        model = _build(synthetic_ssp_wide, norm="independent")
        assert model.spec.agn_model == "composable"

    def test_independent_with_explicitly_zero_ir_frac_builds(self, synthetic_ssp_wide):
        """``Fixed(0.0)`` states "no coupling" -- inert, and legal.

        This is one of the three remedies the refusal names, so it must not be
        refused itself.
        """
        model = _build(synthetic_ssp_wide, norm="independent", ir_frac=Fixed(0.0))
        assert model.spec.agn_model == "composable"

    def test_cigale_joint_with_ir_frac_builds(self, synthetic_ssp_wide):
        """The second remedy: the joint policy the coupling was written for."""
        model = _build(synthetic_ssp_wide, norm="cigale_joint", ir_frac=Fixed(0.3))
        assert model.spec.agn_model == "composable"

    def test_conserving_with_ir_frac_builds(self, synthetic_ssp_wide):
        """The third remedy: the energy-ledger policy, also not refused."""
        model = _build(synthetic_ssp_wide, norm="conserving", ir_frac=Fixed(0.3))
        assert model.spec.agn_model == "composable"


class TestJointPolicyRatioIsMassIndependent:
    """The property the refusal protects, measured live rather than asserted.

    Under ``'cigale_joint'`` with the SKIRTOR torus, disc and torus are tied to
    the SAME ``agn_power`` reference through the template ratio ``R``, so their
    bolometric ratio is a property of the AGN model alone. The refused
    configuration's ratio is instead proportional to ``1/M*`` -- the table in
    this module's docstring. Sweeping the stellar mass is what tells the two
    apart, and it is the sweep, not the guard, that says the ruling was right.
    """

    @staticmethod
    def _disc_over_torus(ssp, *, norm, ir_frac, log_total_mass):
        from tengri.utils.physics_constants import C_AA

        state = _build(
            ssp, norm=norm, ir_frac=ir_frac, log_total_mass=log_total_mass
        ).predict_state({})
        wave = np.asarray(state.wave)
        nu = C_AA / wave
        order = np.argsort(nu)

        def integrate(key):
            return float(np.trapezoid(np.asarray(state.derived[key])[order], nu[order]))

        torus = integrate("sed_agn_torus")
        assert torus > 0.0, "probe setup failed: the torus is not emitting"
        return integrate("sed_agn_disc") / torus

    def test_cigale_joint_disc_over_torus_does_not_move_with_stellar_mass(
        self, synthetic_ssp_wide
    ):
        """1e10x of stellar mass, and the AGN's own ratio does not budge."""
        low = self._disc_over_torus(
            synthetic_ssp_wide, norm="cigale_joint", ir_frac=Fixed(0.3), log_total_mass=0.0
        )
        high = self._disc_over_torus(
            synthetic_ssp_wide, norm="cigale_joint", ir_frac=Fixed(0.3), log_total_mass=10.0
        )
        assert high / low == pytest.approx(1.0, rel=1e-6, abs=0.0), (
            f"int(disc)/int(torus) moved from {low:.9e} at log M* = 0 to "
            f"{high:.9e} at log M* = 10 under agn_norm='cigale_joint': the "
            "disc is no longer tied to the same agn_power reference as the "
            "torus, which is the property that makes the coupling meaningful"
        )

    def test_independent_without_ir_frac_is_also_mass_independent(self, synthetic_ssp_wide):
        """The control: without the coupling, ``'independent'`` is fine too.

        Both components are then on ``agn_log_lbol``-derived scales, so the
        ratio is again a property of the AGN model. This is why the refusal is
        about the COMBINATION and not about the policy.
        """
        low = self._disc_over_torus(
            synthetic_ssp_wide, norm="independent", ir_frac=None, log_total_mass=0.0
        )
        high = self._disc_over_torus(
            synthetic_ssp_wide, norm="independent", ir_frac=None, log_total_mass=10.0
        )
        assert high / low == pytest.approx(1.0, rel=1e-6, abs=0.0), (
            f"int(disc)/int(torus) moved from {low:.9e} to {high:.9e} across "
            "1e10x of stellar mass with no fracAGN coupling active"
        )
