# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the ``sfh={'age_kernel': ...}`` knob selects the SFH→SSP kernel.

Before this knob the choice was welded to an unrelated modeling flag — a
parametric SFH silently meant the cloud-in-cell kernel, a GP-field SFH silently
meant DSPS's histogram kernel (#964). These tests pin the knob's four
guarantees:

1. the default is **bit-identical** to explicit ``'cic'`` (no silent behavior
   change for existing models);
2. ``'dsps'`` genuinely selects the other kernel — end-to-end through
   ``predict_photometry``, not merely in ``predict_state``;
3. the combination the implementation cannot serve raises instead of silently
   returning the other kernel's answer;
4. the two kernels get **distinct compile signatures**, so the shared kernel
   cache cannot hand one model the other's photometry.

Guarantee 4 is a regression guard: ``age_kernel`` changes the numbers without
changing the JIT graph *shape*, which is exactly the shape of bug that has hit
this cache before (#1156, #1392, #1450). Caught in development — with the entry
missing, whichever kernel was built first won for the rest of the process.
"""

import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.contract


SFH_DELAYED = {
    "type": "delayed",
    "tau_gyr": Fixed(1.0),
    "age_gyr": Fixed(5.0),
    "log_total_mass": Fixed(10.0),
    "all_params": FIXED,
}


def _build(ssp, obs=None, **sfh_extra):
    """A minimal dust-free, fixed-z model differing only in the sfh group."""
    kwargs = {}
    if obs is not None:
        kwargs["observation"] = obs
    return SEDModel.build(
        ssp_data=ssp,
        stellar={"logzsol": Fixed(0.0), "all_params": FIXED},
        sfh=dict(SFH_DELAYED, **sfh_extra),
        dust={
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": FIXED,
        },
        redshift=Fixed(0.05),
        **kwargs,
    )


def _age_marginal(model):
    state = model.predict_state({})
    return np.asarray(state.derived["joint_weights"]).sum(axis=0)


class TestAgeKernelSelection:
    def test_default_is_bit_identical_to_explicit_cic(self, synthetic_ssp_wide):
        """Leaving age_kernel unset must not change any existing model."""
        w_default = _age_marginal(_build(synthetic_ssp_wide))
        w_cic = _age_marginal(_build(synthetic_ssp_wide, age_kernel="cic"))
        # Bit-exact, not approx: the default path must be the same code.
        assert np.array_equal(w_default, w_cic)

    def test_dsps_selects_a_different_kernel(self, synthetic_ssp_wide):
        """'dsps' must actually change the weights — a knob that no-ops is a bug."""
        w_cic = _age_marginal(_build(synthetic_ssp_wide, age_kernel="cic"))
        w_dsps = _age_marginal(_build(synthetic_ssp_wide, age_kernel="dsps"))
        assert np.abs(w_cic - w_dsps).sum() > 1e-3

    def test_dsps_reproduces_the_964_old_edge_hole(self, synthetic_ssp_wide):
        """Pins WHY 'cic' is the default: DSPS zeroes the SFH's oldest node.

        DSPS's histogram kernel interpolates log10(M(<t)) in log10(t), which
        annihilates the mass of the table segment straddling the SFH's maximum
        age. The first SSP node older than the SFH start keeps only a residual
        ~1e-5 of the share CIC gives it (#964).
        """
        lg_age = np.asarray(synthetic_ssp_wide.ssp_lg_age_gyr)
        i_above = int(np.searchsorted(10.0**lg_age * 1e9, 5.0e9))  # SFH age = 5 Gyr

        w_cic = _age_marginal(_build(synthetic_ssp_wide, age_kernel="cic"))
        w_dsps = _age_marginal(_build(synthetic_ssp_wide, age_kernel="dsps"))

        assert w_cic[i_above] > 0.01, "CIC must carry real mass at the old edge"
        # Not exactly zero — a residual survives — but negligible beside CIC's
        # share. Assert the ratio, which is what "annihilated" actually means.
        assert w_dsps[i_above] < 0.01 * w_cic[i_above], (
            f"the DSPS hole is the #964 signature: cic={w_cic[i_above]:.5e} "
            f"dsps={w_dsps[i_above]:.5e}"
        )


class TestAgeKernelReachesTheHotPath:
    """The knob must reach predict_photometry, not just predict_state.

    ``predict_photometry`` is the inference hot path and goes through the
    cached compiled observables closure; ``predict_state`` does not. A knob
    verified only on ``predict_state`` can be wholly inert where it matters.
    """

    def test_photometry_differs_between_kernels(self, synthetic_ssp_wide, synthetic_tophat_obs):
        m_cic = _build(synthetic_ssp_wide, synthetic_tophat_obs, age_kernel="cic")
        m_dsps = _build(synthetic_ssp_wide, synthetic_tophat_obs, age_kernel="dsps")
        p_cic = np.asarray(m_cic.predict_photometry({}))
        p_dsps = np.asarray(m_dsps.predict_photometry({}))
        assert np.all(np.isfinite(p_cic)) and np.all(np.isfinite(p_dsps))
        assert not np.array_equal(p_cic, p_dsps), (
            "predict_photometry is identical across kernels — the knob does not "
            "reach the compiled observables closure (compile-signature collision)"
        )

    def test_photometry_is_build_order_independent(self, synthetic_ssp_wide, synthetic_tophat_obs):
        """Whichever kernel is built first must not win for the other.

        The direct symptom of a compile-signature collision: build A then B and
        B returns A's photometry, so the answer depends on construction order.
        """

        def phot(kernel, other_first):
            if other_first is not None:
                _build(
                    synthetic_ssp_wide, synthetic_tophat_obs, age_kernel=other_first
                ).predict_photometry({})
            m = _build(synthetic_ssp_wide, synthetic_tophat_obs, age_kernel=kernel)
            return np.asarray(m.predict_photometry({}))

        assert np.array_equal(phot("dsps", None), phot("dsps", "cic"))
        assert np.array_equal(phot("cic", None), phot("cic", "dsps"))

    def test_compile_signature_distinguishes_kernels(
        self, synthetic_ssp_wide, synthetic_tophat_obs
    ):
        """The cache key itself must separate the two kernels."""
        m_cic = _build(synthetic_ssp_wide, synthetic_tophat_obs, age_kernel="cic")
        m_dsps = _build(synthetic_ssp_wide, synthetic_tophat_obs, age_kernel="dsps")
        assert m_cic.compile_signature() != m_dsps.compile_signature()


class TestAgeKernelRejectsBadInput:
    def test_unknown_kernel_raises_with_the_valid_set(self, synthetic_ssp_wide):
        with pytest.raises(ValueError, match=r"Unknown sfh age_kernel"):
            _build(synthetic_ssp_wide, age_kernel="cci")

    def test_cic_with_field_raises_rather_than_silently_using_dsps(self, synthetic_ssp_wide):
        """An explicit request the implementation cannot serve must not no-op."""
        with pytest.raises(NotImplementedError, match=r"age_kernel='cic'.*GP-field"):
            _build(synthetic_ssp_wide, age_kernel="cic", field={"all_params": FIXED})

    def test_field_default_still_builds(self, synthetic_ssp_wide):
        """Auto-select on the field path keeps resolving to DSPS silently."""
        w = _age_marginal(_build(synthetic_ssp_wide, field={"all_params": FIXED}))
        assert np.all(np.isfinite(w))


class TestAgeKernelIsDiscoverable:
    """A builder-accepted value named by no menu is undiscoverable (#1446).

    The kernel is a structural axis of ``SEDModel.build`` with two valid
    values, so it needs the same discovery surface every other axis has.
    """

    def test_menu_lists_both_kernels(self):
        import tengri

        names = [row["name"] for row in tengri.list_age_kernels()]
        assert set(names) == {"cic", "dsps"}

    def test_every_menu_name_actually_builds(self, synthetic_ssp_wide):
        """The menu must not advertise a value the builder rejects."""
        import tengri

        for row in tengri.list_age_kernels():
            model = _build(synthetic_ssp_wide, age_kernel=row["name"])
            assert np.all(np.isfinite(_age_marginal(model))), row["name"]

    def test_describe_resolves_each_kernel(self):
        """The table prints `tengri.describe(<name>)` — it must not KeyError."""
        import tengri

        for row in tengri.list_age_kernels():
            assert tengri.describe(row["name"])["kind"] == "age_kernel"

    def test_only_cic_is_production_status(self):
        """'dsps' is a comparison tool, and the menu must say so."""
        import tengri

        assert [r["name"] for r in tengri.list_age_kernels(status="production")] == ["cic"]
