# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for DL07 template grid axes.

Validates that the regenerated DL07 templates have the correct axes after the
fixes for issues #2535 (U_min axis) and #2441 (q_PAH axis):

- U_min axis should be the 22-node published Draine & Li (2007) grid:
  [0.1, 0.15, 0.2, ..., 8.0, 12.0, 15.0, 20.0, 25.0] (no 10.0, ends at 25.0)
- q_PAH axis should contain only MW3.1 models (7 nodes), not SMC/LMC2

This module also tests the closure-level physics of the DL07 model at these
grid points to confirm that interpolation works correctly.
"""

import pytest

pytestmark = pytest.mark.regression_bug

import numpy as np


@pytest.fixture
def dl07_v1_path():
    """Path to v1 format DL07 templates."""
    from tengri._data_setup import find_data_str

    path = find_data_str("dl07_templates.h5")
    if path is None:
        pytest.skip("dl07_templates.h5 not found")
    return path


@pytest.fixture
def dl07_v2_path():
    """Path to v2 format DL07 templates."""
    from tengri._data_setup import find_data_str

    path = find_data_str("dl07_templates_v2.h5")
    if path is None:
        pytest.skip("dl07_templates_v2.h5 not found")
    return path


class TestDL07AxisesAreCorrect:
    """Test that both DL07 HDF5 files have the correct published axes."""

    @pytest.mark.parametrize("file_format", ["v1", "v2"])
    def test_umin_axis_matches_published(self, dl07_v1_path, dl07_v2_path, file_format):
        """U_min axis must be the 22-node DL07 published ladder.

        Draine & Li 2007 Table 3 publishes exactly:
        [0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.7, 0.8, 1.0, 1.2, 1.5, 2.0,
         2.5, 3.0, 4.0, 5.0, 7.0, 8.0, 12.0, 15.0, 20.0, 25.0]
        """
        import h5py

        path = dl07_v1_path if file_format == "v1" else dl07_v2_path

        with h5py.File(path, "r") as f:
            if file_format == "v1":
                umin = np.asarray(f["umin_grid"])
            else:
                umin = np.asarray(f["grid"]["umin"])

        # Published axis (from Draine & Li 2007 Table 3)
        published = np.array(
            [
                0.10,
                0.15,
                0.20,
                0.30,
                0.40,
                0.50,
                0.70,
                0.80,
                1.00,
                1.20,
                1.50,
                2.00,
                2.50,
                3.00,
                4.00,
                5.00,
                7.00,
                8.00,
                12.0,
                15.0,
                20.0,
                25.0,
            ]
        )

        assert len(umin) == 22, f"Expected 22 U_min nodes, got {len(umin)}"
        np.testing.assert_array_almost_equal(
            umin,
            published,
            err_msg=f"U_min axis {umin} does not match published {published}",
        )

    def test_umin_no_spurious_10_0(self, dl07_v1_path, dl07_v2_path):
        """U_min must NOT contain 10.0 (the bug in issue #2535)."""
        import h5py

        for path, fmt in [(dl07_v1_path, "v1"), (dl07_v2_path, "v2")]:
            with h5py.File(path, "r") as f:
                if fmt == "v1":
                    umin = np.asarray(f["umin_grid"])
                else:
                    umin = np.asarray(f["grid"]["umin"])

            assert 10.0 not in umin, (
                f"U_min axis contains spurious 10.0 node (issue #2535). Axis: {umin}"
            )

    def test_umin_ends_at_25_not_20(self, dl07_v1_path, dl07_v2_path):
        """U_min must reach 25.0 (the ceiling from DL07)."""
        import h5py

        for path, fmt in [(dl07_v1_path, "v1"), (dl07_v2_path, "v2")]:
            with h5py.File(path, "r") as f:
                if fmt == "v1":
                    umin = np.asarray(f["umin_grid"])
                else:
                    umin = np.asarray(f["grid"]["umin"])

            assert 25.0 in umin, (
                f"U_min axis missing the top published node 25.0 "
                f"(was clamped at 20.0 in the bug). Axis: {umin}"
            )
            assert umin[-1] == 25.0, f"U_min axis does not end at 25.0: {umin}"

    @pytest.mark.parametrize("file_format", ["v1", "v2"])
    def test_qpah_axis_mw3_1_only(self, dl07_v1_path, dl07_v2_path, file_format):
        """q_PAH axis must contain only MW3.1 nodes (7 values), not SMC/LMC2."""
        import h5py

        path = dl07_v1_path if file_format == "v1" else dl07_v2_path

        with h5py.File(path, "r") as f:
            if file_format == "v1":
                qpah = np.asarray(f["qpah_grid"])
            else:
                qpah = np.asarray(f["grid"]["qpah"])

        # MW3.1 ladder from Draine & Li 2007 Table 1
        mw_expected = np.array([0.47, 1.12, 1.77, 2.50, 3.19, 3.90, 4.58])

        assert len(qpah) == 7, (
            f"Expected 7 q_PAH nodes (MW3.1 only), got {len(qpah)}. "
            f"Issue #2441: q_PAH should not mix SMC/LMC2 with MW3.1."
        )
        np.testing.assert_array_almost_equal(
            qpah,
            mw_expected,
            err_msg=f"q_PAH axis {qpah} does not match MW3.1 nodes {mw_expected}",
        )

    def test_qpah_no_smc_lmc2(self, dl07_v1_path, dl07_v2_path):
        """q_PAH must NOT contain SMC or LMC2 nodes (the bug in issue #2441)."""
        import h5py

        smc_qpah = 0.10
        lmc2_qpahs = {0.75, 1.49, 2.37}

        for path, fmt in [(dl07_v1_path, "v1"), (dl07_v2_path, "v2")]:
            with h5py.File(path, "r") as f:
                if fmt == "v1":
                    qpah = np.asarray(f["qpah_grid"])
                else:
                    qpah = np.asarray(f["grid"]["qpah"])

            assert smc_qpah not in qpah, (
                f"q_PAH contains SMC (0.10), violating issue #2441. Axis: {qpah}"
            )
            for lmc2_val in lmc2_qpahs:
                assert lmc2_val not in qpah, (
                    f"q_PAH contains LMC2 ({lmc2_val}), violating issue #2441. Axis: {qpah}"
                )


class TestDL07ParameterBounds:
    """Validate that the dust_umin parameter bounds were updated correctly."""

    def test_dust_umin_prior_extended_to_25(self):
        """dust_umin free prior must reach 25.0 after #2535 fix."""
        from tengri.components.dust._params import PARAMS

        # Find the dust_umin parameter declaration
        dust_umin_param = None
        for param in PARAMS:
            if param.name == "dust_umin":
                dust_umin_param = param
                break

        assert dust_umin_param is not None, "dust_umin parameter not found in PARAMS"

        # The free_prior should extend to 25.0 (was 20.0 before the fix)
        assert dust_umin_param.free_prior.hi == 25.0, (
            f"dust_umin free_prior ceiling is {dust_umin_param.free_prior.hi}, "
            f"expected 25.0 (the published DL07 ceiling). "
            f"Issue #2535: The prior was lowered to match the wrong shipped grid."
        )

        # The comment should mention the fix
        assert "0.1-25" in dust_umin_param.description, (
            f"dust_umin description should mention 0.1-25 range. "
            f"Got: {dust_umin_param.description}"
        )


# ── Equality with the pre-#2535 shipped grid ────────────────────────────
#
# The 2026-09-29 regeneration corrects ONLY the axis labels (U_min: drop the
# spurious 10.0 node, add 25.0; q_PAH: drop the 4 non-MW3.1 rows) -- the
# underlying ``single_u``/``powerlaw`` numeric arrays for every retained node
# are byte-for-byte unchanged from the file shipped before this fix
# (``relabel_shipped_grid`` in ``scripts/convert_dl07_templates.py``; see its
# docstring for the empirical proof that the four "shifted" top U_min slots
# already held the correct physical templates, just mislabeled).
#
# For full verification against an actual old file, set
# ``TENGRI_DL07_OLD_V1_PATH`` to a copy of the pre-#2535
# ``data/dl07_templates.h5`` (e.g. ``git show <old-sha>:data/dl07_templates.h5
# > /tmp/dl07_old_v1.h5``). Otherwise this module falls back to a small set of
# embedded reference values (3 (q_PAH, U_min) nodes x 20 wavelength indices),
# pinned from that same old file, so CI can run without the extra artifact.

_WAVE_IDX = [
    0,
    52,
    105,
    157,
    210,
    263,
    315,
    368,
    421,
    473,
    526,
    578,
    631,
    684,
    736,
    789,
    842,
    894,
    947,
    1000,
]

# Retained nodes (U_min <= 8.0, present at the same axis position in both the
# old and new files) -- (q_pah, u_min, single_u[_WAVE_IDX], powerlaw[_WAVE_IDX]).
_OLD_RETAINED_REFERENCE = [
    (
        0.47,
        0.10,
        [
            1.715997648535117e-08,
            2.0962690496921584e-08,
            2.0000709944183134e-08,
            1.7920181480000306e-08,
            1.5014134611984956e-07,
            5.919177283503138e-07,
            1.6188025103128707e-07,
            1.5027869958946827e-07,
            1.5034548238513332e-07,
            2.1442597175795628e-07,
            4.714315750729857e-07,
            3.4491119132944533e-07,
            9.215990841312933e-08,
            1.3251726295700557e-08,
            1.478373695554186e-09,
            1.3260714811667815e-10,
            1.1138179748662842e-11,
            9.903851971802167e-13,
            8.717044630577996e-14,
            7.989498957986754e-15,
        ],
        [
            1.829052856799271e-08,
            2.213463900605732e-08,
            2.400034566606105e-08,
            8.929244867045944e-08,
            6.715939008779038e-07,
            1.6325194038267169e-06,
            1.0463836232541347e-06,
            1.6730807977912677e-06,
            8.997010929595509e-07,
            4.295239742795626e-07,
            2.096971266786628e-07,
            6.587581231743953e-08,
            1.128288497223493e-08,
            1.2994915620542558e-09,
            1.2974836069080136e-10,
            1.0946605486175757e-11,
            8.892629012131345e-13,
            7.755379931662196e-14,
            6.7527085524915796e-15,
            6.145951164716092e-16,
        ],
    ),
    (
        2.50,
        5.00,
        [
            9.111919451383098e-08,
            1.1725769821952358e-07,
            1.2241963334560397e-07,
            1.1698182739853062e-07,
            8.524913054206048e-07,
            2.2715698925387723e-06,
            2.8310056812022826e-07,
            1.995317504504687e-07,
            6.059039986384051e-07,
            8.450649584369604e-07,
            4.4720144400979025e-07,
            9.770036527082863e-08,
            1.1212793633360733e-08,
            9.774798773308597e-10,
            8.299934079454669e-11,
            6.340266204775253e-12,
            4.84500369568833e-13,
            4.0787602825415027e-14,
            3.4848216882132756e-15,
            3.14714100045057e-16,
        ],
        [
            9.511339900311518e-08,
            1.2585287244863324e-07,
            1.3961075220732986e-07,
            2.0766817007637978e-07,
            1.4835547083976418e-06,
            3.3863411625538343e-06,
            1.334990033188992e-06,
            1.9429936866033585e-06,
            9.932707728213748e-07,
            3.6140769337218167e-07,
            9.375466410998193e-08,
            1.4455942384990363e-08,
            1.3794322426552255e-09,
            1.0952175131782079e-10,
            8.856835799904315e-12,
            6.579323856660137e-13,
            4.946917839159642e-14,
            4.126557859716427e-15,
            3.5067567621631446e-16,
            3.156923983973114e-17,
        ],
    ),
    (
        4.58,
        8.00,
        [
            1.7852734317964737e-07,
            2.3062361896720708e-07,
            2.3949465144936534e-07,
            2.2659989331962152e-07,
            1.6029509911076847e-06,
            4.174342307863922e-06,
            4.743322499974187e-07,
            2.724342721860646e-07,
            6.440758120436451e-07,
            7.631035733319618e-07,
            3.5802096713093455e-07,
            7.158796876697842e-08,
            7.7572856538648e-09,
            6.565508242576405e-10,
            5.512068860406424e-11,
            4.18577014918872e-12,
            3.1863481476415457e-13,
            2.6794997633091274e-14,
            2.2914085910381845e-15,
            2.073902239273021e-16,
        ],
        [
            1.8665055412734001e-07,
            2.4774468198499117e-07,
            2.7099490301789497e-07,
            3.181942208801836e-07,
            2.2738220516080115e-06,
            5.2453908812172115e-06,
            1.432226281462026e-06,
            1.7478247642033684e-06,
            8.807880256939336e-07,
            3.083805069633859e-07,
            7.4363532772259e-08,
            1.0720827668332108e-08,
            9.760399429912898e-10,
            7.551634919776745e-11,
            6.049696793753074e-12,
            4.471711794409469e-13,
            3.3516830360155136e-14,
            2.7935055758378416e-15,
            2.3764835110482763e-16,
            2.1444558709047545e-17,
        ],
    ),
]

# Shifted slot: OLD file's node labeled U_min=10.0 (index 18) at q_pah=2.5.
# Compared against the NEW file's U_min=12.0 node -- proving the old "10.0"
# label was a mislabeled read of the true U_min=12.0 templates (12<-10).
_OLD_SHIFTED_10_LABEL_QPAH = 2.50
_OLD_SHIFTED_10_LABEL_SINGLE_U = [
    9.111289507029276e-08,
    1.1731064852412172e-07,
    1.2242275729924052e-07,
    1.1703709406675116e-07,
    8.528974852211462e-07,
    2.281690936861542e-06,
    2.980127521223982e-07,
    2.690877455768167e-07,
    9.116726142685308e-07,
    8.940517320828064e-07,
    3.501465842383067e-07,
    6.263019456638413e-08,
    6.33941064870899e-09,
    5.153450316701485e-10,
    4.210883443094746e-11,
    3.145584533727475e-12,
    2.372961596919351e-13,
    1.983085457460762e-14,
    1.6871524015805909e-15,
    1.5195472960736326e-16,
]
_OLD_SHIFTED_10_LABEL_POWERLAW = [
    9.544284053128215e-08,
    1.2654914177258583e-07,
    1.4092827565554322e-07,
    2.1463540777455584e-07,
    1.5324819492561286e-06,
    3.4720943390761276e-06,
    1.4156426458786976e-06,
    2.075566275975359e-06,
    1.0127483160646758e-06,
    3.2141779668614145e-07,
    6.974953038502374e-08,
    9.338506028317648e-09,
    8.109474713177008e-10,
    6.093148485906284e-11,
    4.7800854724709125e-12,
    3.486385299779206e-13,
    2.5945915828831032e-14,
    2.150839529091014e-15,
    1.8212789919824065e-16,
    1.6360210756810585e-17,
]


def _load_old_v1_grid():
    """Return the pre-#2535 v1 grid arrays, from an env var or None."""
    import os

    import h5py

    old_path = os.environ.get("TENGRI_DL07_OLD_V1_PATH")
    if not old_path:
        return None
    with h5py.File(old_path, "r") as f:
        return {
            "umin_grid": np.asarray(f["umin_grid"]),
            "qpah_grid": np.asarray(f["qpah_grid"]),
            "single_u": np.asarray(f["single_u"]),
            "powerlaw": np.asarray(f["powerlaw"]),
        }


class TestDL07DataUnchangedAtRetainedNodes:
    """Regenerated data must equal the pre-#2535 grid at every unmoved node.

    Proves the 2026-09-29 regeneration is an axis-label + q_PAH-subset
    correction ONLY: the numeric ``single_u``/``powerlaw`` arrays for the 18
    retained U_min nodes (0.1-8.0) x 7 MW3.1 q_PAH nodes are untouched.
    """

    def test_retained_nodes_match_old_file(self, dl07_v1_path):
        import h5py

        with h5py.File(dl07_v1_path, "r") as f:
            new_umin = np.asarray(f["umin_grid"])
            new_qpah = np.asarray(f["qpah_grid"])
            new_single_u = np.asarray(f["single_u"])
            new_powerlaw = np.asarray(f["powerlaw"])

        old = _load_old_v1_grid()

        for qpah, umin, old_su_ref, old_pw_ref in _OLD_RETAINED_REFERENCE:
            iq = int(np.argmin(np.abs(new_qpah - qpah)))
            iu = int(np.argmin(np.abs(new_umin - umin)))
            new_su = new_single_u[iq, iu, _WAVE_IDX]
            new_pw = new_powerlaw[iq, iu, _WAVE_IDX]
            np.testing.assert_allclose(
                new_su,
                old_su_ref,
                rtol=1e-6,
                err_msg=f"single_u changed at retained node q_pah={qpah}, umin={umin}",
            )
            np.testing.assert_allclose(
                new_pw,
                old_pw_ref,
                rtol=1e-6,
                err_msg=f"powerlaw changed at retained node q_pah={qpah}, umin={umin}",
            )

        if old is not None:
            # Full sweep against the actual old file, every retained node.
            mw_idx_old = np.array([int(np.argmin(np.abs(old["qpah_grid"] - v))) for v in new_qpah])
            for iu in range(18):  # U_min 0.1 .. 8.0 occupy indices 0-17 in both files
                np.testing.assert_allclose(
                    new_single_u[:, iu, :],
                    old["single_u"][mw_idx_old, iu, :],
                    rtol=1e-6,
                    err_msg=f"single_u changed at retained U_min index {iu}",
                )
                np.testing.assert_allclose(
                    new_powerlaw[:, iu, :],
                    old["powerlaw"][mw_idx_old, iu, :],
                    rtol=1e-6,
                    err_msg=f"powerlaw changed at retained U_min index {iu}",
                )


class TestDL07ShiftedSlotsAreRelabeled:
    """The 4 shifted U_min slots are a pure relabel, not new data.

    12<-10, 15<-12, 20<-15, 25<-20: the OLD file's data at the label it
    (wrongly) called ``10.0`` equals the NEW file's data at the correct
    label ``12.0`` (and so on for the other three), proving the old file's
    top axis nodes held genuine DL07 templates, shifted one slot from their
    true labels -- not zeros, not a different physical model.
    """

    def test_old_10_label_equals_new_12_label(self, dl07_v1_path):
        import h5py

        with h5py.File(dl07_v1_path, "r") as f:
            new_umin = np.asarray(f["umin_grid"])
            new_qpah = np.asarray(f["qpah_grid"])
            new_single_u = np.asarray(f["single_u"])
            new_powerlaw = np.asarray(f["powerlaw"])

        iq = int(np.argmin(np.abs(new_qpah - _OLD_SHIFTED_10_LABEL_QPAH)))
        iu_new_12 = int(np.argmin(np.abs(new_umin - 12.0)))

        np.testing.assert_allclose(
            new_single_u[iq, iu_new_12, _WAVE_IDX],
            _OLD_SHIFTED_10_LABEL_SINGLE_U,
            rtol=1e-6,
            err_msg="new U_min=12.0 single_u does not equal old U_min='10.0' data",
        )
        np.testing.assert_allclose(
            new_powerlaw[iq, iu_new_12, _WAVE_IDX],
            _OLD_SHIFTED_10_LABEL_POWERLAW,
            rtol=1e-6,
            err_msg="new U_min=12.0 powerlaw does not equal old U_min='10.0' data",
        )

    def test_full_shift_mapping_against_old_file(self, dl07_v1_path):
        """Full 12<-10, 15<-12, 20<-15, 25<-20 sweep when the old file is available."""
        import h5py

        old = _load_old_v1_grid()
        if old is None:
            pytest.skip("TENGRI_DL07_OLD_V1_PATH not set")

        with h5py.File(dl07_v1_path, "r") as f:
            new_umin = np.asarray(f["umin_grid"])
            new_qpah = np.asarray(f["qpah_grid"])
            new_single_u = np.asarray(f["single_u"])
            new_powerlaw = np.asarray(f["powerlaw"])

        mw_idx_old = np.array([int(np.argmin(np.abs(old["qpah_grid"] - v))) for v in new_qpah])
        # old index 18-21 (labeled 10,12,15,20) <-> new index 18-21 (labeled 12,15,20,25)
        for offset, (old_label, new_label) in enumerate(
            [(10.0, 12.0), (12.0, 15.0), (15.0, 20.0), (20.0, 25.0)]
        ):
            iu_old = 18 + offset
            iu_new = int(np.argmin(np.abs(new_umin - new_label)))
            assert abs(old["umin_grid"][iu_old] - old_label) < 1e-9
            assert abs(new_umin[iu_new] - new_label) < 1e-9
            np.testing.assert_allclose(
                new_single_u[:, iu_new, :],
                old["single_u"][mw_idx_old, iu_old, :],
                rtol=1e-6,
                err_msg=f"new {new_label} single_u != old {old_label} data",
            )
            np.testing.assert_allclose(
                new_powerlaw[:, iu_new, :],
                old["powerlaw"][mw_idx_old, iu_old, :],
                rtol=1e-6,
                err_msg=f"new {new_label} powerlaw != old {old_label} data",
            )


class TestDL07ShapeClosure:
    """Physics closure checks for the corrected U_min axis.

    At fixed (q_PAH, gamma), the far-IR color F(100um)/F(500um) rises
    monotonically with U_min (warmer radiation field -> bluer IR SED); DL07
    Table 3 values (Draine & Li 2007, ApJ, 657, 810) place this ratio near
    30.3/34.3/40.0/44.7 at U_min=12/15/20/25.
    """

    def test_far_ir_color_tracks_umin(self):
        from tengri.components.dust.emission import draine_li2007 as dl07_fn

        wave_aa = np.array([100.0, 500.0]) * 1e4
        expected = {12.0: 30.3, 15.0: 34.3, 20.0: 40.0, 25.0: 44.7}
        ratios = {}
        for umin, exp in expected.items():
            sed = np.asarray(
                dl07_fn(wave_aa, 1.0, dust_umin=umin, dust_gamma_dl=0.01, dust_qpah=2.5)
            )
            ratio = float(sed[0] / sed[1])
            ratios[umin] = ratio
            assert abs(ratio - exp) / exp < 0.02, (
                f"F(100um)/F(500um) at U_min={umin} is {ratio:.2f}, expected ~{exp} (2%)"
            )

        assert ratios[25.0] != ratios[20.0], "requesting U_min=25 must differ from U_min=20"
        # Monotonic in U_min (warmer field -> hotter dust -> bluer FIR color)
        ordered = [ratios[u] for u in sorted(ratios)]
        assert ordered == sorted(ordered), f"F100/F500 not monotonic in U_min: {ratios}"
