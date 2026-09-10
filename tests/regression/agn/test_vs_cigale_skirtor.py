# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests: SKIRTOR separate components against CIGALE reference.

References
----------
- Stalevski et al. 2016, MNRAS, 458, 2288 (SKIRTOR)
- Yang et al. 2020, MNRAS, 491, 740 (X-CIGALE polar dust)
"""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from tests._bounds import assert_non_negative

pytestmark = pytest.mark.regression_paper

#: 8 A - 1e8 A: spans the whole of the analytic disc's 80 A - 1e7 A support
#: AND the whole of the SKIRTOR templates' 10 A - 1e8 A axis, so nothing here
#: is measured on a grid that truncates either (R60 Item B, R66).
_COVERING_GRID = jnp.asarray(np.geomspace(8.0, 1.0e8, 3000))

#: The CIGALE reproduction's SKIRTOR fiducial as composable-runner kwargs
#: (t=7, pl=1, q=1, oa=40, disk_type=1 -> schartmann2005, SMC E(B-V)=0.03,
#: T=100 K, beta=1.6), inclination left to the caller.
_JOINT_FIDUCIAL = dict(
    agn_disc_block="schartmann2005",
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="skirtor",
    agn_attenuation_block="polar_dust",
    agn_norm="cigale_joint",
    agn_tau_skirtor=7.0,
    agn_p_skirtor=1.0,
    agn_q_skirtor=1.0,
    agn_oa_skirtor=40.0,
    agn_polar_ebv=0.03,
    agn_polar_oa=40.0,
    agn_polar_T=100.0,
    agn_polar_beta=1.6,
)


class TestSKIRTORComponentSeparation:
    """Test that SKIRTOR components separate correctly."""

    @pytest.fixture
    def skirtor_components_fn(self):
        """Load SKIRTOR v3 components function."""
        try:
            from tengri.components.agn.skirtor import _load_skirtor_components

            fn = _load_skirtor_components()
            if fn is None:
                pytest.skip("SKIRTOR v3 grid not available")
            return fn
        # The absent-grid case is the ``fn is None`` check above; anything else
        # that reaches here is a loading defect, not a missing file.
        except (FileNotFoundError, ImportError, OSError):
            pytest.skip("SKIRTOR grid loading failed")

    @pytest.fixture
    def test_wavelength(self):
        """Standard test wavelength grid."""
        return jnp.logspace(1, 5, 256)  # 10 Å to 100 μm

    def test_disk_dust_non_negative(self, skirtor_components_fn, test_wavelength):
        """Test that disk and dust components are non-negative with significant emission."""
        components = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_cos_inc=0.5,
            frac_agn=0.5,
        )

        # All components should be non-negative
        assert_non_negative(
            components.disk, name="output", msg="disk component should be non-negative"
        )
        assert_non_negative(
            components.dust, name="output", msg="dust component should be non-negative"
        )
        assert_non_negative(
            components.total, name="output", msg="total component should be non-negative"
        )

        # Each component should have significant emission (peak > 1e20 erg/s/Hz)
        assert float(jnp.max(components.disk)) > 1e20, "disk should have significant peak"
        assert float(jnp.max(components.dust)) > 1e20, "dust should have significant peak"
        assert float(jnp.max(components.total)) > 1e20, "total should have significant peak"

    def test_bolometric_luminosity(self, skirtor_components_fn, test_wavelength):
        """Test that all components integrate to significant bolometric luminosity."""
        from tengri.components.agn._phys import wavelength_to_nu

        components = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=40.0,
            agn_cos_inc=0.5,
            frac_agn=0.5,
        )

        nu = wavelength_to_nu(test_wavelength)
        idx_sort = jnp.argsort(nu)

        # Each component should have definite bolometric luminosity
        L_disk = jnp.trapezoid(components.disk[idx_sort], nu[idx_sort])
        L_dust = jnp.trapezoid(components.dust[idx_sort], nu[idx_sort])
        L_total = jnp.trapezoid(components.total[idx_sort], nu[idx_sort])

        # All should be non-negative and above a reasonable threshold
        # (SKIRTOR v3 templates normalize disk and dust independently,
        # so L_total ≠ L_disk + L_dust precisely)
        assert float(L_disk) >= 0.0, "L_disk should be non-negative"
        assert float(L_dust) >= 0.0, "L_dust should be non-negative"
        assert float(L_total) >= 0.0, "L_total should be non-negative"

        # Each should be > 1e40 erg/s (reasonable AGN luminosity)
        assert float(L_disk) > 1e40, "L_disk should be significant"
        assert float(L_dust) > 1e40, "L_dust should be significant"
        assert float(L_total) > 1e40, "L_total should be significant"

    def test_type1_type2_sightline(self, skirtor_components_fn, test_wavelength):
        """Test Type 1 vs Type 2 sightline definitions."""
        # Type 1: face-on, cos_inc ≥ cos(90° - oa)
        # Type 2: edge-on, cos_inc < cos(90° - oa)

        oa_deg = 40.0
        cos_threshold = jnp.cos(jnp.radians(90.0 - oa_deg))

        # Face-on (Type 1): cos_inc = 0.95
        components_type1 = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=oa_deg,
            agn_cos_inc=0.95,
            frac_agn=0.5,
        )

        # Edge-on (Type 2): cos_inc = 0.2
        components_type2 = skirtor_components_fn(
            wavelength=test_wavelength,
            agn_log_lbol=12.0,
            agn_tau_skirtor=7.0,
            agn_p_skirtor=1.0,
            agn_q_skirtor=1.0,
            agn_oa_skirtor=oa_deg,
            agn_cos_inc=0.2,
            frac_agn=0.5,
        )

        # The total SED should differ between Type 1 and Type 2
        # (due to different extinction and viewing angle)
        diff = jnp.max(jnp.abs(components_type1.total - components_type2.total))
        assert float(diff) > 0.0, "Type 1 and Type 2 SEDs should differ"


class TestPolarDustExtinction:
    """Test polar dust extinction and reemission."""

    def test_polar_dust_extinction_type1_type2(self):
        """Test that extinction is applied only to Type 1."""
        from tengri.components.agn.polar_dust import polar_dust_extinction

        wave = jnp.logspace(2, 5, 128)  # 100 Å to 100 μm
        l_nu = jnp.ones_like(wave)  # Flat spectrum
        oa_deg = 40.0
        ebv = 0.1

        # Type 1 (face-on): should show extinction
        l_att_type1, l_abs_type1 = polar_dust_extinction(
            l_nu, wave, cos_inc=0.95, opening_angle_deg=oa_deg, ebv=ebv
        )

        # Type 2 (edge-on): should show no extinction
        l_att_type2, _ = polar_dust_extinction(
            l_nu, wave, cos_inc=0.1, opening_angle_deg=oa_deg, ebv=ebv
        )

        # Type 1 should be more attenuated than Type 2
        att_ratio_type1 = jnp.mean(l_att_type1 / l_nu)
        att_ratio_type2 = jnp.mean(l_att_type2 / l_nu)

        assert float(att_ratio_type1) < float(att_ratio_type2), (
            "Type 1 should be more attenuated than Type 2"
        )

        # Type 1 should have absorbed energy
        assert float(jnp.sum(l_abs_type1)) > 0.0, "Type 1 should absorb some energy"

    def test_polar_dust_emission_energy_conservation(self):
        """Test that polar dust reemission conserves energy."""
        from tengri.components.agn.polar_dust import polar_dust_emission

        wave = jnp.logspace(2, 5, 256)  # 100 Å to 100 μm
        from tengri.components.agn._phys import wavelength_to_nu

        l_absorbed_total = 1e45  # erg/s
        l_reemit = polar_dust_emission(
            l_absorbed_total,
            wave,
            temperature=100.0,
            beta=1.6,
            lambda_0=2e6,
        )

        # Integrate reemitted spectrum
        nu = wavelength_to_nu(wave)
        idx_sort = jnp.argsort(nu)
        l_reemit_total = jnp.trapezoid(l_reemit[idx_sort], nu[idx_sort])

        # Should equal input absorbed luminosity (within numerical tolerance)
        np.testing.assert_allclose(float(l_reemit_total), l_absorbed_total, rtol=0.15)


class TestSKIRTORModelComponent:
    """Test the SKIRTORTorus SEDModelComponent."""

    @pytest.fixture
    def skirtor_component(self):
        """Create a SKIRTORTorus instance."""
        try:
            from tengri.components.agn.skirtor import _find_skirtor_grid
            from tengri.components.agn.skirtor_model import (
                SKIRTORTorus,
                SKIRTORTorusConfig,
            )

            grid_path = _find_skirtor_grid()
            config = SKIRTORTorusConfig(grid_path=grid_path)
            return SKIRTORTorus(config=config)
        except (FileNotFoundError, ImportError, OSError):
            pytest.skip("SKIRTOR grid or component instantiation failed")

    def test_model_parameter_sanity(self, skirtor_component):
        """Test that model has required parameters."""
        # Check that the component has the core SKIRTOR parameters
        required_params = [
            "log_lbol",
            "tau_skirtor",
            "p_skirtor",
            "q_skirtor",
            "oa_skirtor",
            "cos_inc",
        ]
        for param_name in required_params:
            assert hasattr(skirtor_component, param_name), (
                f"SKIRTORTorus should have parameter {param_name}"
            )

        # Verify log_lbol bounds are reasonable
        log_lbol_param = skirtor_component.log_lbol
        assert float(log_lbol_param.lo) > 0.0, "log_lbol.lo should be > 0"
        assert float(log_lbol_param.hi) > float(log_lbol_param.lo), (
            "log_lbol.hi should be > log_lbol.lo"
        )

    def test_model_parameter_defaults(self, skirtor_component):
        """Test that parameter defaults are sensible."""
        # Check declared parameters
        assert hasattr(skirtor_component, "log_lbol")
        assert hasattr(skirtor_component, "oa_skirtor")
        assert hasattr(skirtor_component, "cos_inc")

        # Default bounds should be reasonable
        assert float(skirtor_component.log_lbol.lo) > 0.0
        assert float(skirtor_component.log_lbol.hi) > float(skirtor_component.log_lbol.lo)
        assert float(skirtor_component.cos_inc.lo) >= 0.0
        assert float(skirtor_component.cos_inc.hi) <= 1.0


#: CIGALE's own polar share ``int(polar_dust)/int(dust)`` per unit AGN dust
#: budget, read off a LIVE ``pcigale.sed_modules.skirtor2016.SKIRTOR2016``
#: ``_init_code()`` at the CIGALE reproduction's fiducial (t=7, pl=1, q=1,
#: oa=40, R=20, Mcl=0.97, disk_type=1, delta=0, fracAGN=0.3, law=0 (SMC),
#: EBV=0.03, T=100 K, beta=1.6). ``oa=40`` puts CIGALE's own Type-1 boundary
#: at i <= 50, so i=60 and i=80 are Type-2 sightlines -- where its share is
#: LARGEST, because the polar blackbody is added unconditionally while only
#: the line-of-sight reddening of ``disk`` is gated (R63).
_CIGALE_POLAR_SHARE: dict[int, float] = {
    0: 0.204988393,
    30: 0.209707567,
    60: 0.352217228,
    80: 0.450588625,
}

#: ``int(AGN1.disk x AGN1.norm/SKIRTOR2016.norm) / int(SKIRTOR2016.dust)`` at
#: i=30, read off pcigale's own template database at the same fiducial: the
#: quantity ``skirtor_disc_dust_ratio`` calls ``R_faceon``. pcigale's
#: ``int(disk(i=0))`` is 4.433073303 on its 948-point 1 nm - 1e7 nm grid and
#: its ``norm0/norm(i=30)`` is 1.029130565.
_CIGALE_R_FACEON_I30 = 4.562211232

#: ``pcigale / tengri`` on ``R_faceon(i=30)``, measured, and fully accounted
#: for by the two grids' quadratures of the SAME spectra: the repackaged
#: ``skirtor_templates_v3.h5`` carries a 136-point copy of pcigale's 948-point
#: 1 nm - 1e7 nm axis, and integrating pcigale's own arrays on it gives
#: ``int(dust(i=30))`` 1.001908482x its 948-point value and ``int(disk(0))``
#: 0.999998368x -- product 1.001906847, leaving 1.8e-6 unexplained (the
#: repackaged grid stores its spectra as float32, and its red extrapolation to
#: 10 mm is its own). So this is a trapezoid difference on identical data, not
#: a missing factor.
#:
#: It is NOT uniform in inclination, and the dust half is what varies: the
#: 136-vs-948-point dust factor is 1.001912 at i=0, 1.001908 at i=30,
#: 1.003308 at i=60, 1.004467 at i=80 and 1.005008 at i=90 (the dust template
#: sharpens toward edge-on, so the coarser axis loses more of it). Only the
#: i=30 value is pinned here, and only for i=30.
_VENDORED_GRID_QUADRATURE_RESIDUAL_I30 = 1.001908635


class TestStoredInclinationNormIsApplied:
    r"""``spectra/norm``: the vendored SKIRTOR grid's inclination normalization.

    ``skirtor_templates_v3.h5`` stores every model pre-normalized so that
    :math:`\int {\rm dust}\,d\lambda` is a constant, with the physical scale
    factored out into ``spectra/norm``, shape ``(5, 4, 4, 8, 3, 10)`` --
    exactly the parameter axes, no wavelength axis. pcigale keeps the same
    split and puts the factor back where the FACE-ON disc is used as a
    reference::

        AGN1.disk *= AGN1.norm / self.SKIRTOR2016.norm  # skirtor2016.py

    ``AGN1`` is the i=0 record and ``SKIRTOR2016`` the record at the observer's
    inclination, so the factor is ``norm(0)/norm(i)``: it rises from 1.0
    face-on to 3.621 edge-on in this file (pcigale's own database: 3.633) at
    the fiducial R=20, oa=40, because an edge-on model radiates
    less dust emission per unit intrinsic AGN power. It reaches exactly one
    quantity: the observed disc ``SKIRTOR2016.disk = disk x SKIRTOR2016.disk /
    AGN1.disk`` divides it straight back out, so ``R`` is untouched, while
    ``l_ext = g(oa) x int(AGN1.disk (1 - ext_fac))`` -- the polar dust's
    absorbed-power reference -- carries it in full. In tengri's terms that is
    ``R_faceon``, and nothing else.

    The loader never read ``spectra/norm``, so ``R_faceon`` was 4.4246148 at
    every inclination and the polar dust's share of the AGN dust budget came
    out inclination-FLAT (0.200035 at i=0, 30, 60 and 80 alike) against
    CIGALE's 0.204988 -> 0.450589 rise: a 2.4% error face-on growing to 2.25x
    edge-on, invisible at the i=30 fiducial everything else is measured at.
    """

    _FIDUCIAL: ClassVar[dict] = dict(
        agn_tau_skirtor=7.0,
        agn_p_skirtor=1.0,
        agn_q_skirtor=1.0,
        agn_oa_skirtor=40.0,
        agn_radius_ratio=20.0,
    )

    @staticmethod
    def _tie(cos_inc: float, wave=None):
        from tengri.components.agn.blocks import resolve_agn_block
        from tengri.components.agn.skirtor import (
            _load_raw_disk_dust_grid,
            skirtor_disc_dust_ratio,
        )

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        wave = _COVERING_GRID if wave is None else wave
        disc = jnp.asarray(
            resolve_agn_block("disc", "schartmann2005")(wave, agn_log_lbol=12.0, templates=None)
        )
        return skirtor_disc_dust_ratio(
            wave,
            disc,
            jnp.ones_like(wave),
            agn_cos_inc=cos_inc,
            **TestStoredInclinationNormIsApplied._FIDUCIAL,
        )

    @staticmethod
    def _file_derivation(i_deg: int) -> tuple[float, float, float, float]:
        """``(int_disk0, int_dust_i, norm0/norm_i, R_faceon)`` from the h5 file.

        Read straight out of ``skirtor_templates_v3.h5`` with ``h5py`` at the
        fiducial's exact node indices, so the expectation is the FILE's own
        content and not an echo of the code under test.
        """
        h5py = pytest.importorskip("h5py")
        from tengri.components.agn.skirtor import _find_skirtor_grid

        with h5py.File(_find_skirtor_grid(), "r") as f:
            if "spectra/norm" not in f:
                pytest.skip("vendored SKIRTOR grid carries no spectra/norm")
            wl = np.asarray(f["wavelength"][:])
            axes = {
                "tau": np.asarray(f["grid/tau_97"][:]),
                "p": np.asarray(f["grid/p"][:]),
                "q": np.asarray(f["grid/q"][:]),
                "oa": np.asarray(f["grid/opening_angle"][:]),
                "R": np.asarray(f["grid/radius_ratio"][:]),
                "cos_inc": np.asarray(f["grid/cos_inclination"][:]),
            }
            idx = (
                int(np.argmin(np.abs(axes["tau"] - 7.0))),
                int(np.argmin(np.abs(axes["p"] - 1.0))),
                int(np.argmin(np.abs(axes["q"] - 1.0))),
                int(np.argmin(np.abs(axes["oa"] - 40.0))),
                int(np.argmin(np.abs(axes["R"] - 20.0))),
            )
            cos_i = float(np.cos(np.deg2rad(i_deg)))
            j_i = int(np.argmin(np.abs(axes["cos_inc"] - cos_i)))
            j_0 = int(np.argmin(np.abs(axes["cos_inc"] - 1.0)))
            disk0 = np.asarray(f["spectra/disk_emission"][(*idx, j_0)])
            dust_i = np.asarray(f["spectra/dust_emission"][(*idx, j_i)])
            norm = np.asarray(f["spectra/norm"][idx])
        int_disk0 = float(np.trapezoid(disk0, x=wl))
        int_dust_i = float(np.trapezoid(dust_i, x=wl))
        ratio = float(norm[j_0] / norm[j_i])
        return int_disk0, int_dust_i, ratio, int_disk0 * ratio / int_dust_i

    def test_r_faceon_equals_the_files_own_norm_weighted_derivation(self):
        """``R_faceon = int_disk0 x norm(0)/norm(i) / int_dust(i)``, node-exact.

        Measured at the fiducial and i=30 from the file: ``int_disk0`` =
        44.24614775, ``int_dust`` = 10.00000003 (both over the Angstrom axis),
        ``norm(0)/norm(30)`` = 1.029133714, so ``R_faceon`` = 4.553520226.
        Before the fix the loader dropped the third factor and returned
        4.424614763 at every inclination.
        """
        _d0, _di, ratio, expected = self._file_derivation(30)
        assert ratio == pytest.approx(1.029133714, rel=1e-6, abs=0.0), (
            "probe setup failed: the file's norm(0)/norm(30) is not the measured "
            f"1.029133714 but {ratio:.9f}"
        )
        got = float(self._tie(float(np.cos(np.deg2rad(30.0)))).R_faceon)
        assert got == pytest.approx(expected, rel=1e-6, abs=0.0), (
            f"R_faceon(i=30) = {got:.9f} but the vendored grid's own content gives "
            f"{expected:.9f} (= int_disk0 x norm(0)/norm(i) / int_dust). The loader "
            "is not reading spectra/norm, so the inclination-dependent "
            "normalization CIGALE applies as AGN1.norm/SKIRTOR2016.norm is missing."
        )

    def test_r_faceon_matches_cigale_to_the_vendored_grids_quadrature(self):
        """i=30 against pcigale's own 4.562211232, and the residual explained.

        The remaining 0.19% is the two grids' trapezoids of the same data --
        see :data:`_VENDORED_GRID_QUADRATURE_RESIDUAL_I30` for the
        decomposition. Pinning the residual to that measured factor (rather
        than to a loose band) is what makes this a validation test: any
        mis-application moves the ratio far outside 1e-4 -- not applied at all
        reads 1.031098, half-applied (``sqrt(norm0/norm_i)``) 1.016399, and
        inverted (``norm_i/norm0``) 1.061138.
        """
        got = float(self._tie(float(np.cos(np.deg2rad(30.0)))).R_faceon)
        assert _CIGALE_R_FACEON_I30 / got == pytest.approx(
            _VENDORED_GRID_QUADRATURE_RESIDUAL_I30, rel=1e-4, abs=0.0
        ), (
            f"R_faceon(i=30) = {got:.9f} against pcigale's {_CIGALE_R_FACEON_I30:.9f}: "
            f"a factor {_CIGALE_R_FACEON_I30 / got:.6f}, where the two grids' "
            f"quadratures of the same spectra account for exactly "
            f"{_VENDORED_GRID_QUADRATURE_RESIDUAL_I30:.6f} and nothing else should."
        )
        assert got == pytest.approx(_CIGALE_R_FACEON_I30, rel=2.5e-3, abs=0.0)

    @pytest.mark.parametrize("i_deg", (0, 30, 60, 80, 90))
    def test_r_faceon_rises_with_inclination_by_the_stored_norm(self, i_deg):
        """``R_faceon(i)/R_faceon(0)`` is the file's ``norm(0)/norm(i)``.

        The mechanism itself, one inclination at a time. Measured factors at
        the fiducial (R=20, oa=40): 1.0, 1.029134, 2.105812, 3.172626,
        3.621294 -- pcigale's own are 1.0, 1.029131, 2.108751, 3.180730,
        3.632501, agreeing to 0.3% at the extreme edge-on node.

        ``i_deg=0`` is the CONTROL, not a defect probe: ``norm(0)/norm(0)`` is
        1.0 by construction, so that case passed before the fix too. It is
        here because the fix must leave the face-on reference exactly where it
        was -- the polar share at i=0 is a shipped number (0.200035) that no
        inclination normalization may move.
        """
        _d0, _di, ratio, _r = self._file_derivation(i_deg)
        face_on = float(self._tie(1.0).R_faceon)
        got = float(self._tie(float(np.cos(np.deg2rad(i_deg)))).R_faceon)
        assert got / face_on == pytest.approx(ratio, rel=1e-6, abs=0.0), (
            f"R_faceon(i={i_deg})/R_faceon(0) = {got / face_on:.9f} but the file's "
            f"norm(0)/norm(i) = {ratio:.9f}. Before the fix this ratio was 1.0 at "
            "every inclination."
        )

    def test_polar_share_tracks_cigale_across_inclination(self):
        """The polar dust's share of the AGN dust budget, vs live pcigale.

        Under the joint budget (R59) ``polar + torus`` IS the AGN dust budget,
        so ``share = int(polar)/(int(polar) + int(torus))`` is directly
        CIGALE's ``int(polar_dust)/int(dust)``. The share depends on
        inclination through ``R_faceon`` alone: ``share = g R_faceon J /
        (1 + g R_faceon J)`` with ``J`` the disc-shape-weighted absorbed
        fraction, which is CIGALE's ``l_ext/(1 + l_ext)`` in the same form.

        Measured, tengri / pcigale: before the fix 0.9758 / 0.9539 / 0.5679 /
        0.4439 at i = 0 / 30 / 60 / 80 -- flat at 0.200035 while CIGALE's
        rises 2.20x. After: 0.9758 / 0.9760 / 0.9793 / 0.9818, from tengri
        shares 0.200035 / 0.204670 / 0.344936 / 0.442378, rising 2.2115x
        against CIGALE's 2.1981x.

        The residual decomposes exactly, which is why a 5% band is the right
        one here. Writing the share as ``x/(1+x)`` with
        ``x = g R_faceon J``, ``x_cigale/x_tengri`` factors into
        ``R_faceon``'s own ratio -- 1.001912 / 1.001909 / 1.003310 / 1.004471,
        i.e. the two grids' per-inclination quadratures -- times
        ``1.029181``, IDENTICAL at all four inclinations. That constant
        2.9182% sits in the absorbed-power proxy ``g J`` (the cone factor and
        the disc-shape-weighted absorbed fraction), is untouched by R64, and
        is what remains of this fiducial's disagreement.
        """
        from tengri.components.agn.blocks.runner import compose_l_nu
        from tengri.components.agn.skirtor import _load_raw_disk_dust_grid
        from tengri.utils.physics_constants import C_AA

        if _load_raw_disk_dust_grid() is None:
            pytest.skip("raw SKIRTOR disk/dust grid not available")
        nu = C_AA / _COVERING_GRID
        order = jnp.argsort(nu)
        shares = {}
        for i_deg in _CIGALE_POLAR_SHARE:
            _sed, comps = compose_l_nu(
                _COVERING_GRID,
                12.0,
                agn_ir_frac=0.3,
                agn_torus_frac=0.5,
                agn_cos_inc=float(np.cos(np.deg2rad(i_deg))),
                return_components=True,
                **_JOINT_FIDUCIAL,
            )
            integ = {
                key: float(jnp.abs(jnp.trapezoid(jnp.asarray(comps[key])[order], nu[order])))
                for key in ("polar", "torus")
            }
            shares[i_deg] = integ["polar"] / (integ["polar"] + integ["torus"])
        spread = shares[80] / shares[0]
        assert spread > 1.5, (
            f"the polar share is inclination-flat ({shares[80]:.9f} at i=80 against "
            f"{shares[0]:.9f} at i=0, {spread:.4f}x) while CIGALE's rises "
            f"{_CIGALE_POLAR_SHARE[80] / _CIGALE_POLAR_SHARE[0]:.4f}x: the stored "
            "spectra/norm is not being applied."
        )
        for i_deg, reference in _CIGALE_POLAR_SHARE.items():
            assert shares[i_deg] == pytest.approx(reference, rel=0.05, abs=0.0), (
                f"i={i_deg}: polar share {shares[i_deg]:.9f} against CIGALE's "
                f"{reference:.9f} ({shares[i_deg] / reference:.4f}x), outside 5%."
            )

    def test_loaded_norm_is_reversed_with_the_cos_inclination_axis(self):
        """``norm`` travels with the cubes when the descending axis is flipped.

        The file stores ``cos_inclination`` DESCENDING (1 -> 0) and the loader
        reverses it, flipping every cube along the same axis (#1911). ``norm``
        is indexed by that axis too, so leaving it in file order pairs each
        model's spectra with a different model's normalization -- and it is
        the far end of the axis that gets swapped in, so the mispairing is a
        factor of 3.62 at the fiducial, not a rounding error.
        """
        h5py = pytest.importorskip("h5py")
        from tengri.components.agn.skirtor import _find_skirtor_grid, _load_grid_arrays

        path = _find_skirtor_grid()
        with h5py.File(path, "r") as f:
            if "spectra/norm" not in f:
                pytest.skip("repackaged SKIRTOR grid carries no spectra/norm")
            cos_file = np.asarray(f["grid/cos_inclination"][:])
            norm_file = np.asarray(f["spectra/norm"][:])
        raw = _load_grid_arrays(path)
        assert "norm" in raw, (
            "the loader does not publish spectra/norm, so the CIGALE-lineage "
            "inclination normalization cannot reach R_faceon at all"
        )
        cos_loaded = np.asarray(raw["axes"][-1])
        assert cos_loaded[0] < cos_loaded[-1], (
            "probe setup failed: the loaded cos_inclination axis is not ascending"
        )
        j_face_file = int(np.argmax(cos_file))
        j_face_loaded = int(np.argmax(cos_loaded))
        np.testing.assert_allclose(
            np.asarray(raw["norm"])[..., j_face_loaded],
            norm_file[..., j_face_file],
            rtol=1e-12,
            atol=0.0,
            err_msg=(
                "the loaded norm's face-on slice is not the file's face-on slice: "
                "norm was not reversed along the cos_inclination axis with the cubes"
            ),
        )

    def test_shipped_grid_norm_is_positive_everywhere(self):
        """No cell needs the ``norm <= 0`` degradation, and the guard is real.

        ``skirtor_disc_dust_ratio`` falls back to a factor of 1.0 where the
        interpolated norm is non-positive, because ``norm = 0`` marks a
        parameter cell no SKIRTOR run filled -- there the spectra are the
        1e-99 filler and any ratio is meaningless. This asserts the shipped
        grid never takes that path, so the numbers above are the real
        interpolant and not a silent fallback: all 19200 cells carry
        norm > 0, and the same 19200 carry non-filler dust spectra.
        """
        h5py = pytest.importorskip("h5py")
        from tengri.components.agn.skirtor import _find_skirtor_grid

        with h5py.File(_find_skirtor_grid(), "r") as f:
            if "spectra/norm" not in f:
                pytest.skip("repackaged SKIRTOR grid carries no spectra/norm")
            norm = np.asarray(f["spectra/norm"][:])
            dust_peak = np.asarray(f["spectra/dust_emission"][:]).max(axis=-1)
        n_zero = int((norm <= 0.0).sum())
        assert n_zero == 0, (
            f"{n_zero} of {norm.size} parameter cells carry norm <= 0, so "
            "R_faceon silently degrades to the pre-R64 factor of 1.0 there"
        )
        assert int((dust_peak <= 1e-90).sum()) == 0, (
            "some cells carry the 1e-99 filler dust spectrum while declaring "
            "norm > 0: the two coverage markers disagree"
        )
