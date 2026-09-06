# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for AGN backends on SEDModelComponent.

Validates:
- Registry registration (name uniqueness)
- isinstance checks against base classes
- declared_parameters matches expected structure
- predict returns valid arrays and published dicts
- Cross-component inputs/outputs are declared correctly
"""

import jax.numpy as jnp
import pytest

from tengri.components.agn.cat3d_torus_model import CAT3DTorus
from tengri.components.agn.kd18_disc_model import KD18Disc
from tengri.components.agn.powerlaw_disc_model import PowerLawDisc
from tengri.components.agn.silva04_model import Silva04Torus
from tengri.components.agn.skirtor_agnfitter_model import SKIRTORAgnfitterTorus
from tengri.components.agn.skirtor_model import SKIRTORTorus
from tengri.components.sed_model_component import SEDModelComponent

pytestmark = pytest.mark.contract


class TestKD18DiscContract:
    """KD18 disc component contract validation."""

    def test_kd18_registered(self):
        """KD18Disc is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "kd18_disc" in _REGISTRY
        assert _REGISTRY["kd18_disc"] is KD18Disc

    def test_kd18_isinstance(self):
        """KD18Disc implements SEDModelComponent and SEDComponent protocols."""
        component = KD18Disc()
        assert isinstance(component, SEDModelComponent)
        # SEDComponent is a protocol; check duck-typing
        assert hasattr(component, "declared_parameters")
        assert hasattr(component, "inputs")
        assert hasattr(component, "outputs")
        assert hasattr(component, "precompute")
        assert hasattr(component, "apply")

    def test_kd18_declared_parameters(self):
        """KD18Disc declares all expected free parameters."""
        component = KD18Disc()
        params = component.declared_parameters()
        param_names = {p.name for p in params}

        expected = {
            "agn_log_lbol",
            "agn_log_mbh",
            "agn_log_ledd",
            "agn_a_spin",
            "agn_cos_inc",
            "agn_f_hard",
            "agn_gamma_warm",
            "agn_kt_warm",
            "agn_gamma_hard",
            "agn_kt_hot",
            "agn_r_warm_ratio",
            "agn_lum_ratio",
        }
        assert param_names == expected

        # Check units are present
        for p in params:
            assert p.units, f"{p.name} missing units"

    def test_kd18_outputs(self):
        """KD18Disc publishes L_agn_disc."""
        component = KD18Disc()
        outputs = component.outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "L_agn_disc"
        assert outputs[0].units == "erg/s"

    def test_kd18_inputs_empty(self):
        """KD18Disc has no required inputs."""
        component = KD18Disc()
        inputs = component.inputs()
        assert len(inputs) == 0

    def test_kd18_bounds_match_canonical_declaration(self):
        """Every KD18Disc free-parameter bound/default derives from the
        canonical PARAMS declaration (Task 11 fix round 1, ruling R25).

        Before this fix, all twelve of these were restated as independent
        literals that had drifted from ``agn/_params.py`` -- e.g.
        ``log_lbol`` default ``11.0`` vs canonical ``10.0``, ``log_mbh``
        default ``8.0`` vs canonical ``7.0``, ``log_ledd`` bounds
        ``(-3.0, 0.0)`` vs canonical ``(-2.0, 0.5)``, ``a_spin`` default
        ``0.5`` vs canonical ``0.0``. Fails if the class ever restates one
        of them as a literal again.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = KD18Disc()
        for attr, canonical_name in (
            ("log_lbol", "agn_log_lbol"),
            ("log_mbh", "agn_log_mbh"),
            ("log_ledd", "agn_log_ledd"),
            ("a_spin", "agn_a_spin"),
            ("cos_inc", "agn_cos_inc"),
            ("f_hard", "agn_f_hard"),
            ("gamma_warm", "agn_gamma_warm"),
            ("kt_warm", "agn_kt_warm"),
            ("gamma_hard", "agn_gamma_hard"),
            ("kt_hot", "agn_kt_hot"),
            ("r_warm_ratio", "agn_r_warm_ratio"),
            ("lum_ratio", "agn_lum_ratio"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr

    def test_kd18_predict_shape(self):
        """KD18Disc.predict returns valid shapes."""
        component = KD18Disc()

        wave = jnp.logspace(0, 5, 100)  # 1 Å to 100 μm
        sed_in = jnp.zeros_like(wave)

        p = {
            "log_lbol": 44.0,
            "log_mbh": 8.0,
            "log_ledd": -1.0,
            "a_spin": 0.0,
            "cos_inc": 0.5,
            "f_hard": 0.02,
            "gamma_warm": 2.5,
            "kt_warm": 0.2,
            "gamma_hard": 1.8,
            "kt_hot": 100.0,
            "r_warm_ratio": 2.0,
            "frac": 1.0,
        }

        sed_out, published = component.predict(p, sed_in, wave)

        assert sed_out.shape == wave.shape
        assert "L_agn_disc" in published
        assert published["L_agn_disc"] >= 0.0


class TestPowerLawDiscContract:
    """PowerLaw disc component contract validation."""

    def test_powerlaw_registered(self):
        """PowerLawDisc is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "powerlaw_disc" in _REGISTRY
        assert _REGISTRY["powerlaw_disc"] is PowerLawDisc

    def test_powerlaw_declared_parameters(self):
        """PowerLawDisc declares all expected free parameters."""
        component = PowerLawDisc()
        params = component.declared_parameters()
        param_names = {p.name for p in params}

        expected = {"agn_log_lbol", "agn_alpha", "agn_T_max", "agn_lum_ratio"}
        assert param_names == expected

    def test_powerlaw_bounds_match_canonical_declaration(self):
        """Every PowerLawDisc free-parameter bound/default derives from the
        canonical PARAMS declaration (Task 11 fix round 1, ruling R25).

        Before this fix, ``log_lbol`` restated default ``11.0`` vs
        canonical ``10.0``; ``alpha`` restated bounds ``(-1.5, -0.5)`` vs
        canonical ``(-2.0, 0.0)``; ``lum_ratio`` restated bounds/default
        ``(0.0, 1.0, 0.5)`` vs canonical ``(0.0, 5.0, 1.0)``. Fails if the
        class ever restates one of them as a literal again.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = PowerLawDisc()
        for attr, canonical_name in (
            ("log_lbol", "agn_log_lbol"),
            ("alpha", "agn_alpha"),
            ("T_max", "agn_T_max"),
            ("lum_ratio", "agn_lum_ratio"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr

    def test_powerlaw_outputs(self):
        """PowerLawDisc publishes L_agn_disc."""
        component = PowerLawDisc()
        outputs = component.outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "L_agn_disc"

    def test_powerlaw_predict_shape(self):
        """PowerLawDisc.predict returns valid shapes."""
        component = PowerLawDisc()

        wave = jnp.logspace(0, 5, 100)
        sed_in = jnp.zeros_like(wave)

        p = {
            "log_lbol": 44.0,
            "alpha": -1.0,
            "T_max": 1e5,
            "frac": 1.0,
        }

        sed_out, published = component.predict(p, sed_in, wave)

        assert sed_out.shape == wave.shape
        assert "L_agn_disc" in published
        assert published["L_agn_disc"] >= 0.0


class TestSilva04TorusContract:
    """Silva+04 torus component contract validation."""

    def test_silva04_registered(self):
        """Silva04Torus is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "silva04" in _REGISTRY
        assert _REGISTRY["silva04"] is Silva04Torus

    def test_silva04_declared_parameters(self):
        """Silva04Torus declares all expected free parameters."""
        component = Silva04Torus()
        params = component.declared_parameters()
        param_names = {p.name for p in params}

        expected = {"agn_log_lbol", "agn_log_nh_silva", "agn_torus_frac"}
        assert param_names == expected

    def test_silva04_log_nh_matches_canonical_declaration(self):
        """log_nh_silva derives from the canonical PARAMS declaration (single source
        of truth, Task 1/M-A). Fails if the class ever restates a literal
        ``Uniform(lo, hi, ...)`` that can drift from ``agn_log_nh_silva`` again,
        the way it drifted to ``(22.0, 25.0)`` while the grid moved to
        ``(21.5, 24.45)``.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        canonical = declared_prior(PARAMS, "agn_log_nh_silva")
        component = Silva04Torus()
        assert component.log_nh_silva.lo == canonical.lo
        assert component.log_nh_silva.hi == canonical.hi
        assert component.log_nh_silva.default == canonical.default

    def test_silva04_log_lbol_and_torus_frac_match_canonical_declaration(self):
        """log_lbol/torus_frac derive from the canonical PARAMS declarations
        (Task 11 fix round 1, ruling R25). Before this fix, ``log_lbol``
        restated default ``11.0`` vs canonical ``10.0`` and ``torus_frac``
        restated bounds/default vs canonical. Fails if the class ever
        restates one of them as a literal again.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = Silva04Torus()
        for attr, canonical_name in (
            ("log_lbol", "agn_log_lbol"),
            ("torus_frac", "agn_torus_frac"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr

    def test_silva04_outputs(self):
        """Silva04Torus publishes L_agn_torus."""
        component = Silva04Torus()
        outputs = component.outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "L_agn_torus"

    def test_silva04_predict_no_grid(self):
        """Silva04Torus.predict gracefully handles missing grid."""
        component = Silva04Torus()  # grid_path is None by default

        wave = jnp.logspace(0, 5, 100)
        sed_in = jnp.zeros_like(wave)

        # Precompute will set self.data = None since grid_path is None
        component.precompute(wave_grid=wave)

        p = {
            "log_lbol": 44.0,
            "log_nh_silva": 23.0,
            "torus_frac": 0.5,
        }

        sed_out, published = component.predict(p, sed_in, wave)

        # Should return zero emission gracefully
        assert sed_out.shape == wave.shape
        assert published["L_agn_torus"] == 0.0


class TestCAT3DTorusContract:
    """CAT3D-Wind torus component contract validation."""

    def test_cat3d_registered(self):
        """CAT3DTorus is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "cat3d_wind" in _REGISTRY
        assert _REGISTRY["cat3d_wind"] is CAT3DTorus

    def test_cat3d_declared_parameters(self):
        """CAT3DTorus declares all expected free parameters."""
        component = CAT3DTorus()
        params = component.declared_parameters()
        param_names = {p.name for p in params}

        expected = {
            "agn_log_lbol",
            "agn_cos_inc",
            "agn_a_cat3d",
            "agn_fwd_cat3d",
            "agn_torus_frac",
        }
        assert param_names == expected

    def test_cat3d_bounds_match_canonical_declaration(self):
        """a_cat3d/fwd_cat3d/cos_inc derive from the canonical PARAMS declarations
        (single source of truth, Task 1/M-A). Fails if the class ever restates a
        literal ``Uniform(lo, hi, ...)`` that can drift from
        ``agn_a_cat3d``/``agn_fwd_cat3d``/``agn_cos_inc`` again, the way
        ``fwd_cat3d`` drifted to a stale ``Uniform(0.0, 1.0)`` while the canonical
        declaration (and the vendored grid) moved to ``(1.0, 2.25)``.
        Extended in Task 11 fix round 1 (ruling R25) to also cover
        ``log_lbol``/``torus_frac``, which restated a stale
        ``default=11.0`` (vs canonical ``10.0``) and stale ``torus_frac``
        bounds/default respectively.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = CAT3DTorus()
        for attr, canonical_name in (
            ("a_cat3d", "agn_a_cat3d"),
            ("fwd_cat3d", "agn_fwd_cat3d"),
            ("cos_inc", "agn_cos_inc"),
            ("log_lbol", "agn_log_lbol"),
            ("torus_frac", "agn_torus_frac"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr

    def test_cat3d_outputs(self):
        """CAT3DTorus publishes L_agn_torus."""
        component = CAT3DTorus()
        outputs = component.outputs()
        assert len(outputs) == 1
        assert outputs[0].name == "L_agn_torus"

    def test_cat3d_predict_no_grid(self):
        """CAT3DTorus.predict gracefully handles missing grid."""
        component = CAT3DTorus()  # grid_path is None by default

        wave = jnp.logspace(0, 5, 100)
        sed_in = jnp.zeros_like(wave)

        component.precompute(wave_grid=wave)

        p = {
            "log_lbol": 44.0,
            "cos_inc": 0.5,
            "a_cat3d": -2.0,
            "fwd_cat3d": 1.75,
            "torus_frac": 0.5,
        }

        sed_out, published = component.predict(p, sed_in, wave)

        # Should return zero emission gracefully
        assert sed_out.shape == wave.shape
        assert published["L_agn_torus"] == 0.0


class TestSKIRTORAgnfitterTorusContract:
    """SKIRTOR_mean_3p (AGNfitter-faithful) torus component contract validation
    (Task 1/M-A fix round 1, finding 1).
    """

    def test_skirtor_agnfitter_registered(self):
        """SKIRTORAgnfitterTorus is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "skirtor_agnfitter" in _REGISTRY
        assert _REGISTRY["skirtor_agnfitter"] is SKIRTORAgnfitterTorus

    def test_skirtor_agnfitter_bounds_match_canonical_declaration(self):
        """oa_skirtor/incl_skirtor/tv_skirtor derive from the canonical PARAMS
        declarations (single source of truth, Task 1/M-A). Numerically equal
        to ``agn_oa_skirtor``/``agn_incl_skirtor``/``agn_tv_skirtor`` today but
        were unprotected literal restatements before this fix round -- fails
        if the class ever restates one of them as a literal again (the same
        way ``CAT3DTorus.fwd_cat3d`` drifted to a stale ``Uniform(0.0, 1.0)``
        while its canonical declaration moved to ``(1.0, 2.25)``).
        Extended in Task 11 fix round 1 (ruling R25) to also cover
        ``log_lbol``/``torus_frac``, which were genuine drifts (``log_lbol``
        default ``11.0`` vs canonical ``10.0``), not merely unprotected.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = SKIRTORAgnfitterTorus()
        for attr, canonical_name in (
            ("oa_skirtor", "agn_oa_skirtor"),
            ("incl_skirtor", "agn_incl_skirtor"),
            ("tv_skirtor", "agn_tv_skirtor"),
            ("log_lbol", "agn_log_lbol"),
            ("torus_frac", "agn_torus_frac"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr


class TestSKIRTORTorusContract:
    """SKIRTOR (X-CIGALE-faithful, default ``skirtor`` block) torus component
    contract validation (Task 1/M-A fix round 1, finding 2).
    """

    def test_skirtor_registered(self):
        """SKIRTORTorus is registered with unique name."""
        from tengri.components.sed_model_component import _REGISTRY

        assert "skirtor" in _REGISTRY
        assert _REGISTRY["skirtor"] is SKIRTORTorus

    def test_skirtor_bounds_match_canonical_declaration(self):
        """oa_skirtor/tau_skirtor/p_skirtor/q_skirtor/cos_inc derive from the
        canonical PARAMS declarations (single source of truth, Task 1/M-A).
        Fails if the class ever restates a literal ``Uniform(lo, hi, ...)``
        that can drift from ``agn_oa_skirtor``/... again, the way
        ``oa_skirtor`` drifted to a stale ``Uniform(20.0, 60.0)`` while both
        the canonical declaration and this class's own vendored grid
        (``data/skirtor_templates_v3.h5``, ``grid/opening_angle`` = [10, 80])
        moved to ``(10.0, 80.0)``.
        """
        from tengri.components.agn._params import PARAMS
        from tengri.protocols.component import declared_prior

        component = SKIRTORTorus()
        for attr, canonical_name in (
            ("oa_skirtor", "agn_oa_skirtor"),
            ("tau_skirtor", "agn_tau_skirtor"),
            ("p_skirtor", "agn_p_skirtor"),
            ("q_skirtor", "agn_q_skirtor"),
            ("cos_inc", "agn_cos_inc"),
        ):
            canonical = declared_prior(PARAMS, canonical_name)
            prior = getattr(component, attr)
            assert prior.lo == canonical.lo, attr
            assert prior.hi == canonical.hi, attr
            assert prior.default == canonical.default, attr
