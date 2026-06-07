# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SEDModel.build resolver consulting _REGISTRY for SEDModelComponent ports."""

import pytest

from tengri import Fixed, SEDModel
from tengri.components.sed_model_component import _REGISTRY

pytestmark = pytest.mark.contract


class TestBuildResolverDustAttenuation:
    """Test that SEDModel.build resolves dust attenuation SEDModelComponent types."""

    def test_dust_calzetti_from_registry(self, ssp_data_bc03):
        """Build with dust type 'calzetti' from registry."""
        assert "calzetti" in _REGISTRY
        # Should not raise ValueError for unknown type
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"type": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_smc_from_registry(self, ssp_data_bc03):
        """Build with dust type 'smc' from registry."""
        assert "smc" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"type": "smc", "tau_v": Fixed(0.2)},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_mw_from_registry(self, ssp_data_bc03):
        """Build with dust type 'mw' (Milky Way) from registry."""
        assert "mw" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"type": "mw", "tau_v": Fixed(0.3)},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_salim18_from_registry(self, ssp_data_bc03):
        """Build with dust type 'salim18' from registry."""
        assert "salim18" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"type": "salim18", "tau_v": Fixed(0.3)},
            redshift=Fixed(0.1),
        )
        assert model is not None


class TestBuildResolverDustEmission:
    """Test that SEDModel.build resolves dust emission SEDModelComponent types."""

    def test_dust_emission_modified_blackbody(self, ssp_data_bc03):
        """Build with dust emission type 'modified_blackbody_ir' from registry."""
        assert "modified_blackbody_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "modified_blackbody_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_emission_dl07(self, ssp_data_bc03):
        """Build with dust emission type 'dl07_ir' from registry."""
        assert "dl07_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "dl07_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_emission_dl14(self, ssp_data_bc03):
        """Build with dust emission type 'dl14_ir' from registry."""
        assert "dl14_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "dl14_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_emission_dale2014(self, ssp_data_bc03):
        """Build with dust emission type 'dale2014_ir' from registry."""
        assert "dale2014_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "dale2014_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_emission_astrodust(self, ssp_data_bc03):
        """Build with dust emission type 'astrodust_ir' from registry."""
        assert "astrodust_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "astrodust_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_dust_emission_draine2021_pah(self, ssp_data_bc03):
        """Build with dust emission type 'draine2021_pah_ir' from registry."""
        assert "draine2021_pah_ir" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"emission": {"type": "draine2021_pah_ir"}},
            redshift=Fixed(0.1),
        )
        assert model is not None


class TestBuildResolverNebular:
    """Test that SEDModel.build resolves nebular backends via the grammar.

    The nebular backend physics lives in SEDModelComponent ports
    (``cue_emulator`` / ``cloudy_grid`` in ``_REGISTRY``), but the canonical
    build surface uses the grammar names ``cue`` / ``cloudy`` / ``cb19`` (the
    ``_REGISTRY`` aliases are not wired into the neb grammar). These assert the
    canonical surface; data-gated backends skip when their grid is absent.
    """

    def test_neb_cue(self, ssp_data_bc03):
        """Build with nebular backend 'cue' (Cue emulator)."""
        try:
            model = SEDModel.build(
                ssp_data=ssp_data_bc03,
                neb={"type": "cue"},
                redshift=Fixed(0.1),
            )
        except FileNotFoundError as exc:
            pytest.skip(f"cue weights not on disk: {exc}")
        except ValueError as exc:
            if any(t in str(exc).lower() for t in ("grid", "requires", "not on disk")):
                pytest.skip(f"cue weights not on disk: {exc}")
            raise
        assert model is not None

    def test_neb_cloudy(self, ssp_data_bc03):
        """Build with nebular backend 'cloudy' (CloudyGrid)."""
        try:
            model = SEDModel.build(
                ssp_data=ssp_data_bc03,
                neb={"type": "cloudy"},
                redshift=Fixed(0.1),
            )
        except FileNotFoundError as exc:
            pytest.skip(f"cloudy grid not on disk: {exc}")
        except ValueError as exc:
            if any(t in str(exc).lower() for t in ("grid", "requires", "not on disk")):
                pytest.skip(f"cloudy grid not on disk: {exc}")
            raise
        assert model is not None

    def test_neb_cb19(self, ssp_data_bc03):
        """Build with nebular backend 'cb19'."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            neb={"type": "cb19"},
            redshift=Fixed(0.1),
        )
        assert model is not None


class TestBuildResolverAGNTorus:
    """Test that SEDModel.build resolves AGN torus SEDModelComponent types."""

    def test_agn_torus_skirtor(self, ssp_data_bc03):
        """Build with AGN torus type 'skirtor' from registry."""
        assert "skirtor" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            agn={"torus": {"type": "skirtor"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_agn_torus_silva04(self, ssp_data_bc03):
        """Build with AGN torus type 'silva04' from registry."""
        assert "silva04" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            agn={"torus": {"type": "silva04"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_agn_torus_cat3d_wind(self, ssp_data_bc03):
        """Build with AGN torus type 'cat3d_wind' from registry."""
        assert "cat3d_wind" in _REGISTRY
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            agn={"torus": {"type": "cat3d_wind"}},
            redshift=Fixed(0.1),
        )
        assert model is not None


class TestBuildResolverAGNDisc:
    """Test that SEDModel.build resolves AGN disc composable-block types.

    Unlike dust/nebular ports (whose ``_REGISTRY`` names *are* the grammar
    keys), AGN disc physics is reached through the composable disc-block
    grammar (ADR-0018) — ``agn={'disc': {'type': 'kubota_done'}}`` — not via the
    bare ``SEDModelComponent`` names (``kd18_disc`` / ``powerlaw_disc``), which
    are not wired into the disc resolver. These assert the canonical surface.
    """

    def test_agn_disc_kubota_done(self, ssp_data_bc03):
        """Build with AGN disc block 'kubota_done' (Kubota & Done 2018)."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            agn={"disc": {"type": "kubota_done"}},
            redshift=Fixed(0.1),
        )
        assert model is not None

    def test_agn_disc_powerlaw(self, ssp_data_bc03):
        """Build with AGN disc block 'powerlaw'."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            agn={"disc": {"type": "powerlaw"}},
            redshift=Fixed(0.1),
        )
        assert model is not None
