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


def _assert_nebular_sedmodelcomponent(model, expected_cls: str) -> None:
    """Assert the build dispatched nebular to its SEDModelComponent port.

    ADR-0011: nebular grammar keys must land on their ``SEDModelComponent``
    class, not the legacy bare-Protocol ``NebularSEDComponent``. Currently RED
    — dispatch migration is incomplete (#738).
    """
    from tengri.components.sed_model_component import SEDModelComponent

    chain = model._build_component_chain()
    classes = {type(c).__name__ for c in chain if isinstance(c, SEDModelComponent)}
    assert expected_cls in classes, (
        f"nebular dispatched to {sorted(type(c).__name__ for c in chain)} — expected "
        f"the {expected_cls} SEDModelComponent, not the legacy NebularSEDComponent (#738)."
    )


class TestBuildResolverNebular:
    """Test that SEDModel.build dispatches nebular grammar keys to their ports.

    The nebular grammar keys ``cue`` / ``cloudy`` / ``cb19`` must dispatch to
    their ``SEDModelComponent`` classes (ADR-0011). Data-gated backends skip
    when their grid is absent. These currently fail — see #738.
    """

    def test_neb_cue(self, ssp_data_bc03):
        """'cue' dispatches to the CueNebularSEDComponent port."""
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
        _assert_nebular_sedmodelcomponent(model, "CueNebularSEDComponent")

    def test_neb_cloudy(self, ssp_data_bc03):
        """'cloudy' dispatches to the CloudyGridSEDComponent port."""
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
        _assert_nebular_sedmodelcomponent(model, "CloudyGridSEDComponent")

    def test_neb_cb19(self, ssp_data_bc03):
        """'cb19' dispatches to the CB19SEDComponent port."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            neb={"type": "cb19"},
            redshift=Fixed(0.1),
        )
        _assert_nebular_sedmodelcomponent(model, "CB19SEDComponent")


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
