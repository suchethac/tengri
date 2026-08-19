# SPDX-License-Identifier: BSD-3-Clause
"""Tests for SEDModel.build resolver consulting _REGISTRY for SEDModelComponent
components."""

import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.sed_model_component import _REGISTRY

pytestmark = pytest.mark.contract


class TestBuildResolverDustAttenuation:
    """Dust attenuation laws are config sub-selectors of the canonical engine.

    Direction B (#738): the thin single-law components (calzetti/smc/mw/salim18)
    were deleted — they were silent no-ops. ``calzetti``/``smc``/… are dust
    *laws* selected via ``law_bc``/``law_diff`` on the ``two_component`` /
    ``single`` / ``wg00`` engine, NOT standalone ``_REGISTRY`` component types.
    """

    @pytest.mark.parametrize("law", ["calzetti", "smc", "mw", "salim18"])
    def test_law_is_not_a_phantom_dust_type(self, ssp_data_bc03, law):
        """``dust={'type': <law>}`` must fail loud — not silently no-op.

        Before #738 these routed through a ``_REGISTRY`` pass-through that set
        ``dust_model='two_component'`` and dropped the law (built ``power_law``).
        """
        assert law not in _REGISTRY
        # Two loud-failure messages are acceptable (#664/#784): an outright
        # "Unknown dust type" (e.g. 'mw', 'salim18' — not registered laws) or
        # the law-vs-type redirect for names that ARE registered attenuation
        # laws ('calzetti', 'smc'): "<law> is a dust attenuation *law*…".
        with pytest.raises(ValueError, match=r"Unknown dust type|is a dust attenuation \*law\*"):
            SEDModel.build(ssp_data=ssp_data_bc03, dust={"type": law}, redshift=Fixed(0.1))

    @pytest.mark.parametrize("law", ["calzetti", "smc", "cardelli", "salim"])
    def test_law_surface_builds_and_threads(self, ssp_data_bc03, law):
        """The canonical surface ``dust={'type':'two_component','law_bc':<law>}``
        builds and threads the chosen law through to the engine (not dropped)."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            dust={"type": "two_component", "law": law, "*": FIXED},
            redshift=Fixed(0.1),
        )
        assert model is not None
        assert model._dust_law_bc == law


class TestBuildResolverDustEmission:
    """Dust IR emission models are engine sub-selectors of the two-component
    engine (resolved via ``_REGISTRY``), not standalone ``_REGISTRY``
    components. The ``*_ir`` SEDModelComponent classes are unused parity
    mirrors — the grammar surface is the engine name (#738).
    """

    @pytest.mark.parametrize(
        "emission",
        ["modified_blackbody", "dl07", "dl14", "dale2014", "astrodust", "draine_li2014"],
    )
    def test_emission_model_builds(self, ssp_data_bc03, emission):
        """``dust={'emission':{'type': <engine name>}}`` builds on the engine."""
        try:
            model = SEDModel.build(
                ssp_data=ssp_data_bc03,
                dust={"law": "power_law", "emission": {"type": emission}},
                redshift=Fixed(0.1),
            )
        except (FileNotFoundError, OSError) as exc:
            pytest.skip(f"{emission!r} template not on disk: {exc}")
        assert model is not None

    @pytest.mark.parametrize(
        "component_name", ["dl07_ir", "schreiber2016_ir", "draine2021_pah_ir"]
    )
    def test_ir_component_names_rejected_as_emission_types(self, ssp_data_bc03, component_name):
        """``*_ir`` SEDModelComponent names are not valid emission grammar —
        not the deleted duplicates (``dl07_ir``) nor the surviving unique
        components (``schreiber2016_ir``/``draine2021_pah_ir``, still in
        ``_REGISTRY``). They used to be silently accepted then fail at predict
        (the removed #738 footgun)."""
        with pytest.raises(ValueError, match="Unknown dust emission type"):
            SEDModel.build(
                ssp_data=ssp_data_bc03,
                dust={"emission": {"type": component_name}},
                redshift=Fixed(0.1),
            )


def _assert_nebular_backend(model, expected_backend: str) -> None:
    """Assert build dispatched nebular to the canonical engine + backend.

    Direction B (#738): nebular grammar keys land on the tested
    ``NebularSEDComponent`` engine with ``config.backend`` set — NOT a
    per-backend ``SEDModelComponent`` component (those duplicates were deleted).
    """
    from tengri.components.nebular.component import NebularSEDComponent

    chain = model._build_component_chain()
    neb = next((c for c in chain if isinstance(c, NebularSEDComponent)), None)
    assert neb is not None, (
        f"no NebularSEDComponent in chain {sorted(type(c).__name__ for c in chain)}"
    )
    assert neb.config.backend == expected_backend, (
        f"nebular dispatched to backend {neb.config.backend!r}, expected {expected_backend!r}."
    )


class TestBuildResolverNebular:
    """Nebular grammar keys dispatch to the canonical ``NebularSEDComponent`` +
    backend (Direction B, #738). ``cue`` → backend ``'cue'``, ``cloudy`` →
    ``'cloudy_grid'``, ``cb19`` → ``'cb19'``. Data-gated backends skip when
    their grid is absent.
    """

    def test_neb_cue(self, ssp_data_bc03):
        """'cue' dispatches to NebularSEDComponent with the 'cue' backend."""
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
        _assert_nebular_backend(model, "cue")

    def test_neb_cloudy(self, ssp_data_bc03):
        """'cloudy' dispatches to NebularSEDComponent with the 'cloudy_grid' backend."""
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
        _assert_nebular_backend(model, "cloudy_grid")

    def test_neb_cb19(self, ssp_data_bc03):
        """'cb19' dispatches to NebularSEDComponent with the 'cb19' backend."""
        model = SEDModel.build(
            ssp_data=ssp_data_bc03,
            neb={"type": "cb19"},
            redshift=Fixed(0.1),
        )
        _assert_nebular_backend(model, "cb19")


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

    Unlike dust/nebular components (whose ``_REGISTRY`` names *are* the grammar
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
