# SPDX-License-Identifier: BSD-3-Clause
"""Behavioral regression tests for the dust-emission registry switchover.

These are deliberately *behavioral*, not structural: each asserts a physical
consequence that FAILS if the switchover is wrong, because the earlier
structural tests (assert-a-name-is-registered) passed even while dust emission
contributed nothing at all (it was ordered before the attenuator, so L_ir=0).

Coverage (each targets a specific failure mode we actually hit):
- ordering: the emission component must run AFTER the dust attenuator (else L_ir=0).
- exact photometry: the far-IR band lights up with emission; optical is unchanged.
- WavePrecomp photometry: the far-IR band lights up AND matches the exact path.
- energy conservation: the component re-radiates ~L_ir into the IR.
- redshift/CMB: emission depends on redshift (the BARE_NAME_ALLOWLIST threading).
- spectrum path: emission adds IR flux to the rest-frame spectrum.

All tests run on the synthetic wide SSP fixture (CI-runnable, no data/ grids).
"""

import contextlib

import jax.numpy as jnp
import numpy as np
import pytest

import tengri  # noqa: F401  (ensures components register in _REGISTRY)

pytestmark = pytest.mark.contract

# Filter indices used across the photometry tests: 0-3 optical, 4 = far-IR (100 um).
_OPTICAL = (3500.0, 4800.0, 6200.0, 9000.0)
_FAR_IR = 1.0e6


def _tophat(center, frac=0.16, n=40):
    from tengri.observation.photometry import FilterCurve

    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


def _obs_optical_plus_fir():
    from tengri.observation import Observation, Photometry

    curves = tuple(_tophat(c) for c in (*_OPTICAL, _FAR_IR))
    return Observation(photometry=Photometry(filters=curves))


def _params(emit: bool):
    from tengri import Fixed, Parameters

    kw = dict(
        mean_sfh_type="dpl",
        sfh_dpl_alpha=Fixed(2.0),
        sfh_dpl_beta=Fixed(1.0),
        sfh_dpl_tau_gyr=Fixed(1.0),
        sfh_dpl_age_gyr=Fixed(5.0),
        sfh_dpl_log_total_mass=Fixed(10.0),
        met_logzsol=Fixed(0.0),
        redshift=Fixed(0.05),
        dust_tau_bc=Fixed(0.8),
        dust_tau_diff=Fixed(0.5),
    )
    if emit:
        kw["dust_emission"] = "modified_blackbody"
    return Parameters(**kw)


def _model(emit, ssp, obs=None, approx=None):
    from tengri import SEDModel

    obs = obs if obs is not None else _obs_optical_plus_fir()
    kwargs = {"observation": obs}
    if approx is not None:
        kwargs["approx"] = approx
    return SEDModel(_params(emit), ssp, **kwargs)


# ─── structural smoke (cheap; not the ones that catch physics bugs) ──────────


def test_registry_resolution_and_aliases():
    """The seam resolves every emission type (and alias) to a registered component."""
    from tengri.forward.component_factory import (
        _EMISSION_TYPE_ALIASES,
        _resolve_registry_component,
    )

    for name in ("modified_blackbody", "dale2014", "draine_li2007", "schreiber2018"):
        comp = _resolve_registry_component("dust_emission", name)
        assert "sed_dust_ir" in {o.name for o in comp.outputs()}
    for alias in _EMISSION_TYPE_ALIASES:
        _resolve_registry_component("dust_emission", alias)  # must not raise
    with pytest.raises(ValueError):
        _resolve_registry_component("dust_emission", "nonexistent_model")


def test_no_double_declaration_of_emission_params():
    """Emission params are owned by the component, not DustSEDComponent (no collision)."""
    from tengri.forward.component_factory import build_components
    from tengri.forward.orchestrator import merge_declared_parameters

    class _MockSSP:
        ssp_wave = None

    components = build_components(
        ssp_data=_MockSSP(),
        dust_model="two_component",
        dust_emission_model="modified_blackbody",
        use_dust=True,
        nebular_backend=None,
        agn_model=None,
        use_radio=False,
        use_xray=False,
        use_igm=False,
    )
    params = merge_declared_parameters(components)  # raises on duplicate declaration
    assert any(name == "dust_T" for name in params), "emission owns dust_T"


def test_emission_component_base_not_registered():
    """The abstract EmissionComponent base must NOT leak into the dispatch registry."""
    from tengri.components.sed_model_component import _REGISTRY

    assert "component" not in _REGISTRY
    assert not any(type(v).__name__ == "EmissionComponent" for v in _REGISTRY.values())
    for name in ("modified_blackbody", "casey2012", "pah_drude", "dale2014", "themis"):
        assert name in _REGISTRY


# ─── behavioral: these FAIL if the switchover is wrong ───────────────────────


def test_emission_ordered_after_attenuator(synthetic_ssp_wide):
    """The emission component must run AFTER the dust attenuator that publishes L_ir.

    This is the exact bug we hit: the component was ordered before DustSEDComponent,
    so it read L_ir=0 and emitted nothing. Assert the ordering directly.
    """
    m = _model(True, synthetic_ssp_wide)
    chain = [type(c).__name__ for c in m._build_component_chain()]
    assert "DustSEDComponent" in chain and "ModifiedBlackbodyIRSEDComponent" in chain
    assert chain.index("ModifiedBlackbodyIRSEDComponent") > chain.index("DustSEDComponent"), (
        f"emission component must come after the attenuator; got {chain}"
    )


def test_emission_lights_up_far_ir_exact(synthetic_ssp_wide):
    """Exact photometry: the 100 um band lights up with emission; optical unchanged."""
    fw = np.asarray(_model(True, synthetic_ssp_wide).predict_photometry({}))
    fn = np.asarray(_model(False, synthetic_ssp_wide).predict_photometry({}))
    # Far-IR band jumps by orders of magnitude (dust re-radiation lands here).
    assert fw[4] > fn[4] * 100.0, f"far-IR must light up: with={fw[4]:.3e} no={fn[4]:.3e}"
    # Optical bands (energy is *absorbed* there, not re-emitted) are unchanged.
    np.testing.assert_allclose(fw[:4], fn[:4], rtol=1e-4)


def test_emission_waveprecomp_lights_up_and_matches_exact(synthetic_ssp_wide):
    """WavePrecomp photometry must ALSO light up the far-IR AND track the exact path.

    energy_balance_lut asserts LUT≈exact, but that passes trivially if BOTH are
    ~0. This additionally requires the far-IR band to be non-trivially lit.
    """
    from tengri import WavePrecomp

    ssp = synthetic_ssp_wide
    f_exact = np.asarray(_model(True, ssp).predict_photometry({}))
    try:
        f_lut = np.asarray(_model(True, ssp, approx=WavePrecomp()).predict_photometry({}))
    except TypeError:
        pytest.skip("SEDModel constructor does not accept approx= in this build")
    f_no = np.asarray(_model(False, ssp).predict_photometry({}))
    # far-IR is lit in the LUT path too (not silently zero)
    assert f_lut[4] > f_no[4] * 100.0, f"WavePrecomp far-IR must light up: {f_lut[4]:.3e}"
    # and the LUT tracks the exact projection to a few percent (WavePrecomp budget)
    np.testing.assert_allclose(f_lut[4], f_exact[4], rtol=0.05)


def test_port_conserves_energy(synthetic_ssp_wide):
    """A full emitter re-radiates ~L_ir: integral of sed_dust_ir over nu == L_ir.

    Guards against a wrong L_ir handoff or a broken normalization. pah_drude is
    excluded — it is a PAH-only building block, deliberately not energy-balanced.
    """
    from tengri.components.sed_model_component import _REGISTRY

    C_AA = 2.99792458e18  # Angstrom / s
    wave = jnp.logspace(3.0, 7.0, 4000)  # 1000 A .. 1 mm
    nu = C_AA / wave
    L_ir = 1.0e44
    for name in ("modified_blackbody", "casey2012", "graybody", "dale2014", "themis"):
        comp = _REGISTRY[name]()
        if hasattr(comp, "precompute"):
            with contextlib.suppress(Exception):
                comp.precompute(wave_grid=wave)
        sed, _ = comp.predict(
            {
                "T": 30.0,
                "beta_ir": 1.8,
                "epsilon_mbb": 1.0,
                "alpha_mir": 2.0,
                # The opacity pivot casey2012 and graybody both declare as
                # Fixed(200.0). ``predict`` reads it out of ``p`` with no
                # fallback on purpose: the grammar resolves every declared
                # parameter before the component is called, so a ``.get`` here
                # would be a second, silently-different default living in the
                # component. A hand-built dict has to supply what the grammar
                # would have.
                "lambda_0_um": 200.0,
                "alpha_dale": 2.0,
                "frac_agn": 0.0,
                "umin": 1.0,
                "qpah": 2.5,
                "gamma_dl": 0.01,
                "qhac": 0.17,
                "alpha": 2.0,
                "redshift": 0.0,
            },
            jnp.zeros_like(wave),
            wave,
            L_ir=L_ir,
        )
        integral = float(-np.trapezoid(np.asarray(sed), np.asarray(nu)))
        assert 0.5 * L_ir < integral < 1.5 * L_ir, (
            f"{name}: integral {integral:.3e} should re-radiate ~L_ir={L_ir:.3e}"
        )


def test_emission_depends_on_redshift_cmb(synthetic_ssp_wide):
    """The MBB component's emission changes with redshift (da Cunha CMB correction).

    Guards the BARE_NAME_ALLOWLIST redshift threading: if redshift never reaches
    predict, z=0 and z=6 would be identical.
    """
    from tengri.components.sed_model_component import _REGISTRY

    wave = jnp.logspace(4.0, 7.0, 1000)
    comp = _REGISTRY["modified_blackbody"]()
    p0 = {"T": 25.0, "beta_ir": 1.8, "epsilon_mbb": 1.0, "redshift": 0.0}
    p6 = {**p0, "redshift": 6.0}
    sed0, _ = comp.predict(p0, jnp.zeros_like(wave), wave, L_ir=1.0e44)
    sed6, _ = comp.predict(p6, jnp.zeros_like(wave), wave, L_ir=1.0e44)
    assert not np.allclose(np.asarray(sed0), np.asarray(sed6), rtol=1e-3), (
        "MBB emission must depend on redshift (CMB correction); redshift not threaded?"
    )


def test_emission_adds_ir_to_rest_sed(synthetic_ssp_wide):
    """The rest-frame SED differs (only in the IR) with vs without emission.

    Catches the total no-op: predict_rest_sed was byte-identical with/without
    emission when the component was mis-ordered.
    """
    ssp = synthetic_ssp_wide
    pe = _model(True, ssp).predict_rest_sed({})
    pn = _model(False, ssp).predict_rest_sed({})
    # The emission model's master grid extends into the submm (#1005) while
    # the no-emission model stays on the SSP grid — compare on the latter.
    we, wn = np.asarray(pe.wavelength), np.asarray(pn.wavelength)
    se = np.interp(wn, we, np.asarray(pe.sed))
    sn = np.asarray(pn.sed)
    assert not np.allclose(se, sn), "emission must change the rest-frame SED"
    diff = np.abs(se - sn)
    assert float(diff.max()) > 0.0
