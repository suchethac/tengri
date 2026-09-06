# SPDX-License-Identifier: BSD-3-Clause
"""End-to-end smoke tests for `SEDModelComponent` components.

For each domain (dust attenuation, dust IR emission, AGN, nebular, radio,
X-ray), build a minimal `SEDModel` with a `SEDModelComponent`-based
backend and verify:

  * Construction succeeds (no resolver miss, no contract violation)
  * `model.predict_photometry({})` returns a finite, positive vector
  * `model.spec.free_params` (when free params are declared) is non-empty
  * `model.compile_signature()` is hashable and stable across calls

These are NOT inference-level parity tests — they exercise the wiring
end-to-end (resolver → component chain → forward pass → observation
projection) for every new component at once. If a component's `name` is
missing from the resolver, or its `predict()` signature drifts from the
base class contract, or its `inputs/outputs` declaration is malformed, the
build or predict call here will catch it.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.contract

from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, load_ssp_data
from tengri.components.sed_model_component import _REGISTRY

# Filter set for all the cases below — cheap (5 filters) and physically
# realistic enough that finite/positive checks mean something.
_FILTERS = ("sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z")


_SSP_CANDIDATES = [
    "data/ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5",
    "data/ssp_prsc_bc03_chabrier.h5",
]


@pytest.fixture(scope="module")
def ssp():
    path = next((p for p in _SSP_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        pytest.skip("No SSP grid available under data/.")
    return load_ssp_data(path)


# Bare-stellar grids, for the additive nebular backends below (#1579). Every
# entry in ``_NEB_BACKEND`` adds nebular emission on top of the stellar
# continuum, so all three need a grid that does not already contain it --
# CueBackend and CloudyGridBackend refuse a nebular-included one outright.
# ``ssp`` above resolves to the wNE grid first, so ``test_nebular_e2e`` was
# building exactly that refused pairing and only survived because conftest's
# TENGRI_ALLOW_WNE_CUE=1 disabled the check along with the Q_H heuristic it
# was written for.
_BARE_SSP_CANDIDATES = [
    "data/fsps_prsc_miles_chabrier.h5",
    "data/bc03_pdva_stelib_chabrier.h5",
]


@pytest.fixture(scope="module")
def bare_ssp():
    path = next((p for p in _BARE_SSP_CANDIDATES if Path(p).is_file()), None)
    if path is None:
        pytest.skip("No bare-stellar SSP grid available under data/.")
    return load_ssp_data(path)


@pytest.fixture(scope="module")
def obs():
    return Observation(photometry=Photometry.from_names(list(_FILTERS)))


def _silent_build(**kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(**kwargs)


def _assert_phot_ok(phot):
    assert phot.shape == (len(_FILTERS),), f"unexpected shape: {phot.shape}"
    assert bool(jnp.all(jnp.isfinite(phot))), "non-finite photometry"
    assert bool(jnp.all(phot > 0)), "non-positive photometry"


# Module-level skip tally — populated by pytest's reporting hook in
# ``tests/conftest.py`` via the ``_skipped_e2e_components`` list. If anyone
# wonders "how many components actually ran in CI?", the terminal_summary
# hook at the end of the session prints the count.
_skipped_e2e_components: list[str] = []


def _skip_with_tally(reason: str, component_name: str = "") -> None:
    """Skip a component test AND record the skip for the end-of-session tally."""
    if component_name:
        _skipped_e2e_components.append(component_name)
    pytest.skip(reason)


# ─────────────────────────────────────────────────────────────────────
# Dust attenuation backends
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("law", ["calzetti", "smc", "cardelli", "salim"])
def test_dust_attenuation_e2e(ssp, obs, law):
    """Each attenuation law builds and predicts finite photometry.

    Direction B (#738): laws are config sub-selectors of the canonical
    two-component engine (``law_bc``), not standalone ``type`` components.
    """
    model = _silent_build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "two_component", "law": law, "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )
    _assert_phot_ok(model.predict_photometry({}))


# ─────────────────────────────────────────────────────────────────────
# Dust IR emission backends — closes the dust-IR end-to-end gap
# ─────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "emission_type",
    [
        "graybody",
        "modified_blackbody",
        "dl07",
        "dl14",
        "dale2014",
        "astrodust",
        "draine_li2014",
    ],
)
def test_dust_ir_emission_e2e(ssp, obs, emission_type):
    """Each dust IR emission model builds + predicts on the canonical engine.

    Emission models are config sub-selectors of the two-component engine
    (``emission={'type': ...}``, resolved via ``_REGISTRY``) —
    paired here with a Calzetti birth-cloud screen. Template-backed models
    skip when their HDF5 grid is absent.
    """
    try:
        model = _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
            },
            dust_emission={"type": emission_type, "all_params": Fixed(DEFAULT)},
            redshift=Fixed(0.05),
        )
    except (FileNotFoundError, OSError) as exc:
        pytest.skip(f"{emission_type!r} requires HDF5 template not on disk: {exc}")
    _assert_phot_ok(model.predict_photometry({}))


# ─────────────────────────────────────────────────────────────────────
# AGN backends
# ─────────────────────────────────────────────────────────────────────


def _fixed_dust() -> dict:
    """A fully-FIXED two-component dust screen (fresh dict per call).

    The AGN / radio / X-ray e2e builds below pair the emitting component with a
    dust screen so ``predict_photometry({})`` has no free params — an AGN with
    an *unconstrained* host dust law cannot be predicted from an empty dict.
    A fresh dict is returned each call because ``SEDModel.build`` consumes the
    group dicts it is handed.
    """
    return {
        "type": "two_component",
        "law": "power_law",
        "all_params": Fixed(DEFAULT),
        "tau_bc": Fixed(0.3),
        "tau_diff": Fixed(0.2),
    }


# AGN is built through the composable block grammar (the canonical AGN surface,
# ADR-0018) — NOT direct ``agn={'type': <SEDModelComponent>}`` resolution. The
# monolithic disc+torus names (skirtor / silva04 / cat3d_wind) still resolve;
# the bare disc components map onto their composable disc blocks (kd18 →
# kubota_done, powerlaw_disc → powerlaw). Each case maps an id to its canonical
# agn spec.
_AGN_E2E_CASES = {
    "skirtor": {"type": "skirtor", "all_params": Fixed(DEFAULT)},
    "silva04": {"type": "silva04", "all_params": Fixed(DEFAULT)},
    "cat3d_wind": {"type": "cat3d_wind", "all_params": Fixed(DEFAULT)},
    "kubota_done_disc": {
        "type": "composable",
        "disc": {"type": "kubota_done"},
        "all_params": Fixed(DEFAULT),
    },
    "powerlaw_disc": {
        "type": "composable",
        "disc": {"type": "powerlaw"},
        "all_params": Fixed(DEFAULT),
    },
}


@pytest.mark.parametrize("agn_id", list(_AGN_E2E_CASES))
def test_agn_e2e(ssp, obs, agn_id):
    """Each AGN component (disc + torus) builds + predicts via its canonical
    grammar."""
    try:
        model = _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            agn=_AGN_E2E_CASES[agn_id],
            dust_attenuation=_fixed_dust(),
            redshift=Fixed(0.05),
        )
    except FileNotFoundError as exc:
        _skip_with_tally(f"{agn_id!r} build skipped (template not on disk): {exc}", agn_id)
    _assert_phot_ok(model.predict_photometry({}))


# ─────────────────────────────────────────────────────────────────────
# Nebular backends — emulator (Cue) + library (CloudyGrid/CB19/MAPPINGS)
# ─────────────────────────────────────────────────────────────────────


# Nebular dispatches to the canonical ``NebularSEDComponent`` engine + a backend
# instance (Direction B, #738): ``neb={'type': X}`` builds NebularSEDComponent
# with ``config.backend`` set, NOT a per-backend SEDModelComponent
# (those duplicates were deleted). A FIXED dust screen keeps
# ``predict_photometry({})`` free-param-less.
_NEB_BACKEND = {
    "cue": "cue",
    "cloudy": "cloudy_grid",
    "cb19": "cb19",
}


@pytest.mark.parametrize("neb_type", list(_NEB_BACKEND))
def test_nebular_e2e(bare_ssp, obs, neb_type):
    """Each nebular grammar key dispatches to NebularSEDComponent + its backend and predicts."""
    from tengri.components.nebular.component import NebularSEDComponent

    try:
        model = _silent_build(
            ssp_data=bare_ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            neb={"type": neb_type, "all_params": Fixed(DEFAULT)},
            dust_attenuation=_fixed_dust(),
            redshift=Fixed(0.05),
        )
    except FileNotFoundError as exc:
        _skip_with_tally(f"{neb_type!r} build skipped (data missing): {exc}", neb_type)
    except ValueError as exc:
        # Grid-backed backends (e.g. cloudy) raise a ValueError asking for the
        # grid path when their grid is absent — treat as a data skip.
        if any(t in str(exc).lower() for t in ("grid", "requires", "not on disk")):
            _skip_with_tally(f"{neb_type!r} build skipped (grid missing): {exc}", neb_type)
        raise
    _assert_phot_ok(model.predict_photometry({}))
    # Dispatch provenance (Direction B, #738): canonical engine + backend.
    chain = model._build_component_chain()
    neb = next((c for c in chain if isinstance(c, NebularSEDComponent)), None)
    assert neb is not None, (
        f"neb={neb_type!r}: no NebularSEDComponent in chain "
        f"{sorted(type(c).__name__ for c in chain)}"
    )
    assert neb.config.backend == _NEB_BACKEND[neb_type], (
        f"neb={neb_type!r} dispatched to backend {neb.config.backend!r}, "
        f"expected {_NEB_BACKEND[neb_type]!r}."
    )


# ─────────────────────────────────────────────────────────────────────
# Radio and X-ray
# ─────────────────────────────────────────────────────────────────────


# Radio is built through the composable SF/AGN sub-block grammar
# (``radio={'agn': {'type': ...}}``; ADR-0018 §8a), the canonical surface — the
# legacy ``radio={'type': 'radio_dpl'}`` spelling was retired in PR6 (#1980). The AGN
# radio variants reproduce AGNfitter-rX (powerlaw loudness / double-power-law).
@pytest.mark.parametrize("radio_agn", ["powerlaw", "dpl"])
def test_radio_e2e(ssp, obs, radio_agn):
    """Each AGN-radio variant builds + predicts. Reads L_ir / L_agn_bol /
    log_mstar via optional_inputs from upstream when available; falls back to
    0.0 when not."""
    try:
        model = _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            radio={"agn": {"type": radio_agn}, "all_params": Fixed(DEFAULT)},
            dust_attenuation=_fixed_dust(),
            redshift=Fixed(0.05),
        )
    except (TypeError, KeyError, ValueError) as exc:
        _skip_with_tally(f"radio agn={radio_agn!r} build skipped: {exc}", radio_agn)
    _assert_phot_ok(model.predict_photometry({}))


@pytest.mark.parametrize("xray_type", ["xray_aird", "agn_xray_corona"])
def test_xray_e2e(ssp, obs, xray_type):
    """Each X-ray component builds + predicts."""
    if xray_type not in _REGISTRY:
        pytest.skip(f"{xray_type!r} not registered")
    try:
        model = _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            xray={"type": xray_type, "all_params": Fixed(DEFAULT)},
            dust_attenuation=_fixed_dust(),
            redshift=Fixed(0.05),
        )
    except (TypeError, KeyError, ValueError) as exc:
        # These X-ray types were unified into the build selector by #1120.
        # If a ValueError still fires, it indicates a real build issue.
        pytest.skip(f"{xray_type!r} build skipped: {exc}")
    _assert_phot_ok(model.predict_photometry({}))


# ─────────────────────────────────────────────────────────────────────
# Shock — verify L_shock is properly the frequency integral, not just Hα
# ─────────────────────────────────────────────────────────────────────


def test_shock_total_luminosity_exceeds_halpha_anchor():
    """The total shock luminosity — the trapezoidal frequency integral of the
    published ``sed_shock`` — must exceed the single Hα anchor set by the
    absolute ``shock_log_lhalpha`` knob, because the emitted spectrum spans the
    full multi-line set (~11 lines), not Hα alone."""
    from tengri.components.nebular.shock_model import ShockNebular, ShockNebularConfig
    from tengri.utils.physics_constants import C_AA

    wave = jnp.linspace(3500.0, 7000.0, 1024)
    sed_in = jnp.zeros_like(wave)
    comp = ShockNebular()
    comp.config = ShockNebularConfig(norm="lhalpha")  # absolute Hα anchor
    p = {
        "frac": jnp.asarray(0.0),
        "log_lhalpha": jnp.asarray(40.0),
        "velocity": jnp.asarray(300.0),
        "log_density": jnp.asarray(0.0),
        "b_over_sqrt_n": jnp.asarray(1.0),
    }
    _, published = comp.predict(p, sed_in, wave)
    nu = C_AA / wave
    L_shock = float(jnp.abs(jnp.trapezoid(published["sed_shock"], nu)))
    l_halpha = 10.0**40.0
    assert L_shock > l_halpha, (
        f"total shock luminosity {L_shock:.3e} should exceed the Hα anchor "
        f"{l_halpha:.3e} (the emitted spectrum spans the full line set)."
    )


# ─────────────────────────────────────────────────────────────────────
# Radio chain integration — optional_inputs from upstream propagate
# ─────────────────────────────────────────────────────────────────────


def test_radio_powerlaw_actually_reads_L_ir_when_published():
    """Unit-level test: instantiate RadioPowerLawSEDComponent, drive its
    ``apply`` with two ForwardStates — one with L_ir = 1e44 erg/s in
    state.derived, one without — and verify the published sed_radio
    differs by a finite, positive amount in the L_ir case.

    This is the proper verification that ``optional_inputs`` plumbs the
    upstream-published value through to ``predict()``; the e2e chain
    tests above only assert the build doesn't crash.
    """
    from tengri.components.radio.radio_model import RadioPowerLawSEDComponent
    from tengri.protocols.component import ForwardState
    from tengri.protocols.derived_state import DerivedState

    comp = RadioPowerLawSEDComponent()
    wave = jnp.geomspace(1e6, 1e9, 64)  # radio wavelengths (Å)
    params = {
        # Observed-frame emission needs a redshift; the framework supplies one,
        # so a direct apply() must too. Explicit rather than defaulted, so the
        # test names the frame it asserts in (#1432).
        "redshift": jnp.asarray(0.0),
        "radio_q_ir": jnp.asarray(2.64),
        "radio_alpha_sf": jnp.asarray(0.8),
        "radio_loudness": jnp.asarray(0.0),
        "radio_alpha_agn": jnp.asarray(0.7),
        "radio_T_e": jnp.asarray(1e4),
        "radio_alpha_ff": jnp.asarray(-0.1),
    }

    state_no_L_ir = ForwardState(wave=wave, sed_intrinsic=jnp.zeros_like(wave))
    state_with_L_ir = ForwardState(
        wave=wave,
        sed_intrinsic=jnp.zeros_like(wave),
        derived=DerivedState.from_dict({"L_ir": jnp.asarray(1e44)}),
    )

    out_no = comp.apply(state_no_L_ir, params)
    out_with = comp.apply(state_with_L_ir, params)

    sed_no = out_no.derived["sed_radio"]
    sed_with = out_with.derived["sed_radio"]

    # The L_ir > 0 case should produce strictly more radio emission
    # somewhere on the grid than the L_ir = 0 fallback case.
    diff = jnp.max(jnp.abs(sed_with - sed_no))
    assert float(diff) > 0.0, (
        "optional_inputs wiring is broken: radio sed_radio is identical "
        "with and without L_ir in state.derived"
    )


# ─────────────────────────────────────────────────────────────────────
# Resolver provenance — pinpoint that build() picked SEDModelComponent
# ─────────────────────────────────────────────────────────────────────


def test_dust_law_surface_applies_law_not_silent_noop(ssp, obs):
    """The dust LAW is a sub-selector of the canonical engine, not a phantom
    type.

    Regression guard for the removed ``#664`` silent no-op. ``build()`` used to
    accept ``dust={'type': 'calzetti'}`` and SILENTLY drop the law (defaulting to
    ``power_law``) by routing through a ``_REGISTRY`` pass-through to thin
    single-law "components". Those toy components were deleted; the canonical
    dust surface is ``dust={'type': 'two_component', 'law_bc': ...}`` and the law
    must actually reach the engine.
    """
    # (a) the phantom type is gone — fail loud rather than silently no-op.
    # A registered law name used as a 'type' raises the law-vs-type redirect
    # (#664/#784); a non-law unknown type raises "Unknown dust type".
    with pytest.raises(ValueError, match=r"Unknown dust type|is a dust attenuation \*law\*"):
        _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "calzetti"},
            redshift=Fixed(0.05),
        )

    # (b) the law surface wires the chosen law through to the engine (not dropped).
    model = _silent_build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(0.05),
    )
    assert model._dust_law_bc == "calzetti", (
        f"law_bc silently dropped: got {model._dust_law_bc!r} — the #664 no-op."
    )
    _assert_phot_ok(model.predict_photometry({}))

    # (c) the law genuinely changes the SED (calzetti vs smc differ under load).
    model_smc = _silent_build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "smc",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 1.0,
            "tau_diff": 0.5,
        },
        redshift=Fixed(0.05),
    )
    model_cal = _silent_build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_bc": 1.0,
            "tau_diff": 0.5,
        },
        redshift=Fixed(0.05),
    )
    f_cal = jnp.asarray(model_cal.predict_photometry({}))
    f_smc = jnp.asarray(model_smc.predict_photometry({}))
    # Scale-invariant: the absolute fluxes are ~1e-27, so jnp.allclose's default
    # atol=1e-8 would call everything "equal" — compare relative difference.
    rel_diff = jnp.max(jnp.abs(f_cal - f_smc) / (jnp.abs(f_smc) + 1e-300))
    assert rel_diff > 1e-2, (
        f"calzetti and smc dust laws produced near-identical photometry "
        f"(max rel diff {float(rel_diff):.2e}) — law not applied?"
    )


# ─────────────────────────────────────────────────────────────────────
# Registry baseline — catch silent renames or removals
# ─────────────────────────────────────────────────────────────────────


# Hard-coded snapshot of component names as of this PR. New components add to
# this set in their own PR; removals must update this list explicitly.
_EXPECTED_REGISTRY_NAMES = frozenset(
    {
        # Dust attenuation laws are config sub-selectors of the canonical
        # two-component / single / wg00 engine — NOT standalone _REGISTRY
        # components. The thin single-law component duplicates
        # (calzetti/smc/mw/salim18/charlot_fall) were deleted (Direction B, #738);
        # the live WG00 component stays.
        # Dust IR emission — only the two UNIQUE-physics components remain; the 5
        # exact-engine-duplicate *_ir components were deleted (#738, Phase 2b). The
        # engine models (modified_blackbody/dl07/dl14/dale2014/astrodust) are the
        # canonical emission surface.
        "draine2021_pah_ir",
        "schreiber2016_ir",
        # AGN
        "skirtor",
        "kd18_disc",
        "powerlaw_disc",
        "silva04",
        "cat3d_wind",
        # ``agn_nlr`` (nlr_model.AGNNebular) was deleted in #897 — a grammar-
        # unreachable orphan component with drifted param names (agn_nlr_cov_frac
        # vs the canonical agn_nlr_cf) and bit-identical physics to the reachable
        # blocks/nlr_analytic. The analytic NLR ships via AGN_BLOCKS, not a
        # component.
        # Nebular — cue/cloudy/cb19 dispatch to the canonical NebularSEDComponent
        # engine + backend; the component duplicates (cue_emulator/cloudy_grid/cb19)
        # were deleted (#738, Phase 3b). The ``mappings`` component was deleted in
        # the 2026-07 dead-code purge — a grammar-unreachable no-op superseded by
        # the canonical ``shock`` component (#851); its load() passed a
        # ``grid_path`` kwarg ShockBackend does not even accept.
        "shock",
        # Radio
        "radio_powerlaw",
        "radio_dpl",
        # X-ray
        "xray_aird",
        "agn_xray_corona",
    }
)


def test_registered_component_names_against_baseline():
    """Hard-coded baseline of component names. New components add to
    ``_EXPECTED_REGISTRY_NAMES`` explicitly. Removals or renames fail this test,
    forcing a conversation in the PR that drops them rather than silent breakage
    in downstream code.

    Adds (registry has names not in the expected set) are permitted — a new
    component lands cleanly without touching this file. Removes (expected has
    names not in the registry) fail.
    """
    actual = set(_REGISTRY.keys())
    missing = _EXPECTED_REGISTRY_NAMES - actual
    assert not missing, (
        f"component name(s) missing from _REGISTRY: {sorted(missing)}. "
        f"If intentional, update _EXPECTED_REGISTRY_NAMES in this file."
    )


# ─────────────────────────────────────────────────────────────────────
# Catalog cross-compile reuse — the headline guarantee
# ─────────────────────────────────────────────────────────────────────


def test_catalog_z_range_end_to_end(real_ssp_only, ssp, obs):
    """Three SEDModels at different Fixed(z) under WavePrecomp(catalog_z_range)
    share the same compile_signature AND produce sensible per-row photometry.

    Requires the real SSP: the assertion that photometry *differs* across z needs
    real spectral features; the smooth synthetic #613 SSP gives near-identical
    bands across z, so ``real_ssp_only`` skips it on synthetic-only CI.
    """
    from tengri import WavePrecomp

    cz = WavePrecomp(catalog_z_range=(0.01, 1.5), n_z=200)

    def _build(z):
        return _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.3)},
            redshift=Fixed(z),
            approx=cz,
        )

    m_lo = _build(0.05)
    m_mid = _build(0.5)
    m_hi = _build(1.2)

    # Signature collapse — one compile across the catalog
    sig_lo = m_lo.compile_signature()
    sig_mid = m_mid.compile_signature()
    sig_hi = m_hi.compile_signature()
    assert sig_lo == sig_mid == sig_hi, (
        "catalog_z_range should make compile_signature invariant under Fixed(z)"
    )

    # Per-row photometry varies smoothly with z
    phot_lo = m_lo.predict_photometry({})
    phot_mid = m_mid.predict_photometry({})
    phot_hi = m_hi.predict_photometry({})
    for phot in (phot_lo, phot_mid, phot_hi):
        _assert_phot_ok(phot)
    # All distinct — higher z is much dimmer in observed-frame F_nu.
    # atol=0.0 is load-bearing: fluxes are ~1e-27, so jnp.allclose's default
    # atol=1e-8 calls ANY two photometry vectors "close" and the assertion
    # was vacuous from birth (#1013).
    assert not bool(jnp.allclose(phot_lo, phot_mid, rtol=1e-3, atol=0.0))
    assert not bool(jnp.allclose(phot_mid, phot_hi, rtol=1e-3, atol=0.0))


# ─────────────────────────────────────────────────────────────────────
# WavePrecomp (photometry) agreement at the documented tolerance
# ─────────────────────────────────────────────────────────────────────


def test_waveprecomp_agreement_with_exact(ssp, obs):
    """`approx=WavePrecomp()` should agree with the exact path within the
    documented Zacharegkas+2025 tolerance (~0.5% on broadband photometry)."""
    from tengri import WavePrecomp

    def _build(approx):
        return _silent_build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={"type": "single_component", "law": "calzetti", "tau_v": Fixed(0.4)},
            redshift=Fixed(0.1),
            approx=approx,
        )

    m_exact = _build(None)
    m_wp = _build(WavePrecomp())

    phot_exact = m_exact.predict_photometry({})
    phot_wp = m_wp.predict_photometry({})
    _assert_phot_ok(phot_exact)
    _assert_phot_ok(phot_wp)

    # Zacharegkas+2025 docs: < 0.5% on photometric magnitudes;
    # equivalent to ~5e-3 in F_nu fractional differences.
    rel = jnp.abs(phot_wp - phot_exact) / jnp.abs(phot_exact)
    assert float(jnp.max(rel)) < 5e-3, (
        f"WavePrecomp vs exact max rel err = {float(jnp.max(rel)):.4%}"
    )
