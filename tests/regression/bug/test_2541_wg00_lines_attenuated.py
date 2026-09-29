# SPDX-License-Identifier: BSD-3-Clause
"""Regression: #2541 — WG00 attenuation reaches every emission-line surface.

``SEDModel._line_dust_component`` used to select the chain's dust component by
NAME (``"dust"`` / ``"dust_attenuation"``); WG00's component is named
``"wg00_attenuation"``, so it was never selected. Every catalog-line surface
(``predict_line_fluxes``, ``predict_line_ratios``, the ``.lines`` / BPT
properties, ``predict_emission_lines``) then passed lines through UNATTENUATED
under ``dust_attenuation={'type': 'wg00'}``, while the continuum and the
SED-measured lines (``measure_line_fluxes`` exact path,
``predict_spectral_indices``) WERE attenuated -- every reference code (FSPS
``dust_type=3``, Bagpipes, CIGALE, Synthesizer) attenuates lines under
WG00-type laws. The fix selects the dust component by CAPABILITY
(``hasattr(component, "attenuate_line_catalog")``) instead of by name, and
gives WG00 an ``attenuate_line_catalog`` that reuses the identical
interpolation closure / ``tau_v`` the continuum path uses, publishing
``log_line_lums_attenuated`` the same way ``single_component`` /
``two_component`` already did.

This module was previously a weak regression guard: it only checked that
``predict_line_fluxes`` "responds to tau_v", which a wrong sign, a wrong
curve, or a missing unit-conversion factor would all still satisfy. It is
rewritten here to pin the EXACT transmission, not just its direction, per
``.claude/jobs/4fa62888/tmp/brief_2541.md`` Sec. "Tests":

1. A class sweep over every attenuator type (``single_component``,
   ``two_component``, ``wg00``) asserting, for EVERY line in the catalog,
   that ``predict_line_fluxes(dust on) / predict_line_fluxes(dust off)``
   equals the attenuator's own transmission curve evaluated independently
   (never through ``attenuate_line_catalog`` itself, which is the method
   under test).
2. WG00: ``predict_line_ratios``, the ``.lines`` / ``balmer_decrement``
   property, and the deprecated ``predict_emission_lines`` all show
   attenuation and agree with ``predict_line_fluxes``.
3. WG00 joint-fit consistency: the Hα dust-on/dust-off ratio from
   ``predict_line_fluxes`` (the discrete-catalog path) must equal the same
   ratio measured off the model spectrum by ``measure_line_fluxes`` (the
   continuum-subtraction path) -- the two paths disagreed under wg00 before
   this fix, which is the joint-fit inconsistency #2541 reports.

WG00-specific subtlety the tests below account for: the WG00 grid tabulates
``A(lambda; tau_v)`` only over ``tau_v in [0.25, 10.0]`` and CLAMPS at the
edges (``wg00.py::create_wg00_from_grid``), so ``dust_tau_v=0`` does NOT mean
zero attenuation for wg00 -- it clamps to the ``tau_v=0.25`` curve. The
"dust off" reference used throughout is therefore the SAME grid closure
evaluated at ``tau_v=0`` (which clamps identically inside and outside the
model), never an assumed transmission of 1.
"""

from __future__ import annotations

import warnings

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.components.dust.attenuation import calzetti
from tengri.components.dust.wg00 import _find_wg00_grid, create_wg00_from_grid
from tengri.observation import LineRatioData
from tengri.observation.line_measurement import DESI_LINES
from tengri.observation.photometry import FilterCurve

HALPHA_AA = 6564.72
HBETA_AA = 4862.71

# Independent WG00 grid closure -- built the same way
# ``WG00AttenuationSEDComponent._build_curve_fn`` builds its own (same grid
# file, same structural selectors: the model below never overrides
# dust_curve/geometry/structure from their WG00AttenuationSEDComponentConfig
# defaults), but constructed here from the PUBLIC loader, never by reaching
# into the live component, so it is a genuinely independent reference curve.
_HALPHA_LINEDEF = next(ld for ld in DESI_LINES if ld.name == "Halpha")

# Exact-equality tolerance for the class-sweep transmission checks (#2541
# Sec. "Tests" item 1). Every comparison here is two independent NumPy/JAX
# evaluations of the SAME deterministic closure (Calzetti polynomial or the
# WG00 triweight grid interpolant) at the SAME (wave, tau) inputs, so
# agreement is expected to floating-point precision; measured deviation
# across all 138 catalog lines for single_component, two_component and wg00
# was <= 1.8e-14 (relative) during development of this test. rtol=1e-6 is
# generous headroom above that measured noise floor, not a physics
# tolerance.
_CURVE_RTOL = 1e-6

# Measured tolerance for the predict_line_fluxes vs measure_line_fluxes
# joint-fit consistency check (#2541 Sec. "Tests" item 3). measure_line_fluxes
# estimates a local continuum from side-bands and subtracts it (a
# spectroscopic-pipeline-style operator), which is NOT bit-identical to the
# discrete-catalog path predict_line_fluxes reads, even though the two now
# apply the same dust screen. Measured relative disagreement between the two
# surfaces' Halpha dust-on/dust-off ratios, at dust_tau_v swept over
# {0.3, 1.0, 2.0, 3.0} against the module's tau_v=2.5 fixture point: 1.3e-5 to
# 5.3e-4, rising with tau_v. _JOINT_FIT_RTOL is set just above the measured
# ceiling at this module's tau_v=2.5 (measured 4.2487e-4).
_JOINT_FIT_RTOL = 1e-3


def _band(center, n=24):
    """Simple transmission filter for testing."""
    wave = np.linspace(center * 0.85, center * 1.15, n)
    trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")


@pytest.fixture(scope="module")
def ssp_bare():
    """Load the bare-stellar grid Cue requires."""
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


@pytest.fixture(scope="module")
def observation():
    """Observation with filters for testing."""
    return Observation(
        photometry=Photometry(filters=tuple(_band(c) for c in (1500.0, 2200.0, 6200.0)))
    )


def _build(ssp_bare, observation, dust_attenuation):
    """A const-SFH, Cue-nebular model at z=0.05 with the given dust config."""
    return SEDModel.build(
        ssp_data=ssp_bare,
        observation=observation,
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        dust_attenuation=dust_attenuation,
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.05),
    )


@pytest.fixture(scope="module")
def model_single(ssp_bare, observation):
    """single_component (Calzetti), the simplest law: no shape parameters."""
    return _build(
        ssp_bare,
        observation,
        {"type": "single_component", "law": "calzetti", "tau_v": Uniform(0.1, 3.0)},
    )


@pytest.fixture(scope="module")
def model_two(ssp_bare, observation):
    """two_component with the DEFAULT nebular_screen ("birth_cloud").

    Both screens use Calzetti (no shape parameters) so the documented
    birth-cloud line screen -- T = exp(-(tau_bc + tau_diff) * k(lambda)),
    ``two_component.py::_screen_transmission`` at ``f_obscuration=0`` --
    reduces to a single closed form independently computable per line,
    letting item 1's exact-equality bar apply to two_component too, not just
    the "ratio < 1, decreasing toward the blue" fallback the brief allows
    when no independent closed form is available.
    """
    return _build(
        ssp_bare,
        observation,
        {
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Uniform(0.1, 1.0),
            "tau_diff": Uniform(0.1, 3.0),
        },
    )


@pytest.fixture(scope="module")
def model_wg00(ssp_bare, observation):
    """WG00 dust model with Cue nebular backend (default mw/shell/homogeneous)."""
    return _build(ssp_bare, observation, {"type": "wg00", "tau_v": Uniform(0.1, 3.0)})


def _sample_params(model, key=None, **overrides):
    """Sample parameters from the model spec, with scalar overrides applied."""
    if key is None:
        key = jax.random.PRNGKey(0)
    params = dict(model.spec.sample(key))
    for name, value in overrides.items():
        params[name] = np.asarray(value)
    return params


def _line_fluxes_ratio(model, params, wave_num, wave_den, redden=True):
    """Compute ratio of line fluxes for two wavelengths."""
    fluxes = np.asarray(
        model.predict_line_fluxes(params, target_wavelengths=[wave_num, wave_den], redden=redden)
    )
    return float(fluxes[0] / fluxes[1])


def _catalog_ratio(model, params_on, params_off):
    """predict_line_fluxes(dust on) / predict_line_fluxes(dust off), every catalog line.

    ``target_wavelengths=None`` returns the FULL backend catalog in
    ``state.derived['line_waves']`` order (no target-matching reindex), so
    the returned ratio array aligns 1:1 with ``line_waves`` from either
    state (the two params dicts share every non-dust key).
    """
    flux_on = np.asarray(
        model.predict_line_fluxes(params_on, target_wavelengths=None, redden=True)
    )
    flux_off = np.asarray(
        model.predict_line_fluxes(params_off, target_wavelengths=None, redden=True)
    )
    return flux_on / flux_off


class TestClassSweepEveryLineMatchesTheAttenuatorsOwnCurve:
    """#2541 Sec. "Tests" item 1: CLASS SWEEP over every dust attenuator type.

    For each attenuator, every catalog line's dust-on/dust-off flux ratio
    must equal that attenuator's own transmission curve, evaluated by an
    INDEPENDENT computation outside the pipeline (never by calling
    ``attenuate_line_catalog``, the method the fix under test lives in).
    """

    def test_single_component_matches_calzetti_at_every_line(self, model_single):
        params_on = _sample_params(model_single, dust_tau_v=2.5)
        # SAME base params dict; only dust_tau_v differs, matching the "same
        # params" contract in the brief.
        params_off = dict(params_on)
        params_off["dust_tau_v"] = np.asarray(0.0)

        line_waves = np.asarray(model_single.predict_state(params_on).derived["line_waves"])
        ratio = _catalog_ratio(model_single, params_on, params_off)

        tau_v = float(params_on["dust_tau_v"])
        # calzetti() is the law's public curve k(lambda); single_component's
        # own attenuate_line_catalog computes exactly
        # log_lums - tau_v * k(lambda) * log10(e), i.e. a linear-space factor
        # of exp(-tau_v * k(lambda)) -- reproduced here from the law function
        # directly, not from attenuate_line_catalog.
        expected = np.exp(-tau_v * np.asarray(calzetti(line_waves)))

        assert expected.min() < 0.5, (
            "FIXTURE FAULT: tau_v=2.5 Calzetti should attenuate the bluest "
            f"catalog lines by more than half; got min transmission {expected.min()}"
        )
        np.testing.assert_allclose(
            ratio,
            expected,
            rtol=_CURVE_RTOL,
            err_msg="single_component predict_line_fluxes ratio disagrees with "
            "the Calzetti curve evaluated directly at the catalog line wavelengths",
        )

    def test_two_component_matches_birth_cloud_screen_at_every_line(self, model_two):
        params_on = _sample_params(model_two, dust_tau_bc=0.6, dust_tau_diff=2.0)
        params_off = dict(params_on)
        params_off["dust_tau_bc"] = np.asarray(0.0)
        params_off["dust_tau_diff"] = np.asarray(0.0)

        line_waves = np.asarray(model_two.predict_state(params_on).derived["line_waves"])
        ratio = _catalog_ratio(model_two, params_on, params_off)

        tau_bc = float(params_on["dust_tau_bc"])
        tau_diff = float(params_on["dust_tau_diff"])
        k = np.asarray(calzetti(line_waves))
        # Default nebular_screen="birth_cloud": _screen_transmission's
        # birth_cloud branch is f_obsc + (1-f_obsc)*exp(-(tau_bc*k_bc +
        # tau_diff*k_diff)); dust_f_obscuration defaults to 0, and law_bc ==
        # law_diff == calzetti here, so k_bc == k_diff == k.
        expected = np.exp(-(tau_bc + tau_diff) * k)

        assert expected.min() < 0.1, (
            "FIXTURE FAULT: this tau sweep should attenuate the bluest lines "
            f"by more than 90%; got min transmission {expected.min()}"
        )
        np.testing.assert_allclose(
            ratio,
            expected,
            rtol=_CURVE_RTOL,
            err_msg="two_component predict_line_fluxes ratio disagrees with its "
            "documented birth_cloud line screen evaluated by hand",
        )

    def test_wg00_matches_grid_transmission_at_every_line(self, model_wg00):
        params_on = _sample_params(model_wg00, dust_tau_v=2.5)
        params_off = dict(params_on)
        params_off["dust_tau_v"] = np.asarray(0.0)

        line_waves = np.asarray(model_wg00.predict_state(params_on).derived["line_waves"])
        ratio = _catalog_ratio(model_wg00, params_on, params_off)

        # Independent reference curve, loaded from the same vendored grid
        # via the PUBLIC loader (not the live component's cached closure).
        # WG00AttenuationSEDComponentConfig defaults are dust_curve="mw",
        # geometry="shell", structure="homogeneous", matching model_wg00
        # (no override passed to SEDModel.build).
        wg00_fn = create_wg00_from_grid(
            _find_wg00_grid(), dust_curve="mw", geometry="shell", structure="homogeneous"
        )
        tau_on = float(params_on["dust_tau_v"])
        tau_off = float(params_off["dust_tau_v"])
        # NOT exp(-A(tau_on)) alone: the WG00 grid tabulates tau_v only over
        # [0.25, 10.0] and clamps at the edges, so tau_v=0 clamps to the
        # tau_v=0.25 curve rather than giving zero attenuation. The correct
        # "dust off" reference is the SAME closure evaluated at tau_v=0,
        # exactly as predict_line_fluxes computes it internally for
        # params_off -- confirmed empirically: a_off ranged [0.0, 1.002]
        # across the catalog, not identically 0.
        a_on = np.asarray(wg00_fn(line_waves, tau_on))
        a_off = np.asarray(wg00_fn(line_waves, tau_off))
        expected = np.exp(-(a_on - a_off))

        assert not np.allclose(a_off, 0.0), (
            "FIXTURE INVARIANT VIOLATED: expected the wg00 grid's tau_v=0 clamp "
            "to be non-zero at the catalog wavelengths (documents the grid's "
            "[0.25, 10.0] range); if this ever becomes all-zero the 'dust off' "
            "reference here needs revisiting, not silently accepted."
        )
        assert expected.min() < 0.9, (
            "FIXTURE FAULT: tau_v 0.0->2.5 under wg00 should attenuate the "
            f"bluest lines measurably; got min transmission {expected.min()}"
        )
        np.testing.assert_allclose(
            ratio,
            expected,
            rtol=_CURVE_RTOL,
            err_msg="wg00 predict_line_fluxes ratio disagrees with the WG00 grid "
            "transmission evaluated directly (independent of attenuate_line_catalog)",
        )


class TestWG00SurfacesAgreeAndShowAttenuation:
    """#2541 Sec. "Tests" item 2: every wg00 line surface sees the same dust.

    predict_line_ratios, the interactive `.lines` / `balmer_decrement`
    property, and the deprecated predict_emission_lines must all (a) change
    with dust_tau_v and (b) agree with predict_line_fluxes at both sweep
    points -- not just move in the same direction.
    """

    @pytest.fixture(scope="class")
    def sweep(self, model_wg00):
        lo = _sample_params(model_wg00, dust_tau_v=0.5)
        hi = dict(lo)
        hi["dust_tau_v"] = np.asarray(2.5)
        return lo, hi

    def test_predict_line_ratios_agrees_with_predict_line_fluxes(self, model_wg00, sweep):
        lo, hi = sweep
        lrd = LineRatioData.from_dict({("Halpha", "Hbeta"): (1.0, 1.0)})
        r_lo = float(np.asarray(model_wg00.predict_line_ratios(lo, lrd))[0])
        r_hi = float(np.asarray(model_wg00.predict_line_ratios(hi, lrd))[0])
        assert r_hi > r_lo, (
            f"wg00 predict_line_ratios Balmer decrement did not increase with "
            f"tau_v: {r_lo} -> {r_hi}"
        )
        for label, params, r in (("lo", lo, r_lo), ("hi", hi, r_hi)):
            d_fluxes = _line_fluxes_ratio(model_wg00, params, HALPHA_AA, HBETA_AA)
            assert r == pytest.approx(d_fluxes, rel=_CURVE_RTOL), (
                f"[{label}] wg00 predict_line_ratios {r} disagrees with "
                f"predict_line_fluxes decrement {d_fluxes}"
            )

    def test_lines_balmer_decrement_agrees_with_predict_line_fluxes(self, model_wg00, sweep):
        lo, hi = sweep
        d_lo = float(np.asarray(model_wg00.predict(lo).lines.balmer_decrement))
        d_hi = float(np.asarray(model_wg00.predict(hi).lines.balmer_decrement))
        assert d_hi > d_lo, (
            f"wg00 .lines.balmer_decrement did not increase with tau_v: {d_lo} -> {d_hi}"
        )
        for label, params, d in (("lo", lo, d_lo), ("hi", hi, d_hi)):
            d_fluxes = _line_fluxes_ratio(model_wg00, params, HALPHA_AA, HBETA_AA)
            assert d == pytest.approx(d_fluxes, rel=_CURVE_RTOL), (
                f"[{label}] wg00 .lines.balmer_decrement {d} disagrees with "
                f"predict_line_fluxes decrement {d_fluxes}"
            )

    def test_predict_properties_balmer_decrement_agrees_with_predict_line_fluxes(
        self, model_wg00, sweep
    ):
        lo, hi = sweep
        p_lo = float(
            np.asarray(
                model_wg00.predict_properties(lo, names=("balmer_decrement",))["balmer_decrement"]
            )
        )
        p_hi = float(
            np.asarray(
                model_wg00.predict_properties(hi, names=("balmer_decrement",))["balmer_decrement"]
            )
        )
        assert p_hi > p_lo, (
            f"wg00 predict_properties balmer_decrement did not increase with "
            f"tau_v: {p_lo} -> {p_hi}"
        )
        for label, params, p in (("lo", lo, p_lo), ("hi", hi, p_hi)):
            d_fluxes = _line_fluxes_ratio(model_wg00, params, HALPHA_AA, HBETA_AA)
            assert p == pytest.approx(d_fluxes, rel=_CURVE_RTOL), (
                f"[{label}] wg00 predict_properties balmer_decrement {p} disagrees "
                f"with predict_line_fluxes decrement {d_fluxes}"
            )

    def test_predict_emission_lines_agrees_with_predict_line_fluxes(self, model_wg00, sweep):
        lo, hi = sweep
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", DeprecationWarning)
            el_lo = model_wg00.predict_emission_lines(lo)
            el_hi = model_wg00.predict_emission_lines(hi)
        d_lo = float(np.asarray(el_lo.halpha / el_lo.hbeta))
        d_hi = float(np.asarray(el_hi.halpha / el_hi.hbeta))
        assert d_hi > d_lo, (
            f"wg00 predict_emission_lines Balmer decrement did not increase with "
            f"tau_v: {d_lo} -> {d_hi}"
        )
        for label, params, d in (("lo", lo, d_lo), ("hi", hi, d_hi)):
            d_fluxes = _line_fluxes_ratio(model_wg00, params, HALPHA_AA, HBETA_AA)
            assert d == pytest.approx(d_fluxes, rel=_CURVE_RTOL), (
                f"[{label}] wg00 predict_emission_lines decrement {d} disagrees "
                f"with predict_line_fluxes decrement {d_fluxes}"
            )


class TestWG00FastNebularFallbackUsesCapabilitySelection:
    """The actual defect surface: ``_line_dust_component``'s selection.

    Under the normal (stateful) forward pass, WG00's own ``apply()`` now
    publishes ``log_line_lums_attenuated`` directly (fix part 2), so
    ``predict_line_fluxes``'s redden branch reads that published key and
    never calls ``_attenuate_line_catalog`` / ``_line_dust_component`` at
    all -- confirmed empirically: reverting ONLY ``_line_dust_component`` to
    the old name-based selection left every test above GREEN, because the
    state path no longer depends on it. ``_line_dust_component`` (fix part
    3) is reached ONLY on the no-state fallback: ``enable_fast_nebular()``'s
    ``predict_line_fluxes`` (#950) and the deprecated
    ``predict_emission_lines`` when no attenuated catalog was published. This
    is the surface ``tests/regression/bug/test_bug_2223_line_screen_kwargs.py``
    tests for two_component; this class is its wg00 analog, and the one test
    in this module that actually fails when ``_line_dust_component`` is
    reverted (see this module's RED/GREEN verification in the PR history).
    """

    def test_fallback_is_exercised_and_matches_the_grid_curve(
        self, ssp_bare, observation, monkeypatch
    ):
        model = _build(ssp_bare, observation, {"type": "wg00", "tau_v": Uniform(0.1, 3.0)})
        target_wavelengths = np.asarray([HBETA_AA, HALPHA_AA])
        model.enable_fast_nebular(target_wavelengths, n_grid=2)

        calls: list[int] = []
        original = type(model)._attenuate_line_catalog

        def _spy(self, params, line_waves, line_lums):
            calls.append(1)
            return original(self, params, line_waves, line_lums)

        monkeypatch.setattr(type(model), "_attenuate_line_catalog", _spy)

        params = _sample_params(model, dust_tau_v=2.5)
        atten = np.asarray(model.predict_line_fluxes(params, redden=True))
        intr = np.asarray(model.predict_line_fluxes(params, redden=False))

        assert calls, (
            "predict_line_fluxes on the enable_fast_nebular() path (no state) "
            "did not call _attenuate_line_catalog; this test is not exercising "
            "the surface _line_dust_component's capability selection guards."
        )

        transmission = atten / intr
        # #2235 snaps each requested target within 0.5 A of a true catalog
        # line to that line's own tabulated wavelength, so the oracle below
        # reads the grid's ACTUAL wavelengths, not HBETA_AA/HALPHA_AA.
        grid_wave = np.asarray(model._nebular_grid_table.wavelengths)
        wg00_fn = create_wg00_from_grid(
            _find_wg00_grid(), dust_curve="mw", geometry="shell", structure="homogeneous"
        )
        # redden=False is the TRUE intrinsic catalog here (no state-carried
        # clamped-tau subtlety: the fallback's attenuated/intrinsic ratio is
        # simply exp(-A(lambda; tau_v)), the grid curve evaluated once).
        expected = np.exp(-np.asarray(wg00_fn(grid_wave, float(params["dust_tau_v"]))))

        assert expected.max() < 0.9, (
            "FIXTURE FAULT: tau_v=2.5 should attenuate Hbeta/Halpha measurably "
            f"on the fast-nebular path; got max transmission {expected.max()}"
        )
        np.testing.assert_allclose(
            transmission,
            expected,
            rtol=_CURVE_RTOL,
            err_msg="enable_fast_nebular() wg00 line transmission disagrees with "
            "the WG00 grid curve evaluated directly; _line_dust_component is not "
            "resolving wg00's attenuate_line_catalog on this no-state path",
        )


class TestWG00JointFitConsistency:
    """#2541 Sec. "Tests" item 3: predict_line_fluxes vs measure_line_fluxes.

    The defect broke exactly this consistency: measure_line_fluxes reads the
    dust-attenuated CONTINUUM (which wg00 always reddened correctly), while
    predict_line_fluxes read the discrete catalog (which wg00 did not). A fit
    combining a line-flux channel with a spectroscopy channel under wg00
    would fit two different dust laws to the same tau_v.
    """

    def test_halpha_dust_ratio_agrees_between_predict_and_measure(self, model_wg00):
        params_on = _sample_params(model_wg00, dust_tau_v=2.5)
        params_off = dict(params_on)
        params_off["dust_tau_v"] = np.asarray(0.0)

        flux_on = float(
            np.asarray(model_wg00.predict_line_fluxes(params_on, target_wavelengths=[HALPHA_AA]))[
                0
            ]
        )
        flux_off = float(
            np.asarray(model_wg00.predict_line_fluxes(params_off, target_wavelengths=[HALPHA_AA]))[
                0
            ]
        )
        ratio_predict = flux_on / flux_off

        measured_on = float(
            np.asarray(model_wg00.measure_line_fluxes(params_on, [_HALPHA_LINEDEF], approx=False))[
                0
            ]
        )
        measured_off = float(
            np.asarray(
                model_wg00.measure_line_fluxes(params_off, [_HALPHA_LINEDEF], approx=False)
            )[0]
        )
        ratio_measure = measured_on / measured_off

        assert ratio_predict < 0.9, (
            "FIXTURE FAULT: tau_v 0.0->2.5 under wg00 should attenuate Halpha "
            f"measurably; got predict_line_fluxes ratio {ratio_predict}"
        )
        assert ratio_measure == pytest.approx(ratio_predict, rel=_JOINT_FIT_RTOL), (
            f"wg00 joint-fit inconsistency: predict_line_fluxes Halpha "
            f"dust-on/dust-off ratio {ratio_predict} disagrees with the same "
            f"ratio from measure_line_fluxes (exact path) {ratio_measure} by more "
            f"than the measured continuum-subtraction noise floor "
            f"(rtol={_JOINT_FIT_RTOL})"
        )
