# SPDX-License-Identifier: BSD-3-Clause
"""Regression: #1867 — the public line surfaces must be dust-reddened.

``Prediction._ensure_lines`` computes the attenuated catalog into
``_cache["line_lums"]``; ``LineProperties.halpha`` then reads
``properties["halpha"]``, which routes through
``PropertyCatalog.__getitem__`` -> ``_halpha_fn`` -> ``_line_luminosity_helper``
-> ``state.derived["line_lums"]`` — the component's INTRINSIC publication
(pre-dust by design, ``forward/component_factory.py``). The attenuation is
computed on every access and discarded, so ``pred.lines.*`` and
``predict_properties(names=("halpha", ...))`` are both un-reddened while
documented as observed. Measured: Balmer decrement pinned at 2.7886 across
``dust_tau_diff`` 0.8 -> 2.5, where ``predict_line_fluxes(redden=True)``
correctly rose 4.3282 -> 7.3814.

This is #313 reappearing on the surface that replaced the one #313 fixed.
#313's own guard (``test_bug_a_nebular_surface.py``) never caught it because
its fixture pointed at ``data/ssp_prsc_miles_chabrier_noNE.h5``, a filename
that does not exist, so it skipped unconditionally on every checkout
including CI. This module resolves the grid through ``tengri.load_ssp()``
instead — the same grid ``download_ssp()`` fetches — so it runs wherever the
default grid is present.

The suite is built so it cannot pass vacuously. Both controls come from the
model's own code rather than from an assumption about the fixture:

    live   predict_line_fluxes(redden=True)   MUST move under the sweep
    inert  predict_line_fluxes(redden=False)  MUST stay flat
    tested pred.lines.* / predict_properties  MUST track the live one

If the live control ever goes flat the fixture stopped exercising dust, and
the tested assertions are meaningless — so it is asserted first, and its
failure is reported as a fixture fault rather than as a defect.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

import tengri
from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform
from tengri.observation.photometry import FilterCurve

HALPHA_AA = 6564.72
HBETA_AA = 4862.71

_TAU_LO = 0.8
_TAU_HI = 2.5

# A dust sweep this large must move the decrement by far more than this;
# the threshold only has to exclude "exactly flat".
_MOVED = 0.05


@pytest.fixture(scope="module")
def ssp_bare():
    """The bare-stellar grid Cue requires, resolved the way users get it.

    Deliberately NOT a hardcoded ``data/*.h5`` path: that is how #313's guard
    came to skip on every checkout for an unknown length of time.
    """
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:  # pragma: no cover - depends on checkout
        pytest.skip(f"default bare-stellar SSP not available: {exc}")


@pytest.fixture(scope="module")
def observation():
    def band(center, n=24):
        wave = np.linspace(center * 0.85, center * 1.15, n)
        trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{center:.4g}")

    return Observation(
        photometry=Photometry(filters=tuple(band(c) for c in (1500.0, 2200.0, 6200.0)))
    )


@pytest.fixture(scope="module")
def model(ssp_bare, observation):
    """Dusty model with a photoionized backend and FREE optical depths.

    ``tau_diff`` is free so the sweep is a change of parameter value on one
    model rather than a rebuild, which keeps every other sampled parameter
    identical between the two points.
    """
    return SEDModel.build(
        ssp_data=ssp_bare,
        observation=observation,
        # Constant SFH at z=0.05. A `dpl` at z=0.5 tripped
        # SFHBeforeBigBangWarning -- 54% of the stellar mass formed before the
        # Big Bang and was silently truncated, so the model under test was not
        # the one the fixture asked for. Ratios would still have moved, which
        # is what makes that class of warning easy to wave through.
        #
        # `start_gyr` is pinned inside cosmic time rather than left at its
        # Fixed(14.0) default, which is a lookback older than the universe at
        # this redshift and re-triggers the same truncation.
        sfh={
            "type": "const",
            "all_params": FIXED,
            "log_total_mass": 10.0,
            "start_gyr": 10.0,
            "end_gyr": 0.0,
        },
        dust={
            "type": "two_component",
            "law_bc": "calzetti",
            "law_diff": "calzetti",
            "tau_bc": Uniform(0.1, 1.0),
            "tau_diff": Uniform(0.1, 3.0),
        },
        neb={"type": "cue", "all_params": FIXED},
        redshift=Fixed(0.05),
    )


@pytest.fixture(scope="module")
def sweep(model):
    """The two parameter dicts, identical but for ``dust_tau_diff``."""
    lo = dict(model.spec.sample(jax.random.PRNGKey(0)))
    lo["dust_tau_bc"] = np.asarray(0.6)
    lo["dust_tau_diff"] = np.asarray(_TAU_LO)
    hi = dict(lo)
    hi["dust_tau_diff"] = np.asarray(_TAU_HI)
    return lo, hi


def _decrement_from_fluxes(model, params, *, redden):
    flux = np.asarray(
        model.predict_line_fluxes(params, target_wavelengths=[HALPHA_AA, HBETA_AA], redden=redden)
    )
    return float(flux[0] / flux[1])


def test_the_dust_sweep_moves_the_reddened_surface(model, sweep):
    """Live control. A failure here indicts the fixture, not the code."""
    lo, hi = sweep
    d_lo = _decrement_from_fluxes(model, lo, redden=True)
    d_hi = _decrement_from_fluxes(model, hi, redden=True)
    assert d_hi - d_lo > _MOVED, (
        "FIXTURE FAULT, not a defect: predict_line_fluxes(redden=True) is the "
        f"model's own reddened surface and it did not move ({d_lo} -> {d_hi}) "
        "across a tau_diff sweep of 0.8 -> 2.5. Every other assertion in this "
        "module is meaningless until this one passes."
    )


def test_the_intrinsic_surface_stays_flat(model, sweep):
    """Inert control: ``redden=False`` must be indifferent to dust."""
    lo, hi = sweep
    d_lo = _decrement_from_fluxes(model, lo, redden=False)
    d_hi = _decrement_from_fluxes(model, hi, redden=False)
    assert d_hi == pytest.approx(d_lo, rel=1e-9), (
        f"redden=False should be intrinsic and dust-independent, got {d_lo} -> {d_hi}"
    )


def test_prediction_lines_are_reddened(model, sweep):
    """#1867: ``pred.lines.*`` must track the reddened surface, not the intrinsic one."""
    lo, hi = sweep
    d_lo = float(np.asarray(model.predict(lo).lines.halpha / model.predict(lo).lines.hbeta))
    d_hi = float(np.asarray(model.predict(hi).lines.halpha / model.predict(hi).lines.hbeta))
    assert d_hi - d_lo > _MOVED, (
        f"pred.lines Balmer decrement did not move under dust ({d_lo} -> {d_hi}). "
        "_ensure_lines reddens into _cache['line_lums'], but LineProperties reads "
        "properties[...] -> state.derived['line_lums'], which is intrinsic."
    )


def test_predict_properties_lines_are_reddened(model, sweep):
    """#1867: the jit/vmap property surface must be reddened too.

    Separate from the interactive path on purpose: patching ``LineProperties``
    alone would make the test above pass and leave this one failing, which is
    the outcome to prevent — this is the surface a fit reads.
    """
    lo, hi = sweep
    names = ("halpha", "hbeta")
    p_lo = model.predict_properties(lo, names=names)
    p_hi = model.predict_properties(hi, names=names)
    d_lo = float(np.asarray(p_lo["halpha"] / p_lo["hbeta"]))
    d_hi = float(np.asarray(p_hi["halpha"] / p_hi["hbeta"]))
    assert d_hi - d_lo > _MOVED, (
        f"predict_properties Balmer decrement did not move under dust ({d_lo} -> {d_hi})"
    )


def test_prediction_lines_agree_with_predict_line_fluxes(model, sweep):
    """The two public surfaces must agree on the ratio, at both sweep points.

    Ratios rather than absolutes: ``predict_line_fluxes`` returns flux and
    ``pred.lines`` returns luminosity, so they differ by 4*pi*d_L^2, which
    cancels here.
    """
    lo, hi = sweep
    for label, params in (("tau_lo", lo), ("tau_hi", hi)):
        pred = model.predict(params)
        d_lines = float(np.asarray(pred.lines.halpha / pred.lines.hbeta))
        d_fluxes = _decrement_from_fluxes(model, params, redden=True)
        assert d_lines == pytest.approx(d_fluxes, rel=1e-6), (
            f"[{label}] pred.lines decrement {d_lines} disagrees with "
            f"predict_line_fluxes(redden=True) {d_fluxes}; both are documented "
            "as observed and single-sourced through _attenuate_line_catalog"
        )


def test_the_published_balmer_decrement_property_rises(model, sweep):
    """#1867: the ``balmer_decrement`` PROPERTY must move, not just a hand ratio.

    Distinct from the assertions above, which divide ``halpha`` by ``hbeta``
    themselves. The published diagnostics — ``balmer_decrement``, ``bpt_nii``,
    ``r23``, ``o32`` — come from ``_line_lums_for_ratios``, a *different*
    reader of the catalog. The first draft of this fix updated the line
    accessors and left that one intrinsic, so every hand-computed ratio here
    went green while the property a user actually reads stayed pinned at its
    Case-B value. Asserting the hand ratio is not asserting the property.
    """
    lo, hi = sweep
    d_lo = float(
        np.asarray(model.predict_properties(lo, names=("balmer_decrement",))["balmer_decrement"])
    )
    d_hi = float(
        np.asarray(model.predict_properties(hi, names=("balmer_decrement",))["balmer_decrement"])
    )
    assert d_hi - d_lo > _MOVED, (
        f"published balmer_decrement did not move under dust ({d_lo} -> {d_hi}); "
        "_line_lums_for_ratios is still reading the intrinsic catalog"
    )


def test_the_log_companion_follows_its_linear_sibling(model, sweep):
    """#1867: ``log_halpha`` must stay ``log10(halpha)`` once dust is applied.

    Line luminosities are ~1e41 erg/s, past float32's ceiling, so the catalog
    carries a ``log_line_lums`` companion (#1534). Reddening the linear array
    and not the companion makes them disagree — which the first draft of this
    fix did, and which
    ``tests/regression/precision/test_log_line_properties.py`` caught. That is
    the same defect class as #1867 itself: a value updated in one place and not
    in its sibling. Pinned here too, against a dusty model specifically, since
    the precision test does not sweep dust.
    """
    lo, hi = sweep
    for label, params in (("tau_lo", lo), ("tau_hi", hi)):
        out = model.predict_properties(params, names=("halpha", "hbeta", "log_halpha"))
        linear = float(np.asarray(out["halpha"]))
        log_companion = float(np.asarray(out["log_halpha"]))
        assert log_companion == pytest.approx(np.log10(linear), rel=1e-9), (
            f"[{label}] log_halpha = {log_companion} but log10(halpha) = "
            f"{np.log10(linear)}; the companion is not the log of its linear sibling"
        )


def test_the_log_companion_is_itself_reddened(model, sweep):
    """#1867: ``log_halpha`` must respond to dust, not merely agree with ``halpha``.

    Agreement alone is satisfiable by leaving BOTH intrinsic, so the previous
    test cannot stand on its own.
    """
    lo, hi = sweep
    v_lo = float(np.asarray(model.predict_properties(lo, names=("log_halpha",))["log_halpha"]))
    v_hi = float(np.asarray(model.predict_properties(hi, names=("log_halpha",))["log_halpha"]))
    assert v_lo - v_hi > 0.05, (
        f"log_halpha did not dim under dust ({v_lo} -> {v_hi} dex); "
        "the log catalog is still intrinsic"
    )
