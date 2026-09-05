# SPDX-License-Identifier: BSD-3-Clause
"""Shock emission under ``approx=WavePrecomp()`` (#1375).

Any model with a ``shock`` component raised ``ComponentIOError`` the moment the
photometry LUT ran — i.e. on every inference call, since ``WavePrecomp`` is the
inference hot path. ``SEDModelComponent`` publishes
``f"{self.name}_phot_lnu_precomp"`` generically, ``DerivedState`` carried no
``shock_`` field, and the write fell through to the untyped ``_extras``
spillover that the ADR-0007 guard rejects.

The fix that #1375 proposed — add the field, add a ``_CANONICAL_UNITS`` entry —
stops the crash and replaces it with something worse: an exact no-op. The
inherited ``predict_precomp`` builds its LUT by calling ``predict`` on a dummy
SED of *zeros*, so the ``norm="frac"`` anchor ``frac * max(1e-3 * L_bol, 1e-30)``
collapses onto its epsilon guard and publishes ~1e-44 erg/s/Hz against a true
~1e29. These tests pin all three properties the real fix has to deliver: it does
not raise, it is not a no-op, and its filter integration is *exact*.

The tests are ordered so a regression tells you which half broke — a failure in
``test_lut_matches_exact_without_dust`` means the integration itself drifted; a
failure in ``test_shock_changes_lut_photometry`` means the contribution was
dropped again.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

SHOCK_FRAC = 0.5
REDSHIFT = 0.1


def _build(ssp, obs, *, shock, approx, tau_bc=0.5):
    """Model with everything fixed except the two knobs under test."""
    from tengri import DEFAULT, Fixed, SEDModel

    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "const", "all_params": Fixed(DEFAULT), "log_total_mass": 10.0},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": Fixed(DEFAULT),
            "tau_bc": tau_bc,
            "tau_diff": tau_bc * 0.4,
        },
        redshift=Fixed(REDSHIFT),
        approx=approx,
    )
    if shock:
        kwargs["shock"] = {"frac": SHOCK_FRAC}
    return SEDModel.build(**kwargs)


def _photometry(model):
    import jax

    return np.asarray(model.predict_photometry(dict(model.spec.sample(jax.random.PRNGKey(0)))))


@pytest.fixture
def wave_precomp():
    from tengri import WavePrecomp

    return WavePrecomp()


def test_shock_model_has_shock_params(ssp_data_wne, synthetic_tophat_obs):
    """Setup guard: the shock group really registers parameters.

    Without this every other test in the file could pass by building a model
    with no shock component at all.
    """
    model = _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=None)
    names = set(model.spec.get_fixed_values()) | set(model.spec.free_params)
    assert any(n.startswith("shock_") for n in names), (
        f"no shock_* parameters in spec; the shock group did not register: {sorted(names)}"
    )


def test_shock_under_waveprecomp_does_not_raise(ssp_data_wne, synthetic_tophat_obs, wave_precomp):
    """The original #1375 symptom: ComponentIOError from the _extras spillover."""
    model = _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=wave_precomp)
    flux = _photometry(model)
    assert np.all(np.isfinite(flux)), f"non-finite photometry: {flux}"


def test_shock_publishes_typed_fields_not_extras(ssp_data_wne, synthetic_tophat_obs, wave_precomp):
    """ADR-0007: the LUT must land on typed DerivedState fields.

    Asserting ``_extras`` is empty is the direct inverse of the guard that
    raised; asserting the key is *present* stops a future refactor from
    satisfying the guard by simply not publishing anything.
    """
    import jax

    model = _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=wave_precomp)
    state = model.predict_state(dict(model.spec.sample(jax.random.PRNGKey(0))))

    assert not getattr(state.derived, "_extras", {}), (
        f"untyped spillover is non-empty: {list(state.derived._extras)}"
    )
    for key in ("shock_phot_lnu_precomp", "shock_restband_lnu_precomp"):
        assert key in state.derived, f"{key} was never published"
        assert np.all(np.isfinite(np.asarray(state.derived[key])))


def test_shock_changes_lut_photometry(ssp_data_wne, synthetic_tophat_obs, wave_precomp):
    """The regression that the #1375 patch would have shipped: a silent no-op.

    Before the fix this difference was exactly 0.0000% — the published LUT was
    ``frac * 1e-30``. The exact path moves by ~1.8% on the same model, so the
    threshold here is deliberately far below that: this test is about "did the
    contribution survive at all", not about its size.
    """
    on = _photometry(_build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=wave_precomp))
    off = _photometry(_build(ssp_data_wne, synthetic_tophat_obs, shock=False, approx=wave_precomp))

    frac_diff = np.max(np.abs(on - off) / np.abs(off))
    assert frac_diff > 1e-4, (
        f"shock changed LUT photometry by {frac_diff:.3e} — it is being dropped. "
        "This is the exact-no-op failure mode of #1375."
    )


def test_shock_lut_matches_exact_without_dust(ssp_data_wne, synthetic_tophat_obs, wave_precomp):
    """With no dust screen the LUT must reproduce the exact path essentially exactly.

    This isolates the filter integration from the screen approximation. The LUT
    publishes an *intrinsic* rest-frame L_nu, so with ``tau = 0`` there is
    nothing for ``predict_via_precomp`` to approximate and the two paths agree
    to roundoff. A regression here means the integration itself broke; the
    dust-on residual is a separate, documented approximation.
    """
    exact = _photometry(
        _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=None, tau_bc=0.0)
    )
    lut = _photometry(
        _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=wave_precomp, tau_bc=0.0)
    )

    rel = np.max(np.abs(lut - exact) / np.abs(exact))
    assert rel < 1e-6, f"shock LUT vs exact (no dust) = {rel:.3e}, expected roundoff-level"


def test_restband_is_a_separate_integral(ssp_data_wne, synthetic_tophat_obs, wave_precomp):
    """The rest band is its own integral at z=0, not the observed value reused.

    Reusing the observed-band number is what made the nebular LUT read 769 %
    high in des_g at z=0.5 (#1148); at z=0.1 the two differ modestly but must
    not be bit-identical.
    """
    import jax

    model = _build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=wave_precomp)
    state = model.predict_state(dict(model.spec.sample(jax.random.PRNGKey(0))))

    observed = np.asarray(state.derived["shock_phot_lnu_precomp"])
    restband = np.asarray(state.derived["shock_restband_lnu_precomp"])
    assert not np.array_equal(observed, restband), (
        "shock rest-band LUT is bit-identical to the observed band — it is being "
        "reused rather than integrated at z=0"
    )


def test_exact_path_still_sees_shock(ssp_data_wne, synthetic_tophat_obs):
    """The fix must not disturb ``approx=None``, which was always correct."""
    on = _photometry(_build(ssp_data_wne, synthetic_tophat_obs, shock=True, approx=None))
    off = _photometry(_build(ssp_data_wne, synthetic_tophat_obs, shock=False, approx=None))

    frac_diff = np.max(np.abs(on - off) / np.abs(off))
    assert frac_diff > 1e-4, f"shock does not affect the exact path either ({frac_diff:.3e})"
