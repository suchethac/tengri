# SPDX-License-Identifier: BSD-3-Clause
r"""``approx=None`` must force the exact path on the batch surfaces too (#2377).

``Fitter._resolve_fit_approx`` documents ``None`` as "force the exact wave-grid path
(overrides a build-time approx)" and "``None`` means exact and stays exact". Its
batch mirror, ``_resolve_batch_fit_approx``, read ``approx is None`` as "leave the
model untouched" and returned it as given -- so a catalog or population fit on a
model built with ``approx=(WavePrecomp(), FeaturePrecomp())`` kept **both** tables on
after the caller asked for the exact path in its documented spelling. Measured:

============================================  ==========================
resolver, on a model built with the pair      ``approx=None`` gave
============================================  ==========================
``Fitter._resolve_fit_approx``                wave False, feature False
``_resolve_batch_fit_approx``                 wave **True**, feature **True**
============================================  ==========================

One word meant two opposite things depending on which fitter you reached for, and
the surface that kept the approximation is the one whose fits are largest.

This is not a speed regression to guard against. It is the contract being honored,
and it is what makes ``PrecompBiasWarning`` actionable: that warning tells the reader
"for final inference at this SNR, rerun with approx=None (the exact path)", advice
these surfaces previously ignored. #1671 is about ``WavePrecomp``'s forward bias
entering the posterior gradient multiplied by SNR -- a reference run is exactly the
case where the difference is load-bearing.
"""

import jax.numpy as jnp
import pytest

pytestmark = pytest.mark.regression_bug


def _build(ssp, approx):
    """A dust-free Cue photometry model.

    Dust-free and Cue on purpose: that is the one configuration where
    ``FeaturePrecomp``'s photometry shortcut actually engages (#1748, and the
    refusal added in #2377's first half), so "both tables attached" is a real state
    to strip rather than one arm being silently inert.
    """
    from tengri import DEFAULT, Fixed, Observation, Photometry, SEDModel, Uniform

    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "sdss_i"])),
        sfh={
            "type": "delayed",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        dust_attenuation={"type": "none"},
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(0.1),
        approx=approx,
    )


def _lut_state(model):
    """``(has_wave, has_feature)`` read off ``model.approx``."""
    state = getattr(model, "approx", None)
    return (
        state is not None and bool(getattr(state, "wave_precomp", False)),
        state is not None and bool(getattr(state, "feature_precomp", False)),
    )


def test_setup_the_build_time_pair_really_attached(ssp_data_fsps):
    """Guard the guard: the build-time model must really carry both tables.

    Without this, every assertion below passes vacuously -- "``approx=None`` left no
    table attached" is equally consistent with "there was never one to strip".
    """
    from tengri import FeaturePrecomp, WavePrecomp

    has_wave, has_feature = _lut_state(_build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp())))
    assert has_wave, "the build-time model does not carry WavePrecomp"
    assert has_feature, (
        "the build-time model does not carry FeaturePrecomp, so this fixture cannot "
        "guard anything. Dust-free Cue photometry is exactly the case #1748 measured "
        "as engaging the shortcut; if that changed, fix the fixture, not the test."
    )


def test_the_two_resolvers_agree_that_none_means_exact(ssp_data_fsps):
    """Both resolvers must answer ``approx=None`` the same way: no table, whatever
    the model carried in."""
    from tengri import FeaturePrecomp, WavePrecomp
    from tengri.inference.fitter import Fitter, _resolve_batch_fit_approx

    model = _build(ssp_data_fsps, (WavePrecomp(), FeaturePrecomp()))
    n_bands = 3

    fitter_wave, fitter_feature = _lut_state(
        Fitter(model, jnp.ones(n_bands), jnp.ones(n_bands), approx=None).model
    )
    assert not fitter_wave and not fitter_feature, (
        "Fitter(approx=None) left a lookup table attached to a model built with one; "
        "approx=None is documented as forcing the exact path (#2377)"
    )

    batch_wave, batch_feature = _lut_state(_resolve_batch_fit_approx(model, None, "photometry"))
    assert not batch_wave and not batch_feature, (
        "_resolve_batch_fit_approx(model, None, ...) left a lookup table attached to "
        "a model built with one, disagreeing with the single-galaxy resolver on the "
        "one value whose contract is 'exact, overrides a build-time approx'. A "
        "catalog or population fit asking for a reference run silently kept the "
        "approximation, whose forward bias enters the gradient multiplied by SNR "
        "(#1671, #2377)."
    )


def test_auto_is_unchanged_on_the_batch_surface(ssp_data_fsps):
    """Scoped to ``None``: the default policy must be untouched."""
    from tengri.inference.fitter import _resolve_batch_fit_approx

    has_wave, has_feature = _lut_state(
        _resolve_batch_fit_approx(_build(ssp_data_fsps, None), "auto", "photometry")
    )
    assert has_wave, "'auto' no longer attaches WavePrecomp on the batch surface"
    assert has_feature, (
        "'auto' no longer tops up FeaturePrecomp on the batch surface for a dust-free "
        "Cue photometry model, where #1748 measured it engaging"
    )


def test_an_explicit_config_is_still_honoured_on_the_batch_surface(ssp_data_fsps):
    """An explicit config is used exactly as given, with no implicit top-up."""
    from tengri import WavePrecomp
    from tengri.inference.fitter import _resolve_batch_fit_approx

    has_wave, has_feature = _lut_state(
        _resolve_batch_fit_approx(_build(ssp_data_fsps, None), (WavePrecomp(),), "photometry")
    )
    assert has_wave, "an explicit WavePrecomp() config was not attached"
    assert not has_feature, (
        "_resolve_batch_fit_approx added FeaturePrecomp to an explicit config that "
        "did not ask for it"
    )
