# SPDX-License-Identifier: BSD-3-Clause
"""Contract: the observable accessors on ``Prediction`` (#1046, Phase 2).

``model.predict(params)`` is the rich exploration surface. It exposes the
observables as uniform callables with defaults:

    pred.photometry()                       # build-time filters, EXACT
    pred.photometry(filters=["jwst_f356w"]) # arbitrary filters, EXACT, warns once
    pred.photometry(approx=True)              # explicit opt-in to the WavePrecomp LUT
    pred.magnitudes(...)                    # AB mags, same signature
    pred.spectrum()                         # instrument grid, LSF-convolved, calibrated
    pred.rest_sed() / pred.obs_sed()            # panchromatic model-grid arrays

Two rules carry real risk and are pinned hard below.

**Exact by default.** The default must integrate the SED through the filters even
when the model was built with ``approx=WavePrecomp(...)``. The tempting
implementation — delegate to ``model.predict_photometry`` — silently returns the
LUT on such a model, so ``pred.photometry()`` would mean "exact" on one model and
"approximate" on another. The two differ by ~6% at z=3, so this is not academic.

**One kernel, not two.** Arbitrary post-build filters must go through the same
``project_photometry`` the build-time path uses. A parallel copy would be a
natural place to forget the IGM transmission (#932), producing plausible but wrong
fluxes at high z — the failure this suite exists to prevent.
"""

from __future__ import annotations

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import FIXED, Fixed, Observation, SEDModel, Spectroscopy, WavePrecomp
from tengri.observation.photometry_config import Photometry
from tengri.utils.magnitudes import fnu_to_ab_mag

pytestmark = pytest.mark.contract

_FILTERS = ["jwst_f150w", "jwst_f200w", "jwst_f277w"]
_EXTRA = "jwst_f356w"
# At z=3 the Lyman limit lands at 3648 A observed and Ly-alpha at 4864 A, so sdss_u
# (3562 A) sits inside the forest and is heavily absorbed. The JWST bands above do
# NOT: they sample rest-frame >3700 A, where IGM transmission is ~1.
_FOREST_FILTERS = ["sdss_u", "sdss_g"]
_SFH = {"type": "dpl", "*": FIXED}
_DUST = {"type": "single_component", "law": "calzetti", "*": FIXED}
# z=3 keeps the IGM transmission live: at low z it is ~1 and would mask a
# projection path that drops it entirely.
_Z = 3.0


@pytest.fixture(scope="module")
def ssp():
    try:
        return tengri.load_ssp()
    except FileNotFoundError as exc:
        pytest.skip(f"SSP data not on disk (CI runner): {exc}")


def _model(ssp, filters=_FILTERS, spectroscopy=None, **kw) -> SEDModel:
    photometry = Photometry.from_names(filters) if filters else None
    if "igm" not in kw:
        kw["igm"] = {"type": "inoue"}
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=photometry, spectroscopy=spectroscopy),
        sfh=_SFH,
        dust_attenuation=_DUST,
        redshift=Fixed(_Z),
        **kw,
    )


def _params(model: SEDModel) -> dict:
    return {name: float(v) for name, v in model.spec.get_fixed_values().items()}


@pytest.fixture(scope="module")
def exact(ssp):
    """A model with no approximation — its photometry IS the exact reference."""
    m = _model(ssp)
    return m, _params(m)


def test_photometry_is_a_method_not_a_property(exact):
    """The Phase-2 clean break: ``pred.photometry`` takes arguments now."""
    model, params = exact
    pred = model.predict(params)
    assert callable(pred.photometry)
    assert np.asarray(pred.photometry()).shape == (len(_FILTERS),)


def test_photometry_default_matches_the_exact_reference(exact):
    model, params = exact
    pred = model.predict(params)
    np.testing.assert_allclose(
        np.asarray(pred.photometry()),
        np.asarray(model.predict_photometry(params)),
        rtol=1e-12,
    )


def test_default_stays_exact_on_a_waveprecomp_model(ssp, exact):
    """The regression that matters: a WavePrecomp build must NOT change the default.

    ``pred.photometry()`` must integrate the SED; only ``approx=True`` may read the
    LUT. If the default were routed through ``model.predict_photometry``, this
    model would silently return the LUT instead — and at z=3 that is a ~6% error,
    not a rounding difference.
    """
    reference, ref_params = exact
    ref = np.asarray(reference.predict_photometry(ref_params))

    fast_model = _model(ssp, approx=WavePrecomp())
    pred = fast_model.predict(_params(fast_model))

    default = np.asarray(pred.photometry())
    lut = np.asarray(pred.photometry(approx=True))

    np.testing.assert_allclose(default, ref, rtol=1e-10)
    # ...and the LUT really is a different number, so the assertion above has teeth.
    assert not np.array_equal(default, lut)
    assert np.max(np.abs(lut / ref - 1.0)) > 1e-6

    # approx=True is the same LUT the lean inference shortcut uses on this model:
    # the fit and the exploration surface must agree about what "fast" means.
    # Not rtol=0 — the two reach the same LUT by different accumulation orders
    # (jitted kernel vs eager accessor), so they agree to ~1 ULP, not to the bit.
    # Forcing bit-equality here would be pinning an XLA scheduling detail.
    np.testing.assert_allclose(
        lut, np.asarray(fast_model.predict_photometry(_params(fast_model))), rtol=1e-12
    )


def test_arbitrary_filters_equal_a_model_built_with_them(ssp, exact):
    """The one-kernel rule: a call-time filter equals the same filter at build time.

    Note this cannot, on its own, prove the kernel is *correct* — both sides run
    through ``project_photometry``, so a bug inside it cancels. That is what
    ``test_photometry_carries_the_igm_attenuation`` is for. This test pins only
    that the two entry points agree.
    """
    model, params = exact
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        runtime = np.asarray(model.predict(params).photometry(filters=[_EXTRA]))

    built_in = _model(ssp, filters=[_EXTRA])
    reference = np.asarray(built_in.predict_photometry(_params(built_in)))

    np.testing.assert_allclose(runtime, reference, rtol=1e-12)


def test_photometry_carries_the_igm_attenuation(ssp):
    """The projection must apply the IGM transmission — checked INDEPENDENTLY.

    Every model-side entry point (``pred.photometry``, ``predict_photometry``,
    call-time filters) now shares one kernel, so comparing them to each other
    cannot detect a bug *inside* that kernel: drop the IGM multiply and they all
    drop it together, in perfect agreement. A parity gate between two paths that
    share the defect proves nothing.

    So the reference here is composed in the test itself, from the published
    transmission and the model-free ``compute_photometry`` engine.

    The filter choice is the whole test. At z=3 the Lyman limit lands at
    912 x 4 = 3648 A and Ly-alpha at 4864 A observed, so ``sdss_u`` (3562 A)
    sits in the forest and is heavily absorbed — while the JWST bands used by the
    rest of this module sit at rest-frame >3700 A, where the transmission is ~1
    and dropping it would change nothing. A test that samples only those bands
    *passes with the IGM factor deleted*; the vacuity guard below is what catches
    that, and it is not decoration.
    """
    from tengri.cosmology import luminosity_distance
    from tengri.observation.photometry import compute_photometry

    model = _model(ssp, filters=_FOREST_FILTERS)
    params = _params(model)
    pred = model.predict(params)
    state = pred._ensure_state()

    igm = state.derived.get("igm_transmission")
    assert igm is not None, "IGM is off — this test would be vacuous"

    z = params["redshift"]
    dl_cm = jnp.asarray(luminosity_distance(z)).reshape(())
    photometry = model.observation.photometry

    with_igm = np.asarray(
        compute_photometry(
            state.sed_intrinsic * igm,
            state.wave,
            list(photometry.filters),
            z,
            dl_cm,
            convention=photometry.convention,
        )
    )
    without_igm = np.asarray(
        compute_photometry(
            state.sed_intrinsic,
            state.wave,
            list(photometry.filters),
            z,
            dl_cm,
            convention=photometry.convention,
        )
    )

    # The two references must genuinely differ, or the assertion below is vacuous.
    assert np.max(np.abs(with_igm / without_igm - 1.0)) > 1e-3

    np.testing.assert_allclose(np.asarray(pred.photometry()), with_igm, rtol=1e-10)


def test_arbitrary_filters_warn_exactly_once(exact):
    """Warn once, not once per call — a hot loop must not be flooded.

    Match on the message, not on ``UserWarning``: unrelated loaders (e.g. the
    wNE-SSP notice) also raise ``UserWarning``, so counting by category would
    silently pass on a guard that never fired.
    """
    model, params = exact
    pred = model.predict(params)
    from tengri.forward import prediction as _p

    _p._WARNED_RUNTIME_PHOTOMETRY = False
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        pred.photometry(filters=[_EXTRA])
        pred.photometry(filters=[_EXTRA])

    ours = [w for w in caught if "exact path" in str(w.message)]
    assert len(ours) == 1, [str(w.message)[:60] for w in caught]


def test_fast_without_waveprecomp_raises(exact):
    model, params = exact
    with pytest.raises(ValueError, match="WavePrecomp"):
        model.predict(params).photometry(approx=True)


def test_fast_with_arbitrary_filters_raises(ssp):
    """A LUT cannot cover filters it was never built for — loud, never a fallback."""
    model = _model(ssp, approx=WavePrecomp())
    with pytest.raises(ValueError):
        model.predict(_params(model)).photometry(approx=True, filters=[_EXTRA])


def test_magnitudes_are_ab_mags_of_the_photometry(exact):
    model, params = exact
    pred = model.predict(params)
    np.testing.assert_allclose(
        np.asarray(pred.magnitudes()),
        np.asarray(fnu_to_ab_mag(pred.photometry())),
        rtol=0.0,
    )


def test_rest_and_obs_sed_are_the_panchromatic_arrays(exact):
    model, params = exact
    pred = model.predict(params)

    rest = np.asarray(pred.rest_sed())
    obs = np.asarray(pred.obs_sed())
    assert rest.shape == obs.shape
    assert np.all(np.isfinite(rest)) and np.all(np.isfinite(obs))
    np.testing.assert_allclose(obs, np.asarray(model.predict_obs_sed(params).sed), rtol=0.0)
    # obs_sed is F_nu at z=3; rest_sed is L_nu. They are not the same quantity.
    assert not np.allclose(rest, obs)


def test_spectrum_uses_the_instrument_grid(ssp):
    from tengri import Spectroscopy

    wave_obs = jnp.linspace(4000.0, 9000.0, 200)
    model = _model(ssp, spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=1000.0))
    params = _params(model)
    spec = np.asarray(model.predict(params).spectrum())
    assert spec.shape == (200,)
    # Both are the exact projector on a model with no approximation, but one runs
    # eagerly off the cached ForwardState and the other inside the jitted kernel,
    # so they agree to ~1 ULP rather than to the bit. rtol=0 here would be pinning
    # an XLA accumulation order, not a physical claim.
    np.testing.assert_allclose(spec, np.asarray(model.predict_spectrum(params)), rtol=1e-12)


def test_missing_blocks_raise_naming_the_block(ssp):
    """A missing observation sub-block must say which one, not return None or NaN."""
    phot_only = _model(ssp)
    with pytest.raises(ValueError, match="spectroscopy"):
        phot_only.predict(_params(phot_only)).spectrum()

    from tengri import Spectroscopy

    spec_only = _model(
        ssp,
        filters=None,
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 9000.0, 100), resolution=1000.0),
    )
    with pytest.raises(ValueError, match="photometry"):
        spec_only.predict(_params(spec_only)).photometry()


# ── Review follow-ups (#1097): three silent defects and a performance inversion ──


@pytest.mark.parametrize("convention", ["bessell", "energy"])
def test_arbitrary_filters_honor_the_models_filter_convention(ssp, convention):
    """Runtime filters must integrate under the model's own convention (ADR-0017).

    ``Photometry.from_names`` defaults to Bessell. Resolving call-time filters
    without passing the model's convention answers a different question than
    ``photometry()`` does — the same filters, two numbers, ~0.5-0.8% apart on an
    energy-convention model. Silent, and invisible to any test that only ever
    builds the default convention.
    """
    photometry = Photometry.from_names(_FILTERS, convention=convention)
    model = SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=photometry),
        sfh=_SFH,
        dust_attenuation=_DUST,
        redshift=Fixed(_Z),
    )
    params = _params(model)
    pred = model.predict(params)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        runtime = np.asarray(pred.photometry(filters=_FILTERS))

    np.testing.assert_allclose(runtime, np.asarray(pred.photometry()), rtol=0.0)


def test_spectrum_default_stays_exact_on_a_spectrumprecomp_model(ssp):
    """The spectrum twin of the photometry exact-by-default rule.

    ``predict_spectrum`` honors the SpectrumPrecomp LUT, so defaulting
    ``pred.spectrum()`` to it made the method mean "exact" on one model and
    "approximate" on another — measured ~5% apart, the same order as the
    photometry LUT error. This test is the counterpart that did not exist, which
    is why the defect survived.
    """
    from tengri.forward.sed_model import SpectrumPrecomp

    wave_obs = jnp.linspace(4000.0, 9000.0, 300)

    def build(**kw):
        return _model(
            ssp,
            filters=None,
            spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=1000.0),
            **kw,
        )

    exact_model = build()
    lut_model = build(approx=SpectrumPrecomp())

    reference = np.asarray(exact_model.predict(_params(exact_model)).spectrum())
    pred = lut_model.predict(_params(lut_model))

    default = np.asarray(pred.spectrum())
    lut = np.asarray(pred.spectrum(approx=True))

    np.testing.assert_allclose(default, reference, rtol=1e-10)
    # ...and the LUT is genuinely a different spectrum, so the assertion has teeth.
    assert np.max(np.abs(lut / reference - 1.0)) > 1e-3


def test_spectrum_fast_without_spectrum_precomp_raises(ssp):
    model = _model(
        ssp,
        filters=None,
        spectroscopy=Spectroscopy(wave_obs=jnp.linspace(4000.0, 9000.0, 100), resolution=1000.0),
    )
    with pytest.raises(ValueError, match="SpectrumPrecomp"):
        model.predict(_params(model)).spectrum(approx=True)


def test_fast_photometry_does_not_materialize_the_forward_state(ssp):
    """``approx=True`` must not be slower than exact — pinned structurally.

    WavePrecomp's saving is that XLA dead-code-eliminates the full-resolution SED
    einsum when only the LUT is consumed. Reaching the LUT via
    ``_ensure_state()`` materializes that state eagerly, spending the saving and
    then doing the LUT lookup on top: measured *0.8x* — slower than the exact
    default, while also returning an approximation.

    Asserting on wall-clock would be flaky, so pin the cause: a fresh Prediction
    asked only for fast photometry must never have built a ForwardState.
    """
    model = _model(ssp, approx=WavePrecomp())
    pred = model.predict(_params(model))

    assert "_state" not in pred._cache
    pred.photometry(approx=True)
    assert "_state" not in pred._cache, (
        "approx=True built the ForwardState — the LUT saving is gone"
    )

    # The exact path, by contrast, legitimately needs it.
    pred.photometry()
    assert "_state" in pred._cache
