# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for :mod:`tengri.measure` — the model-free measurement façade (#1047).

``tengri.measure`` is a thin façade over the verified measurement engines
(``measure_index_jax``, ``measure_line_flux_jax``, ``compute_photometry``). It
adds **no new algorithms**; its whole job is to hide the three conventions an
astronomer gets burned by:

* **frame**    — the engines want *rest-frame* wavelengths;
* **units**    — the engines want rest-frame :math:`L_\\nu` [erg/s/Hz];
* **distance** — ``compute_photometry`` takes ``dl_cm`` while
  ``measure_line_flux_jax`` takes ``log10_four_pi_dl2``. One ``redshift=`` argument
  must derive both.

These tests pin the *conventions*, not the physics (the engines carry their own
physics tests). Each reference is composed **independently** of the façade — a
parity check between two paths that share a kernel would still pass with the
physics deleted (the Phase 2 lesson, #1097).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, SEDModel
from tengri.observation import Observation, Photometry
from tengri.observation.line_measurement import DESI_LINES
from tengri.observation.spectral_indices import STANDARD_INDICES
from tengri.utils.filter_convention import FilterConvention

pytestmark = pytest.mark.contract


# ── fixtures ──────────────────────────────────────────────────────


def _model(ssp, obs):
    """A minimal model with **no IGM**.

    IGM is the one component that would break the ``rest_sed`` -> photometry
    identity these tests lean on: ``project_photometry`` multiplies the
    transmission in from ``state.derived`` *inside* the kernel, so a model with
    IGM has photometry that is not a pure function of its rest SED. Dust is
    harmless — it attenuates ``sed_intrinsic``, which *is* the rest SED.
    """
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FIXED},
        dust={"type": "two_component", "law": "calzetti", "*": FIXED},
        neb={"type": "none"},
    )


def _rest(model, params):
    """``(wave_rest, L_nu)`` — the public rest-frame SED accessor.

    ``Prediction.rest_sed()`` returns the flux array *without* its grid, so the
    grid comes from :meth:`SEDModel.predict_rest_sed`. This asymmetry is exactly
    why ``measure.from_prediction`` exists: a user holding a ``Prediction`` has
    no ergonomic way to pair the SED with the axis it lives on.
    """
    r = model.predict_rest_sed(params)
    return r.wavelength, r.sed


@pytest.fixture
def params():
    return {"redshift": 0.5}


@pytest.fixture
def model_bessell(synthetic_ssp_wide, synthetic_tophat_obs):
    return _model(synthetic_ssp_wide, synthetic_tophat_obs)


@pytest.fixture
def model_energy(synthetic_ssp_wide, synthetic_tophat_obs):
    """The SAME model on the ENERGY convention.

    Phase 2 defect #1 — ``photometry(filters=)`` silently dropping the model's
    FilterConvention — survived review because **no test ever built a
    non-default-convention model**. This fixture exists so that class of bug
    cannot hide again.
    """
    phot = Photometry(
        filters=synthetic_tophat_obs.photometry.filters,
        convention=FilterConvention.ENERGY,
    )
    return _model(synthetic_ssp_wide, Observation(photometry=phot))


# ── photometry: the dl_cm convention + the filter convention ──────


def test_photometry_matches_the_models_exact_path(model_bessell, params):
    """``measure.photometry`` on a model's rest SED == the model's exact photometry.

    NOT a tautology: ``pred.photometry()`` runs ``project_photometry`` ->
    ``compute_flux_density_batch`` (vmapped over a zero-padded filter block),
    while ``measure.photometry`` runs ``compute_photometry`` (a Python loop over
    ``compute_flux_density``). Different code paths, same physics.
    """
    from tengri import measure

    wave, lnu = _rest(model_bessell, params)
    got = measure.photometry(
        wave, lnu, model_bessell.observation.photometry, redshift=params["redshift"]
    )
    expected = model_bessell.predict(params).photometry()

    assert np.all(np.isfinite(got))
    np.testing.assert_allclose(np.asarray(got), np.asarray(expected), rtol=1e-12)


def test_photometry_honors_the_filter_convention(model_energy, params):
    """The façade must not silently answer in BESSELL when the model is ENERGY.

    Phase 2 defect #1 regression guard.
    """
    from tengri import measure

    wave, lnu = _rest(model_energy, params)
    filters = model_energy.observation.photometry.filters
    z = params["redshift"]

    as_energy = measure.photometry(
        wave, lnu, filters, redshift=z, convention=FilterConvention.ENERGY
    )
    as_bessell = measure.photometry(
        wave, lnu, filters, redshift=z, convention=FilterConvention.BESSELL
    )

    # VACUITY GUARD. If the two conventions agreed on this SED, the assertion
    # below would pass on an implementation that ignores ``convention`` outright.
    ratio = np.asarray(as_energy) / np.asarray(as_bessell)
    assert np.max(np.abs(ratio - 1.0)) > 1e-3, (
        "the two conventions agree on this SED, so the check below is vacuous — "
        "pick a filter set / SED slope that discriminates them"
    )

    # Handing over the Photometry config carries its convention: no silent BESSELL.
    via_config = measure.photometry(wave, lnu, model_energy.observation.photometry, redshift=z)
    np.testing.assert_allclose(np.asarray(via_config), np.asarray(as_energy), rtol=1e-12)


def test_from_prediction_inherits_the_models_convention(model_energy, params):
    """``from_prediction`` must never re-derive the convention — it inherits it.

    ``filters=None`` carries ``Prediction.photometry``'s meaning ("the bands the
    model was built with"), so this measures the ENERGY-convention model through
    its own bands. A façade that re-derived the convention would answer in
    BESSELL and miss by ~0.5-0.8 %.
    """
    from tengri import measure

    pred = model_energy.predict(params)
    got = measure.from_prediction(pred, filters=None)

    np.testing.assert_allclose(
        np.asarray(got["photometry"]), np.asarray(pred.photometry()), rtol=1e-12
    )


def test_from_prediction_skips_photometry_when_filters_omitted(model_bessell, params):
    """Omitting ``filters`` entirely is distinct from passing ``None``."""
    from tengri import measure

    pred = model_bessell.predict(params)
    out = measure.from_prediction(pred, indices=("Dn4000",))

    assert "photometry" not in out
    assert set(out) == {"Dn4000"}


# ── spectral indices: the frame convention ───────────────────────


def test_spectral_index_matches_the_model_catalog(model_bessell, params):
    """``measure.spectral_index`` on the rest SED == ``predict_spectral_indices``.

    Bit-exact (rtol=0): ``predict_spectral_indices`` measures on exactly
    ``(state.wave, state.sed_intrinsic)``, which is what ``predict_rest_sed``
    returns. Any drift means the façade measures a different array than the
    model does.
    """
    from tengri import measure

    idx = STANDARD_INDICES["Dn4000"]
    wave, lnu = _rest(model_bessell, params)

    got = measure.spectral_index(wave, lnu, idx)
    expected = model_bessell.predict_spectral_indices(params, [idx])[0]

    assert np.isfinite(got)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_spectral_index_accepts_a_name(model_bessell, params):
    """Astronomers say ``"Dn4000"``, not ``STANDARD_INDICES["Dn4000"]``."""
    from tengri import measure

    wave, lnu = _rest(model_bessell, params)
    by_name = measure.spectral_index(wave, lnu, "Dn4000")
    by_def = measure.spectral_index(wave, lnu, STANDARD_INDICES["Dn4000"])
    np.testing.assert_array_equal(np.asarray(by_name), np.asarray(by_def))


def test_unknown_index_name_lists_the_alternatives(model_bessell, params):
    """Contract §1 spirit: never NaN/None — raise, and say what IS available."""
    from tengri import measure

    wave, lnu = _rest(model_bessell, params)
    with pytest.raises(KeyError, match="Dn4000"):
        measure.spectral_index(wave, lnu, "Dn4OOO")


# ── line flux: the log10_four_pi_dl2 convention ──────────────────


def test_line_flux_derives_log10_four_pi_dl2_from_redshift(model_bessell, params):
    """``redshift=`` must become ``log10(4 pi d_L^2)`` — the engine's actual argument.

    The reference is built INDEPENDENTLY (cosmology called directly, the
    log-of-4-pi-d_L-squared formed by hand) rather than by calling the façade
    twice — and deliberately *without* ``tengri.utils.scale.log10_four_pi_dl2``,
    so this stays a check on the façade rather than a tautology through the
    shared helper (#1859).
    """
    from tengri import measure
    from tengri.cosmology import luminosity_distance
    from tengri.observation.line_measurement import measure_line_flux_jax

    line = DESI_LINES[2]  # Halpha
    z = params["redshift"]
    wave, lnu = _rest(model_bessell, params)

    got = measure.line_flux(wave, lnu, line, redshift=z)

    dl_cm = jnp.asarray(luminosity_distance(jnp.asarray(z))).reshape(())
    log10_4pi_dl2 = jnp.log10(4.0 * jnp.pi) + 2.0 * jnp.log10(dl_cm)
    expected = measure_line_flux_jax(wave, lnu, line, log10_4pi_dl2)

    assert np.isfinite(got)
    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


def test_line_flux_falls_with_distance(model_bessell):
    """Sanity: the same galaxy further away is fainter. Guards a dropped 1/d_L^2."""
    from tengri import measure

    line = DESI_LINES[2]
    w_near, f_near_sed = _rest(model_bessell, {"redshift": 0.5})
    w_far, f_far_sed = _rest(model_bessell, {"redshift": 1.5})

    f_near = measure.line_flux(w_near, f_near_sed, line, redshift=0.5)
    f_far = measure.line_flux(w_far, f_far_sed, line, redshift=1.5)

    assert abs(float(f_far)) < abs(float(f_near))


def test_line_flux_matches_the_model_operator(model_bessell, params):
    """The façade applies the same operator ``model.measure_line_fluxes`` does."""
    from tengri import measure

    wave, lnu = _rest(model_bessell, params)
    got = jnp.array(
        [measure.line_flux(wave, lnu, ln, redshift=params["redshift"]) for ln in DESI_LINES]
    )
    expected = model_bessell.measure_line_fluxes(params, DESI_LINES)

    np.testing.assert_array_equal(np.asarray(got), np.asarray(expected))


# ── the operator IS the deliverable: same ruler on both sides ─────


def test_operators_run_on_arbitrary_arrays_with_no_model():
    """The scope decision on #1047: ship the operator, not a pipeline.

    A user with their OWN reduced ``(wave_rest, flux)`` arrays must be able to
    measure the observed side with the SAME operator tengri uses on the model
    side — that is what closes the different-rulers systematic when fitting the
    ``SpectralIndexData`` path. No SEDModel, no Observation, no Prediction.
    """
    from tengri import measure

    # Arrays from anywhere at all — here a plain numpy power law, no JAX, no model.
    wave_rest = np.linspace(3000.0, 8000.0, 4000)
    flux = (wave_rest / 4000.0) ** -1.7

    value = measure.spectral_index(wave_rest, flux, "Dn4000")

    assert np.isfinite(value)
    assert float(value) > 0.0  # a red power law has flux on both sides of the break


# ── JIT contract ─────────────────────────────────────────────────


def test_operators_are_jit_and_grad_safe(model_bessell, params):
    """Contract §6: the measure operators are pure, JIT-safe array functions."""
    from tengri import measure

    idx = STANDARD_INDICES["Dn4000"]
    wave, lnu = _rest(model_bessell, params)

    jitted = jax.jit(lambda f: measure.spectral_index(wave, f, idx))
    np.testing.assert_allclose(
        np.asarray(jitted(lnu)),
        np.asarray(measure.spectral_index(wave, lnu, idx)),
        rtol=1e-12,
    )

    g = jax.grad(lambda f: measure.spectral_index(wave, f, idx))(lnu)
    assert np.all(np.isfinite(g))
    assert np.any(np.asarray(g) != 0.0), "index is insensitive to flux — window mismatch?"


# ── discoverability ──────────────────────────────────────────────


def test_namespace_is_exported_and_self_describing():
    """``tengri.measure`` is the discoverable surface; the catalogs ride along."""
    import tengri
    from tengri import measure

    assert tengri.measure is measure
    assert "Dn4000" in measure.STANDARD_INDICES
    assert {"Halpha", "Hbeta"} <= {ln.name for ln in measure.DESI_LINES}

    for fn in ("spectral_index", "line_flux", "photometry", "from_prediction"):
        assert fn in measure.__all__
        assert callable(getattr(measure, fn))
