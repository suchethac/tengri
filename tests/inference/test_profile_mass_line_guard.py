# SPDX-License-Identifier: BSD-3-Clause
"""Guard #8 admits a measured line-flux channel, and refuses the rest by name.

``profile_mass`` marginalizes a stellar-mass amplitude analytically. Guard #8
used to refuse whenever *any* emission-line channel was configured, which
lumped three unrelated situations together; only a **measured line-flux**
channel is a plain data channel with a mass-proportional prediction (#2360).

``tests/inference/`` is auto-marked ``slow`` (see ``tests/conftest.py``); run
with ``-m slow``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import (
    ForwardModel,
    Observation,
    Photometry,
    SEDModel,
    builders,
    generate_mock,
    recipes,
)
from tengri.inference.fitter import Fitter
from tengri.inference.loss_functions import _resolve_measured_line_defs
from tengri.inference.mass_profile import (
    _line_flux_schema,
    _reinsert_mass_fn,
)
from tengri.observation import LineFluxData

pytestmark = pytest.mark.contract

_FILTERS = ["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z", "des_g", "des_r", "des_i"]
_LINE_NAMES = ("Halpha", "Hbeta", "OIII_5007")
_MASS_NAME = "sfh_tsnorm_log_total_mass"


def _line_template():
    return LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINE_NAMES})


def _model(ssp_data, line_fluxes=None):
    """Photometry, optionally plus a measured line-flux channel, on a wNE grid."""
    kw = {"photometry": Photometry.from_names(_FILTERS)}
    if line_fluxes is not None:
        kw["line_fluxes"] = line_fluxes
    obs = Observation(**kw)
    recipe = recipes.mock_recovery_minimal()
    recipe["neb"] = builders.neb.ssp()
    return SEDModel.build(ssp_data=ssp_data, observation=obs, **recipe), obs


def _fitter(ssp_data, line_fluxes=None, **fitter_kw):
    model, obs = _model(ssp_data, line_fluxes)
    key_truth, key_mock = jax.random.split(jax.random.PRNGKey(0))
    truth = model.spec.sample(key_truth)
    mock = generate_mock(model, truth, key=key_mock, snr=30.0)
    fitter = Fitter(
        ForwardModel.build(sed=model, observation=obs),
        jnp.asarray(mock["flux_obs"]),
        jnp.asarray(mock["noise"]),
        profile_mass="auto",
        **fitter_kw,
    )
    return fitter, model, truth


class TestGuardDecisions:
    """Which cases engage, which refuse, and whether a refusal says which."""

    def test_measured_line_fluxes_now_engage(self, ssp_data_wne):
        """The case #2360 is about: photometry plus measured lines.

        Refused before this change with "an emission-line channel is
        configured", which named none of the three situations it covered.
        """
        fitter, _, _ = _fitter(ssp_data_wne, line_fluxes=_line_template())
        assert fitter._fits_line_fluxes(fitter.model), "the fixture lost its line channel"
        assert fitter._profile_mass, (
            f"a measured line-flux channel should engage profiling; "
            f"reason was {fitter._profile_mass_reason!r}"
        )
        # The measured deviation rides along on the engage path (#2370), so the
        # margin is inspectable rather than implied.
        assert "linearity" in (fitter._profile_mass_reason or "")

    def test_photometry_only_control_is_unchanged(self, ssp_data_wne):
        """Without a line channel the decision must be what it always was.

        Present so the test above cannot pass by making everything engage.
        """
        fitter, _, _ = _fitter(ssp_data_wne)
        assert fitter._profile_mass

    def test_censored_line_fluxes_refuse_and_say_so(self, ssp_data_wne):
        """Upper limits are ln Phi terms, not a Gaussian chi-square.

        This is the case the pre-existing censored guard cannot catch: it reads
        ``fitter.data_mask``, which covers the photometry/spectroscopy vector
        only, never ``LineFluxData``'s own limit mask.
        """
        tmpl = _line_template()
        censored = LineFluxData(
            names=tmpl.names,
            wavelengths=tmpl.wavelengths,
            fluxes=tmpl.fluxes,
            errors=tmpl.errors,
            is_upper_limit=jnp.asarray([False, True, False]),
        )
        fitter, _, _ = _fitter(ssp_data_wne, line_fluxes=censored)
        assert not fitter._profile_mass
        reason = fitter._profile_mass_reason or ""
        assert "censored" in reason, reason
        # And it must not be mistaken for the photometry-vector censoring guard,
        # whose reason string is a different one.
        assert "data_mask" in reason, reason

    def test_user_supplied_likelihood_refuses_and_says_so(self, ssp_data_wne):
        """An adapter the profiler cannot inspect must not be assumed Gaussian."""

        class _OpaqueLikelihood:
            name = "opaque"

            def log_prob(self, prediction):  # pragma: no cover - never evaluated
                return jnp.asarray(0.0)

        fitter, _, _ = _fitter(
            ssp_data_wne, line_fluxes=_line_template(), likelihood=_OpaqueLikelihood()
        )
        assert not fitter._profile_mass
        assert "user-supplied" in (fitter._profile_mass_reason or "")

    def test_refusal_reasons_are_distinct_per_case(self, ssp_data_wne):
        """Three situations, three reasons -- the point of splitting the guard.

        A single shared string would let this file pass while telling a user
        nothing about which condition blocked their fit.
        """
        tmpl = _line_template()
        censored = LineFluxData(
            names=tmpl.names,
            wavelengths=tmpl.wavelengths,
            fluxes=tmpl.fluxes,
            errors=tmpl.errors,
            is_upper_limit=jnp.asarray([True, False, False]),
        )

        class _OpaqueLikelihood:
            name = "opaque"

            def log_prob(self, prediction):  # pragma: no cover - never evaluated
                return jnp.asarray(0.0)

        r_censored = (_fitter(ssp_data_wne, line_fluxes=censored)[0]._profile_mass_reason) or ""
        r_user = (
            _fitter(ssp_data_wne, line_fluxes=tmpl, likelihood=_OpaqueLikelihood())[
                0
            ]._profile_mass_reason
        ) or ""
        assert r_censored != r_user
        assert r_censored and r_user


class TestReinsertionIsPerGalaxy:
    """The reinsertion program is cached per model and reused across galaxies."""

    def test_line_values_reach_the_cached_reinsertion_program(self, ssp_data_wne):
        """Two galaxies' line fluxes must give two different reinserted masses.

        ``_reinsert_mass_fn`` caches its compiled program on the **model**,
        keyed by ``_engine_cache_key()``, and reuses it for every galaxy -- which
        is why ``data``/``noise``/``presence`` are arguments rather than closure
        values. ``_LineFluxBlock`` carries the line *schema* (model-level, safe
        to close over) together with the measured *values* (per-galaxy, not
        safe), so closing over a whole block would bake the first galaxy's line
        fluxes into the program every later galaxy reuses.

        That failure is silent: identical shapes, no error, a mass drawn from
        the wrong conditional. This test is the detector -- with the values
        closed over, the two calls below return bit-identical draws, because the
        arguments would be ignored.
        """
        fitter, model, truth = _fitter(ssp_data_wne, line_fluxes=_line_template())
        assert fitter._profile_mass, "profiling must engage or this tests nothing"

        schema = _line_flux_schema(fitter)
        assert schema is not None, "the fixture must have a resolvable line schema"

        defs = _resolve_measured_line_defs(model, has_line_fluxes=True)
        lf_true = jnp.asarray(model.measure_line_fluxes(truth, defs))
        err = jnp.abs(lf_true) / 30.0

        # Two galaxies sharing one schema (same names/wavelengths) but carrying
        # different measured values -- so they share the compiled program.
        obs_a = lf_true
        obs_b = lf_true * 1.5

        n_draws = 4
        samples = {
            n: jnp.repeat(jnp.asarray(truth[n])[None], n_draws, axis=0)
            for n in fitter.spec.free_params
            if n != _MASS_NAME
        }
        keys = jax.random.split(jax.random.PRNGKey(11), n_draws)
        data = jnp.asarray(fitter.data)
        noise = jnp.asarray(fitter.noise)

        fn = _reinsert_mass_fn(fitter)
        out_a = fn(samples, keys, data, noise, None, obs_a, err)
        out_b = fn(samples, keys, data, noise, None, obs_b, err)

        assert np.all(np.isfinite(np.asarray(out_a)))
        assert np.all(np.isfinite(np.asarray(out_b)))
        # atol=0.0 deliberately: these are log10-mass values of order 10, but a
        # default absolute tolerance would still be the wrong instrument for an
        # "these must differ" claim.
        rel = float(jnp.max(jnp.abs(out_a - out_b) / jnp.abs(out_b)))
        assert not bool(jnp.allclose(out_a, out_b, rtol=0.0, atol=0.0)), (
            "the reinserted mass did not respond to the line-flux values at all, "
            "which is what closing them over instead of passing them would produce"
        )
        assert rel > 1e-6, (
            f"the reinserted mass barely responded to a 1.5x change in every line "
            f"flux (max relative change {rel:.3e}); the line block may not be "
            f"reaching the quadratic"
        )

    def test_chunk_size_is_still_measured_with_a_line_block(self, ssp_data_wne, caplog):
        """The #2356 chunk width must still come from XLA, not from the fallback.

        ``_compute_reinsertion_chunk_size`` compiles a reference per-chunk
        program and reads its scratch out of XLA's memory analysis. It wraps
        that in ``except Exception`` and falls back to a fixed width on any
        failure, logging a warning -- so a reference program that stops
        compiling degrades the measured width of #2356's OOM fix into a guess
        while the fit still runs.

        That is not hypothetical: the reference program originally passed a
        length-1 placeholder data vector and relied on broadcasting against the
        prediction. Concatenating a line block onto both sides makes a
        ``1 + n_lines`` data vector that no longer broadcasts against an
        ``n_bands + n_lines`` prediction, and the resulting TypeError was
        swallowed exactly this way.

        The load-bearing assertion is the **positive** one: the derivation log
        line is present. Asserting only that the failure string is absent would
        also pass if the function were never called at all.
        """
        import logging

        fitter, _, _ = _fitter(ssp_data_wne, line_fluxes=_line_template())
        assert fitter._profile_mass, "profiling must engage or nothing is derived"

        with caplog.at_level(logging.INFO, logger="tengri.inference.mass_profile"):
            _reinsert_mass_fn(fitter)

        text = caplog.text
        assert "reinsertion chunk size derived" in text, (
            f"the chunk width was not derived from XLA memory analysis; captured log was:\n{text}"
        )
        assert "XLA memory analysis failed" not in text, (
            f"the reference program failed to compile and fell back:\n{text}"
        )
