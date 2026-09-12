# SPDX-License-Identifier: BSD-3-Clause
"""Gradient safety of the log-domain DIG grid mix (#2263 review C1 / I1).

The merge that resolved PR #2263's conflict added a ``log_domain`` branch to
:func:`~tengri.components.nebular.dig.mix_dig_grid_reconstruction`, used by
:meth:`~tengri.forward.sed_model.SEDModel.predict_line_fluxes` on the
``FeaturePrecomp`` fast path so a line luminosity (~1e40 erg/s, out of
float32 range) never has to be exponentiated to be mixed. The first version
of the underlying helper (:func:`~tengri.components.nebular.dig._log10_weighted_mix`)
built ``log10(1 - frac)`` / ``log10(frac)`` as log-space addends and combined
them with a log-sum-exp: finite in the forward direction, but
``d/dfrac log10(frac)`` diverges at ``frac == 0`` or ``frac == 1``, and the
log-sum-exp's own zero weight for the dropped term turns that into a NaN
gradient (``0 * inf``). Those two points are reachable, not measure-zero:
the declared ``neb_dig_frac`` prior is ``Uniform(0, 1)``, and its
unconstrained sampler coordinate saturates to exactly ``1.0`` for any
``|z| >= 9`` in float64 (``|z| >= 8`` in float32) -- ordinary during HMC/NUTS
warmup.

The fixed helper keeps ``frac`` linear inside a factored sum (the same
offset-and-exponentiate trick :func:`~tengri.utils.scale.log10_add` uses),
so ``log10(frac)`` is never differentiated. This module tests that helper
in isolation (synthetic ``reconstruct`` callables, no SSP/Cue data needed)
and through the shipped production surface (a real ``FeaturePrecomp`` Cue
model, gated on ``data/cue_weights.npz``).
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, SEDModel, Uniform, load_ssp_data
from tengri.components.nebular.dig import mix_dig_grid_reconstruction
from tengri.observation.line_flux_data import LineFluxData
from tengri.utils.scale import pow10

pytestmark = pytest.mark.bounds


# --- Part 1: synthetic, data-free tests of the helper + its wiring --------


class _FakeTable:
    """Minimal stand-in for ``NebularGridTable``: only ``axis_names`` is read
    by :func:`mix_dig_grid_reconstruction` itself (the synthetic
    ``reconstruct`` callables below ignore the table entirely)."""

    axis_names = ("neb_logU",)


def _toy_reconstruct_log(amplitude, point, table):
    """Deterministic toy reconstruct: log10 magnitude = amplitude + neb_logU."""
    return jnp.asarray(amplitude) + jnp.asarray(point["neb_logU"])


def _toy_reconstruct_linear(amplitude, point, table):
    """The linear sibling of :func:`_toy_reconstruct_log`: ``10**`` of it."""
    return pow10(_toy_reconstruct_log(amplitude, point, table))


#: HII log magnitude at ``neb_logU = 0`` (the toy reconstruct's base point).
_HII_LOG = 41.0
#: ``neb_dig_delta_logU`` chosen so the DIG lookup lands at 38.0 dex.
_DELTA = -3.0
_DIG_LOG = _HII_LOG + _DELTA


def _mix_log(frac, *, delta=_DELTA):
    return mix_dig_grid_reconstruction(
        _toy_reconstruct_log,
        _HII_LOG,
        {"neb_logU": jnp.asarray(0.0)},
        _FakeTable(),
        neb_dig_frac=frac,
        neb_dig_delta_logU=jnp.asarray(delta),
        log_domain=True,
    )


def _mix_linear(frac, *, delta=_DELTA):
    return mix_dig_grid_reconstruction(
        _toy_reconstruct_linear,
        _HII_LOG,
        {"neb_logU": jnp.asarray(0.0)},
        _FakeTable(),
        neb_dig_frac=frac,
        neb_dig_delta_logU=jnp.asarray(delta),
        log_domain=False,
    )


def test_log_domain_matches_linear_branch_across_frac():
    """``log_domain=True`` agrees with ``log10`` of the linear branch (rtol 1e-12, float64).

    Sweeps ``frac`` across the unit interval (traced, not a Python literal,
    so both lookups run every time) and compares the log-domain result
    directly against ``log10`` of the same computation done through the
    plain linear branch (``log_domain=False``).
    """
    for frac in np.linspace(0.0, 1.0, 21):
        log_result = float(_mix_log(jnp.asarray(frac)))
        linear_result = float(_mix_linear(jnp.asarray(frac)))
        expected = np.log10(linear_result)
        np.testing.assert_allclose(
            log_result,
            expected,
            rtol=1e-12,
            err_msg=f"log_domain branch disagrees with linear branch at frac={frac}",
        )


def test_log_domain_forward_edges_are_exact():
    """``frac=0`` returns the HII value; ``frac=1`` returns the DIG value."""
    result_f0 = float(_mix_log(0.0))
    assert result_f0 == pytest.approx(_HII_LOG, abs=1e-9)

    result_f1 = float(_mix_log(1.0))
    assert result_f1 == pytest.approx(_DIG_LOG, abs=1e-9)


def test_log_domain_gradient_finite_at_every_frac_including_edges():
    """``jax.grad`` w.r.t. ``frac`` is finite at both endpoints and near them.

    RED before the C1 fix: ``frac in {0.0, 1.0}`` gave a NaN gradient through
    ``jax.grad`` (the differentiated argument is always a JAX tracer, so the
    Python-literal short-circuit never engages here -- both lookups run, and
    it is the combine step's own gradient that was broken).
    """
    for frac in (0.0, 1e-6, 0.5, 1.0 - 1e-6, 1.0):
        grad = jax.grad(_mix_log)(jnp.asarray(frac))
        assert np.isfinite(float(grad)), f"grad is not finite at frac={frac}: {grad}"
        assert float(grad) != 0.0, (
            f"grad is exactly zero at frac={frac} -- finite is not enough, a "
            "gradient that has collapsed to zero is as unusable as a NaN one (#2100)"
        )


def test_log_domain_finite_in_float32_where_linear_form_overflows():
    """At ``hii == dig == 40`` dex, the log-domain mix stays finite in float32.

    The linear form of the same magnitude (``10**40``) is ``inf`` in
    float32 -- exactly the overflow #1859/#2269 carry line luminosities in
    log space to avoid.
    """
    hii = jnp.asarray(40.0, dtype=jnp.float32)
    point = {"neb_logU": jnp.asarray(0.0, dtype=jnp.float32)}
    delta = jnp.asarray(0.0, dtype=jnp.float32)  # dig lands at the same 40.0
    frac = jnp.asarray(0.3, dtype=jnp.float32)

    result = mix_dig_grid_reconstruction(
        _toy_reconstruct_log,
        hii,
        point,
        _FakeTable(),
        neb_dig_frac=frac,
        neb_dig_delta_logU=delta,
        log_domain=True,
    )
    assert np.isfinite(float(result)), f"log-domain mix overflowed in float32: {result}"
    assert result.dtype == jnp.float32

    # Sanity check on the premise: the linear form of the same magnitude
    # really does overflow float32, so the log-domain path above is doing
    # the job it exists for.
    linear_hii = pow10(jnp.asarray(40.0, dtype=jnp.float32))
    assert not np.isfinite(float(linear_hii)), "sanity check failed: 10**40 fits in float32?"


# --- Part 2: parity + gradient through the shipped production surface -----

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_LINES = ("Halpha", "Hbeta", "OIII_5007", "NII_6584", "SII_6717")
_LINE_DATA = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
_LW = _LINE_DATA.wavelengths
_BANDS = ["galex_fuv", "galex_nuv", "des_g", "des_r", "des_i", "des_z", "wise_w1", "wise_w2"]
_Z = 0.15

#: Matches the parity ceiling ``test_nebular_grid_precompute.py`` uses
#: throughout its own DIG-mixing parity tests (#2222).
_PARITY_CEILING = 3e-2


def _require_cue_data():
    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    if not Path("data/cue_weights.npz").is_file():
        pytest.skip("Cue weights (data/cue_weights.npz) not present")


def _cue_model(neb):
    """Cue model with dust off, built for :meth:`~SEDModel.enable_fast_nebular`."""
    import warnings

    _require_cue_data()
    ssp = load_ssp_data(_BARE)
    obs = Observation(photometry=Photometry.from_names(_BANDS), line_fluxes=_LINE_DATA)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": Fixed(DEFAULT)},
            dust_attenuation={
                "type": "two_component",
                "law": "calzetti",
                "all_params": Fixed(DEFAULT),
                "tau_diff": Fixed(0.0),
                "tau_bc": Fixed(0.0),
            },
            neb=neb,
            redshift=Fixed(_Z),
        )


def _worst_rel(fast, exact):
    """Worst-case relative error, ignoring entries below 0.1% of the peak."""
    fast = np.asarray(fast)
    exact = np.asarray(exact)
    strong = np.abs(exact) > 1e-3 * np.max(np.abs(exact))
    return float(np.max(np.abs(fast - exact)[strong] / (np.abs(exact)[strong] + 1e-40)))


def test_predict_line_fluxes_log_domain_parity_fixed_frac():
    """``predict_line_fluxes`` through the log-domain FAST grid matches EXACT (#2222 fixture).

    ``neb_dig_frac = Fixed(0.3)``, the same fixture shape
    ``test_reconstruct_matches_exact_with_dig_mixing_fixed_frac`` uses, but
    exercised through the production surface
    (:meth:`~tengri.forward.sed_model.SEDModel.predict_line_fluxes`) with
    :meth:`~tengri.forward.sed_model.SEDModel.enable_fast_nebular` actually
    attached, rather than by calling ``mix_dig_grid_reconstruction`` directly.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": Fixed(0.3),
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _cue_model(neb)
    p = dict(m.spec.sample(jax.random.PRNGKey(2263)))

    exact = np.asarray(m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False))
    m.enable_fast_nebular(_LW, n_grid=14)
    fast = np.asarray(m.predict_line_fluxes(p, target_wavelengths=_LW, redden=False))

    rel = _worst_rel(fast, exact)
    assert rel < _PARITY_CEILING, f"log-domain FAST line parity degraded to {rel:.2e}"


def test_predict_line_fluxes_gradient_wrt_dig_frac_finite_at_edges():
    """``jax.grad`` of ``predict_line_fluxes`` w.r.t. ``neb_dig_frac`` is finite at 0 and 1.

    RED before the C1 fix (measured on this exact fixture at the pre-fix
    commit: NaN at both ``frac = 0.0`` and ``frac = 1.0``). ``neb_dig_frac``
    is FREE here (``Uniform(0, 1)``) so the model is one a fit could actually
    sample; the two edge fractions are exactly the points where the
    sampler's unconstrained coordinate saturates for ``|z| >= 8-9``.
    """
    neb = {
        "type": "cue",
        "all_params": Fixed(DEFAULT),
        "logU": Uniform(-4.0, -1.0),
        "dig_frac": FREE,
        "dig_delta_logU": Fixed(-1.0),
    }
    m = _cue_model(neb)
    m.enable_fast_nebular(_LW, n_grid=14)
    p = dict(m.spec.sample(jax.random.PRNGKey(2263)))

    def objective(frac):
        pp = {**p, "neb_dig_frac": frac}
        return jnp.sum(m.predict_line_fluxes(pp, target_wavelengths=_LW, redden=False))

    for f0 in (0.0, 1.0):
        val = objective(jnp.asarray(f0))
        grad = jax.grad(objective)(jnp.asarray(f0))
        assert np.isfinite(float(val)), f"forward value not finite at neb_dig_frac={f0}"
        assert np.isfinite(float(grad)), f"grad NaN at neb_dig_frac={f0} (#2263 C1)"
        assert float(grad) != 0.0, (
            f"grad is exactly zero at neb_dig_frac={f0} -- finite is not enough, a "
            "gradient that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
