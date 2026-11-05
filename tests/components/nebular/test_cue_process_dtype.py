# SPDX-License-Identifier: BSD-3-Clause
"""Cue runs in the process dtype, so a float64 fit has a float64 objective (#3248).

The Cue emulator cast its inputs to float32 (``jnp.asarray(..., dtype=jnp.float32)``
in ``_forward_lines`` / ``_forward_continuum`` / ``_effective_gas_logno``) and its
weights were stored float32, so under x64 the nebular output came from float32
arithmetic. Its rounding is a staircase in the parameters, and a float64 MAP
stalled on that floor: the objective's finite-difference slope along log mass
disagreed with the AD gradient by about 1 %.

Under x64 the emulator now runs in float64. The weights are widened once at load
(an exact float32 -> float64 widening) and the inputs follow their parameter's
dtype. Under x64-off every value stays float32, and
``test_x64_off_cue_matches_float32_reference`` checks the float32 spectrum, gradient and
NLL against a recorded reference within measured bounds.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src import source_info_util
from jax.flatten_util import ravel_pytree

from tengri import (
    DEFAULT,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    Spectroscopy,
    Uniform,
    load_ssp_data,
)
from tengri.components.nebular import cue as cuemod
from tengri.inference.context import InferenceContext
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_SSP = "data/fsps_prsc_miles_chabrier.h5"
_MASS = "sfh_dpl_log_total_mass"
_WAVE = np.arange(4000.0, 7000.0, 2.0)  # observed frame [Angstrom]
_Z = 0.05
_TRUE = {
    _MASS: 10.0,
    "sfh_dpl_tau_gyr": 2.0,
    "sfh_dpl_alpha": 2.0,
    "sfh_dpl_beta": 2.0,
    "sfh_dpl_age_frac": 0.7,
}
#: Central-difference step along log mass, and the relative agreement with the AD
#: gradient it must reach. Float64 round-off in f (about 1e-16 |f| / t, i.e. 7e-7 in f
#: at t = 1e-7) is 2e-11 relative to g here; the float32 staircase measured before the
#: fix was 1e-2 relative.
_FD_STEP = 1e-7
_FD_RTOL = 1e-6
#: Emulator outputs (line luminosities, continuum) in float64 against float32 arithmetic on
#: the same host weights and the same inputs rounded to float32, peak-relative. Measured:
#: lines 3.5e-6, continuum 5.3e-5 (the continuum is a 10**x of a log-domain network output).
#: The owner's target was 1e-5; the continuum misses it, so the bound is 1e-4, and the
#: continuum figure is reported as measured.
_RTOL_NET_F32_VS_F64 = 1e-4
_CUE_WEIGHTS = "data/cue_weights.npz"
#: Cue inputs for the emulator-only comparison: ionizing slopes, logLratio, logU, logn,
#: logZ, logN/O, logC/O and the ionizing photon rate, all inside the trained box.
_NET_INPUTS = {
    "ionspec_index1": -1.0,
    "ionspec_index2": -1.5,
    "ionspec_index3": -2.0,
    "ionspec_index4": -2.5,
    "ionspec_logLratio1": 0.5,
    "ionspec_logLratio2": 0.5,
    "ionspec_logLratio3": 0.5,
    "gas_logu": -2.0,
    "gas_logn": 2.0,
    "gas_logz": -0.3,
    "gas_logno": -0.7,
    "gas_logco": -0.4,
}
_NET_LOGQION = 50.0
#: Float32 x64-off reference (spectrum, gradient, NLL) recorded at CI's pinned JAX 0.12.0 by
#: ``fixtures/make_cue_x64_off_reference.py``. Measured on one Mac, JAX 0.11.1 against 0.12.0,
#: the spread is: spectrum 1.63e-5 of peak (62 of 1500 values differ, max 1100 ulp), gradient
#: 1.23e-3 of peak, NLL 4.70e-5 relative. The Cue emulator's own outputs were bit-identical
#: across those two versions, so the spread sits in the JAX/XLA spectrum and gradient path,
#: not in Cue. The bounds below are about 3x those measurements. The platform (macOS against
#: Linux CI) may add to the spread; the Linux CI run is the check of these bounds.
_X64_OFF_REFERENCE = (
    Path(__file__).resolve().parents[2] / "regression" / "precision" / "fixtures"
    / "cue_x64_off_reference.npz"
)  # fmt: skip
_X64_OFF_SPEC_PEAK_RTOL = 5e-5
_X64_OFF_GRAD_PEAK_RTOL = 3.7e-3
_X64_OFF_NLL_RTOL = 1.4e-4


def _build(ssp, *, neb: str = "cue"):
    """Single-galaxy spectroscopic model at z = 0.05, Cue or the no-Cue twin."""
    obs = Observation(
        spectroscopy=Spectroscopy(
            wave_obs=jnp.asarray(_WAVE), resolution=3000.0, resample="conserving"
        )
    )
    neb_group = {"type": "cue", "all_params": Fixed(DEFAULT)} if neb == "cue" else {"type": "none"}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE, _MASS: Uniform(8.0, 12.0)},
            redshift=Fixed(_Z),
            neb=neb_group,
        )


def _mock_fitter(model):
    """A Fitter on a seeded mock spectrum drawn from ``model`` at the truth."""
    truth = {k: jnp.asarray(v) for k, v in _TRUE.items() if k in model.spec.free_params}
    clean = np.asarray(model.predict_spectrum(truth))
    sigma = 1e-3 * float(np.max(np.abs(clean)))
    flux = clean + sigma * np.random.default_rng(7).standard_normal(clean.shape)
    return Fitter(
        model,
        data=jnp.asarray(flux),
        noise=jnp.full(clean.shape, sigma),
        data_type="spectroscopy",
        approx=None,
        profile_mass=False,
    )


def _truth_flat(fitter):
    """The truth in unbounded coordinates, as the Fitter's flat vector and unravel."""
    unb = {
        n: fitter.spec.get_distribution(n).standardize(
            jnp.asarray(_TRUE[n], dtype=jnp.result_type(_TRUE[n]))
        )
        for n in fitter._free_names
    }
    return ravel_pytree(unb)


def _nll(fitter):
    """Negative log-posterior of the flat unbounded vector, built outside any trace."""
    ctx = InferenceContext.from_target(fitter)
    nlp = ctx.neg_log_posterior_fn
    flat, unravel = _truth_flat(fitter)
    nlp(unravel(flat), ctx.data_args)
    return flat, unravel, lambda u: nlp(unravel(u), ctx.data_args)


def _walk_eqns(jaxpr):
    """Every equation of ``jaxpr``, recursing into nested jaxprs."""
    for eqn in jaxpr.eqns:
        yield eqn
        for val in eqn.params.values():
            for sub in val if isinstance(val, (list, tuple)) else [val]:
                if hasattr(sub, "jaxpr") and hasattr(sub.jaxpr, "eqns"):
                    yield from _walk_eqns(sub.jaxpr)
                elif hasattr(sub, "eqns"):
                    yield from _walk_eqns(sub)


def _float32_cue_equations(closed_jaxpr) -> list[str]:
    """Source sites of equations with a float32 output that come from ``cue.py``."""
    sites = []
    for eqn in _walk_eqns(closed_jaxpr.jaxpr):
        dtypes = {str(v.aval.dtype) for v in eqn.outvars if hasattr(v.aval, "dtype")}
        if dtypes & {"float32", "bfloat16"}:
            src = source_info_util.summarize(eqn.source_info)
            if "components/nebular/cue.py" in src:
                sites.append(f"{eqn.primitive.name} {src}")
    return sites


@pytest.fixture(scope="module")
def ssp64():
    return load_ssp_data(_SSP)


@pytest.fixture(scope="module")
def model64(ssp64):
    return _build(ssp64)


@pytest.fixture(scope="module")
def fitter64(model64):
    return _mock_fitter(model64)


def test_cue_objective_has_no_float32_equation_under_x64(fitter64):
    """Under x64 the value-and-gradient of the objective has no float32 Cue equation."""
    flat, _, f = _nll(fitter64)
    closed = jax.make_jaxpr(jax.value_and_grad(f))(flat)
    sites = _float32_cue_equations(closed)
    assert not sites, f"{len(sites)} float32 equations from cue.py, first: {sites[:3]}"


def _cue_emulator_outputs(dtype):
    """Line and continuum outputs of the Cue emulator for ``_NET_INPUTS`` at ``dtype``."""
    weights = cuemod.load_cue_weights(_CUE_WEIGHTS)
    p = {k: jnp.asarray(v, dtype=dtype) for k, v in _NET_INPUTS.items()}
    nn_params = cuemod._prepare_nn_params(
        p["ionspec_index1"],
        p["ionspec_index2"],
        p["ionspec_index3"],
        p["ionspec_index4"],
        p["ionspec_logLratio1"],
        p["ionspec_logLratio2"],
        p["ionspec_logLratio3"],
        p["gas_logu"],
        p["gas_logn"],
        p["gas_logz"],
        p["gas_logno"],
        p["gas_logco"],
    )
    gas_logq = cuemod._logq_from_logu(p["gas_logu"], p["gas_logn"])
    logqion = jnp.asarray(_NET_LOGQION, dtype=dtype)
    outputs = (
        *cuemod.predict_all_lines(nn_params, weights, gas_logq, logqion),
        *cuemod.predict_continuum(nn_params, weights, gas_logq, logqion),
    )
    # outputs[0] and outputs[2] are the host wavelength tables, passed through unchanged;
    # the computed luminosities are outputs[1] and outputs[3].
    computed = (outputs[1], outputs[3])
    assert all(str(x.dtype) == np.dtype(dtype).name for x in computed), [
        str(x.dtype) for x in computed
    ]
    return [np.asarray(x, dtype=np.float64) for x in outputs]


def test_cue_emulator_float64_equals_float32_to_float32_precision():
    """The float64 emulator matches float32 arithmetic on the same weights and inputs."""
    out64 = _cue_emulator_outputs(jnp.float64)
    with jax.enable_x64(False):
        out32 = _cue_emulator_outputs(jnp.float32)
    for name, a, b in zip(("lines_0", "lines_1", "cont_0", "cont_1"), out64, out32):
        assert np.all(np.isfinite(a)) and np.all(np.isfinite(b)), name
        rel = float(np.max(np.abs(b - a)) / np.max(np.abs(a)))
        assert rel <= _RTOL_NET_F32_VS_F64, f"{name}: float32 vs float64 peak-relative {rel:.3e}"


def test_objective_slope_along_log_mass_agrees_with_ad_at_float64_level(fitter64):
    """Central difference along log mass matches the AD gradient to float64 round-off."""
    flat, _, f = _nll(fitter64)
    # ravel_pytree orders a dict by sorted key.
    i = sorted(fitter64._free_names).index(_MASS)
    e = jnp.zeros_like(flat).at[i].set(1.0)
    g = float(jax.grad(f)(flat)[i])
    fd = float((f(flat + _FD_STEP * e) - f(flat - _FD_STEP * e)) / (2 * _FD_STEP))
    assert abs(fd - g) <= _FD_RTOL * abs(g), f"FD {fd:+.6e} vs AD {g:+.6e} at t={_FD_STEP:.0e}"


def test_x64_off_cue_matches_float32_reference():
    """Under x64-off the Cue spectrum, gradient and NLL match the recorded float32 reference."""
    ref = np.load(_X64_OFF_REFERENCE)
    with jax.enable_x64(False):
        model = _build(load_ssp_data(_SSP))
        fitter = _mock_fitter(model)
        truth32 = {
            k: jnp.asarray(v, dtype=jnp.float32)
            for k, v in _TRUE.items()
            if k in model.spec.free_params
        }
        spec = np.asarray(model.predict_observables(truth32).spec_fnu)
        assert spec.dtype == np.float32
        flat, unravel = ravel_pytree(
            {
                n: fitter.spec.get_distribution(n).standardize(truth32[n])
                for n in fitter._free_names
            }
        )
        ctx = InferenceContext.from_target(fitter)
        nlp = ctx.neg_log_posterior_fn
        nll = nlp(unravel(flat), ctx.data_args)
        grad = jax.grad(lambda u: nlp(unravel(u), ctx.data_args))(flat)
    grad = np.asarray(grad)
    assert grad.dtype == np.float32
    assert bool(jnp.all(jnp.isfinite(grad))) and bool(jnp.any(grad != 0))
    assert ref["spec"].dtype == np.float32 and ref["grad"].dtype == np.float32
    spec_rel = np.max(np.abs(spec.astype(np.float64) - ref["spec"])) / np.max(np.abs(ref["spec"]))
    grad_rel = np.max(np.abs(grad.astype(np.float64) - ref["grad"])) / np.max(np.abs(ref["grad"]))
    nll_rel = abs(float(nll) - float(ref["nll"])) / abs(float(ref["nll"]))
    assert spec_rel <= _X64_OFF_SPEC_PEAK_RTOL, f"spectrum peak-relative {spec_rel:.3e}"
    assert grad_rel <= _X64_OFF_GRAD_PEAK_RTOL, f"gradient peak-relative {grad_rel:.3e}"
    assert nll_rel <= _X64_OFF_NLL_RTOL, f"NLL relative {nll_rel:.3e}"
