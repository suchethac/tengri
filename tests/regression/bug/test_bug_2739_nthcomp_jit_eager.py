# SPDX-License-Identifier: BSD-3-Clause
"""Regression for #2739: the nthcomp interpolation coordinates keep the caller's dtype.

The Comptonized (``kubota_done``) disc locates ``gamma``, ``kTe`` and ``kTbb`` in the
nthcomp template with a trilinear interpolation. Casting those coordinates to float32
(relative 6e-8) moves the interpolation weight by 6e-8 of a cell, which on a steep
template cell is a 1e-6 step in the shape. A 1e-16 difference between the ``jax.jit``
and eager evaluation orders is enough to flip that rounding, so the disc SED differed
between the two by 1e-6 at M_BH 1e7-1e8, and the central finite difference of
``agn_gamma_warm`` and ``agn_kt_warm`` disagreed with the exact tangent at the 3e-5
level.

Equation: the multilinear interpolant of ``log(shape)`` over (gamma, kTe, kTbb) in
Kubota & Done 2018 (arXiv:1804.00171, Section 2.2), evaluated in the input dtype.

The tangent is compared with the Richardson-extrapolated central difference
``(4 FD(h/2) - FD(h)) / 3``. The plain central difference at the gradient contract's step
carries an O(h^2) term that reaches 5.7e-5 of the error scale for ``agn_kt_warm`` at the
``r_hot_unclipped`` point (``FD(h) - AD`` and ``FD(h/2) - AD`` in the ratio 4.00); the
extrapolated value agrees with the tangent to 3e-9 there.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from tengri import FREE
from tengri.components.agn._nthcomp import load_nthcomp_table, nthcomp_lnu_interp
from tests.physics.gradients import test_disc_block_gradient_contract as _contract

pytestmark = pytest.mark.regression_bug

_JIT_EAGER_RTOL = 1e-9
_AD_FD_RTOL = 1e-5
#: A central difference is second order, so ``FD(h) - AD`` is four times ``FD(h/2) - AD``.
_TRUNCATION_RATIO = 4.0
_TRUNCATION_RATIO_RTOL = 0.10
#: Fraction of the error scale below which an FD error is rounding, where the ratio of two
#: such errors carries no information.
_FD_NOISE_FLOOR = 1e-7
_MBH_GRID = (7.0, 7.5, 8.0)
_SPIN_GRID = (0.0, 0.7)


def _significant(reference: np.ndarray) -> np.ndarray:
    """Mask of the nodes carrying more than 1e-6 of the peak."""
    return reference > 1e-6 * reference.max()


@pytest.mark.parametrize("log_mbh", _MBH_GRID)
@pytest.mark.parametrize("a_spin", _SPIN_GRID)
def test_kubota_done_disc_jit_matches_eager(log_mbh, a_spin):
    """The kubota_done disc L_nu from ``jax.jit`` and from eager agree to 1e-9."""
    from tengri.components.agn.unified import kubota_done_full_agn

    wave = jnp.logspace(0.0, 8.0, 600)
    kwargs = {"agn_log_lbol": 11.5, "agn_log_mbh": log_mbh, "agn_a_spin": a_spin}
    eager = np.asarray(kubota_done_full_agn(wave, **kwargs))
    jitted = np.asarray(jax.jit(kubota_done_full_agn)(wave, **kwargs))

    mask = _significant(eager)
    assert mask.sum() > 50, "vacuous: the disc has no significant nodes"
    rel = np.abs(jitted[mask] - eager[mask]) / np.abs(eager[mask])
    assert rel.max() < _JIT_EAGER_RTOL, (
        f"log M_BH={log_mbh}, a={a_spin}: jit vs eager differ by {rel.max():.3e} "
        f"(limit {_JIT_EAGER_RTOL:.0e})"
    )


@pytest.fixture(scope="module")
def kubota_model():
    """AGN-only composable model with the kubota_done disc, as the gradient contract builds it."""
    import tengri

    if not _contract._SSP.is_file():
        pytest.skip(f"BC03 SSP not found at {_contract._SSP}")
    ssp = tengri.load_ssp(str(_contract._SSP))
    return _contract._build_agn_only(ssp, {"type": "kubota_done", "all_params": FREE})


_POINTS = [
    pytest.param({}, id="default"),
    pytest.param({"agn_f_hard": 0.005}, id="r_hot_unclipped"),
]


@pytest.mark.gradient
@pytest.mark.parametrize("overrides", _POINTS)
def test_warm_comptonization_ad_matches_richardson_central_fd(
    kubota_model, overrides, monkeypatch
):
    """AD equals the Richardson central FD of ``sum(log10 sed_agn)`` to 1e-5 (warm-zone knobs).

    Measured with the contract's own entry, point and error metric (error over the larger
    of ``|FD|`` and 1e-3 of the largest FD derivative of the disc). The central difference
    at the contract's step h has an O(h^2) truncation term, which the reference
    ``(4 FD(h/2) - FD(h)) / 3`` cancels. That the residual of the plain differences is
    truncation and not step noise is asserted: where ``FD(h) - AD`` and ``FD(h/2) - AD``
    both exceed 1e-7 of the error scale they stand in the ratio 4 within 10 %; where one
    of them does not, both plain differences agree with AD to 1e-5.

    At the ``r_hot_unclipped`` point ``agn_kt_warm`` has ``FD(h) - AD`` = 1.187e-3 and
    ``FD(h/2) - AD`` = 2.97e-4 (ratio 4.00; 5.7e-5 and 1.4e-5 of the error scale), and the
    Richardson value agrees with AD to 3e-9.
    """
    full = _contract._grad_and_fd(kubota_model, overrides)
    scale = max(abs(fd) for _, fd in full.values())
    monkeypatch.setattr(
        _contract, "_STEP", {k: 0.5 * v for k, v in _contract._STEP.items()}, raising=True
    )
    half = _contract._grad_and_fd(kubota_model, overrides)

    for name in ("agn_gamma_warm", "agn_kt_warm"):
        ad, fd = full[name]
        ad_half, fd_half = half[name]
        assert np.isclose(ad, ad_half, rtol=1e-12), f"{name}: AD depends on the FD step"
        den = max(abs(fd), _contract._ZERO_FLOOR * scale)
        assert abs(fd) > 0.0, f"vacuous: {name} has no response"
        detail = f"AD={ad:.8e} FD(h)={fd:.8e} FD(h/2)={fd_half:.8e}"

        richardson = (4.0 * fd_half - fd) / 3.0
        err = abs(ad - richardson) / den
        assert err < _AD_FD_RTOL, (
            f"{name} (AD vs Richardson FD): {err:.2e} >= {_AD_FD_RTOL:.0e}; {detail}"
        )

        err_h, err_half = (fd - ad) / den, (fd_half - ad) / den
        if min(abs(err_h), abs(err_half)) > _FD_NOISE_FLOOR:
            assert err_h / err_half == pytest.approx(
                _TRUNCATION_RATIO, rel=_TRUNCATION_RATIO_RTOL
            ), f"{name}: FD errors {err_h:.2e}, {err_half:.2e} are not second order; {detail}"
        else:
            assert max(abs(err_h), abs(err_half)) < _AD_FD_RTOL, (
                f"{name}: an FD error at the noise floor beside one above "
                f"{_AD_FD_RTOL:.0e} ({err_h:.2e}, {err_half:.2e}); {detail}"
            )


def test_float32_mode_is_float32_finite_and_close_to_float64():
    """Pure-float32 evaluation returns float32, finite, within 1e-5 of float64."""
    table = load_nthcomp_table()
    assert table is not None, "nthcomp templates are packaged with the source tree"
    nu = np.logspace(15.0, 19.0, 120)
    args = (2.37, 0.13, 0.005)

    reference = np.asarray(nthcomp_lnu_interp(nu, *args, _template=table))
    assert reference.dtype == np.float64

    with jax.enable_x64(False):
        low = nthcomp_lnu_interp(
            np.asarray(nu, dtype=np.float32),
            *(np.float32(a) for a in args),
            _template=table,
        )
        assert low.dtype == jnp.float32
        low = np.asarray(low)

    assert np.all(np.isfinite(low))
    mask = _significant(reference)
    assert mask.sum() > 20, "vacuous: the probe frequencies carry no flux"
    rel = np.abs(low[mask] - reference[mask]) / reference[mask]
    assert rel.max() < 1e-5, f"float32 vs float64 differ by {rel.max():.3e}"
