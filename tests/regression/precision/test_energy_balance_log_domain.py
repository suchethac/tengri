# SPDX-License-Identifier: BSD-3-Clause
r"""The dust energy balance must not silently zero itself in float32 (issue #1206).

``bolometric_absorbed`` integrates :math:`\int (L_\nu^{\rm intr} - L_\nu^{\rm att})
d\nu`. The integrand is ~1e28 erg/s/Hz and the frequency span ~1e15 Hz, so the
product lands at ~1e43 erg/s -- past the float32 ceiling of 3.4e38. The reduction
therefore returns ``inf``, and the trailing ``jnp.where(jnp.isfinite(signed),
signed, 0.0)`` guard converts that ``inf`` into **0.0**.

That is a silent fail-open: dust IR re-emission switches off completely and
nothing raises. The guard is not the bug -- it exists for genuine ``Inf*0``
artifacts from extreme-metallicity SSP fluxes -- the bug is that overflow
manufactures a non-finite value out of perfectly finite inputs, so the guard
fires on healthy data.

Peak-factoring the integrand keeps every intermediate in range. The absorbed
luminosity itself (~1e43 erg/s) is still not float32-representable, so the
float32-safe contract is the log10 form.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri.forward.energy_balance import bolometric_absorbed, bolometric_absorbed_log10
from tengri.utils.physics_constants import C_AA

pytestmark = pytest.mark.regression_bug


def _grid(n=800):
    """Wavelength grid spanning the Lyman continuum to the far-IR [Angstrom]."""
    return jnp.asarray(np.logspace(np.log10(500.0), np.log10(5.0e6), n))


def _seds(wave, scale=1.0e28, transmission=0.35):
    """Intrinsic and attenuated L_nu [erg/s/Hz] of realistic magnitude."""
    w = jnp.asarray(wave)
    intrinsic = scale * jnp.exp(-(((jnp.log10(w) - 4.0) / 1.2) ** 2))
    return intrinsic, transmission * intrinsic


def _frozen_bolometric_absorbed(sed_intrinsic, sed_attenuated, nu, *, wave, lyman_cutoff_aa=912.0):
    """FROZEN pre-#1206 ``bolometric_absorbed`` (verbatim arithmetic)."""
    absorbed_lnu = sed_intrinsic - sed_attenuated
    if lyman_cutoff_aa is not None:
        absorbed_lnu = jnp.where(wave >= lyman_cutoff_aa, absorbed_lnu, 0.0)
    signed = jnp.trapezoid(absorbed_lnu, nu)
    return jnp.where(jnp.isfinite(signed), signed, 0.0)


def test_energy_balance_f64_exact_vs_frozen():
    """The reformulation must reproduce the pre-change float64 answer."""
    wave = _grid()
    nu = C_AA / wave
    for scale in (1.0e22, 1.0e28, 1.0e33):
        for transmission in (0.0, 0.35, 0.99):
            sed_i, sed_a = _seds(wave, scale, transmission)
            frozen = np.float64(_frozen_bolometric_absorbed(sed_i, sed_a, nu, wave=wave))
            got = np.float64(bolometric_absorbed(sed_i, sed_a, nu, wave=wave))
            assert_allclose(got, frozen, rtol=1e-12)


def test_energy_balance_log_matches_linear_in_f64():
    """``bolometric_absorbed_log10`` is log10 of the magnitude of the linear form."""
    wave = _grid()
    nu = C_AA / wave
    for scale in (1.0e22, 1.0e28, 1.0e33):
        sed_i, sed_a = _seds(wave, scale)
        linear = np.float64(jnp.abs(bolometric_absorbed(sed_i, sed_a, nu, wave=wave)))
        log_form = np.float64(bolometric_absorbed_log10(sed_i, sed_a, nu, wave=wave)[0])
        assert_allclose(log_form, np.log10(linear), rtol=1e-12)


def test_energy_balance_lyman_mask_is_honored_in_log_form():
    """The LyC mask must apply identically in both forms, including ``None``."""
    wave = _grid()
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)
    for cutoff in (912.0, None, 1216.0):
        linear = np.float64(
            jnp.abs(bolometric_absorbed(sed_i, sed_a, nu, wave=wave, lyman_cutoff_aa=cutoff))
        )
        log_form = np.float64(
            bolometric_absorbed_log10(sed_i, sed_a, nu, wave=wave, lyman_cutoff_aa=cutoff)[0]
        )
        assert_allclose(log_form, np.log10(linear), rtol=1e-12)


def test_energy_balance_pure_float32_is_not_silently_zero():
    """Pure float32: the log form is finite and accurate; the linear form was 0.0.

    This is the load-bearing assertion of the whole task -- before the fix the
    frozen form returns exactly 0.0 here (overflow -> ``inf`` -> guard), which
    silently disables dust IR emission.
    """
    wave = _grid()
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)
    ref_log = float(bolometric_absorbed_log10(sed_i, sed_a, nu, wave=wave)[0])
    assert np.isfinite(ref_log)
    assert np.any(ref_log != 0.0), (
        "`ref_log` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )

    with jax.enable_x64(False):
        w32 = jnp.asarray(np.asarray(wave), dtype=jnp.float32)
        i32 = jnp.asarray(np.asarray(sed_i), dtype=jnp.float32)
        a32 = jnp.asarray(np.asarray(sed_a), dtype=jnp.float32)
        assert i32.dtype == jnp.float32  # precondition: genuinely pure float32
        nu32 = C_AA / w32
        got_log = float(bolometric_absorbed_log10(i32, a32, nu32, wave=w32)[0])
        frozen32 = float(_frozen_bolometric_absorbed(i32, a32, nu32, wave=w32))

    assert np.isfinite(got_log), f"log energy balance non-finite in float32: {got_log}"
    assert np.any(got_log != 0.0), (
        "`got_log` is identically zero — finite is not enough, "
        "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
    )
    assert_allclose(got_log, ref_log, atol=5e-3)
    # The frozen form silently returns 0.0 -- proves this test is load-bearing.
    assert frozen32 == 0.0, f"expected the frozen linear form to fail open to 0.0, got {frozen32}"


def test_energy_balance_zero_absorption():
    """No absorption -> linear exactly 0.0, log ``-inf`` (which powers back to 0.0)."""
    wave = _grid(200)
    nu = C_AA / wave
    sed_i, _ = _seds(wave)
    assert float(bolometric_absorbed(sed_i, sed_i, nu, wave=wave)) == 0.0
    assert float(bolometric_absorbed_log10(sed_i, sed_i, nu, wave=wave)[0]) == -np.inf
    assert float(10.0 ** bolometric_absorbed_log10(sed_i, sed_i, nu, wave=wave)[0]) == 0.0


def test_energy_balance_non_finite_input_clamps_linearly_but_not_in_log():
    """Non-finite input: the linear form clamps, the log form reports ``+inf`` (#1527).

    The two spellings agree everywhere except here, deliberately.
    :func:`bolometric_absorbed` keeps the #922 clamp — BUG-NSS-02 behavior, still
    pinned by ``TestFiniteGuard`` — while
    :func:`bolometric_absorbed_log10`, the form every production call site uses,
    refuses to report a corrupt integrand as zero absorption. The full contract
    lives in ``test_energy_balance_fail_open.py``.

    The poisoned index must sit *above* the Lyman cutoff: a non-finite value
    below 912 A is masked out before the integral, so it legitimately does not
    reach the guard at all.
    """
    wave = _grid(200)
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)
    above_lyc = int(np.argmax(np.asarray(wave) >= 912.0)) + 50
    assert float(wave[above_lyc]) > 912.0  # precondition: not masked away

    for bad in (jnp.inf, -jnp.inf, jnp.nan):
        sed_bad = sed_i.at[above_lyc].set(bad)
        frozen = float(_frozen_bolometric_absorbed(sed_bad, sed_a, nu, wave=wave))
        assert frozen == 0.0, f"setup: frozen form should clamp {bad} to 0.0, got {frozen}"
        assert float(bolometric_absorbed(sed_bad, sed_a, nu, wave=wave)) == 0.0
        log_bad = float(bolometric_absorbed_log10(sed_bad, sed_a, nu, wave=wave)[0])
        assert log_bad == np.inf, (
            f"a {bad} in the intrinsic SED gave log10 form {log_bad}, not +inf. -inf here "
            "powers back to exactly 0.0 and silently zeroes the whole IR budget (#1527)"
        )


def test_energy_balance_non_finite_below_lyman_cutoff_is_masked():
    """A non-finite value below 912 A is masked, not clamped — the result stays exact."""
    wave = _grid(200)
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)
    below_lyc = int(np.argmax(np.asarray(wave) >= 912.0)) - 1
    assert float(wave[below_lyc]) < 912.0  # precondition: inside the LyC mask

    clean = np.float64(bolometric_absorbed(sed_i, sed_a, nu, wave=wave))
    poisoned = sed_i.at[below_lyc].set(jnp.inf)
    assert_allclose(
        np.float64(bolometric_absorbed(poisoned, sed_a, nu, wave=wave)), clean, rtol=1e-12
    )


def test_energy_balance_gradient_is_finite():
    """Gradients stay finite through the peak factoring and the where-dummy."""
    wave = _grid(300)
    nu = C_AA / wave

    def loss_linear(tau):
        sed_i, _ = _seds(wave)
        return bolometric_absorbed(sed_i, sed_i * jnp.exp(-tau), nu, wave=wave)

    def loss_log(tau):
        sed_i, _ = _seds(wave)
        return bolometric_absorbed_log10(sed_i, sed_i * jnp.exp(-tau), nu, wave=wave)[0]

    for tau in (0.1, 1.0, 5.0):
        assert np.isfinite(float(jax.grad(loss_linear)(tau))), f"linear grad at tau={tau}"
        assert np.any(float(jax.grad(loss_linear)(tau)) != 0.0), (
            "`float(jax.grad(loss_linear)(tau))` is identically zero — finite is not enough, "
            "a value that has collapsed to zero is as unusable as a NaN one (#2100)"
        )
        assert np.isfinite(float(jax.grad(loss_log)(tau))), f"log grad at tau={tau}"


def test_energy_balance_log_sign_tracks_the_linear_sign():
    """The returned sign must reproduce the linear form's sign exactly.

    The sign is what lets a caller combining two absorbed terms compute
    ``|a + b|`` rather than ``|a| + |b|`` -- they differ whenever an
    attenuation law amplifies rather than attenuates.
    """
    wave = _grid(300)
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)

    # Normal absorption: intrinsic above attenuated.
    linear = float(bolometric_absorbed(sed_i, sed_a, nu, wave=wave))
    _, sign = bolometric_absorbed_log10(sed_i, sed_a, nu, wave=wave)
    assert float(sign) == np.sign(linear) != 0.0
    assert np.all(np.isfinite(np.sign(linear))), (
        "`np.sign(linear)` is non-finite — non-zero is not enough, `nan != 0.0` is True "
        "and a NaN satisfies a non-zero assertion (#2178)"
    )

    # Amplification: the roles swap, so the signed integral flips.
    linear_flipped = float(bolometric_absorbed(sed_a, sed_i, nu, wave=wave))
    _, sign_flipped = bolometric_absorbed_log10(sed_a, sed_i, nu, wave=wave)
    assert float(sign_flipped) == np.sign(linear_flipped)
    assert float(sign_flipped) == -float(sign), "setup: the two cases must differ in sign"

    # Nothing absorbed -> sign 0.0.
    _, sign_zero = bolometric_absorbed_log10(sed_i, sed_i, nu, wave=wave)
    assert float(sign_zero) == 0.0


def test_log_sum_of_two_absorbed_terms_matches_linear_abs_sum():
    """``log10_add`` on two absorbed terms reproduces ``abs(a + b)``.

    This is the exact arithmetic the two-component fast path performs when it
    combines the stellar (LUT) and nebular absorbed luminosities.
    """
    from tengri.utils.scale import log10_add

    wave = _grid(300)
    nu = C_AA / wave
    sed_i, sed_a = _seds(wave)
    sed_neb, sed_neb_att = _seds(wave, scale=4.0e26)

    for stellar_pair, neb_pair in (
        ((sed_i, sed_a), (sed_neb, sed_neb_att)),
        ((sed_i, sed_a), (sed_neb_att, sed_neb)),  # opposite signs -- cancellation
    ):
        linear = float(
            jnp.abs(
                bolometric_absorbed(*stellar_pair, nu, wave=wave)
                + bolometric_absorbed(*neb_pair, nu, wave=wave)
            )
        )
        log_s, sign_s = bolometric_absorbed_log10(*stellar_pair, nu, wave=wave)
        log_n, sign_n = bolometric_absorbed_log10(*neb_pair, nu, wave=wave)
        got = float(log10_add(log_s, log_n, sign_a=sign_s, sign_b=sign_n))
        assert_allclose(got, np.log10(linear), rtol=1e-12)
