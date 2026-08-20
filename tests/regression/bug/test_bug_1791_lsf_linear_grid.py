# SPDX-License-Identifier: BSD-3-Clause
r"""The LSF must broaden by the width it was asked for, on any grid (#1791).

``apply_lsf`` reads its pixel scale once, from the first pair of samples::

    dlnwave = jnp.log(wave_obs[1] / wave_obs[0])

That is the correct scale everywhere only if the grid is uniform in
``ln(lambda)``. On a linearly-spaced grid the true scale grows across the array,
so the kernel is too narrow by ``wave[0] / lambda`` — 0.60 at 5000 A on a
3000-10000 A grid.

This is the same mechanism #1742 fixed in ``velocity_broaden``. That fix guarded
the one copy the forward model does *not* call and left the two it does, because
the guard raises and tengri's own spectroscopy path runs on linear grids.

The remedy here is not the resample #1742's error message suggests. Resampling
onto a log grid and interpolating back does deliver the width, but it moves the
data between grids of different pixel density, and that was measured to cost two
exact properties: flux conservation fell from ~1e-16 to 1.1e-4, and a zero-width
kernel stopped being the identity (a spike cannot survive a round trip through a
non-aligned grid at any refinement — 1.65 at 1x oversampling, still 0.13 at 16x).
Instead the piecewise path, which already convolves in bins, now takes each bin's
pixel scale from the local ``d ln lambda``. Nothing is resampled, so the FFT
normalization is untouched: flux conserves to 3e-11 and the identity to 9e-16,
while a requested 200 km/s comes back as 198.2 at the default ``n_bins=16``.

Direction matters. An under-broadened *model* is compensated by a larger fitted
``sigma_v_kms``, so velocity dispersions come out biased **high** by roughly
``lambda / wave[0]``. It varies smoothly with the blue cutoff, so nothing looks
broken and two instruments merely disagree.

Widths are measured as the increase in quadrature between a ``sigma_v = 0`` run
and a broadened one. That cancels the probe line's own width, which adds in
quadrature, and so measures the kernel rather than the fixture.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.observation.spectrum import apply_lsf

pytestmark = pytest.mark.regression_bug

_C_KM_S = 299792.458
_LAM0 = 5000.0
_SIGMA_V = 200.0

# Enough samples that the kernel is many pixels wide on every grid below. At
# n = 2048 the 3000-10000 A kernel is sub-pixel, the FFT rings, and a second
# moment then measures the ringing instead of the broadening.
_N = 16384

_GRIDS = {
    "log_4500_5500": np.logspace(np.log10(4500.0), np.log10(5500.0), _N),
    "linear_4500_5500": np.linspace(4500.0, 5500.0, _N),
    "linear_3800_9200": np.linspace(3800.0, 9200.0, _N),  # the repo's own fixture range
    "linear_3000_10000": np.linspace(3000.0, 10000.0, _N),  # typical optical
}


def _probe(wave):
    """A narrow Gaussian at ``_LAM0``, in ln(lambda) so it is scale-free."""
    x = np.log(np.asarray(wave, dtype=np.float64))
    sigma0 = 40.0 / _C_KM_S
    return np.exp(-0.5 * ((x - np.log(_LAM0)) / sigma0) ** 2)


def _width_kms(flux, wave, half_width_kms=2000.0):
    """Second moment in ln(lambda), windowed around the line.

    The window excludes the FFT's circular wrap; on a wide grid the ln(lambda)
    lever arm would otherwise let the wrapped tail dominate the moment.
    """
    x = np.log(np.asarray(wave, dtype=np.float64))
    win = np.abs(x - np.log(_LAM0)) < (half_width_kms / _C_KM_S)
    f = np.clip(np.asarray(flux, dtype=np.float64), 0.0, None) * win
    w = f / f.sum()
    mean = float((w * x).sum())
    var = float((w * (x - mean) ** 2).sum())
    return np.sqrt(var) * _C_KM_S


def _added_width(wave, resolution, sigma_v=_SIGMA_V, **kw):
    """Width the LSF actually added, in quadrature — the quantity under test."""
    probe = _probe(wave)
    narrow = apply_lsf(probe, wave, resolution, sigma_v_kms=0.0, **kw)
    broad = apply_lsf(probe, wave, resolution, sigma_v_kms=sigma_v, **kw)
    w0 = _width_kms(narrow, wave)
    w1 = _width_kms(broad, wave)
    return float(np.sqrt(max(w1**2 - w0**2, 0.0)))


# ── the defect ────────────────────────────────────────────────────


@pytest.mark.parametrize("grid_name", list(_GRIDS))
def test_requested_broadening_is_delivered_on_any_grid(grid_name):
    """sigma_v goes in, sigma_v comes out — the whole defect (#1791).

    Pre-fix this passes only for ``log_4500_5500`` and fails the three linear
    grids at exactly ``wave[0] / 5000``: 0.900, 0.760, 0.600.
    """
    wave = _GRIDS[grid_name]
    # R large enough that sigma_inst is negligible, isolating sigma_v.
    added = _added_width(wave, resolution=1.0e8)

    ratio = added / _SIGMA_V
    assert abs(ratio - 1.0) < 0.02, (
        f"{grid_name}: asked for {_SIGMA_V} km/s, got {added:.2f} km/s "
        f"(ratio {ratio:.4f}); wave[0]/lambda = {wave[0] / _LAM0:.4f}"
    )


@pytest.mark.parametrize("grid_name", ["linear_3000_10000", "log_4500_5500"])
def test_the_variable_resolution_path_too(grid_name):
    """The per-pixel-R branch reads the same single ``dlnwave`` and needs the same fix."""
    wave = _GRIDS[grid_name]
    resolution = np.full(wave.shape, 1.0e8)

    added = _added_width(wave, resolution=resolution, n_bins=16)

    ratio = added / _SIGMA_V
    assert abs(ratio - 1.0) < 0.03, (
        f"{grid_name}, variable R: asked for {_SIGMA_V} km/s, got {added:.2f} km/s "
        f"(ratio {ratio:.4f})"
    )


def test_the_log_grid_path_is_not_disturbed():
    """The already-correct path must stay correct.

    A fix that trades one grid for the other is not a fix.
    """
    wave = _GRIDS["log_4500_5500"]
    added = _added_width(wave, resolution=1.0e8)
    assert abs(added / _SIGMA_V - 1.0) < 0.005, (
        f"log grid regressed: {added:.3f} km/s for a requested {_SIGMA_V}"
    )


def test_instrument_resolution_is_broadened_correctly_too():
    """The bug is in the pixel scale, so it hits sigma_inst identically — not just sigma_v."""
    wave = _GRIDS["linear_3000_10000"]
    # R = 1000 -> sigma_inst = c / (2.3548 * 1000) = 127.3 km/s
    expected = _C_KM_S / (2.3548200450309493 * 1000.0)

    probe = _probe(wave)
    narrow = apply_lsf(probe, wave, 1.0e8, sigma_v_kms=0.0)
    broad = apply_lsf(probe, wave, 1000.0, sigma_v_kms=0.0)
    added = float(np.sqrt(max(_width_kms(broad, wave) ** 2 - _width_kms(narrow, wave) ** 2, 0.0)))

    assert abs(added / expected - 1.0) < 0.02, (
        f"instrument LSF: expected {expected:.2f} km/s, got {added:.2f} km/s"
    )


# ── the properties the fix must not cost ──────────────────────────


def test_gradient_still_flows_to_sigma_v():
    """The fit needs d(spectrum)/d(sigma_v).

    An interpolation-based fix must stay differentiable.
    """
    wave = jnp.asarray(_GRIDS["linear_3000_10000"])
    probe = jnp.asarray(_probe(_GRIDS["linear_3000_10000"]))

    def total(sigma_v):
        return jnp.sum(apply_lsf(probe, wave, 1.0e8, sigma_v_kms=sigma_v) ** 2)

    g = float(jax.grad(total)(150.0))
    assert np.isfinite(g), f"non-finite gradient {g}"
    assert g != 0.0, "identically-zero gradient — sigma_v became inert"


def test_still_usable_under_jit_with_the_grid_closed_over():
    """The forward model jits with the instrument grid as a constant; that must keep working."""
    wave = jnp.asarray(_GRIDS["linear_3000_10000"])
    probe = jnp.asarray(_probe(_GRIDS["linear_3000_10000"]))

    @jax.jit
    def run(sigma_v):
        return apply_lsf(probe, wave, 1.0e8, sigma_v_kms=sigma_v)

    out = run(200.0)
    assert out.shape == probe.shape
    assert bool(jnp.all(jnp.isfinite(out)))


def test_output_stays_on_the_input_grid():
    """Resampling is internal: callers get their own grid back, same length."""
    wave = _GRIDS["linear_3800_9200"]
    out = apply_lsf(_probe(wave), wave, 1.0e8, sigma_v_kms=_SIGMA_V)
    assert out.shape == (wave.size,)


# ── it has to engage on the path the model actually runs ──────────


def test_the_local_scale_path_engages_inside_predict_spectrum(ssp_data_wne):
    """A fix that never reaches the forward model reads exactly like a working one.

    That is the #1748 / #1770 failure mode, and it is the reason this asserts on
    a real ``predict_spectrum`` rather than on ``apply_lsf`` alone: the piecewise
    path is chosen only if ``wave_obs`` arrives concrete, which is a fact about
    how the forward model closes over the instrument grid, not about ``apply_lsf``.
    """
    from tengri import FREE, Fixed, SEDModel
    from tengri.observation import Spectroscopy, spectrum as spectrum_module

    seen = []
    original = spectrum_module._is_log_uniform

    def _spy(wave):
        result = original(wave)
        seen.append((isinstance(wave, jax.core.Tracer), result))
        return result

    spectrum_module._is_log_uniform = _spy
    try:
        wave_obs = np.linspace(4000.0, 9000.0, 2000)  # linear, as instruments ship
        model = SEDModel.build(
            ssp_data=ssp_data_wne,
            observation=Spectroscopy(wave_obs=wave_obs, resolution=2000.0),
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "none"},
            redshift=Fixed(0.05),
        )
        model.predict_spectrum(model.spec.sample(jax.random.PRNGKey(0)))
    finally:
        spectrum_module._is_log_uniform = original

    assert seen, "apply_lsf was never reached — this test no longer covers the fix"
    traced_flags = [traced for traced, _ in seen]
    verdicts = [uniform for _, uniform in seen]
    assert not all(traced_flags), (
        "wave_obs arrives traced, so the grid cannot be inspected and the "
        "single-scale path is kept; #1791 would still be live on the forward path"
    )
    assert not all(verdicts), (
        f"a linear grid was judged log-uniform (verdicts {verdicts}), so the "
        f"single-FFT path ran and the broadening is still short by wave[0]/lambda"
    )


# ── the fourth copy: the banded operator ──────────────────────────


@pytest.mark.parametrize("lam_probe", [3500.0, 9000.0])
def test_banded_gaussian_lsf_is_scale_correct_across_the_grid(lam_probe):
    """``gaussian_resolution_bands`` read the same single ``dlnwave`` (#1791, fourth site).

    An explicit banded operator can carry a *local* pixel scale, so unlike the
    FFT path it needs no resampling — only the local ``d ln lambda``. The probe
    runs at both ends because the pre-fix error is ``wave[0]/lambda``: 0.857 at
    3500 A and 0.333 at 9000 A on this grid, so a single-wavelength check near
    the blue end would nearly pass.
    """
    from tengri.observation.banded import banded_matvec, gaussian_resolution_bands

    wave = np.linspace(3000.0, 10000.0, 6000)
    resolution = _C_KM_S / (2.3548200450309493 * _SIGMA_V)  # R giving sigma_v exactly

    bands = gaussian_resolution_bands(wave, resolution, n_diag=61)

    delta = np.zeros(wave.size)
    delta[np.argmin(np.abs(wave - lam_probe))] = 1.0
    out = np.asarray(banded_matvec(bands.offsets, bands.data, jnp.asarray(delta)))

    x = np.log(wave)
    w = np.clip(out, 0.0, None)
    w = w / w.sum()
    mean = float((w * x).sum())
    width = np.sqrt(float((w * (x - mean) ** 2).sum())) * _C_KM_S

    ratio = width / _SIGMA_V
    assert abs(ratio - 1.0) < 0.05, (
        f"banded LSF at {lam_probe:.0f} A: asked for {_SIGMA_V} km/s, got "
        f"{width:.2f} km/s (ratio {ratio:.4f}); wave[0]/lambda = "
        f"{wave[0] / lam_probe:.4f}"
    )


def test_flux_is_conserved_to_machine_precision_on_a_linear_grid():
    """Broadening redistributes flux; it must not create or destroy it.

    Pinned tight deliberately. The rejected resample fix conserved this only to
    ~1e-4, and a 1% tolerance here would have accepted it — the tolerance is the
    whole assertion. Each bin's FFT kernel is normalized and the blending weights
    are divided out, so the guarantee is structural, not approximate.
    """
    wave = _GRIDS["linear_3000_10000"]
    probe = _probe(wave)
    out = np.asarray(apply_lsf(probe, wave, 1.0e8, sigma_v_kms=_SIGMA_V))

    before, after = float(probe.sum()), float(out.sum())
    assert abs(after / before - 1.0) < 1e-9, f"flux changed by {after / before - 1.0:.3e}"


def test_a_zero_width_kernel_is_the_identity_on_a_linear_grid():
    """sigma_eff = 0 must return the spectrum untouched, on any grid.

    Guards the property that ruled out resampling: interpolating onto a log grid
    and back perturbs pixel-scale structure by O(1) however finely it is done, so
    a no-op kernel stopped being a no-op. Convolving in place cannot do that.
    """
    wave = _GRIDS["linear_3000_10000"]
    spike = np.ones(wave.size)
    spike[wave.size // 2] = 10.0

    # sigma_lib exceeds sigma_inst, so sigma_eff clamps to exactly zero.
    out = np.asarray(apply_lsf(spike, wave, 100.0, sigma_lib_kms=5000.0, sigma_v_kms=0.0))
    assert np.max(np.abs(out - spike)) < 1e-10, (
        f"zero-width kernel changed the spectrum by {np.max(np.abs(out - spike)):.3e}"
    )
