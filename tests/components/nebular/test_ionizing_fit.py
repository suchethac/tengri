# SPDX-License-Identifier: BSD-3-Clause
"""The four-segment ionizing-spectrum fit: exact minimizer, gradients, edge cases.

``fit_ionizing_shape`` is the one solver for Cue's shape parameters (Li et al.
2025, ApJ 986, 9): it builds the per-SSP table and fits the composite spectrum of
a population. These tests hold it to the reference objective with an independent
optimizer, and to the limits a fit of this form must satisfy.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.optimize import minimize, minimize_scalar

from tengri.components.nebular._constants import _C_AA, _H_PLANCK
from tengri.components.nebular.ionizing_fit import fit_ionizing_shape, ionizing_layout
from tengri.components.nebular.ionizing_spectrum import SEGMENT_EDGES

pytestmark = pytest.mark.contract

_EDGES = np.asarray(SEGMENT_EDGES)
_WEIGHTS = jnp.arange(1.0, 8.0)  # a generic linear readout of the seven parameters


@pytest.fixture(scope="module")
def ion_grid(ssp_data_fsps):
    """SSP ionizing slice: wavelengths, flux (n_met, n_age, n_ion), log ages, layout."""
    wave = np.asarray(ssp_data_fsps.ssp_wave, dtype=np.float64)
    n_ion = int(np.count_nonzero(wave < 2.0 * 911.76))
    flux = np.asarray(ssp_data_fsps.ssp_flux, dtype=np.float64)[..., :n_ion]
    log_age = np.asarray(ssp_data_fsps.ssp_lg_age_gyr) + 9.0
    return wave[:n_ion], flux, log_age, ionizing_layout(wave[:n_ion], _EDGES)


def _log10(x):
    return np.log10(np.maximum(x, np.finfo(np.float64).tiny))


def _young_composite(flux, log_age, met_index):
    """Constant-SFH composite over ages up to 30 Myr at one metallicity node."""
    young = log_age <= np.log10(3e7)
    return flux[met_index, young].sum(axis=0)


def _segment_objective(wave, flux, k):
    """The reference objective of segment ``k`` in NumPy, independent of the solver."""
    ind = np.array([np.max(np.where(wave <= e)[0]) for e in _EDGES[1:]]) + 1
    ind = np.insert(ind, 0, 0)
    ref = np.median(flux[ind[-1] - 1 : ind[-1] + 1])
    norm = 1e-18 / max(ref, 1e-99)
    nf = np.clip(flux * norm, 1e-70 * norm, np.inf)
    sw, sf = wave[ind[k] : ind[k + 1]], nf[ind[k] : ind[k + 1]]
    x, y = np.log10(sw), np.log10(np.maximum(sf, 1e-99))
    nu = _C_AA / sw
    q = abs(np.trapezoid((sf / (_H_PLANCK * nu))[::-1], x=nu[::-1]))
    log_q = np.log10(max(q, 1e-99))
    lo, hi, n = sw[0], sw[-1], sw.size

    def objective(p):
        a, b = p
        pred_q = b - np.log10(_H_PLANCK) + np.log10(abs((hi**a - lo**a) / a))
        return 0.5 * np.sum((y - b - a * x) ** 2) + 0.5 * n * (log_q - pred_q) ** 2

    return objective, x, y


def test_slopes_are_the_minimizer_of_the_reference_objective(ion_grid):
    """An independent BFGS polish of each segment's 2-D objective lands on the solver's slope."""
    wave, flux, log_age, layout = ion_grid
    for met_index in (0, flux.shape[0] // 2, flux.shape[0] - 1):
        spectrum = _young_composite(flux, log_age, met_index)
        slopes = np.asarray(fit_ionizing_shape(jnp.asarray(_log10(spectrum)), layout).slopes)
        for k in range(4):
            objective, x, y = _segment_objective(wave, spectrum, k)
            start = [slopes[k] + 0.3, float(np.mean(y - (slopes[k] + 0.3) * x))]
            polish = minimize(objective, start, method="BFGS", options={"gtol": 1e-10})
            at_solver = minimize_scalar(lambda b, a=slopes[k], f=objective: f([a, b])).fun
            assert at_solver <= polish.fun + 1e-9 * max(1.0, polish.fun), (met_index, k)
            assert abs(polish.x[0] - slopes[k]) < 1e-4, (met_index, k, polish.x[0], slopes[k])


def test_shape_does_not_depend_on_normalization(ion_grid):
    """Scaling the spectrum by any constant leaves the seven parameters unchanged."""
    _, flux, log_age, layout = ion_grid
    log_flux = jnp.asarray(_log10(_young_composite(flux, log_age, flux.shape[0] // 2)))
    base = fit_ionizing_shape(log_flux, layout).i7
    for shift in (-30.0, -3.0, 12.0):
        np.testing.assert_allclose(
            fit_ionizing_shape(log_flux + shift, layout).i7, base, rtol=0, atol=1e-9
        )


@pytest.mark.limit
def test_pure_power_law_recovers_its_slopes():
    """A spectrum that is a power law on each segment returns those slopes."""
    wave = np.geomspace(1.0, 1800.0, 6000)
    layout = ionizing_layout(wave, _EDGES)
    truth = np.array([3.0, 1.5, 0.5, -0.8])
    seg = np.searchsorted(_EDGES[1:], wave, side="left").clip(0, 3)
    log_flux = truth[seg] * np.log10(wave) - 2.0 * seg
    slopes = np.asarray(fit_ionizing_shape(jnp.asarray(log_flux), layout).slopes)
    np.testing.assert_allclose(slopes, truth, atol=2e-3)


@pytest.mark.gradient
def test_gradient_matches_finite_differences(ion_grid):
    """Reverse-mode gradient of a readout of the shape matches central differences."""
    _, flux, log_age, layout = ion_grid
    x0 = jnp.asarray(_log10(_young_composite(flux, log_age, flux.shape[0] // 2)))

    def readout(lf):
        return jnp.sum(fit_ionizing_shape(lf, layout).i7 * _WEIGHTS)

    grad = np.asarray(jax.grad(readout)(x0))
    assert np.all(np.isfinite(grad)) and np.any(grad != 0.0)
    eps = 1e-6
    for i in np.argsort(-np.abs(grad))[:6]:
        step = jnp.zeros_like(x0).at[i].set(eps)
        fd = (float(readout(x0 + step)) - float(readout(x0 - step))) / (2 * eps)
        assert abs(grad[i] - fd) <= 1e-5 * max(abs(fd), 1.0), (i, grad[i], fd)


@pytest.mark.gradient
@pytest.mark.parametrize("x64", [True, False], ids=["float64", "float32"])
@pytest.mark.parametrize("population", ["old_only", "zero_hard_segment"])
def test_no_ionizing_flux_is_finite_with_finite_gradient(ion_grid, population, x64):
    """Exact zeros (old-only population, or a segment with no flux) stay finite, gradient too.

    The implicit gradient divides by the slope curvature at the root, so a segment
    with no usable flux is the case that could produce an infinite gradient.
    """
    wave, flux, log_age, layout = ion_grid
    met_index = flux.shape[0] // 2
    if population == "old_only":
        spectrum = flux[met_index, log_age >= 9.5].sum(axis=0)
    else:
        spectrum = _young_composite(flux, log_age, met_index)
        spectrum = np.where(wave < _EDGES[1], 0.0, spectrum)

    def readout(t):
        positive = t > 0
        lf = jnp.where(positive, jnp.log10(jnp.where(positive, t, 1.0)), -jnp.inf)
        return jnp.sum(fit_ionizing_shape(lf, layout).i7 * _WEIGHTS)

    dtype = jnp.float64 if x64 else jnp.float32
    with jax.enable_x64(x64):
        total = jnp.asarray(spectrum / spectrum.max(), dtype=dtype)
        value = readout(total)
        grad = jax.grad(readout)(total)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))


def test_float32_within_1e3_of_float64(ion_grid):
    """The fit run in float32 agrees with float64 to 1e-3 in every parameter."""
    _, flux, log_age, layout = ion_grid
    log_flux = _log10(_young_composite(flux, log_age, flux.shape[0] // 2))
    ref = np.asarray(fit_ionizing_shape(jnp.asarray(log_flux, jnp.float64), layout).i7)
    with jax.enable_x64(False):
        got = np.asarray(fit_ionizing_shape(jnp.asarray(log_flux, jnp.float32), layout).i7)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-3)


def test_single_age_composite_reproduces_the_table_fit(ion_grid):
    """One age bin with any weight gives the per-SSP fit the table is built from."""
    from tengri.components.nebular.ionizing_spectrum import fit_ionizing_spectrum

    wave, flux, log_age, layout = ion_grid
    met_index, age_index = flux.shape[0] // 2, int(np.argmin(np.abs(log_age - 6.5)))
    table = fit_ionizing_spectrum(wave, flux[met_index, age_index])
    raw = np.concatenate([table["powerlaw_params"][:, 0], np.diff(table["log_seglum"])])
    got = fit_ionizing_shape(jnp.asarray(_log10(3.7e5 * flux[met_index, age_index])), layout)
    np.testing.assert_allclose(np.asarray(got.i7), raw, rtol=0, atol=1e-9)


def _stationarity_residual(log_flux, layout):
    """``|phi'(alpha*)| / phi''(alpha*)`` per segment at the returned slopes: a Newton step."""
    from tengri.components.nebular.ionizing_fit import _profile, _segment_stats

    stats, *_ = _segment_stats(log_flux, layout)
    slopes = fit_ionizing_shape(log_flux, layout).slopes
    out = []
    for k in range(4):
        s_k = tuple(v[k] for v in stats)
        d1 = jax.grad(_profile)(slopes[k], s_k)
        d2 = jax.grad(jax.grad(_profile))(slopes[k], s_k)
        out.append(float(jnp.abs(d1) / jnp.maximum(jnp.abs(d2), 1e-300)))
    return np.array(out)


def test_newton_reaches_the_stationary_point_on_every_ssp_row_and_composites(ion_grid):
    """At the returned slope the remaining Newton step is below 1e-8 in slope units.

    Every (metallicity, age) row of the SSP grid, and constant-SFR composites over the
    ages below 3, 10, 30 and 100 Myr and all ages at every metallicity.
    """
    _, flux, log_age, layout = ion_grid
    spectra = [flux[m, a] for m in range(flux.shape[0]) for a in range(flux.shape[1])]
    for m in range(flux.shape[0]):
        for cut in (6.5, 7.0, 7.5, 8.0, 11.0):
            spectra.append(flux[m, log_age <= cut].sum(axis=0))
    worst = 0.0
    for spec in spectra:
        if not np.any(spec > 0):
            continue
        worst = max(
            worst, float(np.max(_stationarity_residual(jnp.asarray(_log10(spec)), layout)))
        )
    assert worst < 1e-8, worst


@pytest.mark.gradient
def test_all_zero_spectrum_is_finite_in_float32(ion_grid):
    """A spectrum with no flux anywhere: values and gradients finite, in float32."""
    _, flux, _, layout = ion_grid

    def readout(lf):
        return jnp.sum(fit_ionizing_shape(lf, layout).i7 * _WEIGHTS)

    with jax.enable_x64(False):
        lf = jnp.full(flux.shape[-1], -jnp.inf, dtype=jnp.float32)
        value = readout(lf)
        grad = jax.grad(readout)(lf)
    assert np.isfinite(float(value))
    assert np.all(np.isfinite(np.asarray(grad)))
