# SPDX-License-Identifier: BSD-3-Clause
r"""X-ray and shock emission must be finite in pure float32 (#1206).

Both subsystems failed the same way — an absolute erg/s luminosity that exceeds
the float32 maximum (3.4e38) multiplied by a near-zero spectral shape, giving
``inf * 0 = nan``, even though the *product* (~1e22–1e26 erg/s/Hz) is perfectly
representable:

* **X-ray** (``xray.py``): the XRB and hot-gas normalizations are literal powers
  of ten — ``10**40.28`` (HMXB), ``10**40.276`` (LMXB) and ``10**38.919``
  (hot gas). Each already overflows *before* SFR or M_star is applied. The
  float32 path folds the band integral into the exponent
  (``pow10(log_L - log10(band_int))``) so no out-of-range value forms.
* **Shock** (``shock_model.py``): the shock Hα luminosity is ~1e41 erg/s, from
  either ``shock_log_lhalpha`` directly or ``frac × 1e-3 × L_bol`` (and ``L_bol``
  itself, a ~1e44 erg/s integral, overflows too). Since the shock SED is
  *exactly* linear in that luminosity (verified ×10/dex), the float32 path
  evaluates the shape at unit Hα and re-applies the true scale with
  ``apply_log10_scale``.

Both are float32-gated, so float64 is bit-identical.

The shock gate is worth its own note: it keys on ``sed_in.dtype``, **not**
``wave.dtype``. The wavelength grid arrives as float64 even when the pipeline is
computing in float32, so gating on ``wave`` silently disables the fix — which is
exactly what happened during development, and what this test would catch.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_DUST = {
    "law": "power_law",
    "type": "two_component",
    "all_params": FIXED,
    "tau_diff": 0.5,
    "tau_bc": 0.3,
}
_DUST_EMISSION = {"type": "dale2014", "all_params": FIXED}


def _rest_sed(ssp, dtype, **groups):
    obs = Observation(photometry=Photometry.from_names(["sdss_g", "sdss_r", "wise_w3", "wise_w4"]))
    model = SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={
            "type": "delayed",
            "all_params": FIXED,
            "log_total_mass": Uniform(9.0, 11.0),
            "tau_gyr": 1.0,
            "age_gyr": 5.0,
        },
        redshift=Fixed(0.5),
        **groups,
    )
    p = {"sfh_delayed_log_total_mass": jnp.asarray(10.0, dtype=dtype)}
    return np.asarray(model.predict(p).rest_sed())


def _f32_vs_f64(ssp, tol, **groups):
    with jax.enable_x64(True):
        ref = _rest_sed(ssp, jnp.float64, **groups)
    with jax.enable_x64(False):
        f32 = _rest_sed(ssp, jnp.float32, **groups)
    assert np.all(np.isfinite(f32)), "rest_sed is non-finite in pure float32"
    peak = np.abs(ref).max()
    live = np.abs(ref) > 1e-6 * peak
    rel = np.abs(f32[live] - ref[live]) / np.abs(ref[live])
    assert rel.max() < tol, f"float32 vs float64 max rel = {rel.max():.2e} (tol {tol:.0e})"


@pytest.mark.parametrize("xray_type", ["simple", "lopez24"])
def test_xray_finite_and_matches_f64_in_float32(ssp_bare, xray_type):
    """X-ray emission is finite and float64-accurate in pure float32."""
    _f32_vs_f64(
        ssp_bare,
        1e-4,
        dust_attenuation=_DUST,
        dust_emission=_DUST_EMISSION,
        xray={"type": xray_type},
    )


@pytest.mark.parametrize(
    ("shock_group", "tol"),
    [
        # Sharp emission-line cores land slightly differently on the grid at
        # float32 resolution, so both shock modes carry a small (still tiny)
        # tolerance. The point of the guard is finiteness plus no gross error —
        # before the fix these were NaN, not 1e-3 off.
        pytest.param({"frac": 0.3}, 5e-3, id="norm=frac"),
        pytest.param({"norm": "lhalpha", "log_lhalpha": 41.0}, 5e-3, id="norm=lhalpha"),
    ],
)
def test_shock_finite_and_matches_f64_in_float32(ssp_bare, shock_group, tol):
    """Shock emission is finite and float64-accurate in pure float32."""
    _f32_vs_f64(ssp_bare, tol, dust_attenuation=_DUST, shock=shock_group)


def test_xray_hotgas_kernel_is_finite_in_float32():
    """The bare hot-gas kernel must not form ``10**38.919`` (= 8.3e38 > float32 max)."""
    from tengri.components.xray.xray import xray_total_terms

    wave = np.geomspace(1.0, 1.0e4, 2000)
    with jax.enable_x64(False):
        terms = xray_total_terms(
            jnp.asarray(wave, dtype=jnp.float32),
            sfr=jnp.float32(1.0),
            log_mstar=jnp.float32(10.0),
            redshift=jnp.float32(0.5),
        )
    for name, value in terms.items():
        assert np.all(np.isfinite(np.asarray(value))), (
            f"X-ray '{name}' term is non-finite in float32 — a 10**40-ish "
            "normalization overflowed before being divided by the band integral"
        )
