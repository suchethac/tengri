# SPDX-License-Identifier: BSD-3-Clause
"""``smc`` and ``lmc`` must be usable on the path every fit takes.

Both are ``status='production'`` attenuation laws, and both are among the most
used curves in SED fitting -- SMC-like extinction is the standard choice at high
redshift. Both worked on the exact path and **crashed** under ``WavePrecomp``::

    ValueError: Incompatible shapes for broadcasting: shapes=[(1, 25, 5, 5), (6, 1)]

Every fit surface resolves ``approx="auto"`` to ``WavePrecomp`` for photometry
(``Fitter``, ``PopulationFitter``, ``CatalogFitter``), so selecting ``smc``
raised at **fit** time rather than at build time -- after the user had already
chosen the law, built the model, and started a fit.

Cause: :func:`_pei92_curve` was written for a 1-D wavelength grid. It spelled the
Drude component axis as ``lam_i[:, None]`` and the wavelength axis as
``wave_um[None, :]``, which pins the input to rank 1. ``WavePrecomp`` evaluates
the curve on a rank-4 grid (sub-bands x ages x filters), where the 6-component
axis collided with the filter axis. Every other law is elementwise in wavelength
and broadcast at any rank; only the Pei-92 pair sums over a component axis.

The fix moves the component axis to the trailing position so the reduction is
``axis=-1`` and the function is rank-agnostic.

Assertions here are deliberately three, because "it no longer crashes" is a weak
bar that a fallback to ``calzetti`` would also clear:

1. both laws predict finitely on **both** paths (the crash);
2. the precompute result tracks the exact one (the LUT approximates *this* curve);
3. the result still differs from ``calzetti`` (the law is applied, not swapped).
"""

from __future__ import annotations

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel, SSPData, WavePrecomp
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_PEI92_LAWS = ("smc", "lmc")


@pytest.fixture(scope="module")
def uv_optical_ssp() -> SSPData:
    """SSP spanning 100 A - 10 um, so a UV-steep law has UV to act on."""
    ages = jnp.linspace(-3.0, 1.14, 25)
    lgmet = jnp.array([-4.0, -2.65, -1.3])
    wave = jnp.logspace(2.0, 5.0, 1200)
    base = (5000.0 / wave) ** 2
    flux = (
        base[None, None, :]
        * (1.0 + 0.15 * (ages - ages.mean()))[None, :, None]
        * (1.0 + 0.10 * (lgmet - lgmet.mean()))[:, None, None]
    )
    return SSPData(
        ssp_wave=wave,
        ssp_flux=jnp.abs(flux) + 1e-12,
        ssp_lg_age_gyr=ages,
        ssp_lgmet=lgmet,
    )


@pytest.fixture(scope="module")
def uv_optical_obs() -> Observation:
    """Five bands with UV coverage -- an SMC curve is defined by its UV rise."""

    def _tophat(center: float, frac: float = 0.16, n: int = 40) -> FilterCurve:
        wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
        trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
        return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")

    centers = (1500.0, 2175.0, 3500.0, 6200.0, 9000.0)
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in centers)))


def _photometry(ssp, obs, law: str, approx) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FIXED},
            dust={"type": "single_component", "law_bc": law, "tau_v": Fixed(1.0)},
            redshift=Fixed(0.5),
            approx=approx,
        )
        params = model.spec.sample(jax.random.PRNGKey(0))
        return np.asarray(model.predict_photometry(params))


@pytest.mark.parametrize("law", _PEI92_LAWS)
def test_pei92_law_predicts_on_both_paths(law, uv_optical_ssp, uv_optical_obs):
    """The regression: WavePrecomp raised a broadcasting error, exact did not."""
    exact = _photometry(uv_optical_ssp, uv_optical_obs, law, None)
    precomp = _photometry(uv_optical_ssp, uv_optical_obs, law, WavePrecomp())

    assert np.all(np.isfinite(exact)), f"{law}: exact path produced non-finite photometry"
    assert np.all(np.isfinite(precomp)), (
        f"{law}: WavePrecomp produced non-finite photometry. Every fitter resolves "
        "approx='auto' to WavePrecomp for photometry, so this is the fit path."
    )
    assert np.all(exact > 0.0) and np.all(precomp > 0.0), f"{law}: photometry is not positive"


@pytest.mark.parametrize("law", _PEI92_LAWS)
def test_pei92_precompute_tracks_the_exact_curve(law, uv_optical_ssp, uv_optical_obs):
    """The LUT must approximate *this* law, not merely return something finite."""
    exact = _photometry(uv_optical_ssp, uv_optical_obs, law, None)
    precomp = _photometry(uv_optical_ssp, uv_optical_obs, law, WavePrecomp())

    rel = np.max(np.abs(precomp - exact) / np.abs(exact))
    assert rel < 2e-2, (
        f"{law}: WavePrecomp disagrees with the exact path by {rel:.3e}. The LUT is an "
        "approximation of this curve, not a different curve."
    )


@pytest.mark.parametrize("law", _PEI92_LAWS)
def test_pei92_curve_is_float32_clean_at_every_rank(law):
    """The reduction must hold in float32, at rank 1 and at the rank the LUT uses.

    ``ratio ** (-n_i)`` with ``n_i`` up to 4.5 is a plausible overflow site: at
    the blue end of a 1 A grid the ratio against the 25 um Drude component is
    ~4e-6, and ``(4e-6) ** -4.5`` is ~1e25 -- inside float32's range, but not by
    a wide margin, so it is measured rather than assumed.

    Scope note: this asserts the *curve*, which is what the rank fix touches.
    End-to-end pure-float32 ``predict_photometry`` is non-finite for **every**
    attenuation law including ``calzetti`` (measured), so it is the known
    float32 track (#1206 / #1415 / #1719) rather than anything specific to
    Pei-92, and asserting it here would pin an unrelated defect.
    """
    from tengri.components.dust import attenuation as _att

    args = {
        "smc": (_att._SMC_LAM, _att._SMC_A, _att._SMC_B, _att._SMC_N, _att._SMC_RV),
        "lmc": (_att._LMC_LAM, _att._LMC_A, _att._LMC_B, _att._LMC_N, _att._LMC_RV),
    }[law]

    with jax.enable_x64(True):
        k64 = np.asarray(_att._pei92_curve(jnp.logspace(0.0, 7.0, 3000), *args), dtype=np.float64)

    with jax.enable_x64(False):
        out32 = _att._pei92_curve(jnp.logspace(0.0, 7.0, 3000), *args)
        assert out32.dtype == jnp.float32, f"{law}: expected float32, got {out32.dtype}"
        k32 = np.asarray(out32, dtype=np.float64)

        # the rank the LUT actually evaluates on
        grid = jnp.reshape(jnp.logspace(0.0, 7.0, 1 * 25 * 5 * 5), (1, 25, 5, 5))
        rank4 = _att._pei92_curve(grid, *args)
        assert jnp.all(jnp.isfinite(rank4)), f"{law}: rank-4 curve is non-finite in float32"
        assert rank4.shape == grid.shape

    assert np.all(np.isfinite(k32)), f"{law}: float32 curve is non-finite over 1 A - 1 mm"
    rel = np.max(np.abs(k32 - k64) / np.where(k64 > 0, k64, 1.0))
    assert rel < 1e-4, f"{law}: float32 curve departs from float64 by {rel:.3e}"


@pytest.mark.parametrize("law", _PEI92_LAWS)
def test_pei92_law_is_not_silently_calzetti(law, uv_optical_ssp, uv_optical_obs):
    """A fallback to the default law would satisfy the two assertions above.

    SMC and LMC have no (SMC) or a weak (LMC) 2175 A bump and a much steeper UV
    rise than Calzetti, so a UV band must separate them on both paths.
    """
    for label, approx in (("exact", None), ("WavePrecomp", WavePrecomp())):
        law_phot = _photometry(uv_optical_ssp, uv_optical_obs, law, approx)
        calz = _photometry(uv_optical_ssp, uv_optical_obs, "calzetti", approx)
        rel = np.max(np.abs(law_phot - calz) / np.abs(calz))
        assert rel > 1e-3, (
            f"{law} is indistinguishable from calzetti on the {label} path "
            f"(max rel {rel:.3e}) -- the law is not reaching the curve."
        )
