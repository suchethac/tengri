# SPDX-License-Identifier: BSD-3-Clause
"""Contract: observed-frame IGM/DLA reach photometry, not only predict_obs_sed.

Regression for #932. The IGM (and DLA) transmission is an observed-frame
attenuation that must reach the broadband photometry/spectroscopy projection.
It used to be applied only inside :meth:`SEDModel.predict_obs_sed`, while the
projection (both the exact wave-grid path and the WavePrecomp LUT path) read a
pre-IGM SED — so ``predict_photometry`` returned unattenuated fluxes at high
redshift (on/off ratio 1.0 for every band). These contracts pin:

* IGM attenuates ``predict_photometry`` at z=3 on the exact path,
* IGM attenuates ``predict_photometry`` at z=3 on the WavePrecomp path
  (via the per-filter effective-wavelength approximation), and
* the configured ``igm_model`` is honored (madau dispatches differently from
  Inoue rather than always falling back to Inoue).

Built on the synthetic wide SSP so it runs on CI without the ``data/`` grids.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import (
    FIXED,
    FREE,
    Fixed,
    Observation,
    SEDModel,
    Spectroscopy,
    SpectrumPrecomp,
    WavePrecomp,
)

pytestmark = pytest.mark.contract


def _build(ssp, obs, *, igm_on, approx=None, model="inoue"):
    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(3.0),
        approx=approx,
    )
    if igm_on:
        kwargs["igm"] = {"type": model}
    else:
        kwargs["igm"] = {"type": "none"}
    return SEDModel.build(**kwargs)


def _blue_band_ratio(ssp, obs, *, approx, model="inoue"):
    """on/off ratio of the bluest band (3500 A obs -> rest ~875 A at z=3)."""
    on = _build(ssp, obs, igm_on=True, approx=approx, model=model)
    off = _build(ssp, obs, igm_on=False, approx=approx)
    params = on.spec.sample(jax.random.PRNGKey(1))
    ph_on = np.asarray(on.predict_photometry(params))
    ph_off = np.asarray(off.predict_photometry(params))
    return float(ph_on[0] / max(ph_off[0], 1e-45))


def test_igm_attenuates_photometry_exact_path(synthetic_ssp_wide, synthetic_tophat_obs):
    """Exact wave-grid path: the blue band is strongly IGM-absorbed at z=3."""
    ratio = _blue_band_ratio(synthetic_ssp_wide, synthetic_tophat_obs, approx=None)
    assert ratio < 0.5, f"IGM not reaching predict_photometry (exact): on/off={ratio:.3f}"


def test_igm_attenuates_photometry_waveprecomp_path(synthetic_ssp_wide, synthetic_tophat_obs):
    """WavePrecomp LUT path: IGM applied via the effective-wavelength factor."""
    ratio = _blue_band_ratio(synthetic_ssp_wide, synthetic_tophat_obs, approx=WavePrecomp())
    assert ratio < 0.9, f"WavePrecomp photometry silently drops IGM: on/off={ratio:.3f}"


def test_igm_model_selection_is_honored(synthetic_ssp_wide, synthetic_tophat_obs):
    """madau and Inoue give *different* blue-band attenuation (not both Inoue)."""
    r_inoue = _blue_band_ratio(
        synthetic_ssp_wide, synthetic_tophat_obs, approx=None, model="inoue"
    )
    r_madau = _blue_band_ratio(
        synthetic_ssp_wide, synthetic_tophat_obs, approx=None, model="madau"
    )
    assert not np.isclose(r_inoue, r_madau, rtol=1e-3), (
        f"igm_model ignored: inoue={r_inoue:.4f} == madau={r_madau:.4f}"
    )


def test_predict_obs_sed_runs_with_igm_and_dla(synthetic_ssp_wide, synthetic_tophat_obs):
    """predict_obs_sed must accept the unified use_dla/dla_* dispatch.

    Guards against the emission_helpers shim regression: predict_obs_sed calls
    the flat igm_absorption with use_dla/dla_* kwargs, which a stale wrapper
    copy did not accept — crashing every IGM-enabled call.
    """
    model = SEDModel.build(
        ssp_data=synthetic_ssp_wide,
        observation=synthetic_tophat_obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(3.0),
        igm={"type": "inoue", "dla": {"log_n_hi": Fixed(21.0)}},
    )
    params = model.spec.sample(jax.random.PRNGKey(1))
    sed = np.asarray(model.predict_obs_sed(params).sed)
    assert np.all(np.isfinite(sed)) and sed.shape[0] > 0


def test_igm_attenuates_spectrum_precomp_path(synthetic_ssp_wide):
    """SpectrumPrecomp LUT path applies IGM per-pixel (exact, not per-band)."""
    import jax.numpy as jnp

    wave_obs = jnp.linspace(4000.0, 9000.0, 200)  # z=3 -> rest 1000-2250 A
    obs = Observation(spectroscopy=Spectroscopy(wave_obs=wave_obs, resolution=500.0))
    common = dict(
        ssp_data=synthetic_ssp_wide,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust_attenuation={"type": "two_component", "law": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        redshift=Fixed(3.0),
        approx=SpectrumPrecomp(),
    )
    on = SEDModel.build(igm={"type": "inoue"}, **common)
    off = SEDModel.build(igm={"type": "none"}, **common)
    params = on.spec.sample(jax.random.PRNGKey(1))
    s_on = np.asarray(on.predict_spectrum(params))
    s_off = np.asarray(off.predict_spectrum(params))
    # blue end (4000 A obs -> rest ~1000 A, below Ly-alpha at z=3) is absorbed.
    assert s_on[:30].sum() < 0.9 * s_off[:30].sum(), "SpectrumPrecomp drops IGM"


def _igm_parity_ratios(ssp, obs, *, redshift_spec, params=None, approx=None):
    """Per-band WavePrecomp/exact flux ratio with IGM on and dust zeroed.

    Dust-free so the only LUT-path approximation in play is the IGM
    band factor — the dust Taylor projection (#731) stays out of the
    comparison.
    """
    common = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_bc": 0.0,
            "tau_diff": 0.0,
        },
        neb={"type": "none"},
        redshift=redshift_spec,
        igm={"type": "inoue"},
    )
    exact = SEDModel.build(**common)
    lut = SEDModel.build(approx=approx or WavePrecomp(), **common)
    params = params or {}
    ph_exact = np.asarray(exact.predict_photometry(params))
    ph_lut = np.asarray(lut.predict_photometry(params))
    return ph_lut / ph_exact


# Per-band |LUT/exact − 1| bounds at z=3 on the synthetic tophats. Band 1
# (4800 Å) straddles the Lyman-α edge (4864 Å observed) — the #1026 case: a
# point sample at λ_eff was 18% off; the filter-weighted mean ⟨T⟩ tracks the
# exact path to the in-band SED-slope × T covariance. Band 0 (2940–4060 Å,
# rest 735–1015 Å) straddles the *Lyman limit* — in-band T contrast ~1 against
# the synthetic λ⁻² SED rising into the absorption, the worst case for any
# separable band factor — so it keeps a percent-level covariance residual
# (documented in ``predict_via_precomp``; ``approx=None`` is the precision
# path for dropout bands). Bands 2–4 are unabsorbed and must stay exact.
_PARITY_TOL = np.array([0.08, 0.02, 0.01, 0.01, 0.01])


def test_waveprecomp_igm_is_band_averaged_fixed_z(synthetic_ssp_wide, synthetic_tophat_obs):
    """Regression #1026 (fixed-z LUT): IGM must be filter-averaged, not point-sampled."""
    ratios = _igm_parity_ratios(synthetic_ssp_wide, synthetic_tophat_obs, redshift_spec=Fixed(3.0))
    assert np.all(np.abs(ratios - 1.0) < _PARITY_TOL), (
        f"WavePrecomp IGM band factor biased vs exact path: LUT/exact = {ratios}"
    )


def test_waveprecomp_igm_is_band_averaged_free_z(synthetic_ssp_wide, synthetic_tophat_obs):
    """Regression #1026 (free-z ztable): same contract through the z-interp path."""
    from tengri import Uniform

    ratios = _igm_parity_ratios(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        redshift_spec=Uniform(2.0, 3.5),
        params={"redshift": 3.0},
    )
    assert np.all(np.abs(ratios - 1.0) < _PARITY_TOL), (
        f"free-z WavePrecomp IGM band factor biased vs exact path: LUT/exact = {ratios}"
    )
