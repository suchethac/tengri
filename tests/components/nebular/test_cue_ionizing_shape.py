# SPDX-License-Identifier: BSD-3-Clause
"""Contract: Cue's effective ionizing-spectrum shape is the Q_H-weighted, luminosity
additive combination of the young age bins — continuous, differentiable, and faithful
to the composite spectrum (#1018).

Cue is trained on *time-averaged* ionizing spectra, so the shape it receives must
represent the whole young population. The former ``i7 = ionspec_all[argmax(weighted_qh)]``
picked ONE age bin, which:

* made the forward **discontinuous** — the dominant bin flips as metallicity (or the
  SFH) varies, stepping [OIII] by ~33 % in 0.001 dex, and
* forced ``d(shape)/d(ssp_weights) == 0`` (``argmax`` has no gradient, and
  ``ionspec_all`` does not depend on the weights), silently starving HMC.

Note the correct rule is NOT an arithmetic mean of the 7 parameters: ``logLratio`` is a
log of a ratio of *integrated* segment luminosities, and luminosities add linearly, so a
plain mean is a geometric mean where an arithmetic one is required (it biases [OIII] by
~12 %, worse than the argmax). Segment luminosities add; slopes blend by per-segment
luminosity weight.

Data-gated (needs a bare-stellar SSP for Cue); skips in CI.
"""

from __future__ import annotations

from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, load_ssp_data
from tengri.observation.line_flux_data import LineFluxData

pytestmark = pytest.mark.contract

_BARE = "data/fsps_prsc_miles_chabrier.h5"
_LINES = ["Halpha", "Hbeta", "OIII_5007"]
# On the FSPS/MILES grid the old argmax flipped the dominant age bin (18 -> 0) here.
_FLIP_MET = -1.0955


def _model():
    import warnings

    if not Path(_BARE).is_file():
        pytest.skip(f"missing bare SSP {_BARE}")
    ssp = load_ssp_data(_BARE)
    ld = LineFluxData.from_dict({n: (1e-16, 1e-17) for n in _LINES})
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r"]), line_fluxes=ld)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation=None,
            neb={"type": "cue", "all_params": FIXED},
            redshift=Fixed(0.15),
        )
    return m, ld.wavelengths, ssp


def _o3_per_qh(m, lw, params):
    st = m.predict_state(params)
    q = jnp.sum(st.derived["nion"])
    return m.predict_line_fluxes(params, target_wavelengths=lw, state=st)[2] / q


def test_ionizing_shape_is_continuous_across_the_old_argmax_flip():
    """[OIII]/Q_H must not step where the dominant age bin used to flip.

    Regression for the ~33 % jump: a dense sweep (0.001 dex) straddling the flip
    must be smooth. Q_H and Hbeta were always continuous (a sum, and a
    shape-insensitive recombination line) — only the shape-sensitive [OIII] broke.
    """
    import warnings

    m, lw, _ = _model()
    base = dict(m.spec.sample(jax.random.PRNGKey(0)))
    mets = np.arange(_FLIP_MET - 0.010, _FLIP_MET + 0.0101, 0.001)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        vals = np.array(
            [
                float(_o3_per_qh(m, lw, {**base, "met_logzsol": jnp.asarray(float(x))}))
                for x in mets
            ]
        )
    steps = np.abs(vals[1:] / vals[:-1] - 1.0)
    assert steps.max() < 2e-2, (
        f"[OIII]/Q_H steps by {steps.max():.1%} across met={mets[int(steps.argmax())]:.4f} "
        "— the ionizing shape is discontinuous (argmax age-bin flip regressed)"
    )
    # and it must be monotone here, not merely small-stepped
    assert np.all(np.diff(vals) < 0) or np.all(np.diff(vals) > 0), "shape is non-monotone"


def test_ionizing_shape_gradient_wrt_sfh_is_nonzero_and_exact():
    """d([OIII]/Q_H)/d(SFH shape) must be nonzero and match finite differences.

    [OIII]/Q_H isolates the ionizing-spectrum shape (Q_H divides out). Under the old
    ``argmax`` the shape was piecewise-constant in the SFH, so this gradient was
    identically zero — HMC saw no SFH -> shape -> forbidden-line signal at all.
    """
    import warnings

    m, lw, _ = _model()
    base = dict(m.spec.sample(jax.random.PRNGKey(0)))
    base["met_logzsol"] = jnp.asarray(-1.20)  # away from any cell edge
    p_name = "sfh_dpl_tau_gyr"
    t0 = float(base[p_name])

    def shape_metric(tau):
        return _o3_per_qh(m, lw, {**base, p_name: tau})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        g_ad = float(jax.grad(shape_metric)(jnp.asarray(t0)))
        h = 1e-5 * t0
        g_fd = (
            float(shape_metric(jnp.asarray(t0 + h))) - float(shape_metric(jnp.asarray(t0 - h)))
        ) / (2 * h)

    assert abs(g_ad) > 0.0, "d(shape)/d(SFH) is zero — argmax shape selection regressed"
    assert abs(g_ad / g_fd - 1.0) < 1e-3, f"gradient wrong: autodiff {g_ad:.6e} vs FD {g_fd:.6e}"


def test_effective_shape_matches_a_refit_of_the_composite_spectrum():
    """The combined 7 params must reproduce a direct fit of the true composite spectrum.

    Ground truth: sum the young, met-weighted SSP spectra and re-fit the broken power
    law. The luminosity-additive combination lands within a few percent; an arithmetic
    mean of the parameters does not (it biases the log-ratios).
    """
    import warnings

    from tengri.components.nebular.ionizing_spectrum import (
        MAX_NEB_LOG_AGE,
        fit_ionizing_spectrum,
        interpolate_ionizing_params,
        interpolate_ionizing_seglum,
    )
    from tengri.parameters.translate import LOG10_ZSUN

    m, _lw, ssp = _model()
    be = m._nebular_backend
    met = -1.20
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    p["met_logzsol"] = jnp.asarray(met)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        st = m.predict_state(p)
    jw = np.asarray(st.derived["joint_weights"])
    log_ages = np.log10(np.asarray(st.derived["ssp_ages_yr"]))
    young = log_ages <= MAX_NEB_LOG_AGE
    wave = np.asarray(ssp.ssp_wave)

    # ground truth: fit the composite young ionizing spectrum
    spec_a = np.einsum("ma,maw->aw", jw, np.asarray(ssp.ssp_flux))
    truth = fit_ionizing_spectrum(wave, spec_a[young].sum(axis=0))
    t_idx = np.array([truth[f"ionspec_index{i}"] for i in (1, 2, 3, 4)])
    t_lr = np.array([truth[f"ionspec_logLratio{i}"] for i in (1, 2, 3)])

    # the rule the backend uses, rebuilt from the same tables
    log_z = met + LOG10_ZSUN
    i_all, _q = jax.vmap(
        lambda la: interpolate_ionizing_params(
            be._ionspec_table, be._logqion_table, be._ssp_lgmet, be._ssp_log_age_yr, log_z, la
        )
    )(jnp.asarray(log_ages))
    seg = jax.vmap(
        lambda la: interpolate_ionizing_seglum(
            be._seglum_table, be._ssp_lgmet, be._ssp_log_age_yr, log_z, la
        )
    )(jnp.asarray(log_ages))
    aw = np.asarray(st.derived["age_weights"])
    w = np.where(young & (aw > 0), aw, 0.0)[:, None]
    seg_w = w * np.asarray(10.0**seg)
    Lk = seg_w.sum(axis=0)
    alpha = (seg_w * np.asarray(i_all)[:, :4]).sum(axis=0) / np.maximum(Lk, 1e-300)
    lr = np.diff(np.log10(np.maximum(Lk, 1e-300)))

    assert np.max(np.abs(alpha - t_idx)) < 0.5, f"slopes off: {alpha} vs {t_idx}"
    assert np.max(np.abs(lr - t_lr)) < 0.05, f"logLratios off: {lr} vs {t_lr}"
