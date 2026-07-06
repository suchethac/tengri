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

from tengri import FIXED, FREE, Fixed, SEDModel, WavePrecomp

pytestmark = pytest.mark.contract


def _build(ssp, obs, *, apply_igm, approx=None, model="inoue"):
    kwargs = dict(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "*": FREE},
        dust={"type": "two_component", "law_bc": "calzetti", "*": FIXED},
        neb={"type": "none"},
        redshift=Fixed(3.0),
        approx=approx,
    )
    if apply_igm:
        kwargs["apply_igm"] = True
        kwargs["igm"] = {"type": model}
    else:
        kwargs["apply_igm"] = False
    return SEDModel.build(**kwargs)


def _blue_band_ratio(ssp, obs, *, approx, model="inoue"):
    """on/off ratio of the bluest band (3500 A obs -> rest ~875 A at z=3)."""
    on = _build(ssp, obs, apply_igm=True, approx=approx, model=model)
    off = _build(ssp, obs, apply_igm=False, approx=approx)
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
