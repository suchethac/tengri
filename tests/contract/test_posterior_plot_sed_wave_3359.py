# SPDX-License-Identifier: BSD-3-Clause
"""``posterior_plot_sed`` pairs every flux with the wavelength of its own call (#3359).

The rest-frame grid is the backend's, not always ``ssp_data.ssp_wave``: a Cue model's
SED is on its own grid. Plotting the SED against ``ssp_wave`` raised a shape error
for such models before any posterior band was drawn.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.analysis.plotting.sed import posterior_plot_sed

pytestmark = pytest.mark.contract

N_DRAWS = 9


@pytest.fixture(scope="module")
def samples():
    key = jax.random.PRNGKey(3359)
    return {"sfh_dpl_log_total_mass": jax.random.uniform(key, (N_DRAWS,), minval=9.0, maxval=11.0)}


def _build(ssp, obs, neb):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        approx=WavePrecomp(),
        sfh={"type": "dpl", "all_params": Fixed(DEFAULT), "log_total_mass": Uniform(9.0, 11.0)},
        neb=neb,
        redshift=Fixed(DEFAULT),
    )


def _posterior(model, samples):
    params = {k: v[0] for k, v in samples.items()}
    return SimpleNamespace(samples=samples, params=params, _model=model, diagnostics={})


def test_cue_model_is_plotted_on_its_own_rest_grid(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    model = _build(
        synthetic_ssp_wide,
        synthetic_tophat_obs,
        {"type": "cue", "all_params": Fixed(DEFAULT)},
    )
    sed = model._predict_rest_sed(_posterior(model, samples).params)
    wave = np.asarray(sed.wavelength)
    # The premise: this model's grid is not the SSP grid.
    assert wave.shape[0] != np.asarray(synthetic_ssp_wide.ssp_wave).shape[0]

    fig, axes = posterior_plot_sed(_posterior(model, samples))
    try:
        ax_sed = axes.flat[0]
        assert len(ax_sed.lines) >= 1
        for line in ax_sed.lines:
            np.testing.assert_array_equal(np.asarray(line.get_xdata()), wave)
    finally:
        plt.close(fig)


def test_non_cue_model_still_plots_on_the_ssp_grid(
    synthetic_ssp_wide, synthetic_tophat_obs, samples
):
    model = _build(synthetic_ssp_wide, synthetic_tophat_obs, {"type": "none"})
    ssp_wave = np.asarray(synthetic_ssp_wide.ssp_wave)
    post = _posterior(model, samples)
    expected_map = np.asarray(model._predict_rest_sed(post.params).sed)

    fig, axes = posterior_plot_sed(post)
    try:
        ax_sed = axes.flat[0]
        map_line = next(ln for ln in ax_sed.lines if ln.get_label() == "MAP SED")
        np.testing.assert_array_equal(np.asarray(map_line.get_xdata()), ssp_wave)
        np.testing.assert_allclose(
            np.asarray(map_line.get_ydata()), expected_map, rtol=1e-10, atol=0.0
        )
    finally:
        plt.close(fig)
