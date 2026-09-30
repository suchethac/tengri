# SPDX-License-Identifier: BSD-3-Clause
"""#2561: predict_line_fluxes on a photometry-only nebular grid.

A per-Q_H nebular grid (``enable_fast_nebular`` / the ``approx="auto"`` LUT
policy on a photometry-only ``Fitter``) tabulates only the wavelengths its own
build call named. A grid built for a photometry-only fit (no line targets
requested) tabulates none, so ``grid.wavelengths`` is empty: the target match
in ``predict_line_fluxes`` had nothing to compute an ``argmin`` over and
raised ``ValueError: attempt to get argmin of an empty sequence`` instead of
either answering from the catalog or refusing with the existing "line not in
the catalog" message. Fixed by treating an empty grid as no grid: it falls
through to the exact catalog path, the same way ``predict_photometry`` serves
a channel the LUT does not carry from the exact model.

A NON-empty grid that simply does not carry the one target asked for was
never affected (matching a non-empty candidate list already reaches the
``tolerance_aa`` guard); pinned here anyway so the two cases are not
confused.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import tengri
from tengri import DEFAULT, FREE, Fixed, Observation, Photometry, WavePrecomp
from tengri.inference.fitter import Fitter

pytestmark = pytest.mark.regression_bug

_HALPHA = 6564.61
_HBETA = 4862.69
_OIII = 5008.24


def _build(ssp, *, approx):
    obs = Observation(photometry=Photometry.from_names(["des_g", "des_r", "des_i"]))
    return tengri.SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        redshift=Fixed(0.2),
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "start_gyr": Fixed(1.0),
            "end_gyr": Fixed(0.0),
            "log_total_mass": FREE,
        },
        neb={"type": "cue", "all_params": Fixed(DEFAULT)},
        approx=approx,
    )


@pytest.fixture(scope="module")
def ssp():
    return tengri.load_ssp("fsps_prsc_miles_chabrier", download=False)


def test_empty_grid_falls_through_and_matches_exact_model(ssp):
    """A photometry-only fit's grid tabulates zero lines; must answer, not crash."""
    m = _build(ssp, approx=WavePrecomp())
    truth = dict(m.spec.sample(jax.random.PRNGKey(0)))
    phot = m.predict_photometry(truth)
    fitter = Fitter(
        m,
        data=jnp.asarray(phot),
        noise=jnp.asarray(0.05 * jnp.abs(phot)),
        data_type="photometry",
        approx="auto",
    )
    assert fitter.model._nebular_grid_table is not None
    assert fitter.model._nebular_grid_table.wavelengths.shape[0] == 0

    p = dict(fitter.model.spec.sample(jax.random.PRNGKey(0)))
    targets = jnp.asarray([_HALPHA, _HBETA])

    got = np.asarray(fitter.model.predict_line_fluxes(p, target_wavelengths=targets))
    exact = np.asarray(
        fitter.model.with_approx(None).predict_line_fluxes(p, target_wavelengths=targets)
    )
    np.testing.assert_allclose(got, exact, rtol=1e-10)


def test_nonempty_grid_missing_the_target_raises_the_tolerance_message(ssp):
    """A grid tabulating unrelated lines refuses with the plain tolerance message."""
    m = _build(ssp, approx=None)
    m.enable_fast_nebular(jnp.asarray([_HALPHA]))
    assert m._nebular_grid_table.wavelengths.shape[0] == 1

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    with pytest.raises(ValueError, match="have no match within tolerance_aa"):
        m.predict_line_fluxes(p, target_wavelengths=jnp.asarray([_OIII]))


def test_grid_with_all_targets_answers_from_the_grid(ssp):
    """A grid tabulating every requested target keeps taking the fast path."""
    targets = jnp.asarray([_HALPHA, _HBETA])
    m = _build(ssp, approx=None)
    m.enable_fast_nebular(targets)
    assert m._nebular_grid_table.wavelengths.shape[0] == 2

    p = dict(m.spec.sample(jax.random.PRNGKey(0)))
    got = np.asarray(m.predict_line_fluxes(p, target_wavelengths=targets))
    exact = np.asarray(m.with_approx(None).predict_line_fluxes(p, target_wavelengths=targets))
    # The #950 grid is itself an interpolated approximation (not bit-exact
    # with the exact path even when it does carry the target), see
    # test_2520_igm_line_flux_attenuation.py::test_fast_grid_matches_exact_path.
    np.testing.assert_allclose(got, exact, rtol=2e-2)
