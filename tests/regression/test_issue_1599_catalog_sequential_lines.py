# SPDX-License-Identifier: BSD-3-Clause
"""Per-galaxy line fluxes must reach ``cat.fit()`` on the sequential path (#1599).

``Catalog(line_cols=[...])`` ingests per-galaxy emission-line fluxes, and the
vmapped MCMC branch of :class:`~tengri.inference.catalog_fitter.CatalogFitter`
consumes them. The sequential branch -- which includes ``method="map"``,
``Catalog.fit``'s own default -- built its per-galaxy ``Fitter`` without them,
so every galaxy was scored against the *template* Observation's line flux.

That is the substitution #1480 added a guard against; the guard only fires
when ``line_cols`` is absent, so supplying them defeated it.

Note on method
--------------
The obvious test -- compare galaxy 0 against galaxy 1 inside one catalog --
gives the WRONG answer here. Those two galaxies get different init keys, and
on this fixture the key alone moves a fitted parameter by ~1.0. The A/A
control below pins that noise floor, and the real assertions compare the
*same galaxy index* across two catalogs under the *same* key, so the line
flux is the only thing that differs.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tests.contract._line_catalog_fixture import build_two_galaxy_catalog

pytestmark = pytest.mark.regression_bug


def _fit_params(cat, method, key, **kw):
    """Fitted parameters per galaxy, as plain floats."""
    post = cat.fit(method=method, key=key, **kw)
    return [{k: float(np.asarray(v)) for k, v in post[i].params.items()} for i in range(2)]


def _max_delta(a: dict, b: dict) -> float:
    return max(abs(a[k] - b[k]) for k in a)


def test_per_galaxy_line_flux_changes_the_sequential_fit(synthetic_ssp_wide, synthetic_tophat_obs):
    """Changing one galaxy's observed Halpha must change that galaxy's fit.

    The clean comparison: same galaxy index, same PRNG key, two catalogs
    that differ only in galaxy 1's line flux (1x vs 4x).
    """
    common = dict(ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs)
    flat, _ = build_two_galaxy_catalog(halpha=(1.0, 1.0), **common)
    contrast, _ = build_two_galaxy_catalog(halpha=(1.0, 4.0), **common)

    key = jax.random.PRNGKey(0)
    p_flat = _fit_params(flat, "map", key, n_steps=40)
    p_contrast = _fit_params(contrast, "map", key, n_steps=40)

    delta = _max_delta(p_flat[1], p_contrast[1])

    assert delta > 1e-10, (
        "galaxy 1's fitted parameters are unchanged when its observed Halpha "
        f"goes 1x -> 4x (max delta {delta:g}) -- the per-galaxy line fluxes "
        "ingested from line_cols never reach the objective on the sequential "
        "path, so every galaxy is scored against the template line flux"
    )


def test_aa_control_pins_the_init_key_noise_floor(synthetic_ssp_wide, synthetic_tophat_obs):
    """Two identical galaxies still fit differently -- that is the key, not the data.

    Without this control the naive galaxy-0-vs-galaxy-1 comparison reads as
    evidence that per-galaxy lines work, when the difference is entirely the
    per-galaxy init key. Recorded so nobody re-derives the false positive.
    """
    identical, _ = build_two_galaxy_catalog(
        halpha=(1.0, 1.0), ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs
    )
    ca = identical._catalog_arrays
    assert np.allclose(ca.line_flux_obs[0], ca.line_flux_obs[1]), "fixture is not A/A"
    assert np.allclose(ca.flux[0], ca.flux[1]), "fixture photometry is not identical"

    p = _fit_params(identical, "map", jax.random.PRNGKey(0), n_steps=40)
    aa_floor = _max_delta(p[0], p[1])

    assert aa_floor > 0.0, (
        "two galaxies with byte-identical data fit identically, so this "
        "fixture can no longer demonstrate the init-key confound"
    )


def test_sequential_and_vmapped_paths_agree_on_line_sensitivity(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """The two engine branches must not disagree about whether lines matter.

    ``mcmc_nuts`` (vmapped) always honored the per-galaxy values; ``map``
    (sequential) did not. Both must now respond.
    """
    common = dict(ssp=synthetic_ssp_wide, obs_base=synthetic_tophat_obs)
    flat, _ = build_two_galaxy_catalog(halpha=(1.0, 1.0), **common)
    contrast, _ = build_two_galaxy_catalog(halpha=(1.0, 4.0), **common)

    key = jax.random.PRNGKey(0)
    seq = _max_delta(
        _fit_params(flat, "map", key, n_steps=40)[1],
        _fit_params(contrast, "map", key, n_steps=40)[1],
    )
    vmapped = _max_delta(
        _fit_params(flat, "mcmc_nuts", key, n_warmup=8, n_samples=8)[1],
        _fit_params(contrast, "mcmc_nuts", key, n_warmup=8, n_samples=8)[1],
    )

    assert (seq > 1e-10) == (vmapped > 1e-10), (
        f"sequential delta {seq:g} and vmapped delta {vmapped:g} disagree "
        "about whether per-galaxy line fluxes reach the objective -- the "
        "same catalog must mean the same thing on both engine branches"
    )
