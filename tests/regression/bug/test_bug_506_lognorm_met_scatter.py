# SPDX-License-Identifier: BSD-3-Clause
"""Regression: fittable lognormal metallicity-scatter parameter (#506).

BAGPIPES / Carnall+2018 §3.2 model chemical-mixing inhomogeneity as a
Gaussian-in-log10(Z) metallicity distribution function (MDF) of width sigma
about the mean metallicity. DSPS's ``calc_rest_sed_sfh_table_lognormal_mdf`` /
``..._met_table`` kernels already apply exactly this lognormal MDF via their
``gal_lgmet_scatter`` argument — tengri consumed it only as a *fixed* build-time
config (``lgmet_scatter``, default 0.1), not a fittable parameter.

This exposes it as the optional public parameter ``met_logzsol_scatter`` (dex),
threaded into both the delta and per-age-metallicity DSPS calls. No new physics
— the SSP-weighting kernel already integrates the lognormal MDF; #506 just makes
its width fittable (like Bagpipes' ``lognorm`` chemical-enrichment mode).

These tests pin: (1) the parameter is free and threaded (not a silent no-op),
(2) the sigma -> 0 limit converges to a single-metallicity (delta) population,
(3) a model that does not free it is byte-identical to the historical fixed
0.1 default, and (4) the SED is differentiable w.r.t. the scatter (fittable).

References
----------
.. [1] A. C. Carnall et al., "Inferring the star formation histories of massive
   quiescent galaxies with BAGPIPES," MNRAS, 480, 4379 (2018), §3.2.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, Uniform
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


# Solar log10(Z) offset (Asplund 2009 / MIST); mirrors translate.LOG10_ZSUN so
# the synthetic grid is centered such that met_logzsol=0.0 lands mid-grid (not
# clipped to an edge, which would zero the scatter gradient).
_LOG10_ZSUN = -1.848


def _synthetic_ssp():
    """Synthetic SSP with 5 metallicity bins (CI-safe) so the MDF can spread.

    The absolute-log10(Z) grid is centered on ``_LOG10_ZSUN`` so a mean of
    ``met_logzsol = 0.0`` sits on the middle bin with room to spread both ways.
    """
    n_met, n_age, n_wave = 5, 20, 120
    key = jax.random.PRNGKey(7)
    lgmet = _LOG10_ZSUN + jnp.array([-1.0, -0.5, 0.0, 0.5, 1.0])  # centered on solar
    base = jnp.abs(jax.random.normal(key, (n_met, n_age, n_wave))) * 1e-3 + 1e-5
    # Distinct spectral tilt per Z bin so a wider MDF (mixing more bins) yields a
    # materially different CSP SED.
    tilt = jnp.linspace(0.5, 1.5, n_wave)[None, None, :] ** jnp.arange(n_met)[:, None, None]
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, n_wave),
        ssp_flux=base * tilt,
        ssp_lg_age_gyr=jnp.linspace(-3.0, 1.1, n_age),
        ssp_lgmet=lgmet,
    )


def _model(scatter_dist):
    return SEDModel.build(
        _synthetic_ssp(),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        met={"logzsol": Fixed(0.0), "logzsol_scatter": scatter_dist, "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.1),
    )


def _sed(model, scatter):
    p = dict(model.spec.sample(jax.random.PRNGKey(0)))
    out = model.predict_rest_sed({**p, "met_logzsol_scatter": jnp.asarray(scatter)})
    return np.asarray(out.sed)


def test_scatter_is_a_free_parameter():
    """Freeing met_logzsol_scatter registers it as a fittable parameter (routing)."""
    m = _model(Uniform(0.0, 0.6))
    assert "met_logzsol_scatter" in list(m.spec.free_params)


def test_scatter_changes_the_sed_not_a_noop():
    """A wider MDF mixes more metallicity bins → materially different CSP SED."""
    m = _model(Uniform(0.0, 0.6))
    narrow, wide = _sed(m, 0.02), _sed(m, 0.5)
    band = narrow > narrow.max() * 1e-3
    max_frac = float(np.max(np.abs((wide - narrow) / narrow)[band]))
    assert max_frac > 1e-3, f"scatter appears to be a silent no-op (max frac diff {max_frac:.2e})"


def test_sigma_to_zero_recovers_delta_metallicity():
    """As sigma -> 0 the MDF collapses to a single-Z population (converges)."""
    m = _model(Uniform(0.0, 0.6))
    s_tiny, s_tinier = _sed(m, 1e-4), _sed(m, 1e-5)
    band = s_tiny > s_tiny.max() * 1e-3
    converged = float(np.max(np.abs((s_tiny - s_tinier) / s_tiny)[band]))
    wide_diff = float(np.max(np.abs((_sed(m, 0.5) - s_tiny) / s_tiny)[band]))
    assert converged < 1e-3, f"sigma->0 limit not converged (delta) — {converged:.2e}"
    assert wide_diff > 1e-2, "wide MDF should differ from the delta limit"


def test_fixed_default_is_byte_identical_to_historical():
    """A model that does not free the scatter matches the historical fixed 0.1."""
    m_default = SEDModel.build(
        _synthetic_ssp(),
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(10.0),
            "*": FIXED,
        },
        met={"logzsol": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.1),
    )
    assert "met_logzsol_scatter" not in list(m_default.spec.free_params)
    p = dict(m_default.spec.sample(jax.random.PRNGKey(0)))
    default_sed = np.asarray(m_default.predict_rest_sed(p).sed)
    at_0p1 = _sed(_model(Uniform(0.0, 0.6)), 0.1)
    band = default_sed > default_sed.max() * 1e-6
    assert np.allclose(default_sed[band], at_0p1[band], rtol=1e-6)


def test_sed_is_differentiable_wrt_scatter():
    """The scatter is fittable: jax.grad through the SED is finite and non-zero."""
    m = _model(Uniform(0.0, 0.6))
    p = dict(m.spec.sample(jax.random.PRNGKey(0)))

    def total(s):
        return jnp.sum(m.predict_rest_sed({**p, "met_logzsol_scatter": s}).sed)

    g = float(jax.grad(total)(jnp.asarray(0.2)))
    assert np.isfinite(g) and g != 0.0
