# SPDX-License-Identifier: BSD-3-Clause
"""Regression tests for the Cloudy-grid NLR adapters.

Verifies that `compute_nlr_sed_feltre` produces line ratios consistent
with the Feltre+2016 photoionization grid (BEAGLE parity) and that it
plugs into the `unified_nlr_blr` model via the `nlr_fn` slot.

Skips silently if `data/feltre_grid.h5` is not present locally — the
grid is gitignored and is built/downloaded via
`scripts/build_feltre_grid.py`.

References
----------
Feltre, A., Charlot, S., Gutkin, J. 2016, MNRAS, 456, 3354.
https://doi.org/10.1093/mnras/stv2794
"""

from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest


def _feltre_grid_available() -> bool:
    from tengri.components.nebular.agn_nebular import _DEFAULT_FELTRE_GRID_PATH

    return Path(_DEFAULT_FELTRE_GRID_PATH).is_file()


_GRID_SKIP = pytest.mark.skipif(
    not _feltre_grid_available(),
    reason=("data/feltre_grid.h5 not present; build via `scripts/build_feltre_grid.py`"),
)


@_GRID_SKIP
@pytest.mark.regression_paper
def test_feltre_nlr_line_ratios_seyfert2():
    """[O III]5007/Hβ and Hα/Hβ at canonical Seyfert 2 conditions.

    At α=-1.7, log U=-2, log n_H=3, Z=Z_sun, ξ_d=0.3 the Feltre+2016
    grid should produce [O III]/Hβ in the range 10–15 (paper Fig. 6,
    high-ionization Seyfert 2 regime) and Hα/Hβ >= 2.86 (Case B floor
    plus collisional excitation).
    """
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre

    wave = jnp.linspace(3000.0, 7500.0, 5000)
    sed = compute_nlr_sed_feltre(
        wave,
        l_disc_bol_erg=1e45,
        covering_fraction=0.1,
        fwhm_kms=500.0,
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,  # log10(Z_sun)
        xi_d=0.3,
    )
    assert sed.shape == wave.shape
    assert bool(jnp.all(jnp.isfinite(sed)))

    def peak(w0: float) -> float:
        return float(sed[int(jnp.argmin(jnp.abs(wave - w0)))])

    o3 = peak(5007.0)
    hb = peak(4861.0)
    ha = peak(6563.0)

    assert hb > 0.0, "Hβ peak should be non-zero in Sy2 conditions"
    assert 5.0 <= o3 / hb <= 20.0, (
        f"[O III]/Hβ = {o3 / hb:.2f} outside Seyfert 2 plausible range 5–20"
    )
    assert ha / hb >= 2.86, f"Hα/Hβ = {ha / hb:.2f} violates Case B floor 2.86"


@_GRID_SKIP
@pytest.mark.regression_paper
def test_feltre_nlr_scales_linearly_with_lbol():
    """L_nu is linear in disc bolometric luminosity at fixed covering / geometry."""
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre

    wave = jnp.linspace(4000.0, 7000.0, 1000)
    kw = dict(
        covering_fraction=0.1,
        fwhm_kms=500.0,
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
    )
    sed_1e44 = compute_nlr_sed_feltre(wave, l_disc_bol_erg=1e44, **kw)
    sed_1e45 = compute_nlr_sed_feltre(wave, l_disc_bol_erg=1e45, **kw)
    ratio = jnp.max(sed_1e45) / jnp.max(sed_1e44)
    np.testing.assert_allclose(float(ratio), 10.0, rtol=0.05)


@_GRID_SKIP
@pytest.mark.regression_paper
def test_feltre_nlr_plugs_into_unified_nlr_blr():
    """`nlr_fn=compute_nlr_sed_feltre` swaps in cleanly for `unified_nlr_blr`."""
    from tengri.components.agn import unified_nlr_blr
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre

    wave = jnp.linspace(1000.0, 1.0e5, 5000)
    sed_default = unified_nlr_blr(
        wave,
        agn_log_lbol=12.0,
        agn_cos_inc=0.95,
        agn_theta_torus=30.0,
        agn_torus_frac=0.5,
        agn_lum_ratio=1.0,
    )
    sed_feltre = unified_nlr_blr(
        wave,
        agn_log_lbol=12.0,
        agn_cos_inc=0.95,
        agn_theta_torus=30.0,
        agn_torus_frac=0.5,
        agn_lum_ratio=1.0,
        nlr_fn=compute_nlr_sed_feltre,
        alpha_pl=-1.7,
        neb_logU=-2.0,
        neb_logn=3.0,
        neb_logZ_gas=-1.8477,
        xi_d=0.3,
    )
    # Both must be finite, non-negative, and differ meaningfully in optical
    # (where NLR lines dominate)
    assert bool(jnp.all(jnp.isfinite(sed_default)))
    assert bool(jnp.all(jnp.isfinite(sed_feltre)))
    assert bool(jnp.all(sed_default >= 0.0))
    assert bool(jnp.all(sed_feltre >= 0.0))
    opt_mask = (wave >= 4500.0) & (wave <= 7500.0)
    diff = jnp.max(jnp.abs(sed_default[opt_mask] - sed_feltre[opt_mask]))
    assert float(diff) > 0.0, "Feltre and analytic NLR should differ at optical line wavelengths"


@_GRID_SKIP
@pytest.mark.regression_paper
def test_feltre_nlr_singleton_caching():
    """Calling the adapter twice should reuse the same backend instance."""
    from tengri.components.agn.nlr_cloudy import compute_nlr_sed_feltre, get_feltre_backend

    wave = jnp.linspace(4000.0, 7000.0, 500)
    _ = compute_nlr_sed_feltre(wave, l_disc_bol_erg=1e45)
    b1 = get_feltre_backend()
    _ = compute_nlr_sed_feltre(wave, l_disc_bol_erg=1e45)
    b2 = get_feltre_backend()
    assert b1 is b2, "Feltre backend should be cached as a process-level singleton"
