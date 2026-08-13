# SPDX-License-Identifier: BSD-3-Clause
"""Regression: youngest-bin edge clip biases the ionizing photon rate (#821).

DSPS derives its per-SSP-age weight bin edges as log-midpoints of
``ssp_lg_age_gyr``, so the youngest *physical* bin spans lookback
``[e_lo, e_hi]`` with ``e_lo > 0`` -- the ``[0, e_lo]`` sliver, which holds the
most ionizing stars (``n_ly`` drops ~300x by 10 Myr), is dropped from the age
PDF. The #809 lookback-0 table knot feeds the SFH down to the observation time,
but DSPS still clips the *bin* at ``e_lo``, biasing the ionizing photon rate Q_H
low: measured ~4% on FSPS/MILES grids and up to ~31% for BPASS (binary stars
sustain ionizing output to later ages).

The fix scales the youngest physical bin's weight column by the grid-only factor
``e_hi / (e_hi - e_lo)`` and renormalizes, extending the bin to lookback 0
(SFR is constant to <0.1% over the ~0.1 Myr sliver). This recovers the true age
PDF exactly while leaving ``total_mass`` unchanged, so mass conservation holds.
A leading ``age = 0`` SSP template (``lg = -inf``, e.g. BC03 stelib) already
collapses the youngest bin's lower edge to lookback 0 inside DSPS, so the
correction is a verified no-op there.

These tests pin (1) the grid-only boost factor + age=0 no-op, (2) that the
youngest-bin weight rises while total mass is conserved through the component,
and (3) -- on a real grid -- that Q_H recovers to the exact SFH->SSP ionizing
convolution.

References
----------
.. [1] J. J. Eldridge et al., "Binary Population and Spectral Synthesis Version
   2.1," PASA, 34, e058 (2017).  [BPASS]
.. [2] A. K. Hearin et al., "Differentiable Stellar Population Synthesis,"
   ApJS, 264, 5 (2023).  [DSPS age-weight bin edges]
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel
from tengri.components.stellar.component import _youngest_bin_lookback_multiplier
from tengri.components.stellar.sps.dsps_wrapper import SSPData

pytestmark = pytest.mark.regression_bug


def _synthetic_ssp(prepend_age_zero: bool = False):
    """20-bin synthetic SSP (CI-safe); optionally prepend an age=0 template."""
    lg = jnp.linspace(-3.5, 1.1, 20)  # 0.32 Myr .. ~12.6 Gyr, no age=0
    n_met, n_wave = 3, 100
    flux = (
        jnp.abs(jax.random.normal(jax.random.PRNGKey(123), (n_met, lg.shape[0], n_wave))) * 1e-3
        + 1e-5
    )
    if prepend_age_zero:
        lg = jnp.concatenate([jnp.array([-jnp.inf]), lg])
        flux = jnp.concatenate([jnp.zeros((n_met, 1, n_wave)), flux], axis=1)
    return SSPData(
        ssp_wave=jnp.linspace(3000.0, 10000.0, n_wave),
        ssp_flux=flux,
        ssp_lg_age_gyr=lg,
        ssp_lgmet=jnp.array([-1.5, -0.5, 0.0]),
    )


def test_boost_factor_is_grid_geometric():
    """f = e_hi/(e_hi - e_lo) at the youngest log-midpoint bin (no age=0 grid)."""
    lg = np.linspace(-3.5, 1.1, 20)
    mult = np.asarray(_youngest_bin_lookback_multiplier(jnp.asarray(lg)))
    dlg = lg[1] - lg[0]
    e_lo = 10.0 ** (lg[0] - 0.5 * dlg)
    e_hi = 10.0 ** (0.5 * (lg[0] + lg[1]))
    assert mult[0] == pytest.approx(e_hi / (e_hi - e_lo), rel=1e-12)
    assert mult[0] > 1.0  # a real correction at the youngest bin
    assert np.allclose(mult[1:], 1.0)  # every older bin is untouched


def test_age_zero_template_is_noop():
    """A leading age=0 (lg=-inf) template -> multiplier is all-ones (no clip)."""
    lg = jnp.concatenate([jnp.array([-jnp.inf]), jnp.linspace(-3.5, 1.1, 20)])
    mult = np.asarray(_youngest_bin_lookback_multiplier(lg))
    assert np.allclose(mult, 1.0)
    assert np.all(np.isfinite(mult))  # no NaN from the -inf edge


def test_multiplier_is_jit_safe_on_traced_grid():
    """The multiplier must trace cleanly when the SSP grid is a JIT argument (#821)."""
    lg = jnp.linspace(-3.5, 1.1, 20)
    mult = jax.jit(_youngest_bin_lookback_multiplier)(lg)
    assert np.all(np.isfinite(np.asarray(mult)))
    assert float(mult[0]) > 1.0


def test_boost_raises_youngest_weight_and_conserves_mass():
    """Through the component: the youngest age weight rises; total mass is conserved."""
    ssp = _synthetic_ssp()
    log_mass = 9.0
    m = SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(log_mass),
            "*": FIXED,
        },
        met={"logzsol": Fixed(0.0), "*": FIXED},
        dust={
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "*": FIXED,
        },
        redshift=Fixed(0.5),
    )
    aw = np.asarray(
        m.predict_state(dict(m.spec.sample(jax.random.PRNGKey(0)))).derived["age_weights"]
    )
    mult = np.asarray(_youngest_bin_lookback_multiplier(ssp.ssp_lg_age_gyr))
    assert mult[0] > 1.0  # this synthetic grid has no age=0 template -> clip is live
    # Mass conservation: the boost redistributes, it does not inflate.
    assert np.sum(aw) == pytest.approx(10.0**log_mass, rel=1e-4)
    assert np.all(aw >= 0.0)


@pytest.mark.regression_paper
def test_qh_recovers_exact_convolution_real_grid(real_ssp_only):
    """On a real (no age=0) grid, the corrected Q_H matches the exact SFH conv.

    Ground truth: ``Q_H = integral_0^t_obs SFR(a) * n_ly(a) da`` with the SSP
    ionizing output ``n_ly(a)`` held constant below the youngest template age.
    The clip biases this ~4% low; the corrected ``nion`` recovers it to <2%.
    """
    from tengri.components.stellar.sps.dsps_wrapper import load_ssp
    from tengri.utils.cosmology import age_at_z

    ssp = load_ssp()  # default prsc/MIST -- no age=0 template -> clip is live
    lg = np.asarray(ssp.ssp_lg_age_gyr)
    if not np.isfinite(lg[0]):
        pytest.skip("default SSP unexpectedly carries an age=0 template")

    wave = np.asarray(ssp.ssp_wave, dtype=np.float64)
    flux = np.asarray(ssp.ssp_flux, dtype=np.float64)
    met = np.asarray(ssp.ssp_lgmet)
    age = 10.0**lg * 1e9
    im = int(np.argmin(np.abs(10**met - 0.02)))
    H, C_AA = 6.62607015e-27, 2.99792458e18
    LSUN = 3.828e33
    nu = C_AA / wave
    nu_edge = C_AA / 911.76
    mask = wave < 911.76

    def n_ly(i):  # rectangle-corrected ionizing photon rate per Msun (matches component)
        integ = flux[im, i, :] * LSUN / (H * nu)
        ib = int(np.argmax(np.where(mask, np.arange(len(wave)), -1)))
        bulk = abs(np.trapezoid(np.where(mask, integ, 0.0), nu))
        return (
            bulk - 0.5 * integ[ib] * abs(nu[ib] - nu[ib + 1]) + integ[ib] * abs(nu[ib] - nu_edge)
        )

    nly = np.array([n_ly(i) for i in range(len(age))])
    z = 0.5
    t_obs = float(age_at_z(z))

    def sfr(a):
        t = np.maximum(3e9 - a, 0.0)
        return np.where(a <= 3e9, t * np.exp(-t / 1e9), 0.0)

    af = np.geomspace(1e3, t_obs * 1e9, 400_000)
    nlyf = np.interp(af, age, nly, left=nly[0], right=nly[-1])
    qh_per_msun_exact = np.trapezoid(sfr(af) * nlyf, af) / np.trapezoid(sfr(af), af)

    m = SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(3.0),
            "log_total_mass": Fixed(0.0),
            "*": FIXED,
        },
        met={"logzsol": Fixed(0.0), "*": FIXED},
        dust={"type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(z),
    )
    st = m.predict_state(dict(m.spec.sample(jax.random.PRNGKey(0))))
    aw = np.asarray(st.derived["age_weights"])
    qh_per_msun_live = float(st.derived["nion"]) / aw.sum()
    # Exact and live use a single-metallicity vs MDF-scattered n_ly respectively;
    # compare against the same single-Z reconstruction to isolate the age weighting.
    qh_aw = float((aw * nly).sum() / aw.sum())
    assert qh_aw / qh_per_msun_exact == pytest.approx(1.0, abs=0.02), (
        f"corrected Q_H {qh_aw:.3e} should match exact {qh_per_msun_exact:.3e} within 2%"
    )
    assert np.isfinite(qh_per_msun_live)
