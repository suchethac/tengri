# SPDX-License-Identifier: BSD-3-Clause
"""Contract tests for the standalone surviving-mass helper.

``predict_surviving_mass`` lets the prior layer put priors on
``log_mass_surviving`` without running the full SED forward pass. Tests
here pin down the contract:

* ``M_surv = ∫ SFR(t_lb) × f_surv(t_lb) dt_lb``
* When ``f_surv ≡ 1``: ``M_surv == 10**log_total_mass`` exactly.
* When ``f_surv ≡ c`` (constant): ``M_surv == c × 10**log_total_mass``.
* Linear scaling: ``d(log M_surv) / d(log_total_mass) == 1`` exactly
  (since ``M_surv ∝ M_formed``).
* JIT- and grad-compatible.

These guard the helper's correctness independently of any specific SSP
library, using a synthetic SSP with a hand-crafted mass-remaining table.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from tengri import predict_surviving_mass
from tengri.components.stellar.sfh.mean_sfh import (
    declining_exponential,
    dpl,
    gaussian,
    lognormal,
)
from tengri.components.stellar.sps.dsps_wrapper import SSPData

# Age of the universe today [yr], from the default cosmology — never a
# literal. SFH formation anchor (age_gyr) for dpl/lnorm shape tests.
from tengri.cosmology import age_at_z0 as _age_at_z0

_AGE_UNIV_YR = float(_age_at_z0()) * 1e9

pytestmark = pytest.mark.conservation


# ── Synthetic SSP fixtures ────────────────────────────────────────


def _make_synthetic_ssp(f_surv_value: float | None = None) -> SSPData:
    """Build a minimal SSPData with a hand-crafted mass-remaining table.

    Parameters
    ----------
    f_surv_value : float or None
        If a float, ``ssp_mass_remaining`` is set to that constant
        across (n_met, n_age). If ``None``, uses a physically-motivated
        monotonic decline from 1.0 at youngest age to 0.5 at oldest.
    """
    n_met, n_age, n_wave = 3, 64, 16
    ssp_lg_age_gyr = jnp.linspace(-3.0, 1.14, n_age)  # 1 Myr → 13.8 Gyr
    ssp_lgmet = jnp.array([-2.0, -1.5, -1.0])  # log10(Z), absolute
    ssp_wave = jnp.logspace(2.5, 4.5, n_wave)
    ssp_flux = jnp.ones((n_met, n_age, n_wave))  # unused here
    if f_surv_value is not None:
        mr = jnp.full((n_met, n_age), f_surv_value)
    else:
        # Monotonic decline 1.0 → 0.5 across ages, identical across metallicities.
        per_age = jnp.linspace(1.0, 0.5, n_age)
        mr = jnp.broadcast_to(per_age, (n_met, n_age))
    return SSPData(
        ssp_wave=ssp_wave,
        ssp_flux=ssp_flux,
        ssp_lg_age_gyr=ssp_lg_age_gyr,
        ssp_lgmet=ssp_lgmet,
        ssp_mass_remaining=mr,
    )


T_LOOKBACK = jnp.logspace(5, 10.14, 2000)


# ── Tests ─────────────────────────────────────────────────────────


class TestSurvivingMassHelper:
    """Contract: predict_surviving_mass implements ∫ SFR(t) × f_surv(t) dt."""

    def test_unit_survivor_fraction_recovers_mformed(self):
        """When f_surv ≡ 1, M_surv = M_formed = 10**log_total_mass exactly."""
        ssp = _make_synthetic_ssp(f_surv_value=1.0)
        sfr = declining_exponential(T_LOOKBACK, log_total_mass=10.0, tau=2e9, age=12e9)
        m_surv = float(predict_surviving_mass(sfr, T_LOOKBACK, ssp, log_z_zsun=-1.5))
        m_formed = float(jnp.trapezoid(sfr, T_LOOKBACK))
        assert_allclose(m_surv, m_formed, rtol=1e-6)
        assert_allclose(m_surv, 1e10, rtol=1e-3)

    @pytest.mark.parametrize("f_surv", [0.25, 0.5, 0.75])
    def test_constant_survivor_fraction_scales_linearly(self, f_surv):
        """f_surv ≡ c ⇒ M_surv = c × M_formed."""
        ssp = _make_synthetic_ssp(f_surv_value=f_surv)
        sfr = dpl(T_LOOKBACK, alpha=1.5, beta=1.0, tau=3e9, age=_AGE_UNIV_YR, log_total_mass=10.0)
        m_surv = float(predict_surviving_mass(sfr, T_LOOKBACK, ssp))
        m_formed = float(jnp.trapezoid(sfr, T_LOOKBACK))
        assert_allclose(m_surv, f_surv * m_formed, rtol=1e-6)

    def test_log_mass_gradient_equals_one(self):
        """d(log M_surv) / d(log_total_mass) == 1 exactly (linear scaling)."""
        ssp = _make_synthetic_ssp()  # non-trivial decline

        def log_mstar_surv(log_total_mass):
            sfr = gaussian(T_LOOKBACK, log_total_mass=log_total_mass, peak_lbt=4e9, width=1.5e9)
            return jnp.log10(predict_surviving_mass(sfr, T_LOOKBACK, ssp))

        g = float(jax.grad(log_mstar_surv)(10.5))
        assert_allclose(g, 1.0, rtol=1e-5)

    def test_msurv_le_mformed_always(self):
        """Physical: surviving mass can never exceed formed mass."""
        ssp = _make_synthetic_ssp()  # monotonic decline 1.0 → 0.5
        for log_M in (8.0, 10.0, 11.5):
            sfr = lognormal(
                T_LOOKBACK, log_total_mass=log_M, peak=5e9, width=0.4, age=_AGE_UNIV_YR
            )
            m_surv = float(predict_surviving_mass(sfr, T_LOOKBACK, ssp))
            m_formed = float(jnp.trapezoid(sfr, T_LOOKBACK))
            assert m_surv <= m_formed + 1e-6, f"M_surv {m_surv} > M_formed {m_formed}"
            assert m_surv > 0.4 * m_formed, "older-skewed SFH should keep most of mass"

    def test_jit_compatible(self):
        """The helper closure JIT-compiles cleanly."""
        ssp = _make_synthetic_ssp(f_surv_value=0.7)

        @jax.jit
        def m_surv_jit(log_total_mass, tau):
            sfr = declining_exponential(
                T_LOOKBACK, log_total_mass=log_total_mass, tau=tau, age=12e9
            )
            return predict_surviving_mass(sfr, T_LOOKBACK, ssp)

        out = float(m_surv_jit(10.0, 2e9))
        assert out > 0 and np.isfinite(out)

    def test_raises_without_mass_remaining_table(self):
        """Helper must error loudly when SSP lacks ssp_mass_remaining."""
        ssp_no_mr = SSPData(
            ssp_wave=jnp.logspace(2.5, 4.5, 8),
            ssp_flux=jnp.ones((2, 8, 8)),
            ssp_lg_age_gyr=jnp.linspace(-3, 1.14, 8),
            ssp_lgmet=jnp.array([-2.0, -1.0]),
            ssp_mass_remaining=None,
        )
        sfr = dpl(T_LOOKBACK, alpha=1.5, beta=1.0, tau=3e9, age=_AGE_UNIV_YR, log_total_mass=10.0)
        with pytest.raises(ValueError, match="ssp_mass_remaining"):
            predict_surviving_mass(sfr, T_LOOKBACK, ssp_no_mr)

    def test_metallicity_is_read_relative_to_solar(self):
        """``log_z_zsun`` selects the node at ``log_z_zsun + LOG10_ZSUN`` on the absolute axis.

        The table rises with metallicity (0.5, 0.6, 0.7 at the three nodes), so each
        node gives a different surviving mass and a value clipped to an end node
        would be visible.
        """
        from tengri.utils.physics_constants import LOG10_ZSUN

        base = _make_synthetic_ssp()
        n_age = base.ssp_lg_age_gyr.shape[0]
        table = jnp.asarray([0.5, 0.6, 0.7])[:, None] * jnp.ones((1, n_age))
        ssp = base._replace(ssp_mass_remaining=table)
        sfr = gaussian(T_LOOKBACK, log_total_mass=10.0, peak_lbt=5e9, width=2e9)
        formed = float(jnp.trapezoid(sfr, T_LOOKBACK))

        nodes = np.asarray(ssp.ssp_lgmet)
        for node, fraction in zip(nodes, (0.5, 0.6, 0.7)):
            m_surv = float(
                predict_surviving_mass(sfr, T_LOOKBACK, ssp, log_z_zsun=float(node) - LOG10_ZSUN)
            )
            assert_allclose(m_surv / formed, fraction, rtol=1e-6)

        halfway = 0.5 * (nodes[0] + nodes[1]) - LOG10_ZSUN
        m_half = float(predict_surviving_mass(sfr, T_LOOKBACK, ssp, log_z_zsun=float(halfway)))
        assert_allclose(m_half / formed, 0.55, rtol=1e-6)
