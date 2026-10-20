# SPDX-License-Identifier: BSD-3-Clause
r"""The kubota_done Mdot comes from the disc's own dissipation; the SED closes on L_bol.

Kubota & Done (2018, MNRAS 480, 1247; arXiv:1804.00171) fix the hot-flow power
:math:`L_{\rm diss,hot} = f_{\rm hard} L_{\rm Edd}` (Eq. 2, Table 3). The Page & Thorne (1974)
dissipation integral of the disc is :math:`L_0 h(\infty) \approx 1.02\,\eta \dot M c^2` at zero
spin and rises with spin, so :math:`\dot M = L_{\rm bol}/(\eta c^2)` gave a corona share
:math:`L_{\rm hot}/L_{\rm bol}` of 1.127 inside the declared priors (spin 0.998,
:math:`f_{\rm hard} = 0.1`) and a total near 1.25 :math:`L_{\rm bol}`. :math:`\dot M` is now
inferred so that :math:`L_0 h(\infty) = L_{\rm bol}` (owner ruling 2026-10-20, refined): the share
is at most 0.99, the corona carries exactly :math:`L_{\rm hot}` and the disc :math:`(1 - s)
L_{\rm bol}`.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import disc as disc_module

pytestmark = pytest.mark.regression_bug

_LSUN = 3.828e33  # [erg/s]
_RTOL = 1.0e-4  # the energy-ledger contract's tolerance
_BASE = dict(
    agn_log_mbh=8.0,
    agn_cos_inc=0.5,
    agn_gamma_warm=2.5,
    agn_kt_warm=0.55,
    agn_gamma_hard=2.0,
    agn_kt_hot=175.0,
    agn_r_warm_ratio=3.0,
)
#: (log L_bol, a_spin, f_hard): the first two were saturated on the base (share 1.12 and
#: 1.02); the last two are unsaturated controls (share 0.03 and 0.96).
_POINTS = [(11.0, 0.998, 0.1), (11.0, 0.9, 0.1), (13.0, 0.998, 0.1), (11.0, 0.499, 0.05)]


def _run(log_lbol, a_spin, f_hard):
    wave = jnp.geomspace(1.0e-3, 1.0e8, 30000)
    kw = {k: jnp.asarray(v) for k, v in _BASE.items()}
    out = disc_module.kubota_done_disc(
        wave,
        _return_parts=True,
        agn_log_lbol=jnp.asarray(log_lbol),
        agn_a_spin=jnp.asarray(a_spin),
        agn_f_hard=jnp.asarray(f_hard),
        **kw,
    )
    return wave, [np.asarray(x, dtype=np.float64) for x in out]


@pytest.mark.parametrize(("log_lbol", "a_spin", "f_hard"), _POINTS)
def test_share_is_at_most_one_and_sed_closes_on_lbol(log_lbol, a_spin, f_hard):
    with jax.enable_x64(True):
        wave, parts = _run(log_lbol, a_spin, f_hard)
        share = parts[3]
        nu = 2.99792458e18 / np.asarray(wave)
        total = abs(np.trapezoid(parts[0], nu))
    assert 0.0 < share <= 0.99
    assert total == pytest.approx(10.0**log_lbol * _LSUN, rel=_RTOL)


def test_f_hard_gradient_is_finite_and_nonzero_across_the_former_saturation_boundary():
    # At log L_bol = 11.45, log M = 8 the old share L_hot / L_bol crossed 1 near f_hard = 0.0858.
    # The share is linear in f_hard below the 0.99 cap (reached at f_hard = 0.085 here), so the
    # gradient is finite and nonzero below it; above it the hot flow saturates (the unchanged
    # min()), the gradient is exactly zero and finite, and no 1/(1 - share) singularity remains.
    with jax.enable_x64(True):
        for f in (0.05, 0.07, 0.08):
            g = float(jax.grad(_grad_target)(jnp.asarray(f)))
            assert np.isfinite(g)
            assert g != 0.0
        for f in (0.085, 0.09, 0.1):
            assert np.isfinite(float(jax.grad(_grad_target)(jnp.asarray(f))))


def _grad_target(f_hard):
    kw = {k: jnp.asarray(v) for k, v in _BASE.items()}
    wave = jnp.geomspace(30.0, 1.0e7, 400)
    out = disc_module.kubota_done_disc(
        wave,
        _return_parts=True,
        agn_log_lbol=jnp.asarray(11.45),
        agn_a_spin=jnp.asarray(0.998),
        agn_f_hard=f_hard,
        **kw,
    )
    return out[3]
