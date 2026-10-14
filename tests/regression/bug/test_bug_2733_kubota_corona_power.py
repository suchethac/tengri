# SPDX-License-Identifier: BSD-3-Clause
r"""The kubota_done corona carries exactly its counted power; the total closes on L_acc (#2733).

The three-zone model splits the accretion power into the hot corona, :math:`H_\nu`, and the
disc and warm zones, :math:`D_\nu`. The corona is counted at :math:`l_{\rm hot}` (the
Page-Thorne dissipation inside :math:`R_{\rm hot}`, equal to :math:`f_{\rm hard} L_{\rm Edd}`
at the defaults). The corona takes exactly that power, and the disc and warm zones carry the
rest, :math:`L_{\rm acc} - l_{\rm hot}`, so that

.. math::

    \int H_\nu\,{\rm d}\nu = l_{\rm hot}, \qquad
    \int D_\nu\,{\rm d}\nu = L_{\rm acc} - l_{\rm hot}, \qquad
    \int (2\cos i\,D_\nu + H_\nu)\,{\rm d}\nu \big|_{\cos i = 0.5} = L_{\rm acc}.

Every expected value is written from these identities, never read from a pin. The spectra
are integrated on a converged log grid over :math:`10^{-2}`--:math:`10^{9}` A.
"""

from __future__ import annotations

import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn import disc as disc_module
from tengri.components.agn.blocks.runner import compose_l_nu
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.regression_bug

jax.config.update("jax_enable_x64", True)

_LOG_LBOL = 12.0
_L_ACC = 10.0**_LOG_LBOL * L_SUN  # [erg/s]
_WAVE = np.geomspace(1.0e-2, 1.0e9, 24001)  # [A], converged log grid
_NU = C_AA / _WAVE  # [Hz]
_ORDER = np.argsort(_NU)
_NONE = dict(
    agn_nlr_block="none",
    agn_blr_block="none",
    agn_feii_block="none",
    agn_torus_block="none",
    agn_attenuation_block="none",
)
#: Relative tolerance on each identity. The measured residuals (see the test docstrings and
#: the commit) are below 1e-5; 1e-4 is the acceptance bar, not a fitted number.
_TOL = 1e-4


def _integral_over_lbol(spectrum: np.ndarray) -> float:
    """Frequency integral of a spectrum on the converged grid, in units of L_acc."""
    return float(np.trapezoid(np.asarray(spectrum)[_ORDER], _NU[_ORDER]) / _L_ACC)


def _l_hot_counted_over_lbol() -> float:
    """The corona power the normalizer counts, f_hard L_Edd, as a fraction of L_acc.

    Taken from the kubota_done defaults and the Eddington law directly, not from the output
    spectrum, so the check cannot pass by construction.
    """
    defaults = inspect.signature(disc_module.kubota_done_disc).parameters
    l_edd = 10.0 ** disc_module._log10_eddington_luminosity(defaults["agn_log_mbh"].default)
    return float(defaults["agn_f_hard"].default * l_edd / _L_ACC)


def _spectrum(cos_inc: float):
    _, components = compose_l_nu(
        jnp.asarray(_WAVE),
        _LOG_LBOL,
        agn_disc_block="kubota_done",
        agn_cos_inc=cos_inc,
        return_components=True,
        **_NONE,
    )
    return np.asarray(components["disc"])


def test_corona_spectrum_integrates_to_its_counted_power():
    """H_nu (isotropic, cos i = 0) integrates to l_hot, the power the normalizer counts."""
    corona = _spectrum(0.0)
    assert _integral_over_lbol(corona) == pytest.approx(_l_hot_counted_over_lbol(), rel=_TOL)


def test_disc_and_warm_zones_carry_the_rest_of_the_accretion_power():
    """2 cos i D_nu at cos i = 1 has the disc power L_acc - l_hot, so D_nu = L_acc - l_hot."""
    disc_only = _spectrum(1.0) - _spectrum(0.0)  # 2 D_nu
    assert _integral_over_lbol(disc_only) / 2.0 == pytest.approx(
        1.0 - _l_hot_counted_over_lbol(), rel=_TOL
    )


def test_total_line_of_sight_power_at_half_cos_i_is_l_acc():
    """L_nu(i) = 2 cos i D_nu + H_nu integrates to L_acc at cos i = 0.5."""
    assert _integral_over_lbol(_spectrum(0.5)) == pytest.approx(1.0, rel=_TOL)
