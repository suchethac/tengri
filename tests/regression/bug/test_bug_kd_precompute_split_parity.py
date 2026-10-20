# SPDX-License-Identifier: BSD-3-Clause
r"""The preintegrated kubota_done path carries the same power split as the exact path.

The exact (full-wavelength) path gives the corona exactly :math:`l_{\rm hot}` and the disc and
warm zones :math:`L_{\rm bol} - l_{\rm hot}` (Kubota & Done 2018, MNRAS 480, 1247, Eq. 2 for
:math:`l_{\rm hot} = L_{\rm diss,hot}`; #2733). The preintegrated path scaled both by a single
factor :math:`L_{\rm bol} / (D + l_{\rm hot})`, so wherever :math:`D + l_{\rm hot} \ne L_{\rm bol}`
(a high spin, where the Page-Thorne disc radiates 1.019 eta Mdot c^2 in its own frame) the disc
was off by that ratio. Band photometry now agrees with the exact path to the filter-table
resolution at every point below: the worst bands read 21% (spin 0.998) and 5.8% (f_hard 0.5)
before, and at most 2.1% after.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.disc import kubota_done_disc
from tengri.components.agn.kd_precompute import (
    kubota_done_disc_preintegrated,
    preintegrate_kd_components,
)
from tests.components.agn.test_kd_preintegrate import _make_sdss_like_filters, _np_trapezoid

pytestmark = pytest.mark.regression_bug

_Z = 0.1
_BASE = dict(
    agn_log_lbol=11.0,
    agn_log_mbh=8.0,
    agn_log_ledd=-1.0,
    agn_a_spin=0.499,
    agn_cos_inc=0.5,
    agn_f_hard=0.02,
    agn_gamma_warm=2.5,
    agn_kt_warm=0.2,
    agn_gamma_hard=1.8,
    agn_kt_hot=100.0,
    agn_r_warm_ratio=2.0,
    agn_lum_ratio=1.0,
)
_POINTS = {
    "edge": dict(
        agn_log_ledd=-0.75,
        agn_f_hard=0.05,
        agn_lum_ratio=2.5,
        agn_kt_warm=0.55,
        agn_r_warm_ratio=3.0,
        agn_kt_hot=175.0,
        agn_gamma_hard=2.0,
    ),
    "f_hard_half": dict(agn_f_hard=0.5),
    "spin_0": dict(agn_a_spin=0.0),
    "spin_0998": dict(agn_a_spin=0.998),
    # Hot flow saturated: the share is capped at 0.99 of L_bol (L_0 = L_bol / h(inf)).
    "saturated_0998": dict(agn_a_spin=0.998, agn_f_hard=0.1),
    "saturated_09": dict(agn_a_spin=0.9, agn_f_hard=0.1),
}
#: Largest per-band difference measured after the fix is 2.5% (saturated_0998, share capped at
#: 0.99, where the 50-ring quadrature adds to the corona-table residual), at the default
#: resolution. The
#: residual comes from the preintegrated path's own discretization, not from the power split: its
#: 50-ring radial quadrature (spin 0.998 reads -1.8% at 50 rings and -0.4% at 200) and the coarse
#: corona table grid (20/15/16 nodes; f_hard 0.5 reads +0.3% at 40/30/32). The 3% bound is for
#: that default resolution.
_BAND_TOLERANCE = 0.03


@pytest.fixture(scope="module")
def tables():
    fw, ft = _make_sdss_like_filters()
    return fw, ft, preintegrate_kd_components(fw, ft, redshift=_Z, n_T=300)


@pytest.mark.parametrize("name", sorted(_POINTS))
def test_preintegrated_bands_match_the_exact_path(tables, name):
    """Per-band relative difference from the exact path stays within ``_BAND_TOLERANCE``."""
    fw, ft, kd_data = tables
    kw = {**_BASE, **_POINTS[name]}
    wave = np.geomspace(0.124, 1.24e5, 20000)
    l_nu = np.asarray(kubota_done_disc(jnp.asarray(wave), **kw))
    wave_obs = wave * (1.0 + _Z)
    exact = np.array(
        [
            _np_trapezoid(np.interp(f, wave_obs, l_nu, left=0, right=0) * t * f, f)
            / _np_trapezoid(t * f, f)
            for f, t in zip(fw, ft)
        ]
    )
    pre = np.asarray(kubota_done_disc_preintegrated(kd_data, **kw))
    assert np.all(np.isfinite(pre))
    assert np.max(np.abs(pre / exact - 1.0)) < _BAND_TOLERANCE, (name, pre / exact - 1.0)
