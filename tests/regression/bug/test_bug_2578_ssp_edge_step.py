# SPDX-License-Identifier: BSD-3-Clause
"""
The stellar SED is a step at the SSP grid edge on the exact photometry path (#2578).

Invariant: on a model whose master grid continues past the SSP grid (any radio
block), a band that lies just above the SSP edge reads the same flux on the exact
path as on the ``WavePrecomp`` path. The SSP has no stellar flux there, so both
paths must see zero stellar contribution: the exact path must not ramp the SSP's
last node down across the first master-grid cell.

Mechanism: the stellar projection onto the master grid is exact at the SSP edge
node and zero at the next master node. Linear interpolation between them, in the
filter integral, is a ramp across the whole cell (12.5 % wide at 20 nodes per
decade), so a band 1 % above the edge read the SSP's flux that does not exist.

The radio block is kept so the master grid extends past the edge. Its term is
the same on both paths (it is a rank-1 band response on the LUT and a band
integral on the exact path, and it agrees to round-off in every band above the
edge), so the exact-minus-LUT difference isolates the stellar seam.

Measured on ``main`` (wNE grid, bell2003 + powerlaw radio, const SFH, Calzetti):
the band rest 1.01-1.05 cm read 5.0056e-28 exact against 4.8897e-28 on the
``WavePrecomp`` path (ratio 1.0237 at the first cell; the issue's 1.67x is the
radio-thermal-on configuration).
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel, WavePrecomp
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

pytestmark = pytest.mark.regression_bug

_Z = 0.05


def _tophat(lo_rest, hi_rest, name, n=200):
    """Top-hat transmission over a rest-frame band, observed at redshift ``_Z``."""
    lo, hi = lo_rest * (1.0 + _Z), hi_rest * (1.0 + _Z)
    wave = np.linspace(lo, hi, n)
    trans = np.sin(np.linspace(0.0, np.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=name)


_OBS = Observation(
    photometry=Photometry(
        filters=(
            _tophat(5.0e3, 6.0e3, "opt"),  # rest 5000-6000 A: well below the edge
            _tophat(1.01e8, 1.05e8, "cm1"),  # rest 1.01-1.05 cm: first cell above the edge
            _tophat(1.06e8, 1.10e8, "cm2"),  # rest 1.06-1.10 cm: the same cell
        )
    )
)

# Exact-path flux of the optical band (index 0) on this configuration, as measured
# on ``main`` before the fix. Nothing below the edge moves, so it must hold to 1e-10.
_OPT_EXACT_MAIN = 3.8029878990438525e-27


def _build(ssp, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=_OBS,
        sfh={"type": "const", "all_params": Fixed(DEFAULT)},
        dust_attenuation={
            "type": "two_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
        },
        neb={"type": "ssp"},
        radio={
            "sf": {"type": "bell2003"},
            "agn": {"type": "powerlaw"},
            "all_params": Fixed(DEFAULT),
        },
        redshift=Fixed(_Z),
        approx=approx,
    )


def _flux(model):
    return np.asarray(model.predict_photometry(dict(model.spec.sample(jax.random.PRNGKey(0)))))


def test_band_above_the_ssp_edge_matches_the_lut(ssp_data_wne):
    """Bands in the first master-grid cell above the SSP edge agree with WavePrecomp.

    The stellar part is zero there on both paths; the radio part is shared. Any
    exact-minus-LUT difference is the stellar ramp.
    """
    exact = _flux(_build(ssp_data_wne, approx=None))
    lut = _flux(_build(ssp_data_wne, approx=WavePrecomp()))

    np.testing.assert_allclose(exact[1:], lut[1:], rtol=1e-6, atol=0.0)


def test_band_well_below_the_ssp_edge_is_unchanged(ssp_data_wne):
    """A band far below the edge keeps its exact-path flux to 1e-10 relative."""
    exact = _flux(_build(ssp_data_wne, approx=None))

    np.testing.assert_allclose(exact[0], _OPT_EXACT_MAIN, rtol=1e-10, atol=0.0)
