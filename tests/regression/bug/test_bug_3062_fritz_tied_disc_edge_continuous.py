# SPDX-License-Identifier: BSD-3-Clause
r"""The tied Fritz disc is continuous at the library's 10 A edge, up to the X-ray column (#3062).

Below the edge the tied disc carried the Lambertian :math:`2\cos i\,T_X`, a step of
:math:`2\cos i` against the library's isotropic ratio above it. Owner ruling (2026-11-12,
option B): below the edge the tied Fritz disc takes the held library ratio times :math:`T_X`,
as CIGALE's ``fritz2006`` (``sed_modules/fritz2006.py``, no :math:`2\cos i`) and the isotropic
Fritz et al. (2006) library imply; the untied path keeps :math:`2\cos i`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import torus_library_transmission as tlt

pytestmark = pytest.mark.regression_bug

_PARAMS = {"agn_fritz_oa": 60.0}
_EDGE = tlt.library_edge_wavelength("fritz")


@pytest.mark.parametrize("cos_inc", [0.94, 0.70, 0.30])
def test_tied_fritz_disc_is_continuous_at_the_edge_up_to_t_x(cos_inc):
    """Just below the edge / (just above x T_X) is 1; the angular law is the library's."""
    wave = jnp.asarray([_EDGE * 0.999, _EDGE * 1.001, 5500.0])
    sight = tlt.library_torus_sightline("fritz", wave, cos_inc=cos_inc, params=_PARAMS)
    ratio = np.asarray(sight.ratio)
    tied = np.asarray(tlt.tied_incl_with_library_edge(sight.ratio, wave, sight, "fritz"))
    t_x = float(np.asarray(sight.transmission)[0])
    np.testing.assert_allclose(tied[1], ratio[1], rtol=1e-12)
    np.testing.assert_allclose(tied[0] / (tied[1] * t_x), 1.0, rtol=1e-2)


@pytest.mark.parametrize("cos_inc", [0.94, 0.70])
def test_unabsorbed_tied_fritz_disc_has_no_two_cos_i_step(cos_inc):
    """Face-on sightlines are unabsorbed (T_X ~ 1): the step must not be 2 cos i."""
    wave = jnp.asarray([_EDGE * 0.999, _EDGE * 1.001])
    sight = tlt.library_torus_sightline("fritz", wave, cos_inc=cos_inc, params=_PARAMS)
    tied = np.asarray(tlt.tied_incl_with_library_edge(sight.ratio, wave, sight, "fritz"))
    assert abs(tied[0] / tied[1] - 1.0) < 1e-2
    assert abs(tied[0] / tied[1] - 2.0 * cos_inc) > 0.2
