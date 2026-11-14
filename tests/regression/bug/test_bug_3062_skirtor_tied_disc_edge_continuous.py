# SPDX-License-Identifier: BSD-3-Clause
r"""The tied SKIRTOR disc is continuous at the library's 10 A edge, up to the X-ray column (#3062).

Below the edge the tied disc carried the Lambertian :math:`2\cos i\,T_X`, a step against the
library's anisotropic ratio above it, :math:`\eta\propto\cos i\,(1+2\cos i)/3` (Stalevski et al.
2016, MNRAS 458, 2288). The owner's #3062 ruling (option B) is extended to SKIRTOR: below the
edge the tied disc takes the held library ratio times :math:`T_X`, as CIGALE's ``skirtor2016``
(``sed_modules/skirtor2016.py``, disc times :math:`{\rm disk}_i/{\rm disk}_0`, no :math:`2\cos i`)
implies; the untied path keeps :math:`2\cos i`.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from tengri.components.agn.blocks import torus_library_transmission as tlt

pytestmark = pytest.mark.regression_bug

_EDGE = tlt.library_edge_wavelength("skirtor")


def _tied(cos_inc):
    wave = jnp.asarray([_EDGE * 0.999, _EDGE * 1.001, 5500.0])
    sight = tlt.library_torus_sightline("skirtor", wave, cos_inc=cos_inc, params={})
    tied = np.asarray(tlt.tied_incl_with_library_edge(sight.ratio, wave, sight, "skirtor"))
    return sight, tied


@pytest.mark.parametrize("cos_inc", [0.94, 0.70, 0.30])
def test_tied_skirtor_disc_is_continuous_at_the_edge_up_to_t_x(cos_inc):
    """Just below the edge / (just above x T_X) is 1; the angular law is the library's."""
    sight, tied = _tied(cos_inc)
    ratio = np.asarray(sight.ratio)
    t_x = float(np.asarray(sight.transmission)[0])
    np.testing.assert_allclose(tied[1], ratio[1], rtol=1e-12)
    np.testing.assert_allclose(tied[0] / (tied[1] * t_x), 1.0, rtol=1e-2)


@pytest.mark.parametrize("cos_inc", [0.94, 0.70])
def test_tied_skirtor_disc_has_no_two_cos_i_step(cos_inc):
    """The step across the edge, T_X divided out, is not the Lambertian 2 cos i."""
    sight, tied = _tied(cos_inc)
    t_x = float(np.asarray(sight.transmission)[0])
    assert abs(tied[0] / (tied[1] * t_x) - 1.0) < 1e-2
    assert abs(tied[0] / (tied[1] * t_x) - 2.0 * cos_inc) > 0.2
