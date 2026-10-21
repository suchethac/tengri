# SPDX-License-Identifier: BSD-3-Clause
"""The Fritz (2006) torus keeps the library's anisotropy between viewing angles (#2851).

CIGALE stores every Fritz model divided by its own dust power and keeps the divisor as
``norm`` (pcigale ``model.norm``). Normalizing each record to ``f L_bol`` on its own, as the
torus kernel did, gave every sightline the same torus power and erased the radiative
transfer's result that an edge-on torus radiates less (measured on the shipped grid: the
equatorial record carries 0.06 to 0.98, median 0.52, of the polar record's dust power). The
records multiplied by ``norm`` share one scale (the face-on disc power of 98 % of the 2400
models lies within -0.5 % and +12.5 % of the median), and the kernel now divides them by
ONE sphere-mean power per parameter point, so the angle-integrated torus power is ``f L_bol`` and every sightline keeps
the library's ratio.

The reference (``tests/physics/_template_torus_reference.py``) reads the file with ``h5py``,
interpolates with SciPy's PCHIP and integrates the power laws in closed form.
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tengri._data_setup import find_data
from tengri.utils.physics_constants import L_SUN
from tests.physics._agn_publication import ln_nu_integral
from tests.physics._template_torus_reference import RawTorus

pytestmark = pytest.mark.regression_bug

_WIDE = np.geomspace(5.0, 1.0e8, 20000)
_KW = {"agn_log_lbol": 10.5, "agn_torus_frac": 0.6}
#: (r_ratio, tau, beta, gamma, opening half-angle) node points, and an off-node one.
_POINTS = (
    (60.0, 1.0, -0.5, 4.0, 40.0),
    (10.0, 6.0, 0.0, 0.0, 20.0),
    (150.0, 0.1, -1.0, 6.0, 60.0),
    (45.0, 1.5, -0.6, 3.0, 50.0),
)


@pytest.fixture(scope="module")
def fritz():
    path = find_data("fritz2006_torus_grid.h5")
    if path is None:
        pytest.skip("template data unavailable (fritz2006_torus_grid.h5)")
    from tengri.components.agn import fritz as mod

    return mod, mod.load_fritz_default_grid(), RawTorus("fritz", str(path))


def _sed(fritz, point, psy):
    mod, grid, _ = fritz
    keys = ("agn_fritz_r_ratio", "agn_fritz_tau", "agn_fritz_beta", "agn_fritz_gamma")
    kw = dict(zip(keys, point[:4], strict=True))
    return np.asarray(
        mod.fritz_sed_from_grid(
            grid, jnp.asarray(_WIDE), agn_fritz_oa=point[4], agn_fritz_psy=psy, **kw, **_KW
        )
    )


@pytest.mark.parametrize("point", _POINTS[:3])
def test_node_sightlines_keep_the_records_power_ratio(fritz, point):
    """At library nodes, P(psi_a)/P(psi_b) of the emitted torus is norm_a/norm_b (1e-5).

    Each stored dust record integrates to the same value, so the ratio of the norm-restored
    records' powers is the ratio of their ``norm``; it was exactly 1 before the fix.
    """
    _, _, raw = fritz
    idx = tuple(
        int(np.argmin(np.abs(ax - v))) for ax, v in zip(raw.other_axes, point, strict=True)
    )
    psy = raw.inc_nodes
    emitted = [ln_nu_integral(_sed(fritz, point, float(p)), _WIDE) for p in psy]
    stored = [raw.node_power(idx, j) for j in range(psy.size)]
    for j in range(psy.size):
        assert emitted[j] / emitted[-1] == pytest.approx(stored[j] / stored[-1], rel=1e-5), j
    assert emitted[0] < emitted[-1], "the equatorial torus must be dimmer than the polar one"


@pytest.mark.parametrize("point", _POINTS)
@pytest.mark.parametrize("mu", [0.05, 0.5, 0.95])
def test_sightline_power_is_the_raw_record_over_its_sphere_mean(fritz, point, mu):
    """The emitted power along ``mu`` is ``f L P(mu) / mean P`` from the raw library (2e-5)."""
    _, _, raw = fritz
    psy = float(np.degrees(np.arcsin(mu)))
    emitted = ln_nu_integral(_sed(fritz, point, psy), _WIDE)
    expected = (
        10.0 ** _KW["agn_log_lbol"]
        * L_SUN
        * _KW["agn_torus_frac"]
        * raw.power(point, mu)
        / raw.sphere_mean(point)
    )
    assert emitted == pytest.approx(float(expected), rel=2e-5)
