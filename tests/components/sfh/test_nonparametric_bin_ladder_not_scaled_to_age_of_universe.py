# SPDX-License-Identifier: BSD-3-Clause
"""Known gap: the default nonparametric bin ladder is not scaled to age(z) (#2521).

``continuity``, ``dirichlet``, ``bursty_continuity``, and ``prospector_beta``
all default to the same ``DEFAULT_BIN_EDGES_GYR = [0.0, 0.03, 0.1, 0.3, 1.0,
3.0, 6.0, 13.7]`` Gyr ladder (:func:`~tengri.components.stellar.sfh
.nonparametric.sfh_bin_edges_yr`) unless a caller passes its own
``bin_edges_gyr`` (e.g. via ``tengri.make_agebins_from_zred(zred=...)``,
which ``prospector_beta``'s own registry entry documents but does not apply
automatically). Because the ladder's outer edge sits at today's age of the
universe, a bin can lie entirely beyond ``age_at_z(z)`` at any z > 0: 0 of 7
bins at z=0.5 (age 8.61 Gyr, still above every edge but the last), 2 of 7 at
z=2.5 (age 2.62 Gyr: the [3.0, 6.0] and [6.0, 13.7] bins), and 3 of 7 at z=6
(age 0.93 Gyr: additionally [1.0, 3.0]).

``_mass_conserving_total`` (#2521) still pins the formed mass to the
request -- an unreachable bin's nominal mass share is redistributed across
the bins that remain within cosmic time rather than the ladder omitting the
bin -- so this is a SHAPE defect, not a mass-conservation one: see
``tests/physics/conservation/test_sfh_support_bounded_to_age_of_universe.py``
for the mass-side contract this file does not duplicate.

This ladder is deliberately NOT rescaled in this change (#2521): rescaling
it changes the meaning of every bin-indexed parameter (``sfh_*_ratio_i``,
the stick-breaking fractions) for every existing fit and recipe that uses
these four families, which is a larger, separately-reviewable change. These
tests instead pin the correct target (zero bins beyond age(z)) as an
``xfail(strict=True)`` naming this issue, so implementing the rescale turns
each one into an unexpected pass -- the repository's convention for a
known, tracked gap (see e.g. ``tests/contract/test_agn_template_threading.py``'s
``_KNOWN_BAKING`` xfails) -- rather than silently landing unnoticed.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri.components.stellar.sfh.nonparametric import sfh_bin_edges_yr
from tengri.components.stellar.sfh.registry import SFH_REGISTRY
from tengri.utils.cosmology import age_at_z

pytestmark = pytest.mark.bounds

_FAMILIES = ("continuity", "dirichlet", "bursty_continuity", "prospector_beta")
_REDSHIFTS = (0.5, 2.5, 6.0)

#: Measured present state: bins (of 7) whose full [start, end) interval lies
#: at or beyond age_at_z(z) -- i.e. entirely before the Big Bang at that z --
#: for the shared default ladder. The physically correct value is 0 at every
#: redshift; see the module docstring for why this is pinned via xfail
#: instead of fixed in this change.
_MEASURED_BINS_BEYOND_AGE = {0.5: 0, 2.5: 2, 6.0: 3}


def _n_bins_beyond_age(family: str, z: float) -> int:
    fn = SFH_REGISTRY[family].fn
    edges_yr = sfh_bin_edges_yr(fn, {})
    assert edges_yr is not None, f"{family}: no known default edge set"
    edges_gyr = np.asarray(edges_yr) / 1e9
    starts_gyr = edges_gyr[:-1]
    age_gyr = float(age_at_z(z))
    return int(np.sum(starts_gyr >= age_gyr))


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("z", _REDSHIFTS)
def test_default_ladder_present_state_matches_the_measured_gap(family, z):
    """Pins today's actual count, so a silent change to the shared ladder is caught.

    Not itself the physics assertion (that is the xfail below) -- this
    guards against ``sfh_bin_edges_yr`` or ``DEFAULT_BIN_EDGES_GYR`` moving
    without anyone updating either this dict or the xfail target.
    """
    assert _n_bins_beyond_age(family, z) == _MEASURED_BINS_BEYOND_AGE[z]


@pytest.mark.parametrize("family", _FAMILIES)
def test_no_default_bin_lies_entirely_before_the_big_bang_at_low_z(family):
    """At z=0.5 (age 8.61 Gyr) the ladder's own 13.7 Gyr outer edge is not yet
    the binding constraint: every bin is already reachable, so this is NOT
    part of the tracked gap and is asserted directly (no xfail).
    """
    assert _n_bins_beyond_age(family, 0.5) == 0


@pytest.mark.parametrize("family", _FAMILIES)
@pytest.mark.parametrize("z", (2.5, 6.0))
@pytest.mark.xfail(
    strict=True,
    reason=(
        "#2521: the default nonparametric bin ladder is not rescaled to "
        "age_at_z(z); see this file's module docstring for the measured "
        "per-redshift bin counts and why the rescale is out of scope here."
    ),
)
def test_no_default_bin_lies_entirely_before_the_big_bang(family, z):
    """The physically correct target: every default bin is reachable at every z."""
    assert _n_bins_beyond_age(family, z) == 0
