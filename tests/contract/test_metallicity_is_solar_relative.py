# SPDX-License-Identifier: BSD-3-Clause
"""A metallicity a user reads back must be in the units it was set in (#1703).

``met_logzsol`` is ``log10(Z/Zsun)``. Every summary of it — the property
catalog, ``predict_properties``, ``pred.sfh.*``, the deprecated
``predict_sfh_quantities`` — returned **absolute** ``log10(Z)``, 1.85 dex
lower, a factor of 70 in Z. Three separate docstrings, the registry entry and
``docs/api/_property_table.md`` all said ``log10(Z/Zsun)``.

The argument is not about which convention is nicer. In ``met_mode='delta'``
the SFH carries **one** metallicity, so a mass- or luminosity-weighted mean
over it must reproduce that value. It did not::

    met_logzsol   reported     reported - input
        -1.000   -2.847712        -1.847712
        -0.500   -2.347712        -1.847712
         0.000   -1.847712        -1.847712
         0.250   -1.597712        -1.847712
         0.500   -1.347712        -1.847712

    fit: reported = 1.000000 * met_logzsol + (-1.847712)
    LOG10_ZSUN  =                            -1.8477116556169435

Slope exactly 1, intercept exactly ``LOG10_ZSUN``. A mean over a single value
that does not return that value is broken under any naming convention.

Cause: ``param_map`` **adds** ``LOG10_ZSUN`` on the way in and nothing
subtracted it on the way out. The translation was one-directional.

What must NOT change: ``state.derived["log_metallicity_history"]``, the SSP
grid and the interpolators all work in absolute ``log10(Z)``. The conversion
belongs at the publish boundary, and this file pins both halves — the six
user-facing surfaces are relative, the internal history stays absolute.
"""

from __future__ import annotations

import numpy as np
import pytest

from tengri import FIXED, Fixed, Observation, Photometry, SEDModel
from tengri.forward.properties import PROPERTY_REGISTRY
from tengri.utils.conversions import log_z_abs_to_logzsol, logzsol_to_log_z_abs
from tengri.utils.physics_constants import LOG10_ZSUN

pytestmark = [pytest.mark.contract, pytest.mark.regression_bug]

#: Sweeping beats a single point: a lone value at met_logzsol=0 cannot tell
#: "absolute" from "some unrelated constant". The slope matters too.
_LOGZSOLS = (-1.0, -0.5, 0.0, 0.25, 0.5)


@pytest.fixture(scope="module")
def delta_model(ssp_data_fsps):
    """One metallicity for the whole SFH, so any weighted mean must equal it."""
    obs = Observation(photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r"]))
    return SEDModel.build(
        ssp_data=ssp_data_fsps,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        dust={"type": "two_component", "law_bc": "calzetti", "all_params": FIXED},
        neb={"type": "none"},
        stellar={"met_mode": "delta"},
        redshift=Fixed(0.1),
    )


def _fit_against_input(model, getter):
    """Fit ``reported = a * met_logzsol + b`` over the sweep."""
    xs, ys = [], []
    for z in _LOGZSOLS:
        params = dict(model.spec.get_fixed_values())
        params["met_logzsol"] = z
        xs.append(z)
        ys.append(float(getter(model, params)))
    slope, intercept = np.polyfit(xs, ys, 1)
    return slope, intercept


#: Every user-facing surface that reports a weighted metallicity. Discovered
#: names come from the registry; the accessor and deprecated paths are listed
#: because they are reached by attribute, not by name.
_SURFACES = {
    "properties[mass_weighted_metallicity]": lambda m, p: m.predict(p).properties[
        "mass_weighted_metallicity"
    ],
    "properties[luminosity_weighted_metallicity]": lambda m, p: m.predict(p).properties[
        "luminosity_weighted_metallicity"
    ],
    "predict_properties(mass_weighted)": lambda m, p: m.predict_properties(
        p, names=("mass_weighted_metallicity",)
    )["mass_weighted_metallicity"],
    "pred.sfh.mass_weighted_metallicity": lambda m, p: m.predict(p).sfh.mass_weighted_metallicity,
    "pred.sfh.luminosity_weighted_metallicity": lambda m, p: (
        m.predict(p).sfh.luminosity_weighted_metallicity
    ),
    "predict_sfh_quantities().mass_weighted_metallicity": lambda m, p: (
        m.predict_sfh_quantities(p).mass_weighted_metallicity
    ),
}


class TestTheCensusIsComplete:
    def test_every_registered_metallicity_property_is_covered(self):
        """A new weighted-metallicity property must not skip this file."""
        registered = {
            name for name in PROPERTY_REGISTRY if "metallicity" in name and "weighted" in name
        }
        covered = {n for n in registered if any(n in key for key in _SURFACES)}
        assert registered <= covered, (
            f"{sorted(registered - covered)} report a weighted metallicity and "
            f"are not in _SURFACES, so nothing checks their convention."
        )

    def test_the_surfaces_are_not_all_the_same_object(self, delta_model):
        """Five of six delegate, but the deprecated path recomputes.

        If every entry resolved to one implementation this file would be one
        test wearing six hats — and the deprecated path is exactly where the
        fix was nearly missed.
        """
        params = dict(delta_model.spec.get_fixed_values())
        params["met_logzsol"] = 0.3
        values = {label: float(fn(delta_model, params)) for label, fn in _SURFACES.items()}
        assert len(values) == 6, values


class TestEverySurfaceIsSolarRelative:
    @pytest.mark.parametrize("label", sorted(_SURFACES))
    def test_a_single_metallicity_reads_back_as_itself(self, label, delta_model):
        slope, intercept = _fit_against_input(delta_model, _SURFACES[label])
        assert np.isclose(slope, 1.0, atol=1e-6), (
            f"{label}: reported metallicity does not track met_logzsol "
            f"one-for-one (slope={slope:.6f})."
        )
        assert abs(intercept) < 1e-6, (
            f"{label}: a delta-metallicity SFH carries one metallicity, so the "
            f"weighted mean must equal the input. Measured offset "
            f"{intercept:+.6f} dex; LOG10_ZSUN is {LOG10_ZSUN:+.6f}, so this is "
            f"the grid's absolute log10(Z) escaping to a surface documented as "
            f"log10(Z/Zsun) — a factor of "
            f"{10 ** abs(intercept):.0f} in Z."
        )


class TestTheInternalHistoryStaysAbsolute:
    """The SSP grid's convention must not be 'fixed' along with the readout."""

    def test_log_metallicity_history_is_still_absolute(self, delta_model):
        params = dict(delta_model.spec.get_fixed_values())
        params["met_logzsol"] = 0.0
        history = np.asarray(delta_model.predict_state(params).derived["log_metallicity_history"])
        assert np.allclose(history, LOG10_ZSUN, atol=1e-9), (
            f"state.derived['log_metallicity_history'] should hold absolute "
            f"log10(Z) — the SSP grid, the interpolators and the nebular "
            f"backends all read it that way. Got {history[:3]} for solar."
        )


class TestTheConverterPair:
    def test_round_trip(self):
        for z in _LOGZSOLS:
            assert np.isclose(log_z_abs_to_logzsol(logzsol_to_log_z_abs(z)), z, atol=1e-12)

    def test_the_offset_is_log10_zsun_and_not_a_free_constant(self):
        assert np.isclose(float(logzsol_to_log_z_abs(0.0)), LOG10_ZSUN, atol=0.0)
        assert np.isclose(float(log_z_abs_to_logzsol(0.0)), -LOG10_ZSUN, atol=0.0)
