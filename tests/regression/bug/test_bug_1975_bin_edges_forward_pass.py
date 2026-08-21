# SPDX-License-Identifier: BSD-3-Clause
"""Regression test for #1975: ``sfh['bin_edges_gyr']`` never reached the forward pass.

The nested-dict grammar accepted the key and stored it on ``spec.bin_edges_gyr``
(#337), but the live stellar component resolved its SFH as the bare
``SFH_REGISTRY[...].fn`` rather than the ``resolve_sfh`` partial that carries the
edges, so ``predict_photometry`` was bit-identical with and without them.

The closed #337 shipped a guard named ``TestForwardPassPropagation`` that asserted
only on spec attributes and recorded "no exception is the contract", which is why
the suite stayed green. These tests assert on model output instead.
"""

import itertools

import numpy as np
import pytest

import tengri
from tengri import FIXED, FREE, Fixed, Observation, Photometry, SEDModel, Uniform

pytestmark = pytest.mark.regression_bug

_Z = 1.5
_BANDS = ["jwst_f090w", "jwst_f150w", "jwst_f200w", "jwst_f356w", "jwst_f444w"]


def _build(ssp, sfh_dict):
    return SEDModel.build(
        ssp_data=ssp,
        observation=Observation(photometry=Photometry.from_names(_BANDS)),
        redshift=Fixed(_Z),
        sfh=sfh_dict,
        met={"logzsol": Uniform(-1.5, 0.3)},
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "all_params": FIXED,
            "tau_diff": Uniform(0.0, 2.0),
        },
    )


def _params(model):
    p = dict(model.spec.get_fixed_values())
    mass_key = next(k for k in model.spec.free_params if k.endswith("log_total_mass"))
    p[mass_key] = 10.3
    p["met_logzsol"] = -0.3
    p["dust_tau_diff"] = 0.3
    for k in model.spec.free_params:
        if "ratio" in k:
            p[k] = 0.4
    return p


@pytest.fixture(scope="module")
def edges_z15():
    """Eight edges out to the age of the universe at z=1.5: 7 bins, 6 ratios."""
    from tengri.cosmology import age_at_z

    t_univ = float(age_at_z(_Z))
    return np.concatenate([[0.0, 0.03], np.logspace(np.log10(0.1), np.log10(t_univ), 6)])


class TestBinEdgesReachForwardPass:
    def test_photometry_differs_with_custom_edges(self, synthetic_ssp_wide, edges_z15):
        """Bit-identical photometry is the signature of a dropped config.

        Both models are built in one process on purpose. ``predict_photometry``
        caches its compiled function on ``compile_signature``, so this also
        guards the second half of #1975: with the edges plumbed but absent from
        that signature, these two structurally identical models shared one
        compiled kernel and the second silently returned the first's fluxes.
        """
        default = _build(synthetic_ssp_wide, {"type": "continuity", "all_params": FREE})
        custom = _build(
            synthetic_ssp_wide,
            {"type": "continuity", "all_params": FREE, "bin_edges_gyr": edges_z15},
        )
        f_default = np.asarray(default.predict_photometry(_params(default)))
        f_custom = np.asarray(custom.predict_photometry(_params(custom)))

        assert np.all(np.isfinite(f_default)) and np.all(np.isfinite(f_custom))
        # Relative, not np.allclose: these fluxes are ~1e-14, far under
        # allclose's atol=1e-8, so it would call any two of them equal.
        max_rel = float(np.max(np.abs(f_custom - f_default) / np.abs(f_default)))
        assert max_rel > 1e-3, (
            "photometry is unchanged by bin_edges_gyr — the edges never reached "
            f"the forward pass (#1975). max relative difference {max_rel:.3e}; "
            f"default={f_default}, custom={f_custom}"
        )

    def test_sfh_steps_land_on_supplied_edges(self, synthetic_ssp_wide, edges_z15):
        """The SFR step boundaries must sit at the edges the user supplied."""
        model = _build(
            synthetic_ssp_wide,
            {"type": "continuity", "all_params": FREE, "bin_edges_gyr": edges_z15},
        )
        state = model.predict_state(_params(model))
        lbt_gyr = np.asarray(state.derived["sfh_grid_lbt_yr"]) / 1e9
        sfr = np.asarray(state.derived["sfr_history"])

        # SFR must be constant strictly inside every supplied bin.
        for lo, hi in itertools.pairwise(edges_z15[1:]):
            inside = sfr[(lbt_gyr > lo) & (lbt_gyr < hi)]
            if inside.size < 2:
                continue
            assert np.allclose(inside, inside[0]), "SFR is not constant within a supplied bin"

        # The default ladder has an edge at 6.0 Gyr that the custom one does not,
        # and the custom ladder ends at ~4.28 Gyr, so nothing may step near 6 Gyr.
        near_six = sfr[(lbt_gyr > 5.0) & (lbt_gyr < 7.0)]
        if near_six.size >= 2:
            assert np.allclose(near_six, near_six[0]), (
                "SFR steps at 6.0 Gyr, an edge of the default ladder, so the "
                "default edges are still in force (#1975)"
            )

    def test_default_still_uses_default_ladder(self, synthetic_ssp_wide):
        """Not passing edges must keep the documented default behavior."""
        model = _build(synthetic_ssp_wide, {"type": "continuity", "all_params": FREE})
        assert model.spec.bin_edges_gyr is None
        assert np.all(np.isfinite(np.asarray(model.predict_photometry(_params(model)))))


class TestBinEdgesValidation:
    def test_wrong_edge_count_raises(self, synthetic_ssp_wide):
        """Too few edges leaves declared ratio parameters with no bin: refuse it."""
        bad = np.array([0.0, 0.1, 1.0, 4.0])  # 3 bins, 2 ratios, but 6 are declared
        with pytest.raises((ValueError, tengri.TengriError)):
            _build(
                synthetic_ssp_wide,
                {"type": "continuity", "all_params": FREE, "bin_edges_gyr": bad},
            )
