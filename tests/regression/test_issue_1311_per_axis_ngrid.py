# SPDX-License-Identifier: BSD-3-Clause
"""`FeaturePrecomp(n_grid=...)` takes a per-axis dict, and validates it (#1311).

The Cue fast grid resolves up to three ionization axes (``met_logzsol``,
``neb_logU``, ``neb_logZ_gas``), and its build cost is the **product** over the
free ones — so one global resolution either over-samples the axes that barely
move the lines or under-samples the one that does. ``n_grid`` therefore accepts
a dict, mirroring the shape ``ranges`` already had.

The per-axis lookup existed in the builder before this change but was unreachable
from the documented public surface and wholly unvalidated: ``n_grid.get(name, 16)``
silently swallowed a misspelled axis and handed back the default grid, and
``n_grid=0`` / ``n_grid={'neb_logU': 'x'}`` were accepted at construction only to
fail (or not) somewhere inside the build. Both are pinned here.

Only one test builds a model — a Cue grid build is a multi-GB transient
(measured ~850 MB above the Cue-exact path for a single axis), and the
regression shard is already OOM-marginal (#1346). The rest are pure
construction-time validation and cost nothing.
"""

import jax
import numpy as np
import pytest

from tengri import DEFAULT, FREE, FeaturePrecomp, Fixed, SEDModel, Uniform, WavePrecomp
from tengri.components.nebular.nebular_grid_precompute import (
    _CANDIDATE_AXES,
    _DEFAULT_N_GRID,
    validate_n_grid,
)

pytestmark = pytest.mark.regression_bug


class TestValidation:
    """Construction-time validation — no model, no grid, no cost."""

    def test_unknown_axis_name_raises(self):
        """LOAD-BEARING: the silent no-op this issue is really about.

        Before #1311 a misspelled axis was dropped by ``dict.get(name, default)``
        and the user silently got a default-resolution grid.

        Neuter: delete the ``unknown`` check in ``validate_n_grid`` and this fails.
        """
        with pytest.raises(ValueError, match="not griddable ionization axes"):
            FeaturePrecomp(n_grid={"neb_logu": 8})  # lowercase 'u' — a real typo

    def test_unknown_axis_error_names_the_valid_axes(self):
        """The error is the documentation — it must say what IS accepted."""
        with pytest.raises(ValueError) as exc:
            FeaturePrecomp(n_grid={"logU": 8})
        message = str(exc.value)
        for axis in _CANDIDATE_AXES:
            assert axis in message, f"error does not name the valid axis {axis!r}: {message}"

    @pytest.mark.parametrize("bad", [0, 1, -5])
    def test_resolution_below_two_raises(self, bad):
        """One knot cannot interpolate; zero and negatives are nonsense."""
        with pytest.raises(ValueError, match="at least two knots"):
            FeaturePrecomp(n_grid=bad)
        with pytest.raises(ValueError, match="at least two knots"):
            FeaturePrecomp(n_grid={"neb_logU": bad})

    @pytest.mark.parametrize("bad", ["x", 3.5, None, True])
    def test_non_integer_raises(self, bad):
        """``bool`` is an ``int`` subclass in Python — reject it explicitly."""
        with pytest.raises(TypeError, match="must be an integer"):
            FeaturePrecomp(n_grid=bad)

    def test_scalar_and_dict_forms_both_accepted(self):
        """The scalar form is unchanged; the dict form is now first-class."""
        assert FeaturePrecomp(n_grid=24).n_grid == 24
        assert FeaturePrecomp(n_grid={"neb_logU": 8}).n_grid == {"neb_logU": 8}
        validate_n_grid(16)
        validate_n_grid({axis: 4 for axis in _CANDIDATE_AXES})


class TestPerAxisReachesTheBuilder:
    """The one test that builds a grid."""

    def test_per_axis_resolution_and_default_fallback(self, ssp_data_fsps, synthetic_tophat_obs):
        """LOAD-BEARING: a per-axis number must actually size that axis.

        Asking for ``neb_logU=2`` while omitting ``neb_logZ_gas`` discriminates
        every failure mode at once: if the dict were ignored, ``neb_logU`` would
        come back at the default instead of 2; if the fallback were broken,
        ``neb_logZ_gas`` would not be at the default.

        Neuter: make ``_axis_n`` ignore the dict branch and ``neb_logU`` returns
        ``_DEFAULT_N_GRID`` instead of 2.
        """
        model = SEDModel.build(
            ssp_data=ssp_data_fsps,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FREE},
            dust_attenuation={"type": "none"},
            redshift=0.05,
            met={"logzsol": FREE},
            neb={
                "type": "cue",
                "all_params": Fixed(DEFAULT),
                "logU": Uniform(-3.5, -1.5),
                "logZ_gas": Uniform(-1.0, 0.4),
            },
            approx=(WavePrecomp(), FeaturePrecomp(n_grid={"neb_logU": 2})),
        )
        table = model._nebular_grid_table
        sizes = {name: axis.shape[0] for name, axis in zip(table.axis_names, table.axes)}

        assert sizes["neb_logU"] == 2, (
            f"explicit n_grid={{'neb_logU': 2}} did not size that axis: {sizes}"
        )
        assert sizes["neb_logZ_gas"] == _DEFAULT_N_GRID, (
            f"omitted axis did not fall back to the default {_DEFAULT_N_GRID}: {sizes}"
        )
        # met_logzsol is snapped to the SSP metallicity nodes (#1020), so its
        # length is set by the SSP grid rather than by the request — assert only
        # that it is present and non-degenerate, never a hard-coded node count.
        assert sizes["met_logzsol"] >= 2, sizes

        # The resulting model must still predict. A dict makes the frozen
        # FeaturePrecomp unhashable where the scalar default was hashable, so a
        # config reaching a compile signature or a cache key by hash would fail
        # here and nowhere in the shape assertions above.
        phot = np.asarray(model.predict_photometry(model.spec.sample(jax.random.PRNGKey(0))))
        assert np.all(np.isfinite(phot)), "per-axis grid model produced non-finite photometry"
