# SPDX-License-Identifier: BSD-3-Clause
"""`approx=FeaturePrecomp()` must reach Cue's fast grid for photometry-only fits (#1353).

`SEDModel.enable_fast_nebular` (#950) attaches a per-Q_H grid that replaces the Cue
forward for **lines and photometry** — `predict_photometry` reconstructs the nebular
broadband as ``Q_H x interp(grid)``. The declarative route to it,
``approx=FeaturePrecomp()``, nevertheless resolved line wavelengths *before* looking at
the backend and raised when none were available. A photometry-only Cue model therefore
could not reach the grid through the public grammar at all: it silently stayed on the
un-tabulated path, measured at **~14x** the cost (3751 us vs 278 us warm), with no
warning.

Line wavelengths are only needed for line *observables*; the broadband speedup needs
none. These tests pin that the photometry-only path is reachable and correct.

The assertions are **structural, not timed** — they check the grid is attached and the
answer is unchanged. Wall-clock on a shared runner is worthless for this (the same
comparison measured 13.5x and 15.0x on consecutive runs of one machine).
"""

import jax
import numpy as np
import pytest

from tengri import FIXED, FREE, FeaturePrecomp, SEDModel, WavePrecomp


def _cue_model(ssp, obs, approx):
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FREE},
        dust={"type": "none"},
        redshift=0.05,
        neb={"type": "cue", "all_params": FIXED},
        approx=approx,
    )


def _grid_attached(model):
    """True when the cached chain's nebular component carries a grid_table.

    ``enable_fast_nebular`` rebuilds the chain and swaps in a
    ``dataclasses.replace(c, grid_table=table)`` nebular component, storing it on
    ``_cached_component_chain`` (sed_model.py:4425).
    """
    chain = getattr(model, "_cached_component_chain", None)
    if not chain:
        return False
    return any(getattr(c, "grid_table", None) is not None for c in chain)


@pytest.mark.regression_bug
class TestCuePhotometryOnlyPrecomp:
    def test_feature_precomp_reaches_the_grid_without_lines(
        self, ssp_data_fsps, synthetic_tophat_obs
    ):
        """LOAD-BEARING: the declarative grammar must attach the grid, not raise.

        Neuter: restore the unconditional `raise` on missing lines in
        `_apply_feature_precomp` and this fails with ValueError.
        """
        model = _cue_model(ssp_data_fsps, synthetic_tophat_obs, (WavePrecomp(), FeaturePrecomp()))
        assert _grid_attached(model), (
            "approx=(WavePrecomp(), FeaturePrecomp()) did not attach the per-Q_H "
            "nebular grid for a photometry-only Cue model — it is still on the "
            "un-tabulated Cue path (#1353)."
        )

    def test_the_fast_path_does_not_change_the_answer(self, ssp_data_fsps, synthetic_tophat_obs):
        """The grid is an approximation with a documented budget, not a free lunch.

        `enable_fast_nebular`'s docstring documents 0.42% between grid nodes; a
        photometry-only attachment must stay within that, or the speedup is a bug.
        """
        exact = _cue_model(ssp_data_fsps, synthetic_tophat_obs, WavePrecomp())
        fast = _cue_model(ssp_data_fsps, synthetic_tophat_obs, (WavePrecomp(), FeaturePrecomp()))
        params = exact.spec.sample(jax.random.PRNGKey(0))

        a = np.asarray(exact.predict_photometry(params), dtype=np.float64)
        b = np.asarray(fast.predict_photometry(params), dtype=np.float64)
        # atol=0: these fluxes are ~1e-30 and numpy's default atol=1e-8 would call
        # any two of them equal.
        rel = np.max(np.abs(b - a) / np.abs(a))
        print(f"max relative difference fast vs exact Cue: {rel:.2e}")
        assert rel < 4.2e-3, (
            f"photometry-only fast nebular drifts {rel:.2e} from the exact Cue "
            "forward, beyond the 0.42% the grid documents"
        )

    def test_non_cue_backend_still_refuses_without_lines(self, ssp_data_wne, synthetic_tophat_obs):
        """The relaxation is scoped to Cue-like backends only.

        A backend that is not linear in Q_H cannot be reconstructed by the grid, so
        `FeaturePrecomp` must still raise its teaching error rather than silently
        attaching nothing.
        """
        with pytest.raises(ValueError, match="FeaturePrecomp"):
            SEDModel.build(
                ssp_data=ssp_data_wne,
                observation=synthetic_tophat_obs,
                sfh={"type": "dpl", "all_params": FREE},
                dust={"type": "none"},
                redshift=0.05,
                approx=(WavePrecomp(), FeaturePrecomp()),
            )
