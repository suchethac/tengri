# SPDX-License-Identifier: BSD-3-Clause
"""compile_signature must distinguish the composable AGN blocks (#1450).

``compile_signature`` keyed the AGN axis on ``agn_model`` and
``agn_luminosity_mode`` alone. ``agn_model`` carries no discriminating power:
the composable surface is the only non-deprecated one, so ``list_agn_models()``
returns exactly one selectable entry and every composable model hashes to the
same string. The six block selectors ARE the axis, and each swaps the emitting
physics without changing the graph shape — so the whole AGN axis was unkeyed
and the first-built kernel won, the same cache-collision class as
#1135/#1149/#1163/#1166 and the fixed-z case.

Measured on the parent commit: ``torus='skirtor'`` and ``torus='cat3d_wind'``
agreed bit-for-bit *within* a process and disagreed by 60 % in W4 *across*
processes, depending only on which was built first.

Two models per test, deliberately. A single-model test passes on the broken
code no matter which kernel it received — a cache collision and a physics no-op
are indistinguishable from one build, and differ only in whether the answer
depends on build order.
"""

import numpy as np
import pytest

import tengri  # noqa: F401
from tengri import FIXED, Fixed, SEDModel

pytestmark = pytest.mark.regression_bug


# Each entry: the grammar sub-block, and two implementations that must not
# share a compiled kernel. Values come from the build grammar's own menus.
_BLOCK_PAIRS = [
    ("torus", "skirtor", "cat3d_wind"),
    ("disc", "grahsp_sbpl", "powerlaw"),
    ("nlr", "cue", "none"),
    ("blr", "analytic", "none"),
    ("feii", "boroson_green", "none"),
    ("atten", "smc_prevot", "none"),
]


def _model(ssp, obs, **block):
    """A composable-AGN model with one block overridden.

    The disc is always present: downstream blocks normalize to the disc's
    lambda*L_lambda(5100 A), so with ``disc='none'`` every other block emits
    exactly zero and any comparison between them is vacuous — the builder
    warns about precisely this.
    """
    agn = {
        "type": "composable",
        "disc": {"type": "grahsp_sbpl"},
        "all_params": FIXED,
    }
    for key, value in block.items():
        agn[key] = {"type": value}
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh={"type": "dpl", "all_params": FIXED},
        agn=agn,
        redshift=Fixed(0.05),
    )


@pytest.mark.parametrize(("block", "first", "second"), _BLOCK_PAIRS)
def test_each_agn_block_selector_changes_the_signature(
    synthetic_ssp_wide, synthetic_tophat_obs, block, first, second
):
    """Every composable slot must be part of the structural fingerprint.

    Parametrized over all six slots rather than spot-checking the torus: the
    bug was that the axis as a whole was unkeyed, so a guard naming one slot
    would leave the other five free to regress.
    """
    sig_a = _model(synthetic_ssp_wide, synthetic_tophat_obs, **{block: first}).compile_signature()
    sig_b = _model(synthetic_ssp_wide, synthetic_tophat_obs, **{block: second}).compile_signature()
    assert sig_a != sig_b, (
        f"agn {block!r}={first!r} and {block!r}={second!r} share a compile_signature, "
        "so they share a cached kernel and the first one built wins"
    )


def test_agn_norm_policy_changes_the_signature(synthetic_ssp_wide, synthetic_tophat_obs):
    """'cigale_joint' vs 'independent' is a different SED, not a flag (#556).

    One ties disc/torus/polar to a single energy-conserving reference; the
    other puts each on its own luminosity scale. Same graph shape, so it must
    be keyed or it leaks exactly like the block selectors.
    """

    def build(norm):
        return SEDModel.build(
            ssp_data=synthetic_ssp_wide,
            observation=synthetic_tophat_obs,
            sfh={"type": "dpl", "all_params": FIXED},
            agn={
                "type": "composable",
                "disc": {"type": "grahsp_sbpl"},
                "torus": {"type": "skirtor"},
                "norm": norm,
                "all_params": FIXED,
            },
            redshift=Fixed(0.05),
        )

    assert build("cigale_joint").compile_signature() != build("independent").compile_signature()


def test_two_torus_libraries_do_not_return_the_same_photometry(
    synthetic_ssp_wide, synthetic_tophat_obs
):
    """End-to-end: the collision was observable in the public API.

    ``predict_photometry`` is the surface #1450 was reported against. Both
    models are built in this one process on purpose — under the bug the second
    inherits the first's kernel and the two agree bit-for-bit, so inequality
    here is exactly the property that regressed.
    """
    import jax

    model_a = _model(synthetic_ssp_wide, synthetic_tophat_obs, torus="skirtor")
    model_b = _model(synthetic_ssp_wide, synthetic_tophat_obs, torus="cat3d_wind")

    def photometry(model):
        params = {
            **model.spec.get_fixed_values(),
            **model.spec.sample(jax.random.PRNGKey(0)),
        }
        return np.asarray(model.predict_photometry(params))

    phot_a, phot_b = photometry(model_a), photometry(model_b)

    # Assert the setup before the conclusion: if the AGN emitted nothing, the
    # two would agree for a reason that has nothing to do with the cache.
    assert np.all(np.isfinite(phot_a)) and np.all(np.isfinite(phot_b))
    assert np.any(phot_a > 0.0), "no flux — the comparison below would be vacuous"

    assert not np.array_equal(phot_a, phot_b), (
        "skirtor and cat3d_wind returned bit-identical photometry: the second "
        "model inherited the first's compiled kernel"
    )
