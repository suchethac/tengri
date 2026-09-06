# SPDX-License-Identifier: BSD-3-Clause
r"""``agn.feii='qsogen_balmer'`` was selectable, buildable, and inert (#2175).

Reproduced on this worktree (disc=multicolor, torus/nlr/blr fixed,
``agn_log_lbol=Fixed(12.0)``, ``norm='independent'``): at each FeII type's own
``Fixed(DEFAULT)`` parameters, ``feii='qsogen_balmer'`` photometry was bit-exact
equal to ``feii='boroson_green'`` (``sed_agn`` digests identical to full float64
precision, max abs diff ``0.0``) while ``feii='grahsp'`` differs. That
coincidence is mathematically forced, not a dispatch bug: both blocks' own
strength knob (``agn_bcnorm`` / ``agn_fe2_strength``) declares default ``0.0``
("disabled"), so both blocks legitimately emit an all-zero contribution by
default and any two null contributions summed into the same baseline are
bit-identical.

The genuine defect is upstream of that coincidence: ``qsogen_balmer_block``
(``components/agn/blocks/qsogen_blocks.py``) read its enabling knob via
``params.get("agn_bcnorm", ...)`` out of a loose ``**params`` catch-all instead
of naming it in its signature. Both parameter-routing mechanisms that discover
a composable AGN block's own freeable parameters --
``tengri.parameters.groups._agn_subblock_declared_params`` (sub-block
``'all_params': FREE``, via ``inspect.signature``) and
``tengri.components.agn.blocks._consumes.AGN_BLOCK_CONSUMES`` (the top-level
``agn={'all_params': FREE}`` wildcard) -- need a *named* parameter to find, and
``AGN_BLOCK_CONSUMES[("feii", "qsogen_balmer")]`` was an empty ``frozenset()``
besides. So the block's own knob was invisible to both, permanently pinned at
its disabled default: selectable, buildable, and (via the grammar wildcard)
unfittable.

Fix: name ``agn_bcnorm`` as an explicit keyword-only parameter on
``qsogen_balmer_block`` (matching ``boroson_green_feii_block``'s own pattern),
and add it to ``AGN_BLOCK_CONSUMES[("feii", "qsogen_balmer")]``. This restores
the top-level ``agn={'type': 'composable', 'all_params': FREE, ...}`` wildcard
path (measured below: ``agn_bcnorm`` is now freed and has a non-zero gradient).

RESOLVED (Task 16, item 7): the *sub-block*-scoped wildcard
(``agn.feii={'type': 'qsogen_balmer', 'all_params': FREE}``, exercised by
``tests/contract/test_agn_subblock_wildcard_scoping.py``) needed
``agn_bcnorm`` entered in ``_AGN_PARTITION`` in
``src/tengri/parameters/groups.py`` (mapped to ``"agn.feii"``, alongside the
existing ``"agn_fe2_strength": "agn.feii"`` entry) -- that entry has now
landed. One side effect: ``agn_bcnorm``'s CANONICAL location is now the
``feii`` sub-block, not the top level, so
``_build_agn_search_view``'s documented precedence rule ("a sub-block
wildcard takes precedence over the top-level one") applies to it like every
other sub-block-owned parameter. ``test_qsogen_balmer_wildcard_frees_and_grad_is_live``
below previously paired a top-level ``agn={'all_params': FREE}`` with an
EXPLICIT ``feii={'all_params': Fixed(DEFAULT)}`` -- a recipe that only
worked because ``agn_bcnorm`` was (mis)classified as a shared, top-level
parameter at the time. It now demonstrates the sub-block's own explicit
disposition correctly winning, not the top-level reach-in this test means
to exercise; updated to omit the sub-block's own wildcard entirely (its
structural ``'type'`` selection alone), letting the top-level wildcard's
documented inheritance path govern it, matching how a caller actually
frees a sub-block-owned parameter through an ancestor wildcard.
``test_q2_wired_against_siblings[feii/qsogen_balmer]`` in that other file
is unaffected: item 7 (issue #2175, prior_median evaluation) already made
it pass for real -- see this repository's Task 16 report.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytestmark = pytest.mark.regression_bug

from tengri import DEFAULT, FREE, Fixed, SEDModel
from tengri.components.agn.blocks._protocol import AGN_BLOCKS
from tengri.observation import Observation, Photometry
from tengri.observation.photometry import FilterCurve

#: A representative rest-frame grid and disc-side λL_λ(5100Å) [erg/s] shared by
#: the two block-level tests below, so they compare the registered block
#: FUNCTIONS directly (bypassing SEDModel.build) with every OTHER input held
#: bit-identical -- the precise "two differently-named models, same inputs"
#: comparison the issue makes. A full-model-build comparison confounds this:
#: the two builds' internal l5100_disc/WavePrecomp bookkeeping differ by a few
#: percent for reasons unrelated to the FeII slot, which is too noisy to catch
#: a block silently re-routed to another block's physics.
_BLOCK_WAVE = jnp.logspace(2.0, 7.0, 1600)
_BLOCK_L5100_DISC = jnp.asarray(1.0e40)
_BLOCK_AGN_LOG_LBOL = 12.0


def _feii_block(block_type: str):
    return AGN_BLOCKS["feii"][block_type]


def _tophat(center: float, frac: float = 0.05, n: int = 40) -> FilterCurve:
    """Narrow band straddling the Balmer edge region so the FeII/Balmer slot's
    contribution is not swamped by the rest of the panchromatic SED (broadband
    photometry is known to swamp BLR/FeII categories, see
    ``tests/contract/test_agn_subblock_wildcard_scoping.py``'s module docstring)."""
    wave = jnp.linspace(center * (1.0 - frac), center * (1.0 + frac), n)
    trans = jnp.sin(jnp.linspace(0.0, jnp.pi, n)) * 0.6
    return FilterCurve(wave=wave, trans=trans, name=f"b{int(center)}")


#: UV bands straddling the Balmer edge at 3646 A (rest, redshift=0 here).
_CENTERS = (3000.0, 3400.0, 3646.0, 4000.0, 5000.0, 9000.0)


@pytest.fixture(scope="module")
def near_uv_obs() -> Observation:
    return Observation(photometry=Photometry(filters=tuple(_tophat(c) for c in _CENTERS)))


def _build(ssp, observation, feii_group, *, agn_all_params=None):
    """Minimal composable AGN + negligible stellar mass, so the AGN's own FeII
    contribution is not swamped by the stellar continuum (the same "photometry
    cannot see it" trap the FeII/BLR categories hit at realistic stellar mass)."""
    if agn_all_params is None:
        agn_all_params = Fixed(DEFAULT)
    agn = {
        "type": "composable",
        "disc": {"type": "multicolor", "all_params": Fixed(DEFAULT)},
        "nlr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "blr": {"type": "analytic", "all_params": Fixed(DEFAULT)},
        "feii": feii_group,
        "all_params": agn_all_params,
        "agn_log_lbol": Fixed(12.0),
        "norm": "independent",
    }
    dust_group = {
        "type": "two_component",
        "law": "calzetti",
        "all_params": Fixed(DEFAULT),
    }
    return SEDModel.build(
        ssp_data=ssp,
        observation=observation,
        sfh={
            "type": "const",
            "all_params": Fixed(DEFAULT),
            "log_total_mass": -5.0,
            "start_gyr": 1.0,
        },
        dust_attenuation=dust_group,
        agn=agn,
        redshift=Fixed(0.0),
    )


def _max_rel_diff(a: np.ndarray, b: np.ndarray) -> float:
    denom = np.maximum(np.abs(b), 1e-300)
    with np.errstate(over="ignore"):
        return float(np.max(np.abs(a - b) / denom))


def test_qsogen_balmer_differs_from_boroson_green_at_matched_params():
    """#2175: calling the two REGISTERED feii block functions directly with
    bit-identical wavelength / agn_log_lbol / l5100_disc and each block's own
    strength knob at 2.0 (top of its declared [0, 2] prior), qsogen_balmer's
    Balmer-continuum physics (Grandi 1982) and boroson_green's FeII multiplet
    template (Boroson & Green 1992) must produce measurably different output
    -- two differently-named physical models producing exactly equal output
    (the pre-fix, and still-default-params, behavior) is the signature of a
    block that never reaches the graph."""
    qsogen_balmer = _feii_block("qsogen_balmer")
    boroson_green = _feii_block("boroson_green")

    out_qsogen = np.asarray(
        qsogen_balmer(_BLOCK_WAVE, _BLOCK_AGN_LOG_LBOL, _BLOCK_L5100_DISC, agn_bcnorm=2.0)
    )
    out_boroson = np.asarray(
        boroson_green(_BLOCK_WAVE, _BLOCK_AGN_LOG_LBOL, _BLOCK_L5100_DISC, agn_fe2_strength=2.0)
    )

    assert np.all(np.isfinite(out_qsogen)) and np.all(np.isfinite(out_boroson))
    max_rel_diff = _max_rel_diff(out_qsogen, out_boroson)
    # 1e-2 is far above any floating-point noise floor (the pre-fix value was
    # EXACTLY 0.0 -- confirmed bit-exact array equality on this worktree) and
    # far below the measured post-fix value, so this is not a threshold tuned
    # to the fix.
    assert max_rel_diff > 1e-2, (
        f"feii='qsogen_balmer' (agn_bcnorm=2.0) is indistinguishable from "
        f"feii='boroson_green' (agn_fe2_strength=2.0) at identical wavelength/ "
        f"agn_log_lbol/l5100_disc: max rel diff {max_rel_diff:.3e}. Two "
        f"differently-named physical models producing the same output means "
        f"the block does not apply its own physics (#2175)."
    )


def test_qsogen_balmer_own_parameter_moves_its_output():
    """The block's own knob, swept end to end at fixed wavelength/agn_log_lbol
    /l5100_disc, must move the Balmer-continuum contribution far beyond the
    pre-fix EXACT 0.0 (bit-exact array equality, confirmed on this worktree)."""
    qsogen_balmer = _feii_block("qsogen_balmer")
    out_on = np.asarray(
        qsogen_balmer(_BLOCK_WAVE, _BLOCK_AGN_LOG_LBOL, _BLOCK_L5100_DISC, agn_bcnorm=2.0)
    )
    out_off = np.asarray(
        qsogen_balmer(_BLOCK_WAVE, _BLOCK_AGN_LOG_LBOL, _BLOCK_L5100_DISC, agn_bcnorm=0.0)
    )
    assert np.array_equal(out_off, np.zeros_like(out_off)), (
        "agn_bcnorm=0.0 (disabled) must emit an exactly-zero Balmer continuum."
    )
    max_rel_diff = _max_rel_diff(out_on, out_off)
    assert max_rel_diff > 1e-2, (
        f"Sweeping agn_bcnorm 0.0 -> 2.0 moved the block's own output by only "
        f"{max_rel_diff:.3e} (pre-fix: exactly 0.0)."
    )


def test_qsogen_balmer_wildcard_frees_and_grad_is_live(synthetic_ssp_wide, near_uv_obs):
    """#2175 secondary claim: under the top-level ``agn={'all_params': FREE}``
    wildcard, ``agn_bcnorm`` must actually be freed (not silently pinned at its
    disabled default) and must have a non-zero gradient of
    ``predict_photometry`` -- "selectable, unfittable, and inert" made false.

    The ``feii`` sub-dict states only its structural ``'type'``, no
    ``'all_params'`` of its own (Task 16, item 7's ``_AGN_PARTITION`` entry
    made ``agn_bcnorm``'s canonical home the ``feii`` sub-block; an explicit
    sub-block wildcard there would take precedence over the top-level one
    per ``_build_agn_search_view``'s documented rule, which is not the
    top-level reach-in this test means to exercise -- see the module
    docstring's RESOLVED note)."""
    model = _build(
        synthetic_ssp_wide,
        near_uv_obs,
        {"type": "qsogen_balmer"},
        agn_all_params=FREE,
    )
    free_agn = {p for p in model.spec.free_params if p.startswith("agn_")}
    assert "agn_bcnorm" in free_agn, (
        f"agn.feii='qsogen_balmer' under agn={{'all_params': FREE}} did not "
        f"free agn_bcnorm; free AGN params were {sorted(free_agn)} (#2175)."
    )

    params = dict(model.spec.sample(jax.random.PRNGKey(0)))

    def objective(bcnorm_value):
        p = {**params, "agn_bcnorm": bcnorm_value}
        return jnp.log(jnp.sum(model.predict_photometry(p)) + 1e-300)

    grad = float(jax.grad(objective)(jnp.asarray(params["agn_bcnorm"])))
    assert grad != 0.0, "agn_bcnorm was freed but is dead (grad=0 on predict_photometry) (#2175)."
