# SPDX-License-Identifier: BSD-3-Clause
"""#2634: the Synthesizer-grid NLR/BLR blocks forward ``agn_log_mbh`` / ``agn_log_ledd``.

``nlr_synthesizer_spectra_block`` and ``blr_synthesizer_spectra_block`` called the
grid backend without ``log_bh_mass=`` / ``log_eddington=``, so the grid's ``mass``
and ``mdot_Edd`` axes sat at the backend defaults (8.0, -0.3) whatever the model
declared: both parameters were inert on the composable block path. The U / n_H /
Z axes were made drivable by #931; these two were not.

The backend ``compute_nlr_sed_synthesizer_spectra`` is the reference. Cells:

(a) the block equals the backend at explicit (log M_BH, log lambda_Edd), both regions;
(b) the axes move the spectrum by the backend-measured amounts;
(c) the same through ``SEDModel.build``, value and ``jax.grad``;
(d) the #931 U / n / Z forwarding is unchanged.

Grid-gated: the Synthesizer AGN test grids are not shipped, so the module skips
only when the specific grid file is missing.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn.blocks.blr import blr_synthesizer_spectra_block
from tengri.components.agn.blocks.nlr import (
    _resolve_synthesizer_grid,
    nlr_synthesizer_spectra_block,
)
from tengri.components.agn.nlr_cloudy import compute_nlr_sed_synthesizer_spectra
from tengri.config.exceptions import TengriIOError
from tengri.parameters.priors import Uniform
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.contract

_LOG_LBOL = 13.0
_WAVE = np.linspace(900.0, 30000.0, 29101)
_O3_WINDOW = (_WAVE >= 4995.0) & (_WAVE <= 5020.0)

#: region -> (block, per-region photoionization knobs in the block's own names).
_BLOCKS = {
    "nlr": (nlr_synthesizer_spectra_block, "agn_nlr", -2.0),
    "blr": (blr_synthesizer_spectra_block, "agn_blr", -1.0),
}
_REGIONS = tuple(_BLOCKS)

_BASE = (8.0, -0.3)

#: Measured from the BACKEND on the two-node test grids (``_WAVE``, L_bol = 1e13 L_sun,
#: cf = 0.1, region default U / n=4 / Z=-2): ratio of the modified spectrum to the
#: (8.0, -0.3) baseline, bolometric trapezoid and [O III] 5007 pixel sum (4995-5020 A).
_RATIOS = {
    ("nlr", "ledd"): (0.8765613891643922, 0.821758212949826),
    ("nlr", "mbh"): (0.8251230154217267, 0.757977443821904),
    ("blr", "ledd"): (0.879708808683233, 0.8244935449320386),
    ("blr", "mbh"): (0.8292592457445129, 0.760726440329576),
}
_AXIS_POINTS = {"ledd": (8.0, -1.0), "mbh": (9.0, -0.3)}


@pytest.fixture(scope="module", autouse=True)
def _grids_present():
    """Skip only when a specific Synthesizer AGN grid file is missing."""
    for kind in _REGIONS:
        try:
            _resolve_synthesizer_grid(kind)
        except TengriIOError:
            pytest.skip(f"Synthesizer AGN {kind} test grid not found")


def _knobs(region, logU=None, logn=4.0, logZ=-2.0):
    _, prefix, logU_default = _BLOCKS[region]
    return {
        f"{prefix}_cf": 0.1,
        f"{prefix}_logU": logU_default if logU is None else logU,
        f"{prefix}_logn": logn,
        f"{prefix}_logZ": logZ,
    }


def _block(region, **extra):
    """The block's isotropic-channel L_lambda (second element of its return)."""
    fn = _BLOCKS[region][0]
    maskable, isotropic = fn(_WAVE, _LOG_LBOL, 0.0, **_knobs(region), **extra)
    assert not np.any(np.asarray(maskable)), "the Synthesizer line regions are isotropic"
    return np.asarray(isotropic)


def _backend(region, mbh, ledd, logU=None, logn=4.0, logZ=-2.0):
    """Reference: the grid backend L_nu converted to L_lambda as the block does."""
    _, _, logU_default = _BLOCKS[region]
    l_nu = compute_nlr_sed_synthesizer_spectra(
        _WAVE,
        l_disc_bol_erg=10.0**_LOG_LBOL * L_SUN,
        covering_fraction=0.1,
        grid_path=_resolve_synthesizer_grid(region),
        log_bh_mass=mbh,
        log_eddington=ledd,
        neb_logU=logU_default if logU is None else logU,
        neb_logn=logn,
        neb_logZ_gas=logZ,
        region=region,
    )
    return np.asarray(l_nu) * C_AA / _WAVE**2


def _bolometric(l_lambda):
    return float(np.trapezoid(l_lambda, _WAVE))


@pytest.mark.parametrize("region", _REGIONS)
@pytest.mark.parametrize("mbh, ledd", [(8.0, -1.0), (9.0, -1.0), (8.0, -0.3)])
def test_block_equals_backend_at_declared_axes(region, mbh, ledd):
    """(a) The block reproduces the backend at the (mass, Eddington) it is given."""
    got = _block(region, agn_log_mbh=mbh, agn_log_ledd=ledd)
    want = _backend(region, mbh, ledd)
    assert np.max(want) > 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=1e-12,
        atol=0,
        err_msg=f"{region}: block ignores agn_log_mbh={mbh}, agn_log_ledd={ledd}",
    )


@pytest.mark.parametrize("region", _REGIONS)
@pytest.mark.parametrize("axis", ["ledd", "mbh"])
def test_axes_move_the_spectrum_by_the_backend_amount(region, axis):
    """(b) Bolometric and [O III] 5007 ratios vs the (8.0, -0.3) baseline."""
    mbh, ledd = _AXIS_POINTS[axis]
    bol_want, o3_want = _RATIOS[(region, axis)]
    base = _block(region, agn_log_mbh=_BASE[0], agn_log_ledd=_BASE[1])
    moved = _block(region, agn_log_mbh=mbh, agn_log_ledd=ledd)
    bol = _bolometric(moved) / _bolometric(base)
    o3 = float(moved[_O3_WINDOW].sum() / base[_O3_WINDOW].sum())
    assert abs(bol - 1.0) > 1e-2, f"{region}/{axis}: bolometric ratio {bol} (axis inert)"
    assert abs(o3 - 1.0) > 1e-2, f"{region}/{axis}: [O III] ratio {o3} (axis inert)"
    np.testing.assert_allclose(bol, bol_want, rtol=1e-6)
    np.testing.assert_allclose(o3, o3_want, rtol=1e-6)


def _build(ssp, ledd):
    """Composable AGN with the grid NLR; ``log_ledd`` sits in the disc scope."""
    return SEDModel.build(
        ssp,
        sfh={
            "type": "delayed",
            "tau_gyr": Fixed(1.0),
            "age_gyr": Fixed(5.0),
            "log_total_mass": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        dust_attenuation={
            "law": "power_law",
            "type": "two_component",
            "tau_bc": Fixed(0.0),
            "tau_diff": Fixed(0.0),
            "all_params": Fixed(DEFAULT),
        },
        agn={
            "type": "composable",
            "all_params": Fixed(DEFAULT),
            "log_lbol": _LOG_LBOL,
            "disc": {"type": "multicolor", "log_ledd": ledd, "all_params": Fixed(DEFAULT)},
            "nlr": {"type": "synthesizer_spectra", "all_params": Fixed(DEFAULT)},
        },
        redshift=Fixed(0.0),
    )


_LINES = "sed_agn_lines"  # nlr + blr + feii light of the composable AGN


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:agn_log_ledd has no effect:UserWarning")
def test_agn_log_ledd_moves_the_nlr_light_through_the_grammar(synthetic_ssp_wide):
    """(c) Value: Fixed(-1.0) vs Fixed(-0.3) differ by more than 1e-2 (relative max)."""

    def _lines(ledd):
        model = _build(synthetic_ssp_wide, Fixed(ledd))
        return np.asarray(model.predict_state({}).derived[_LINES])

    a, b = _lines(-1.0), _lines(-0.3)
    assert b.max() > 0.0, "the grid NLR produced no light"
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert rel > 1e-2, f"agn_log_ledd is inert through SEDModel.build (max rel diff {rel})"


@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:agn_log_ledd has no effect:UserWarning")
def test_agn_log_ledd_gradient_reaches_the_nlr_light(synthetic_ssp_wide):
    """(c) With ``agn_log_ledd`` free, d(sum lines)/d(agn_log_ledd) is finite and non-zero."""
    free = _build(synthetic_ssp_wide, Uniform(-1.5, -0.1))
    assert free.spec.free_params == ["agn_log_ledd"]
    params = free.spec.sample(jax.random.PRNGKey(0))
    grad = jax.grad(lambda p: jnp.sum(free.predict_state(p).derived[_LINES]))(params)
    g = np.asarray(grad["agn_log_ledd"])
    assert np.isfinite(g), f"d(sum {_LINES})/d agn_log_ledd is not finite: {g}"
    assert g != 0.0, "agn_log_ledd has an exactly zero gradient (inert parameter)"


@pytest.mark.parametrize("region", _REGIONS)
def test_photoionization_axes_still_forwarded(region):
    """(d) #931 unchanged (GREEN before and after): U / n / Z reach the backend.

    Mass and Eddington ratio are passed at the pre-fix hard-coded node (8.0, -0.3),
    so the cell is independent of the declared defaults.
    """
    knobs = {"logU": -1.5, "logn": 3.0, "logZ": -1.0}
    _, prefix, _ = _BLOCKS[region]
    got = _BLOCKS[region][0](
        _WAVE,
        _LOG_LBOL,
        0.0,
        agn_log_mbh=_BASE[0],
        agn_log_ledd=_BASE[1],
        **{f"{prefix}_cf": 0.1, **{f"{prefix}_{k}": v for k, v in knobs.items()}},
    )[1]
    want = _backend(region, *_BASE, knobs["logU"], knobs["logn"], knobs["logZ"])
    np.testing.assert_allclose(np.asarray(got), want, rtol=1e-12, atol=0)
