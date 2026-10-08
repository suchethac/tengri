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
(d) the #931 U / n / Z forwarding is unchanged;
(e) grid-free: the ``synthesizer_spectra`` blocks hand the declared ``agn_log_mbh`` /
    ``agn_log_ledd`` to the backend as ``log_bh_mass=`` / ``log_eddington=``, both regions.

Cells (a)-(d) are grid-gated: the Synthesizer AGN test grids are not shipped, so each
of them skips when the specific grid file is missing. Cell (e) replaces the backend
with a recorder and runs without the grids.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tengri import DEFAULT, Fixed, SEDModel
from tengri.components.agn.blocks import blr as blr_module, nlr as nlr_module
from tengri.components.agn.blocks.blr import (
    DEFAULT_F_BOL_5100,
    blr_synthesizer_block,
    blr_synthesizer_spectra_block,
)
from tengri.components.agn.blocks.nlr import (
    _resolve_synthesizer_grid,
    nlr_synthesizer_block,
    nlr_synthesizer_spectra_block,
)
from tengri.components.agn.nlr_cloudy import (
    compute_blr_sed_synthesizer,
    compute_nlr_sed_synthesizer,
    compute_nlr_sed_synthesizer_spectra,
)
from tengri.config.exceptions import TengriIOError
from tengri.parameters.priors import Uniform
from tengri.utils.physics_constants import C_AA, L_SUN

pytestmark = pytest.mark.contract

#: Marks the cells that read the Synthesizer AGN test grids.
needs_grids = pytest.mark.usefixtures("_grids_present")

_LOG_LBOL = 13.0
_WAVE = np.linspace(900.0, 30000.0, 29101)
_O3_WINDOW = (_WAVE >= 4995.0) & (_WAVE <= 5020.0)

#: (variant, region) -> block, its knob prefix, the backend and the backend's own kwargs.
#: ``synthesizer_spectra`` reads the grid's ``/spectra/nebular``; ``synthesizer`` broadens the
#: ``/lines`` table. The BLR ``synthesizer`` block takes its bolometric from ``l5100_disc``.
_CASES = {
    ("synthesizer_spectra", "nlr"): {
        "block": nlr_synthesizer_spectra_block,
        "prefix": "agn_nlr",
        "backend": compute_nlr_sed_synthesizer_spectra,
        "knobs": {"logU": -2.0, "logn": 4.0, "logZ": -2.0},
        "extra": {"region": "nlr"},
    },
    ("synthesizer_spectra", "blr"): {
        "block": blr_synthesizer_spectra_block,
        "prefix": "agn_blr",
        "backend": compute_nlr_sed_synthesizer_spectra,
        "knobs": {"logU": -1.0, "logn": 4.0, "logZ": -2.0},
        "extra": {"region": "blr"},
    },
    ("synthesizer", "nlr"): {
        "block": nlr_synthesizer_block,
        "prefix": "agn_nlr",
        "backend": compute_nlr_sed_synthesizer,
        "knobs": {"logU": -2.0, "logZ": -1.8477},
        "extra": {"fwhm_kms": 500.0},
    },
    ("synthesizer", "blr"): {
        "block": blr_synthesizer_block,
        "prefix": "agn_blr",
        "backend": compute_blr_sed_synthesizer,
        "knobs": {"logU": -1.0, "logZ": -1.8477},
        "extra": {"fwhm_kms": 5000.0},
    },
}
_KEYS = tuple(_CASES)
_REGIONS = ("nlr", "blr")
_L_BOL_ERG = 10.0**_LOG_LBOL * L_SUN

_BASE = (8.0, -0.3)

#: Measured from the BACKEND on the two-node test grids (``_WAVE``, L_bol = 1e13 L_sun,
#: cf = 0.1, region default U / n=4 / Z=-2): ratio of the modified spectrum to the
#: (8.0, -0.3) baseline, bolometric trapezoid and [O III] 5007 pixel sum (4995-5020 A).
#: The ``synthesizer`` (``/lines``) [O III] ratios were re-pinned when the Cloudy air labels of
#: ``lines/wavelength`` began converting to vacuum at load: the lines moved +1.4 A against the
#: fixed pixel window (4995-5020 A, vacuum), and the old code with the window moved by
#: ``vac_to_air`` reproduces the new ratios to 4e-7 on a 0.005 A grid. The bolometric ratios and
#: the ``synthesizer_spectra`` rows (a continuum mesh, not labels) are unchanged.
_RATIOS = {
    ("synthesizer_spectra", "nlr", "ledd"): (0.8765613891643922, 0.821758212949826),
    ("synthesizer_spectra", "nlr", "mbh"): (0.8251230154217267, 0.757977443821904),
    ("synthesizer_spectra", "blr", "ledd"): (0.879708808683233, 0.8244935449320386),
    ("synthesizer_spectra", "blr", "mbh"): (0.8292592457445129, 0.760726440329576),
    ("synthesizer", "nlr", "ledd"): (0.8862299605022597, 0.847658401515718),
    ("synthesizer", "nlr", "mbh"): (0.8276594927031491, 0.7815635352506513),
    ("synthesizer", "blr", "ledd"): (0.8928151729780258, 0.972631807964777),
    ("synthesizer", "blr", "mbh"): (0.8358174204085614, 0.9508063300277503),
}
_AXIS_POINTS = {"ledd": (8.0, -1.0), "mbh": (9.0, -0.3)}


@pytest.fixture(scope="module")
def _grids_present():
    """Skip the requesting cell when a specific Synthesizer AGN grid file is missing."""
    for kind in _REGIONS:
        try:
            _resolve_synthesizer_grid(kind)
        except TengriIOError:
            pytest.skip(f"Synthesizer AGN {kind} test grid not found")


def _block_kwargs(key, knobs=None):
    case = _CASES[key]
    values = {**case["knobs"], **(knobs or {})}
    kwargs = {f"{case['prefix']}_cf": 0.1}
    kwargs.update({f"{case['prefix']}_{k}": v for k, v in values.items()})
    return kwargs


def _l5100(key):
    """``l5100_disc`` for the block: only the BLR ``synthesizer`` block reads it."""
    if key == ("synthesizer", "blr"):
        return _L_BOL_ERG / DEFAULT_F_BOL_5100
    return 0.0


def _block(key, knobs=None, **extra):
    """The block's isotropic-channel L_lambda (the NLR blocks return a pair)."""
    out = _CASES[key]["block"](_WAVE, _LOG_LBOL, _l5100(key), **_block_kwargs(key, knobs), **extra)
    if isinstance(out, tuple):
        maskable, out = out
        assert not np.any(np.asarray(maskable)), "the Synthesizer line regions are isotropic"
    return np.asarray(out)


def _backend(key, mbh, ledd, knobs=None):
    """Reference: the grid backend L_nu converted to L_lambda as the block does."""
    case = _CASES[key]
    values = {**case["knobs"], **(knobs or {})}
    kwargs = {"neb_logU": values["logU"], "neb_logZ_gas": values["logZ"], **case["extra"]}
    if "logn" in values:
        kwargs["neb_logn"] = values["logn"]
    l_nu = case["backend"](
        _WAVE,
        l_disc_bol_erg=_L_BOL_ERG,
        covering_fraction=0.1,
        grid_path=_resolve_synthesizer_grid(key[1]),
        log_bh_mass=mbh,
        log_eddington=ledd,
        **kwargs,
    )
    return np.asarray(l_nu) * C_AA / _WAVE**2


def _bolometric(l_lambda):
    return float(np.trapezoid(l_lambda, _WAVE))


@needs_grids
@pytest.mark.parametrize("key", _KEYS, ids=lambda k: "/".join(k))
@pytest.mark.parametrize("mbh, ledd", [(8.0, -1.0), (9.0, -1.0), (8.0, -0.3)])
def test_block_equals_backend_at_declared_axes(key, mbh, ledd):
    """(a) The block reproduces the backend at the (mass, Eddington) it is given."""
    got = _block(key, agn_log_mbh=mbh, agn_log_ledd=ledd)
    want = _backend(key, mbh, ledd)
    assert np.max(want) > 0.0
    np.testing.assert_allclose(
        got,
        want,
        rtol=1e-12,
        atol=0,
        err_msg=f"{key}: block ignores agn_log_mbh={mbh}, agn_log_ledd={ledd}",
    )


@needs_grids
@pytest.mark.parametrize("key", _KEYS, ids=lambda k: "/".join(k))
@pytest.mark.parametrize("axis", ["ledd", "mbh"])
def test_axes_move_the_spectrum_by_the_backend_amount(key, axis):
    """(b) Bolometric and [O III] 5007 ratios vs the (8.0, -0.3) baseline."""
    mbh, ledd = _AXIS_POINTS[axis]
    bol_want, o3_want = _RATIOS[(*key, axis)]
    base = _block(key, agn_log_mbh=_BASE[0], agn_log_ledd=_BASE[1])
    moved = _block(key, agn_log_mbh=mbh, agn_log_ledd=ledd)
    bol = _bolometric(moved) / _bolometric(base)
    o3 = float(moved[_O3_WINDOW].sum() / base[_O3_WINDOW].sum())
    assert abs(bol - 1.0) > 1e-2, f"{key}/{axis}: bolometric ratio {bol} (axis inert)"
    assert abs(o3 - 1.0) > 1e-2, f"{key}/{axis}: [O III] ratio {o3} (axis inert)"
    np.testing.assert_allclose(bol, bol_want, rtol=1e-6)
    np.testing.assert_allclose(o3, o3_want, rtol=1e-6)


def _build(ssp, ledd, variant):
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
            "nlr": {"type": variant, "all_params": Fixed(DEFAULT)},
        },
        redshift=Fixed(0.0),
    )


_VARIANTS = ("synthesizer_spectra", "synthesizer")
_LINES = "sed_agn_lines"  # nlr + blr + feii light of the composable AGN


@needs_grids
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:agn_log_ledd has no effect:UserWarning")
@pytest.mark.parametrize("variant", _VARIANTS)
def test_agn_log_ledd_moves_the_nlr_light_through_the_grammar(synthetic_ssp_wide, variant):
    """(c) Value: Fixed(-1.0) vs Fixed(-0.3) differ by more than 1e-2 (relative max)."""

    def _lines(ledd):
        model = _build(synthetic_ssp_wide, Fixed(ledd), variant)
        return np.asarray(model.predict_state({}).derived[_LINES])

    a, b = _lines(-1.0), _lines(-0.3)
    assert b.max() > 0.0, "the grid NLR produced no light"
    rel = np.max(np.abs(a - b)) / np.max(np.abs(b))
    assert rel > 1e-2, f"agn_log_ledd is inert through SEDModel.build (max rel diff {rel})"


@needs_grids
@pytest.mark.filterwarnings("ignore::DeprecationWarning")
@pytest.mark.filterwarnings("ignore:agn_log_ledd has no effect:UserWarning")
@pytest.mark.parametrize("variant", _VARIANTS)
def test_agn_log_ledd_gradient_reaches_the_nlr_light(synthetic_ssp_wide, variant):
    """(c) With ``agn_log_ledd`` free, d(sum lines)/d(agn_log_ledd) is finite and non-zero."""
    free = _build(synthetic_ssp_wide, Uniform(-1.5, -0.1), variant)
    assert free.spec.free_params == ["agn_log_ledd"]
    params = free.spec.sample(jax.random.PRNGKey(0))
    grad = jax.grad(lambda p: jnp.sum(free.predict_state(p).derived[_LINES]))(params)
    g = np.asarray(grad["agn_log_ledd"])
    assert np.isfinite(g), f"d(sum {_LINES})/d agn_log_ledd is not finite: {g}"
    assert g != 0.0, "agn_log_ledd has an exactly zero gradient (inert parameter)"


@needs_grids
@pytest.mark.parametrize("key", _KEYS, ids=lambda k: "/".join(k))
def test_photoionization_axes_still_forwarded(key):
    """(d) #931 unchanged (GREEN before and after): U / Z reach the backend.

    Mass and Eddington ratio are passed at the pre-fix hard-coded node (8.0, -0.3),
    so the cell is independent of the declared defaults.
    """
    knobs = {"logU": -1.5, "logZ": -1.0}
    got = _block(key, knobs, agn_log_mbh=_BASE[0], agn_log_ledd=_BASE[1])
    want = _backend(key, *_BASE, knobs)
    np.testing.assert_allclose(got, want, rtol=1e-12, atol=0)


@pytest.mark.parametrize(
    "module, block, region, prefix",
    [
        (nlr_module, nlr_synthesizer_spectra_block, "nlr", "agn_nlr"),
        (blr_module, blr_synthesizer_spectra_block, "blr", "agn_blr"),
    ],
    ids=["nlr", "blr"],
)
def test_spectra_blocks_forward_declared_mbh_and_ledd_to_the_backend(
    monkeypatch, module, block, region, prefix
):
    """(e) Grid-free: the backend receives the declared (log M_BH, log lambda_Edd) verbatim."""
    seen = {}

    def _recorder(wave_aa, **kwargs):
        seen.update(kwargs)
        return jnp.ones_like(wave_aa)

    monkeypatch.setattr(module, "compute_nlr_sed_synthesizer_spectra", _recorder)
    monkeypatch.setattr(module, "_resolve_synthesizer_grid", lambda kind: f"{kind}.hdf5")
    block(
        _WAVE,
        _LOG_LBOL,
        0.0,
        **{f"{prefix}_cf": 0.1},
        agn_log_mbh=9.37,
        agn_log_ledd=-1.23,
    )
    assert seen.get("log_bh_mass") == 9.37, f"{region}: agn_log_mbh not forwarded as log_bh_mass"
    assert seen.get("log_eddington") == -1.23, (
        f"{region}: agn_log_ledd not forwarded as log_eddington"
    )
    assert seen["region"] == region
