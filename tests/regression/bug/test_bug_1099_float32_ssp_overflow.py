# SPDX-License-Identifier: BSD-3-Clause
"""Issue #1099: a float32 SSP grid silently overflowed the erg/s mass scale.

``stellar_mass_scale = total_mass x L_sun`` is ~1e42 erg/s for a 1e9 Msun
galaxy. float32 tops out at 3.4e38, so on any grid stored as float32 on disk
the product overflowed to ``inf`` — with no exception, no NaN, and no warning.
The same product scales the ionizing SED handed to the nebular backends, so
Cue and CloudyGrid received a poisoned ionizing continuum and returned
finite-but-meaningless line luminosities (wrong by ~50 orders of magnitude)
while the stellar continuum stayed correct and every continuum-level check
stayed green.

The stellar SED never tripped it because it evaluates
``total_mass * ssp_flux_at_age * LSUN_ERG_PER_S`` left to right — the small
factor lands first, so the huge constant only ever multiplies an already-small
number. The two broken sites materialized ``(total_mass * L_sun)`` as a
standalone scalar. Identical algebra, different association order, and only one
of them overflows.

Three shipped grids stored float32 (``bc03_*``, ``pgny_*``); the ``fsps_*`` and
``ssp_*`` catalog grids are float64 and were never affected — which, together
with CI never rendering the reproduction notebooks, is why nothing tripped.

The guard downcasts a real grid to float32 rather than shipping a fixture, so
the *only* difference from the healthy case is the on-disk dtype.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from tengri import FIXED, Fixed, SEDModel, load_ssp_data

pytestmark = [pytest.mark.regression_bug, pytest.mark.physics]

C_ANGSTROM_PER_S = 2.99792458e18
L_SUN_ERG_PER_S = 3.828e33

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
#: Bare-stellar grid — the one Cue needs, and (not coincidentally) float32 on disk.
_BC03 = _DATA_DIR / "bc03_pdva_stelib_chabrier.h5"
#: Whatever grid the session has. #613 guarantees this one exists even on CI,
#: so the overflow assertions below are not data-gated into invisibility.
_ANY = _DATA_DIR / "ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0.h5"


def _to_float32(src: Path, dest_dir: Path) -> str:
    """Copy of ``src`` with every float array downcast to float32.

    Keeps the original basename: the loader resolves IMF, wNE provenance, and
    the native-L_sun convention from the *filename*, so a renamed copy would
    differ from its source in more than the dtype and stop being a clean A/B.
    """
    dst = dest_dir / src.name
    with h5py.File(src, "r") as fin, h5py.File(dst, "w") as fout:
        for key in fin:
            data = fin[key][:]
            if np.issubdtype(data.dtype, np.floating):
                data = data.astype(np.float32)
            fout.create_dataset(key, data=data)
        for k, v in fin.attrs.items():
            fout.attrs[k] = v
    return str(dst)


@pytest.fixture(scope="module")
def f32_any(tmp_path_factory):
    """A float32 copy of any available grid — runs everywhere, including CI."""
    if not _ANY.is_file():
        pytest.skip(f"no SSP grid at {_ANY}")
    return _to_float32(_ANY, tmp_path_factory.mktemp("f32"))


@pytest.fixture(scope="module")
def f32_bc03(tmp_path_factory):
    """A float32 copy of the bare-stellar BC03 grid — required for Cue."""
    if not _BC03.is_file():
        pytest.skip(f"bare-stellar BC03 grid not on disk: {_BC03}")
    return _to_float32(_BC03, tmp_path_factory.mktemp("f32bc"))


def _build(ssp, **neb):
    return SEDModel.build(
        ssp_data=ssp,
        met={"logzsol": Fixed(0.0), "*": FIXED},
        sfh={
            "type": "const",
            "start_gyr": Fixed(0.01),
            "end_gyr": Fixed(0.0),
            "log_total_mass": Fixed(9.0),
            "*": FIXED,
        },
        dust={"law": "power_law", "type": "two_component", "tau_bc": Fixed(0.0), "tau_diff": Fixed(0.0), "*": FIXED},
        redshift=Fixed(0.0),
        **neb,
    )


def test_float32_grid_loads_at_working_precision(f32_any):
    """The loader upcasts, so no downstream product can overflow."""
    ssp = load_ssp_data(f32_any)
    assert ssp.ssp_flux.dtype == np.float64
    assert ssp.ssp_lg_age_gyr.dtype == np.float64
    assert ssp.ssp_wave.dtype == np.float64


def test_stellar_mass_scale_finite_on_float32_grid(f32_any):
    """The headline symptom: ``inf`` before the fix, 3.828e42 after."""
    state = _build(load_ssp_data(f32_any)).predict_state({})
    scale = float(np.asarray(state.derived["stellar_mass_scale"]))
    assert np.isfinite(scale), "stellar_mass_scale overflowed float32 (#1099)"
    assert scale == pytest.approx(1.0e9 * L_SUN_ERG_PER_S, rel=1e-5)


def test_upcast_preserves_stored_values(f32_any):
    """The upcast changes dtype and nothing else.

    Every float32 is exactly representable as a float64, so promoting on load
    must reproduce the stored numbers bit for bit. This is the invariant that
    makes the fix safe to apply to grids already in the wild — it removes an
    overflow without perturbing anyone's templates.

    (Note what this does *not* claim: float32 **storage** is lossy, and on a
    wNE grid it costs ~0.3% near sharp lines because ``ssp_wave`` itself gets
    rounded. That loss is baked into the file; the loader cannot undo it, and
    is not trying to.)
    """
    ssp = load_ssp_data(f32_any)
    with h5py.File(f32_any, "r") as f:
        for name, loaded in (
            ("ssp_flux", ssp.ssp_flux),
            ("ssp_wave", ssp.ssp_wave),
            ("ssp_lg_age_gyr", ssp.ssp_lg_age_gyr),
            ("ssp_lgmet", ssp.ssp_lgmet),
        ):
            stored = f[name][:]
            assert stored.dtype == np.float32, f"{name} is not float32 on disk"
            np.testing.assert_array_equal(
                np.asarray(loaded), stored.astype(np.float64), err_msg=f"{name} altered on load"
            )


def test_cue_balmer_decrement_on_float32_grid(f32_bc03):
    """Case B fixes Ha/Hb ~ 2.86. Before the fix this came out at 0.41.

    The Balmer decrement is the cleanest single assertion that the ionizing SED
    reaching the nebular backend is sane: recombination physics pins it near
    2.86 regardless of SSP, metallicity, or ionization parameter, so it cannot
    be satisfied by an accidentally-rescaled continuum.
    """
    state = _build(
        load_ssp_data(f32_bc03),
        neb={"type": "cue", "neb_logU": Fixed(-2.0), "neb_logZ_gas": Fixed(0.0), "*": FIXED},
    ).predict_state({})

    wave = np.asarray(state.wave)
    neb = np.asarray(state.derived["sed_nebular"])
    assert np.isfinite(neb).all()

    def line(center: float, half: float = 12.0) -> float:
        m = (wave >= center - half) & (wave <= center + half)
        order = np.argsort(wave[m])
        lam = wave[m][order]
        l_lambda = neb[m][order] * C_ANGSTROM_PER_S / lam**2
        return float(np.trapezoid(np.clip(l_lambda - l_lambda.min(), 0.0, None), lam))

    h_beta = line(4862.68)
    assert h_beta > 0.0, "no Hbeta on the nebular SED"

    decrement = line(6564.61) / h_beta
    assert 2.5 < decrement < 3.3, (
        f"Ha/Hb = {decrement:.2f} violates Case B (~2.86) — the ionizing SED "
        f"handed to Cue is corrupt (#1099)"
    )
