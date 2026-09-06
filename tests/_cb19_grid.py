# SPDX-License-Identifier: BSD-3-Clause
"""Synthetic CB_19 line-ratio grids for tests.

The real grid is a 20-60 minute download from 3MdB_17
(``scripts/download_cb19_templates.py``) and is not tracked, so the suite has
always run CB_19 against a stand-in. Two stand-ins are needed and they are not
interchangeable:

* :func:`write_synthetic_cb19_grid` writes a grid that varies along **every**
  interpolation axis. This is what the backend needs to be exercised at all:
  interpolating a constant slab returns the same value at every query point,
  so a parameter indexing a constant axis is bit-exactly inert and any test
  that sweeps it passes for every implementation.
* :func:`write_flat_cb19_grid` writes the degenerate placeholder (every ratio
  identical), which is what :class:`CB19DegenerateGridError` exists to refuse.

Until #2181 the suite had only a third shape: per-line Case B ratios
*broadcast* across all seven axes. That is non-degenerate by the whole-slab
test and constant along every physical axis, which is why nothing in the suite
could observe five inert nebular parameters.
"""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np

__all__ = [
    "CB19_LINE_WAVES_AA",
    "write_flat_cb19_grid",
    "write_synthetic_cb19_grid",
]

#: Rest-frame vacuum wavelengths [Angstrom] of the ten lines the stand-in
#: carries, matching the real grid's stored catalog. [NII] 6548/6584 are
#: tabulated in air upstream and repeated here so the fixture stays faithful to
#: the file the backend actually reads.
CB19_LINE_WAVES_AA: tuple[float, ...] = (
    1215.67,  # Ly-alpha
    1549.0,  # C IV (blend)
    3727.0,  # [OII] (blend)
    4340.47,  # H-gamma
    4862.68,  # H-beta  (reference line; ratio must be exactly 1.0)
    5008.24,  # [OIII] 5007
    6300.30,  # [OI]
    6548.05,  # [NII] 6548
    6564.61,  # H-alpha
    6583.45,  # [NII] 6584
)

#: Index of H-beta in :data:`CB19_LINE_WAVES_AA`; its ratio is 1.0 by
#: definition and several tests assert exactly that.
_HBETA_INDEX = 4

#: Rough Case B ratios to H-beta (Osterbrock & Ferland 2006, Tables 4.4/4.10).
#: Not production-grade; enough that the ten lines are distinguishable.
_BASE_RATIOS = np.array(
    [10.0, 0.50, 2.00, 0.47, 1.00, 4.00, 0.10, 0.10, 2.87, 0.30],
    dtype=np.float32,
)

#: Grid shape ``(N_OH, N_age, N_U, N_nH, N_CO, N_dNO, N_HbFrac, N_lines)``.
#: ``n_age >= 11`` and ``n_oh, n_u, n_nh = 7, 6, 4`` are asserted by the CB_19
#: contract tests; ``HbFrac = [0.0, 1.0]`` makes ``hbfrac=1.0`` snap exactly
#: and ``hbfrac=0.42`` snap with a gap wide enough to warn.
_SHAPE = (7, 11, 6, 4, 3, 3, 2, 10)


def _axes() -> dict[str, np.ndarray]:
    """The six interpolation axes plus HbFrac, on the real grid's node values."""
    n_oh, n_age, n_u, n_nh, n_co, n_dno, _, _ = _SHAPE
    return {
        "log_OH_total": np.linspace(-5.06, -2.58, n_oh).astype(np.float32),
        "log_age_yr_ssp": np.linspace(6.0, 10.0, n_age).astype(np.float32),
        "log_U": np.linspace(-4.0, -1.5, n_u).astype(np.float32),
        "log_nH": np.linspace(1.0, 4.0, n_nh).astype(np.float32),
        "log_CO": np.linspace(-1.0, 0.15, n_co).astype(np.float32),
        "dNO": np.linspace(-0.25, 0.25, n_dno).astype(np.float32),
        "HbFrac": np.array([0.0, 1.0], dtype=np.float32),
    }


def _write(path: Path, ratios: np.ndarray) -> None:
    """Write one line-ratio slab into the CB_19 HDF5 layout."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    axes = _axes()
    with h5py.File(path, "w") as f:
        ax = f.create_group("axes")
        for name, values in axes.items():
            ax.create_dataset(name, data=values)
        f.create_dataset(
            "line_wavelengths_aa", data=np.array(CB19_LINE_WAVES_AA, dtype=np.float32)
        )
        grp = f.create_group("grids/SSP/Kroupa01/mu100")
        grp.create_dataset("line_ratios", data=ratios.astype(np.float32))


def write_synthetic_cb19_grid(path: str | Path) -> Path:
    """Write a CB_19 grid that varies along every interpolation axis.

    Parameters
    ----------
    path : str or Path
        Destination HDF5 file; parent directories are created.

    Returns
    -------
    Path
        The path written.

    Notes
    -----
    Each of the five parameter-indexed axes (``log_OH``, ``log_U``,
    ``log_nH``, ``log_CO``, ``dNO``) carries a smooth monotone factor, so
    sweeping the parameter that indexes it moves the prediction and a
    finite-difference gradient is comparable to the analytic one. The age axis
    is left flat: the SSP grid indexes it, not a fitted parameter.

    H-beta stays exactly 1.0 everywhere, since it is the reference line the
    ratios are defined against and the contract tests assert that value.

    The factors are illustrative, not physical: this is a plumbing fixture,
    and the real grid is a download away (``scripts/download_cb19_templates.py``).
    """
    axes = _axes()
    ratios = np.ones(_SHAPE, dtype=np.float32)
    ratios = ratios * _BASE_RATIOS[None, None, None, None, None, None, None, :]

    # (axis position in the slab, node values, dex per unit of that axis).
    for axis, values, slope in (
        (0, axes["log_OH_total"], 0.30),
        (2, axes["log_U"], 0.50),
        (3, axes["log_nH"], 0.20),
        (4, axes["log_CO"], 0.40),
        (5, axes["dNO"], 0.60),
    ):
        factor = 10.0 ** (slope * (values - values.mean()))
        shape = [1] * ratios.ndim
        shape[axis] = values.size
        ratios = ratios * factor.reshape(shape).astype(np.float32)

    ratios[..., _HBETA_INDEX] = 1.0
    _write(Path(path), ratios)
    return Path(path)


def write_flat_cb19_grid(path: str | Path, value: float = 1.0) -> Path:
    """Write the degenerate CB_19 placeholder: every line ratio identical.

    Parameters
    ----------
    path : str or Path
        Destination HDF5 file; parent directories are created.
    value : float, optional
        The single ratio every entry carries. Default 1.0, the value the
        shipped placeholder holds.

    Returns
    -------
    Path
        The path written.
    """
    _write(Path(path), np.full(_SHAPE, value, dtype=np.float32))
    return Path(path)
