# SPDX-License-Identifier: BSD-3-Clause
"""HDF5 loader for the GRAHSP template bundle.

The bundle is built once with ``tools/build_grahsp_hdf5.py`` from the upstream
``JohannesBuchner/GRAHSP`` template files (FeII forest, Mor & Netzer line
table, torus wave grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path

import h5py
import jax.numpy as jnp
from jax import Array

__all__ = ["DEFAULT_TEMPLATE_PATH", "GRAHSPTemplates", "load_grahsp_templates"]

DEFAULT_TEMPLATE_PATH: Path = (
    Path(__file__).resolve().parents[5]
    / "data"
    / "grahsp"
    / "grahsp_templates.h5"
)


@dataclass(frozen=True)
class GRAHSPTemplates:
    """Immutable bundle of GRAHSP-specific templates.

    Attributes
    ----------
    feii_wave_nm : ndarray, shape (n_feii,)
        Bruhweiler+Verner 2008 FeII template wavelengths [nm], de-redshifted.
    feii_lumin : ndarray, shape (n_feii,)
        FeII relative intensity (arbitrary units; multiply by ``A_FeII *
        L_Hb_broad`` per upstream).
    line_wave_nm : ndarray, shape (36,)
        Netzer 1990 / Mor & Netzer 2012 line central wavelengths [nm].
    line_broad, line_narrow_sy2, line_narrow_liner : ndarray, shape (36,)
        Line strengths relative to H-beta.
    line_names : tuple[str, ...]
        Line names.
    torus_wave_nm : ndarray, shape (n_torus,)
        Fixed torus wave grid [nm] (from upstream ``activategtorus``).
    """

    feii_wave_nm: Array
    feii_lumin: Array
    line_wave_nm: Array
    line_broad: Array
    line_narrow_sy2: Array
    line_narrow_liner: Array
    line_names: tuple[str, ...] = field(repr=False)
    torus_wave_nm: Array


@lru_cache(maxsize=4)
def load_grahsp_templates(
    path: Path | str = DEFAULT_TEMPLATE_PATH,
) -> GRAHSPTemplates:
    """Load and cache the GRAHSP template HDF5 bundle.

    Parameters
    ----------
    path : path-like, optional
        HDF5 file path. Default: ``data/grahsp/grahsp_templates.h5`` next
        to the installed package.

    Returns
    -------
    templates : GRAHSPTemplates
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"GRAHSP template bundle not found at {path}. Run "
            "`python tools/build_grahsp_hdf5.py` to regenerate."
        )
    with h5py.File(path, "r") as f:
        feii_wave = jnp.asarray(f["feii_bruhweiler2008/wave_nm"][:])
        feii_lumin = jnp.asarray(f["feii_bruhweiler2008/lumin"][:])
        line_wave = jnp.asarray(f["netzer1990_lines/wave_nm"][:])
        line_broad = jnp.asarray(f["netzer1990_lines/broad"][:])
        line_narrow_sy2 = jnp.asarray(f["netzer1990_lines/narrow_sy2"][:])
        line_narrow_liner = jnp.asarray(f["netzer1990_lines/narrow_liner"][:])
        line_names = tuple(
            n.decode("utf-8") if isinstance(n, bytes) else n
            for n in f["netzer1990_lines/name"][:]
        )
        torus_wave = jnp.asarray(f["torus/wave_nm"][:])
    return GRAHSPTemplates(
        feii_wave_nm=feii_wave,
        feii_lumin=feii_lumin,
        line_wave_nm=line_wave,
        line_broad=line_broad,
        line_narrow_sy2=line_narrow_sy2,
        line_narrow_liner=line_narrow_liner,
        line_names=line_names,
        torus_wave_nm=torus_wave,
    )
