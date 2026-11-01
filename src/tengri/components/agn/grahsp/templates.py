# SPDX-License-Identifier: BSD-3-Clause
"""HDF5 loader for the GRAHSP template bundle.

The bundle is built once with ``tools/build_grahsp_hdf5.py`` from the upstream
``JohannesBuchner/GRAHSP`` template files (FeII forest, Mor & Netzer line
table, torus wave grid).
"""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from functools import lru_cache
from pathlib import Path

import h5py
import jax
import numpy as np
from jax import Array

from tengri._data_setup import package_or_env_data_path

__all__ = ["DEFAULT_TEMPLATE_PATH", "GRAHSPTemplates", "load_grahsp_templates"]

#: Honors $TENGRI_DATA_DIR before the package's own ``data/`` (#1431).
DEFAULT_TEMPLATE_PATH: Path = package_or_env_data_path("grahsp/grahsp_templates.h5")


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
    torus_mn12_wave_nm : ndarray, shape (n_mn12,)
        Mor & Netzer 2012 template-torus continuum grid [nm].
    torus_mn12_avg, torus_mn12_lo, torus_mn12_hi : ndarray, shape (n_mn12,)
        Mean / 25th-percentile / 75th-percentile :math:`L_\\lambda` torus
        templates, normalized to 1 at 12 µm (from upstream ``activatetorus``).
    torus_mn12_si_wave_nm : ndarray, shape (n_si,)
        Mullaney+2011 silicate-feature grid [nm].
    torus_mn12_si_lumin : ndarray, shape (n_si,)
        Silicate difference spectrum, normalized by the 12 µm continuum.
    feii_vc04_wave_nm : ndarray, shape (n_vc04,)
        Veron-Cetty+2004 FeII template wavelengths [nm] (rest frame).
    feii_vc04_lumin : ndarray, shape (n_vc04,)
        Veron-Cetty+2004 FeII :math:`L_\\lambda`, normalized at rest 4575 Å.
    disc_wave_nm : ndarray, shape (n_disc_wave,)
        Netzer accretion-disc grid [nm].
    disc_lumin : ndarray, shape (16, n_disc_wave)
        Netzer disc :math:`L_\\lambda` per (M, a, Mdot) model, normalized to 1
        at 510 nm (inc=0).
    disc_m, disc_a, disc_mdot : tuple[str, ...], shape (16,)
        Disc grid labels: log10 M_BH/Msun, spin a, Eddington ratio Mdot.

    Notes
    -----
    The template-path fields (``torus_mn12_*``, ``feii_vc04_*``, ``disc_*``)
    default to ``None`` for backward compatibility with bundles built before
    the GRAHSP parity implementation; regenerate via ``tools/build_grahsp_hdf5.py``.

    The bundle is immutable and a registered pytree. ``disc_wave_nm``, ``feii_wave_nm`` and
    ``feii_vc04_wave_nm`` are static pytree data, so they must be concrete host arrays: the
    normalization integrals select their nodes on the host, also inside ``jit``.
    """

    feii_wave_nm: Array
    feii_lumin: Array
    line_wave_nm: Array
    line_broad: Array
    line_narrow_sy2: Array
    line_narrow_liner: Array
    line_names: tuple[str, ...] = field(repr=False)
    torus_wave_nm: Array = None
    torus_mn12_wave_nm: Array | None = None
    torus_mn12_avg: Array | None = None
    torus_mn12_lo: Array | None = None
    torus_mn12_hi: Array | None = None
    torus_mn12_si_wave_nm: Array | None = None
    torus_mn12_si_lumin: Array | None = None
    feii_vc04_wave_nm: Array | None = None
    feii_vc04_lumin: Array | None = None
    disc_wave_nm: Array | None = None
    disc_lumin: Array | None = None
    disc_m: tuple[str, ...] | None = field(default=None, repr=False)
    disc_a: tuple[str, ...] | None = field(default=None, repr=False)
    disc_mdot: tuple[str, ...] | None = field(default=None, repr=False)


class _StaticAxis:
    """A host wavelength axis held as pytree aux data: hashable and comparable by value.

    The normalization integrals choose their nodes on the host, so these axes must be
    concrete inside ``jit``. Aux data has to hash and compare equal for a jit cache hit; the
    wrapper does both on the axis bytes.
    """

    __slots__ = ("_key", "array")

    def __init__(self, array: np.ndarray):
        self.array = array
        self._key = (array.shape, array.dtype.str, array.tobytes())

    def __hash__(self) -> int:
        return hash(self._key)

    def __eq__(self, other) -> bool:
        return isinstance(other, _StaticAxis) and self._key == other._key


#: Fields read on the host by the normalization (node selection); static in the pytree.
_STATIC_AXES = ("disc_wave_nm", "feii_wave_nm", "feii_vc04_wave_nm")
_LABEL_FIELDS = ("line_names", "disc_m", "disc_a", "disc_mdot")
_CHILD_FIELDS = tuple(
    f.name for f in fields(GRAHSPTemplates) if f.name not in _STATIC_AXES + _LABEL_FIELDS
)


def _flatten_templates(bundle: GRAHSPTemplates):
    children = tuple(getattr(bundle, name) for name in _CHILD_FIELDS)
    axes = tuple(
        None if getattr(bundle, name) is None else _StaticAxis(np.asarray(getattr(bundle, name)))
        for name in _STATIC_AXES
    )
    labels = tuple(getattr(bundle, name) for name in _LABEL_FIELDS)
    return children, (axes, labels)


def _unflatten_templates(aux, children) -> GRAHSPTemplates:
    axes, labels = aux
    values = dict(zip(_CHILD_FIELDS, children, strict=True))
    values.update(
        {
            name: None if ax is None else ax.array
            for name, ax in zip(_STATIC_AXES, axes, strict=True)
        }
    )
    values.update(dict(zip(_LABEL_FIELDS, labels, strict=True)))
    return GRAHSPTemplates(**values)


# The bundle is a pytree so the AGN runner can pass it to the jitted forward model as an
# argument. The tables are children; the label tuples and the wavelength axes the
# normalization reads on the host are static aux data.
jax.tree_util.register_pytree_node(GRAHSPTemplates, _flatten_templates, _unflatten_templates)


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
        feii_wave = np.asarray(f["feii_bruhweiler2008/wave_nm"][:])
        feii_lumin = np.asarray(f["feii_bruhweiler2008/lumin"][:])
        line_wave = np.asarray(f["netzer1990_lines/wave_nm"][:])
        line_broad = np.asarray(f["netzer1990_lines/broad"][:])
        line_narrow_sy2 = np.asarray(f["netzer1990_lines/narrow_sy2"][:])
        line_narrow_liner = np.asarray(f["netzer1990_lines/narrow_liner"][:])
        line_names = tuple(
            n.decode("utf-8") if isinstance(n, bytes) else n for n in f["netzer1990_lines/name"][:]
        )
        torus_wave = np.asarray(f["torus/wave_nm"][:])

        def _arr(key: str):
            return np.asarray(f[key][:]) if key in f else None

        def _strs(key: str):
            if key not in f:
                return None
            return tuple(n.decode("utf-8") if isinstance(n, bytes) else n for n in f[key][:])

        mn12_wave = _arr("torus_mn12/wave_nm")
        mn12_avg = _arr("torus_mn12/avg")
        mn12_lo = _arr("torus_mn12/lo")
        mn12_hi = _arr("torus_mn12/hi")
        mn12_si_wave = _arr("torus_mn12/si_wave_nm")
        mn12_si_lumin = _arr("torus_mn12/si_lumin")
        vc04_wave = _arr("feii_veroncetty2004/wave_nm")
        vc04_lumin = _arr("feii_veroncetty2004/lumin")
        disc_wave = _arr("netzer_disc/wave_nm")
        disc_lumin = _arr("netzer_disc/lumin")
        disc_m = _strs("netzer_disc/m")
        disc_a = _strs("netzer_disc/a")
        disc_mdot = _strs("netzer_disc/mdot")
    return GRAHSPTemplates(
        feii_wave_nm=feii_wave,
        feii_lumin=feii_lumin,
        line_wave_nm=line_wave,
        line_broad=line_broad,
        line_narrow_sy2=line_narrow_sy2,
        line_narrow_liner=line_narrow_liner,
        line_names=line_names,
        torus_wave_nm=torus_wave,
        torus_mn12_wave_nm=mn12_wave,
        torus_mn12_avg=mn12_avg,
        torus_mn12_lo=mn12_lo,
        torus_mn12_hi=mn12_hi,
        torus_mn12_si_wave_nm=mn12_si_wave,
        torus_mn12_si_lumin=mn12_si_lumin,
        feii_vc04_wave_nm=vc04_wave,
        feii_vc04_lumin=vc04_lumin,
        disc_wave_nm=disc_wave,
        disc_lumin=disc_lumin,
        disc_m=disc_m,
        disc_a=disc_a,
        disc_mdot=disc_mdot,
    )
