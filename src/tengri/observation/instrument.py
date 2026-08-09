# SPDX-License-Identifier: BSD-3-Clause
"""Lightweight Instrument bundles + a registry of common premade instruments.

An :class:`Instrument` packages a filter set (or spectroscopic config) with
the associated noise-floor and calibration defaults, so a notebook can say
``Instrument.JWST_NIRCam()`` instead of hand-rolling those pieces every time.

This is a *thin convenience layer* over the existing :class:`Photometry`,
:class:`Spectroscopy`, and :class:`NoiseModel` classes — it does not introduce
new physics, and the underlying objects remain the source of truth.

Examples
--------
>>> from tengri import Instrument
>>> inst = Instrument.JWST_NIRCam()  # premade
>>> obs = inst.observation()  # ready-to-fit Observation
>>> Instrument.list()  # registry as a table

Custom instruments::

>>> from tengri.observation import Photometry
>>> custom = Instrument(
...     name="my_jwst_subset",
...     photometry=Photometry.from_names(["jwst_f200w", "jwst_f444w"]),
...     description="Two-band JWST NIRCam subset for a quick fit.",
... )
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable

from tengri.observation.noise_model import NoiseModel
from tengri.observation.observation import Observation
from tengri.observation.photometry_config import Photometry
from tengri.observation.spectroscopy import Spectroscopy
from tengri.registry import _RegistryTable

__all__ = ["Instrument", "list_instruments"]


@dataclasses.dataclass(frozen=True)
class Instrument:
    """A named bundle of (photometry, spectroscopy, noise) defaults.

    Attributes
    ----------
    name : str
        Short identifier (e.g. ``"JWST_NIRCam"``).
    photometry : Photometry, optional
        Photometric filter set. ``None`` for spectroscopy-only instruments.
    spectroscopy : Spectroscopy, optional
        Spectroscopic configuration. ``None`` for photometry-only instruments.
    noise : NoiseModel, optional
        Default noise model for this instrument. Users can override at fit time.
    description : str
        One-line human description.

    Notes
    -----
    Frozen dataclass — safe to share across processes / treat as immutable.

    Premade instruments live on the class itself as ``@classmethod`` factories
    (``Instrument.JWST_NIRCam()``, ``Instrument.SDSS()``, ...). They lazily
    construct their underlying :class:`Photometry` so import is cheap; the
    actual filter curves only load on first call.
    """

    name: str
    photometry: Photometry | None = None
    spectroscopy: Spectroscopy | None = None
    noise: NoiseModel | None = None
    description: str = ""

    def observation(self) -> Observation:
        """Return an :class:`Observation` populated with this instrument's pieces."""
        return Observation(
            photometry=self.photometry,
            spectroscopy=self.spectroscopy,
            noise=self.noise,
        )

    @property
    def filter_names(self) -> tuple[str, ...]:
        """Filter short-names for the photometric portion (empty if spectro-only)."""
        return tuple(self.photometry.names) if self.photometry is not None else ()

    # ── Premade factories ─────────────────────────────────────────────

    @classmethod
    def SDSS(cls) -> Instrument:
        """SDSS five-band optical photometry (u, g, r, i, z)."""
        return cls(
            name="SDSS",
            photometry=Photometry.from_names(["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"]),
            description="SDSS ugriz; the canonical low-z optical photometric set.",
        )

    @classmethod
    def TWOMASS(cls) -> Instrument:
        """2MASS near-infrared photometry (J, H, Ks)."""
        return cls(
            name="2MASS",
            photometry=Photometry.from_names(["2mass_j", "2mass_h", "2mass_ks"]),
            description="2MASS JHKs near-infrared.",
        )

    @classmethod
    def GALEX(cls) -> Instrument:
        """GALEX FUV + NUV ultraviolet photometry."""
        return cls(
            name="GALEX",
            photometry=Photometry.from_names(["galex_fuv", "galex_nuv"]),
            description="GALEX FUV + NUV; the canonical low-z UV set.",
        )

    @classmethod
    def WISE(cls) -> Instrument:
        """WISE four-band mid-infrared photometry (W1–W4)."""
        return cls(
            name="WISE",
            photometry=Photometry.from_names(["wise_w1", "wise_w2", "wise_w3", "wise_w4"]),
            description="WISE W1–W4 mid-IR.",
        )

    @classmethod
    def SPITZER_IRAC(cls) -> Instrument:
        """Spitzer IRAC four-channel photometry (3.6, 4.5, 5.8, 8.0 µm)."""
        return cls(
            name="Spitzer_IRAC",
            photometry=Photometry.from_names(["irac_36", "irac_45", "irac_58", "irac_80"]),
            description="Spitzer IRAC 3.6/4.5/5.8/8.0 µm.",
        )

    @classmethod
    def HERSCHEL(cls) -> Instrument:
        """Herschel PACS+SPIRE far-infrared photometry (100–500 µm)."""
        return cls(
            name="Herschel",
            photometry=Photometry.from_names(
                ["herschel_100", "herschel_160", "herschel_250", "herschel_350", "herschel_500"]
            ),
            description="Herschel PACS 100/160 + SPIRE 250/350/500 µm.",
        )

    @classmethod
    def UKIDSS(cls) -> Instrument:
        """UKIDSS five-band near-infrared photometry (Y, Z, J, H, K)."""
        return cls(
            name="UKIDSS",
            photometry=Photometry.from_names(
                ["ukidss_z", "ukidss_y", "ukidss_j", "ukidss_h", "ukidss_k"]
            ),
            description="UKIDSS ZYJHK near-infrared.",
        )

    @classmethod
    def HST_ACS_WFC3(cls) -> Instrument:
        """HST ACS+WFC3 broad-band optical/NIR photometry."""
        return cls(
            name="HST_ACS_WFC3",
            photometry=Photometry.from_names(
                [
                    "hst_f435w",
                    "hst_f606w",
                    "hst_f775w",
                    "hst_f814w",
                    "hst_f850lp",
                    "hst_f105w",
                    "hst_f125w",
                    "hst_f140w",
                    "hst_f160w",
                ]
            ),
            description="HST ACS optical + WFC3 NIR; the workhorse high-z imaging set.",
        )

    @classmethod
    def JWST_NIRCam(cls) -> Instrument:
        """JWST NIRCam wide-band photometry (8 broad bands, 0.7–4.4 µm)."""
        return cls(
            name="JWST_NIRCam",
            photometry=Photometry.from_names(
                [
                    "jwst_f070w",
                    "jwst_f090w",
                    "jwst_f115w",
                    "jwst_f150w",
                    "jwst_f200w",
                    "jwst_f277w",
                    "jwst_f356w",
                    "jwst_f444w",
                ]
            ),
            description="JWST NIRCam wide bands (F070W–F444W); the canonical JADES/CEERS set.",
        )

    @classmethod
    def list(cls) -> _RegistryTable:
        """Return every premade instrument as a table.

        Returns
        -------
        _RegistryTable
            One row per premade instrument, with columns ``name``,
            ``n_bands`` and ``description``. Returned a plain
            ``list[dict]`` before #1574; every discovery verb returns a
            table (#1285).
        """
        return _RegistryTable(
            [
                {
                    "name": fac().name,
                    "kind": "instrument",
                    "n_bands": len(fac().filter_names),
                    "description": fac().description,
                    "use": f"tengri.Instrument.{fac.__name__}()",
                }
                for fac in _PREMADE_FACTORIES
            ]
        )


_PREMADE_FACTORIES: tuple[Callable[[], Instrument], ...] = (
    Instrument.GALEX,
    Instrument.SDSS,
    Instrument.TWOMASS,
    Instrument.UKIDSS,
    Instrument.WISE,
    Instrument.SPITZER_IRAC,
    Instrument.HERSCHEL,
    Instrument.HST_ACS_WFC3,
    Instrument.JWST_NIRCam,
)


def list_instruments() -> _RegistryTable:
    """Module-level alias for :meth:`Instrument.list`."""
    return Instrument.list()
