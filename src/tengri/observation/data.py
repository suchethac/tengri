# SPDX-License-Identifier: BSD-3-Clause
"""The measurement record: what came back from the telescope.

``Observation`` is the schema (instrument: filters, wave grid, noise
character, which lines); ``Data`` is one record conforming to it.
Validation happens in exactly one place — ``validate_against`` — so
shape errors, boolean-censor traps, NaNs, and unknown line names all
fail loudly with the offending channel named. See the API spec
(2026-07-23) sections 3.2-3.3 and issue #1321.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


class ValidatedData(NamedTuple):
    flux: jnp.ndarray | None  # (n_filters,) [erg/s/cm^2/Hz]
    noise: jnp.ndarray | None  # (n_filters,)
    spec_flux: jnp.ndarray | None  # (n_pix,)
    spec_noise: jnp.ndarray | None  # (n_pix,)
    censor: jnp.ndarray | None  # (n_filters,) in {0, 1, -1}
    line_values: dict | None  # name -> (value, err)


def _check_uncertainties(sigma, channel: str, names) -> None:
    """Reject uncertainty arrays that cannot produce a meaningful likelihood.

    One rule for both channels, rather than two copies that drift: the
    photometry branch used to check its flux and the spectroscopy branch used to
    check neither, so a NaN or non-positive sigma reached the fit unremarked
    (#1365).

    Parameters
    ----------
    sigma : ndarray, shape (n,)
        1-sigma uncertainties [same units as the channel's flux].
    channel : str
        ``"photometry"`` or ``"spectrum"``, used in the message.
    names : sequence of str or None
        Per-element labels (filter names) when the channel has them, so the
        message can point at the offending band rather than an index.

    Raises
    ------
    ValueError
        If any entry is non-finite, or is <= 0 — a zero sigma divides by zero
        and a negative one gives a negative variance. Both yield a plausible-
        looking fit rather than an error, which is the failure mode this guards.
    """

    def _label(idx):
        if names is not None:
            return [names[i] for i in idx]
        return [int(i) for i in idx]

    bad = np.flatnonzero(~np.isfinite(sigma))
    if bad.size:
        raise ValueError(
            f"NaN/inf uncertainty in {channel} {_label(bad)}: a non-finite sigma "
            "poisons the whole likelihood, not just its own term."
        )
    nonpos = np.flatnonzero(sigma <= 0.0)
    if nonpos.size:
        raise ValueError(
            f"Non-positive uncertainty in {channel} {_label(nonpos)}: sigma must "
            "be > 0 (zero divides by zero, negative gives a negative variance). "
            "To ignore a datum, drop it from the Observation rather than "
            "zeroing its error."
        )


@dataclasses.dataclass(frozen=True)
class Data:
    """One galaxy's measurements, validated against an ``Observation``.

    Parameters
    ----------
    photometry : tuple of (flux, err) or None
        Each ``array_like, shape (n_filters,)`` [erg/s/cm^2/Hz].
    spectrum : tuple of (flux, err) or None
        Each ``array_like, shape (n_pix,)`` [erg/s/cm^2/Hz].
    lines : dict or None
        ``{line_name: (value, err)}`` [erg/s/cm^2]; names must be a
        subset of the observation's ``LineList``.
    censor : array_like or None
        Per-band censoring flags, shape ``(n_filters,)``: ``0`` =
        detected, ``1`` = upper limit, ``-1`` = lower limit. Boolean
        arrays are rejected (they silently invert the semantics).
    """

    photometry: tuple | None = None
    spectrum: tuple | None = None
    lines: dict | None = None
    censor: object | None = None

    def validate_against(self, observation) -> ValidatedData:
        if self.photometry is None and self.spectrum is None and not self.lines:
            raise ValueError("Data is empty: provide photometry=, spectrum=, or lines=.")
        flux = noise = spec_flux = spec_noise = censor = None
        if self.photometry is not None:
            phot_schema = getattr(observation, "photometry", None)
            if phot_schema is None:
                raise ValueError(
                    "Data has photometry but the Observation declares no photometric filters."
                )
            flux, noise = (jnp.asarray(a) for a in self.photometry)
            n = phot_schema.n_filters
            if flux.shape != (n,) or noise.shape != (n,):
                raise ValueError(
                    f"photometry shape {flux.shape} does not match the "
                    f"observation's {n} filters "
                    f"({', '.join(phot_schema.names)})."
                )
            bad = np.flatnonzero(~np.isfinite(np.asarray(flux)))
            if bad.size:
                names = [phot_schema.names[i] for i in bad]
                raise ValueError(
                    f"NaN/inf flux in bands {names}: a single-galaxy Data "
                    "must be complete — drop the filter from the "
                    "Observation instead (spec 3.3)."
                )
            # The NOISE array was checked for shape but never for value (#1365).
            # A NaN sigma poisons the whole likelihood, and a non-positive sigma
            # is a division by zero or a negative variance — both produce a
            # meaningless fit rather than an error.
            _check_uncertainties(np.asarray(noise), "photometry", phot_schema.names)
        if self.censor is not None:
            c = np.asarray(self.censor)
            if c.dtype == bool:
                raise ValueError(
                    "censor must use flags 0/1/-1 (0=detected, 1=upper, "
                    "-1=lower); boolean arrays are rejected because True "
                    "would silently mean 'upper limit'."
                )
            if flux is None or c.shape != flux.shape:
                raise ValueError(f"censor must align with photometry, got shape {c.shape}.")
            censor = jnp.asarray(c)
        if self.spectrum is not None:
            spec_schema = getattr(observation, "spectroscopy", None)
            if spec_schema is None:
                raise ValueError(
                    "Data has a spectrum but the Observation declares no spectroscopy."
                )
            spec_flux, spec_noise = (jnp.asarray(a) for a in self.spectrum)
            npix = spec_schema.wave_obs.shape[0]
            if spec_flux.shape != (npix,):
                raise ValueError(
                    f"spectrum shape {spec_flux.shape} does not match the wave grid ({npix} pix)."
                )
            # The spectroscopy channel was validated on flux SHAPE alone (#1365):
            # a mis-shaped noise array, a NaN flux, or a non-positive sigma all
            # passed. Photometry has checked its flux since spec 3.3; this brings
            # the second channel to the same contract instead of leaving it to
            # fail somewhere downstream, or not at all.
            if spec_noise.shape != (npix,):
                raise ValueError(
                    f"spectrum noise shape {spec_noise.shape} does not match the "
                    f"wave grid ({npix} pix); the flux array is {spec_flux.shape}."
                )
            n_bad = int(np.count_nonzero(~np.isfinite(np.asarray(spec_flux))))
            if n_bad:
                raise ValueError(
                    f"NaN/inf in {n_bad} of {npix} spectrum flux pixels: a "
                    "single-galaxy Data must be complete — mask the pixels out of "
                    "the Spectroscopy wave grid instead (spec 3.3)."
                )
            _check_uncertainties(np.asarray(spec_noise), "spectrum", None)
        if self.lines:
            declared = getattr(observation, "lines", None)
            declared_names = set(getattr(declared, "names", []) or [])
            unknown = set(self.lines) - declared_names
            if unknown:
                raise ValueError(
                    f"lines {sorted(unknown)} are not declared in the "
                    "Observation's LineList — declare WHICH lines on the "
                    "schema; supply their VALUES here (spec 3.2)."
                )
        return ValidatedData(flux, noise, spec_flux, spec_noise, censor, self.lines)
