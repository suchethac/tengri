# SPDX-License-Identifier: BSD-3-Clause
"""The measurement record: what came back from the telescope.

``Observation`` is the schema (instrument: filters, wave grid, noise
character, which lines); ``Data`` is one record conforming to it.
Validation happens in exactly one place (``validate_against``) so
shape errors, boolean-censor traps, NaNs, and unknown line names all
fail loudly with the offending channel named. See the API spec
(2026-07-23) sections 3.2-3.3 and issue #1321.
"""

from __future__ import annotations

import dataclasses
from typing import NamedTuple

import jax.numpy as jnp
import numpy as np


def _bad_indices(values, predicate) -> np.ndarray:
    """Indices where ``predicate`` flags a value as unusable."""
    return np.flatnonzero(predicate(np.asarray(values)))


def _describe(bad: np.ndarray, names=None, limit: int = 5) -> str:
    """Name the offending channels, band names when known, else indices."""
    if names is not None:
        labels = [str(names[i]) for i in bad[:limit]]
    else:
        labels = [f"index {i}" for i in bad[:limit]]
    more = f" (+{bad.size - limit} more)" if bad.size > limit else ""
    return ", ".join(labels) + more


def _reject_nonfinite(values, channel: str, names=None) -> None:
    """Raise if any value is NaN or inf, naming the offending channel."""
    bad = _bad_indices(values, lambda a: ~np.isfinite(a))
    if bad.size:
        raise ValueError(
            f"NaN/inf {channel} at {_describe(bad, names)}: a single-galaxy "
            "Data must be complete, drop the channel from the Observation "
            "instead of passing a placeholder (spec 3.3)."
        )


def _reject_nonpositive_sigma(values, channel: str, names=None) -> None:
    """Raise if any uncertainty is <= 0.

    A zero sigma divides by zero; a *negative* one is worse, because ``chi^2``
    squares the sign away, the fit then runs to completion and reports a
    confidently wrong answer with no warning anywhere.
    """
    bad = _bad_indices(values, lambda a: np.isfinite(a) & (a <= 0))
    if bad.size:
        raise ValueError(
            f"non-positive {channel} at {_describe(bad, names)}: uncertainties "
            "must be > 0. A negative sigma is squared away by chi^2 and would "
            "silently weight that channel as if the sign were positive."
        )


class ValidatedData(NamedTuple):
    flux: jnp.ndarray | None  # (n_filters,) [erg/s/cm^2/Hz]
    noise: jnp.ndarray | None  # (n_filters,)
    spec_flux: jnp.ndarray | None  # (n_pix,)
    spec_noise: jnp.ndarray | None  # (n_pix,)
    censor: jnp.ndarray | None  # (n_filters,) in {0, 1, -1}
    line_values: dict | None  # name -> (value, err)


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
        ``{line_name: (value, err)}``, or
        ``{line_name: (value, err, 'upper'|'lower')}`` [erg/s/cm^2]; names
        must be a subset of the observation's ``LineList``. The optional
        third element marks the flux as a censored limit rather than a
        detection, using the same vocabulary as
        :meth:`~tengri.observation.line_flux_data.LineFluxData.from_dict`.
        Line limits belong here, not in ``censor``, that field is
        per-photometric-band.
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
                    "must be complete, drop the filter from the "
                    "Observation instead (spec 3.3)."
                )
            # The uncertainties are half the record and were previously unchecked,
            # so a NaN or sign-flipped error bar reached the likelihood untouched.
            _reject_nonfinite(noise, "photometry uncertainty", phot_schema.names)
            _reject_nonpositive_sigma(noise, "photometry uncertainty", phot_schema.names)
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
            # Reject garbage flags (-99, 2, 0.5, NaN, ...) here, at the seam.
            # ``censored_neg_log_likelihood`` dispatches with
            # ``jnp.where(mask == 1, upper, jnp.where(mask == -1, lower, detected))``,
            # so every unrecognized value falls through to the DETECTED branch:
            # a sentinel-coded or mis-scaled column silently turns upper limits
            # into detections and biases the fit. ``ingest_catalog`` has always
            # rejected these on the catalog side; the two seams must agree.
            in_range = np.isin(c, (-1, 0, 1))
            if not in_range.all():
                bad = np.unique(c[~in_range]).tolist()
                raise ValueError(
                    f"censor has invalid flag value(s) {bad}; allowed: 0 (detected), "
                    "1 (upper limit), -1 (lower limit). Unrecognized values would be "
                    "silently treated as detections by the censored likelihood."
                )
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
            # Only the flux length was checked before, so a short or long noise array
            # broadcast silently against the residual instead of failing.
            if spec_noise.shape != (npix,):
                raise ValueError(
                    f"spectrum noise shape {spec_noise.shape} does not match the "
                    f"wave grid ({npix} pix); flux and uncertainty must align."
                )
            _reject_nonfinite(spec_flux, "spectrum flux")
            _reject_nonfinite(spec_noise, "spectrum uncertainty")
            _reject_nonpositive_sigma(spec_noise, "spectrum uncertainty")
        if self.lines:
            declared = getattr(observation, "lines", None)
            declared_names = set(getattr(declared, "names", []) or [])
            unknown = set(self.lines) - declared_names
            if unknown:
                raise ValueError(
                    f"lines {sorted(unknown)} are not declared in the "
                    "Observation's LineList, declare WHICH lines on the "
                    "schema; supply their VALUES here (spec 3.2)."
                )
            # Line fluxes carry the same NaN / sign-flip exposure as the continuum,
            # and each is named individually so the offending line is obvious.
            for line_name, pair in self.lines.items():
                value, err, *limit = pair
                # A line is either a detection (two elements) or a censored
                # limit carrying the same marker vocabulary as
                # ``LineFluxData.from_dict``. Anything else would fall through
                # to the detected branch of the censored likelihood and fit a
                # non-detection as a measurement (#1460, the line-side twin of
                # the #1321 photometry trap).
                if len(limit) > 1:
                    raise ValueError(
                        f"line {line_name!r}: expected (flux, err) or "
                        f"(flux, err, 'upper'|'lower'), got {len(pair)} elements."
                    )
                if limit and limit[0] not in ("upper", "lower"):
                    raise ValueError(
                        f"line {line_name!r}: limit marker must be 'upper' or "
                        f"'lower', got {limit[0]!r}. An unrecognized marker "
                        "would be treated as a detection, biasing the fit "
                        "toward flux the galaxy does not have."
                    )
                _reject_nonfinite([value], f"{line_name} line flux", [line_name])
                _reject_nonfinite([err], f"{line_name} line uncertainty", [line_name])
                _reject_nonpositive_sigma([err], f"{line_name} line uncertainty", [line_name])
        return ValidatedData(flux, noise, spec_flux, spec_noise, censor, self.lines)
