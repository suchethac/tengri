# SPDX-License-Identifier: BSD-3-Clause
"""ObservationModel protocol: how a forward-model SED meets data.

An :class:`ObservationModel` takes a :class:`ForwardState` (rest-frame
SED + observer-frame F_nu produced by the chain of
:class:`SEDComponent` objects) and produces *predicted observables* in the
data's native format — broadband photometric points, spectroscopic flux
samples, emission-line equivalent widths, etc.

Nothing in `tengri` consumes this protocol yet; the existing
:mod:`tengri.observation.observation` and friends remain the active
implementations until the orchestrator path is the primary surface.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Protocol, runtime_checkable

import jax.numpy as jnp

from tengri.protocols.component import ForwardState

__all__ = ["ObservationModel"]


@runtime_checkable
class ObservationModel(Protocol):
    """Contract for the data-side of the forward model.

    Concrete implementations:

    - ``PhotometryObservation``: convolves observer-frame F_nu with a
      :class:`tengri.observation.LineList`-equivalent filter set.
    - ``SpectroscopyObservation``: applies the line-spread function,
      wavelength mask, and calibration polynomial.
    - ``JointObservation``: combines both for objects with photometry
      and spectroscopy.

    Required attributes
    -------------------
    name : str
        Stable identifier for diagnostics. Examples: ``"photometry"``,
        ``"spectroscopy"``, ``"joint"``.

    Required methods
    ----------------
    predict(state, params) -> mapping
        Pure JAX. Returns a dict of predicted observables keyed by
        observation channel (e.g. ``{"phot_fnu": ..., "spec_fnu": ...,
        "lines_flux": ...}``). The keys must match the keys the
        :class:`Likelihood` expects.

    declared_parameters() -> list[ParamSpec]
        Any free parameters the observation model itself owns
        (e.g. ``noise_f_cal`` for calibration polynomial coefficients,
        ``eline_sigma_kms`` for line broadening). Domain prefix:
        ``noise_`` or ``eline_``.

    Notes
    -----
    **JIT-compatible:** :meth:`predict` is pure JAX. Static
    configuration (filter curves, wavelength masks) is held as Python
    attributes set at construction.
    """

    name: str

    def declared_parameters(self) -> list[Any]:
        """Free parameters the observation model owns."""
        ...

    def predict(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        """Predicted observables.

        Returns
        -------
        mapping of str -> array
            Keys identify the observation channel. Standard keys:

            - ``"phot_fnu"``: F_nu in cgs at filter pivots
            - ``"spec_fnu"``: F_nu in cgs at the spec wavelength grid
            - ``"lines_flux"``: emission-line integrated fluxes
            - ``"indices"``: Lick indices and similar

            Implementations are free to add more keys; the
            :class:`Likelihood` decides which it consumes.
        """
        ...
