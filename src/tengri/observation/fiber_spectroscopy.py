# SPDX-License-Identifier: BSD-3-Clause
"""FiberSpectroscopyObservation, fiber-aperture-aware spectroscopy adapter.

Wraps an existing :class:`tengri.Observation` (configured for
spectroscopy) and scales its ``spec_fnu`` output by the aperture
fraction computed from the spatial profile in
``state.derived["spatial_profile_2d"]``.

This is the physically-correct replacement for the flat-slab scaling
that classical SED-fitting codes use when comparing a fiber spectrum
against total-flux photometry. The aperture fraction depends on the
galaxy's spatial profile, the fiber radius in arcsec, and the source
redshift via the angular diameter distance.

Example
-------

.. code-block:: python

    from tengri.observation import Observation, FiberSpectroscopyObservation
    from tengri.forward.spatial_model import SpatialModel
    from tengri.components.spatial.sersic import Sersic

    # Existing observation configured for spectroscopy (and possibly photometry)
    base_obs = Observation(spectroscopy=Spectroscopy(...), photometry=Photometry(...))

    obs = FiberSpectroscopyObservation(
        observation=base_obs,
        fiber_radius_arcsec=1.0,  # SDSS-like 2-arcsec-diameter fiber
    )

    spatial = SpatialModel(components=[Sersic()])
    forward = ForwardModel.build(sed=..., spatial=spatial, observation=obs)
    pred = forward.predict_observables(params)
    # pred["spec_fnu"] is now the aperture-fraction-scaled spectrum;
    # pred["phot_fnu"] (if present) is unchanged (total-flux convention).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax.numpy as jnp

from tengri.observation.fiber_aperture import aperture_fraction, arcsec_to_kpc
from tengri.parameters.resolve import require_redshift
from tengri.protocols.component import ForwardState

__all__ = ["FiberSpectroscopyObservation"]


@dataclass(frozen=True)
class FiberSpectroscopyObservation:
    """Spectroscopy observation that scales by the fiber aperture fraction.

    Wraps any object exposing ``predict(state, params) → dict`` (the
    :class:`tengri.protocols.ObservationModel` shape) and, if the
    result has a ``"spec_fnu"`` key AND the state has a
    ``spatial_profile_2d`` key, multiplies the spectrum by the
    aperture fraction.

    Parameters
    ----------
    observation: object
        Wrapped observation. Must expose ``predict(state, params)``.
    fiber_radius_arcsec: float
        Fiber radius in arcseconds (e.g. 1.0 for an SDSS-like
        2-arcsec-diameter fiber).
    fiber_center_arcsec: tuple of (float, float), default (0.0, 0.0)
        Fiber center offset in arcsec (x, y) from the galaxy nucleus.
    softness: float, default 0.01
        Sigmoidal edge softening (fraction of aperture radius). Set
        to 0 for a hard top-hat (non-differentiable at the edge).
    name: str, default "fiber_spec"
        Identifier for diagnostics.

    Notes
    -----
    JIT/grad/vmap-compatible. The aperture-fraction integral is
    differentiable in the redshift (via angular diameter distance)
    and the fiber radius, useful when either is a fitted parameter.

    When the state has no ``spatial_profile_2d`` key (i.e. the user
    didn't wire up a spatial sub-model), the wrapped predict is
    returned unchanged. This degenerates gracefully to the flat-slab
    behavior the classical codes use.
    """

    observation: Any
    fiber_radius_arcsec: float
    fiber_center_arcsec: tuple[float, float] = (0.0, 0.0)
    softness: float = 0.01
    name: str = "fiber_spec"

    def predict(
        self,
        state: ForwardState,
        params: Mapping[str, jnp.ndarray],
    ) -> Mapping[str, jnp.ndarray]:
        """Predicted observables with the spectrum aperture-corrected."""
        pred = dict(self.observation.predict(state, params))
        if "spec_fnu" not in pred:
            return pred
        if "spatial_profile_2d" not in state.derived:
            return pred

        profile = state.derived["spatial_profile_2d"]
        grid = state.derived["spatial_grid_xy_kpc"]

        z = jnp.asarray(require_redshift(params, "observation.fiber_spectroscopy.predict"))
        radius_kpc = arcsec_to_kpc(self.fiber_radius_arcsec, z)
        center_kpc = (
            float(arcsec_to_kpc(self.fiber_center_arcsec[0], z)),
            float(arcsec_to_kpc(self.fiber_center_arcsec[1], z)),
        )
        frac = aperture_fraction(profile, grid, radius_kpc, center_kpc, self.softness)
        pred["spec_fnu"] = pred["spec_fnu"] * frac
        return pred
