# SPDX-License-Identifier: BSD-3-Clause
"""PhotometryObservationModel: first :class:`ObservationModel` adapter.

Phase II-1, ninth pass. Validates the
:class:`tengri.protocols.observation.ObservationModel` Protocol with a real
implementation: takes the rest-frame SED produced by the
:class:`SEDComponent` chain, redshifts it, applies cosmological
distance dimming, convolves through a filter set, and returns
broadband fluxes.

Scope
-----
- Reads ``state.sed_attenuated`` (post-dust rest-frame L_nu) if
  present, else falls back to ``state.sed_intrinsic``.
- Reads ``redshift`` from params (bare-name allowlist).
- Computes luminosity distance from a fixed cosmology held on
  ``self`` (default Planck18). Cosmology is *not* a free parameter —
  it's a Python attribute set at construction.
- Returns ``{"phot_fnu": ndarray, shape (n_filters,)}`` in cgs.

What this is NOT
----------------
- Not the spectroscopic observation model. ``SpectroscopyObservationModel``
  lands as the second adapter (the "two-adapter rule" graduates this
  protocol seam from hypothetical to real).
- Not the calibration polynomial layer — that lives in a separate
  adapter once spectroscopy is migrated.
- Not the noise model — :class:`Likelihood` consumes the prediction
  + a separate :class:`tengri.NoiseModel`.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import jax.numpy as jnp

from tengri.observation.photometry import (
    FilterCurve,
    compute_flux_density_batch,
    pad_filters,
)
from tengri.protocols.component import ParamDeclaration, PipelineState
from tengri.utils.cosmology import CosmoParams, luminosity_distance

__all__ = ["PhotometryObservationModel"]


@dataclass(frozen=True)
class PhotometryObservationModel:
    r"""Forward-model SED → broadband fluxes.

    Notes
    -----
    **JIT-compatible**: yes — :meth:`predict` is pure JAX once the
    padded filter arrays are computed (eagerly at construction).

    **Cosmology**: held as a frozen :class:`CosmoParams` attribute,
    not a parameter. Changing it requires constructing a new model.

    **Source-frame SED selection**: prefers
    ``state.sed_attenuated`` (after dust), falls back to
    ``state.sed_intrinsic`` (before dust). If both are ``None`` the
    prediction is identically zero.

    Parameters
    ----------
    filters : sequence of :class:`tengri.observation.photometry.FilterCurve`
        The broadband filters to convolve through. Order is preserved
        in the output array.
    cosmo : :class:`tengri.utils.cosmology.CosmoParams`, optional
        Cosmology used for the luminosity distance. Defaults to
        Planck18 if ``None``.
    name : str
        Diagnostic identifier; default ``"photometry"``.
    """

    filters: Sequence[FilterCurve]
    cosmo: CosmoParams | None = None
    name: str = "photometry"
    parameter_prefix: str = "phot_"

    # Padded filter arrays cached at construction so :meth:`predict`
    # is JIT-friendly (no Python loops).
    _fw_padded: jnp.ndarray = field(init=False, repr=False, compare=False)
    _ft_padded: jnp.ndarray = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        # `frozen=True` blocks normal attribute assignment; route
        # through object.__setattr__ as is idiomatic for frozen
        # dataclasses with computed fields.
        fw, ft, _n_valid = pad_filters(
            [f.wave for f in self.filters],
            [f.trans for f in self.filters],
        )
        object.__setattr__(self, "_fw_padded", fw)
        object.__setattr__(self, "_ft_padded", ft)

    def declared_parameters(self) -> list[ParamDeclaration]:
        r"""Free parameters this observation model owns.

        Currently empty: the cosmology is fixed and there are no
        broadband calibration nuisance parameters in this minimal
        adapter. When zeropoint offsets land they will appear here as
        ``phot_zp_offset_<filter_name>`` declarations.
        """
        return []

    def predict(
        self,
        state: PipelineState,
        params: Mapping[str, Any],
    ) -> Mapping[str, jnp.ndarray]:
        r"""Convolve the chain's rest-frame L_nu through the filters.

        Parameters
        ----------
        state : :class:`PipelineState`
            Must carry rest-frame ``wave`` (Å). Reads
            ``state.sed_attenuated`` if not ``None`` else
            ``state.sed_intrinsic``.
        params : mapping
            Must contain ``redshift``. The orchestrator threads it via
            :data:`tengri.protocols.BARE_NAME_ALLOWLIST`.

        Returns
        -------
        mapping
            ``{"phot_fnu": ndarray, shape (n_filters,)}`` in
            erg/s/cm²/Hz, observer frame.
        """
        sed_rest = (
            state.sed_attenuated if state.sed_attenuated is not None else state.sed_intrinsic
        )
        if sed_rest is None:
            return {"phot_fnu": jnp.zeros(len(self.filters))}

        z = jnp.asarray(params["redshift"])
        dl_cm = luminosity_distance(z, cosmo=self.cosmo)

        flux = compute_flux_density_batch(
            sed_rest,
            state.wave,
            self._fw_padded,
            self._ft_padded,
            z,
            dl_cm,
        )
        return {"phot_fnu": flux}
