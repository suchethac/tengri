# SPDX-License-Identifier: BSD-3-Clause
"""Observables, the dual of :class:`Observation` for predicted quantities.

An :class:`Observables` NamedTuple is synthesized per-model at
:class:`SEDModel` construction time from the ``Observation`` contents:
which sub-blocks the observation carries determines which fields exist
on the prediction. Missing channels are absent attributes (``AttributeError``
on access), not silent zeros.

Used by :meth:`SEDModel.predict_observables` and :meth:`Observation.predict`.

Part of the forward-projection unification
(``docs/dev/archive/photometry_path_unification.md``).
"""

from __future__ import annotations

from typing import Any, NamedTuple

import jax.numpy as jnp


def build_observables_class(observation) -> type:
    """Synthesize the per-model :class:`Observables` NamedTuple subclass.

    The returned class is a :class:`typing.NamedTuple` subclass whose
    fields mirror the observation's configured sub-blocks. Magnitude
    properties (``mag_apparent``, ``mag_absolute``) attach when
    photometry is present.

    Parameters
    ----------
    observation : Observation
        Configured observation. Used only to read capability flags
        (``can_do_photometry``, ``can_do_spectroscopy``,
        ``has_line_fluxes``, ``has_spectral_indices``); the data
        contents are irrelevant, only the structural fingerprint
        matters for the class shape.

    Returns
    -------
    type
        A :class:`typing.NamedTuple` subclass. Fields are populated
        positionally or by keyword at instantiation. Pytree-registered
        automatically by JAX (NamedTuples are leaves-or-trees by default).

    Notes
    -----
    Field set (conditional on observation contents):

    - ``phot_fnu``: observed-frame F_nu [erg/s/cm²/Hz], shape ``(n_filt,)``
    - ``phot_rest_fnu``: rest-frame F_nu at d_L=10 pc, same filters, shape ``(n_filt,)``
    - ``spec_fnu``: observed-frame F_nu [erg/s/cm²/Hz], shape ``(n_pix,)``
    - ``lines_flux``: integrated line fluxes [erg/s/cm²], shape ``(n_lines,)``
    - ``indices``: spectral indices, shape ``(n_indices,)``

    Properties (attached when photometry is configured):

    - ``mag_apparent``: AB mag from ``phot_fnu``
    - ``mag_absolute``: AB mag from ``phot_rest_fnu``

    Examples
    --------
    >>> obs = Observation(photometry=Photometry.from_names(["sdss_r"]))
    >>> Observables = build_observables_class(obs)
    >>> o = Observables(phot_fnu=jnp.array([1e-26]), phot_rest_fnu=jnp.array([2e-26]))
    >>> o.phot_fnu
    DeviceArray([1.e-26], dtype=float64)
    >>> o.mag_apparent
    DeviceArray([23.4], dtype=float64)
    >>> o.spec_fnu  # raises AttributeError, spectroscopy not configured
    """
    fields: list[tuple[str, Any]] = []

    has_phot = bool(getattr(observation, "can_do_photometry", False))
    has_spec = bool(getattr(observation, "can_do_spectroscopy", False))

    # ``Observables`` is the *projection* output only (phot_fnu / spec_fnu and
    # their rest-frame variants). Scalar measurables, line fluxes, line
    # ratios, spectral indices, are NOT fields here: they are computed
    # separately (``predict_line_fluxes`` / ``predict_line_ratios`` /
    # ``predict_spectral_indices``) and fed to the likelihood via the
    # prediction dict, so the projection NamedTuple stays a clean
    # (phot, spec) container regardless of which measurables are configured.
    if has_phot:
        fields.append(("phot_fnu", jnp.ndarray))
        fields.append(("phot_rest_fnu", jnp.ndarray))
    if has_spec:
        fields.append(("spec_fnu", jnp.ndarray))

    if not fields:
        # Defensive: no observable channels, give an empty NamedTuple
        # so callers can still write `model.predict_observables(params)`
        # and get back a structurally-valid (but empty) result.
        class Observables(NamedTuple):
            pass

        return Observables

    base = NamedTuple("Observables", fields)

    # Attach magnitude properties when photometry is configured.
    # Done by subclassing rather than monkey-patching so the NamedTuple
    # remains a clean pytree leaf-or-tree from JAX's perspective.
    if has_phot:
        from tengri.units import fnu_to_ab_mag

        class Observables(base):  # type: ignore[misc, valid-type]
            __slots__ = ()

            @property
            def mag_apparent(self) -> jnp.ndarray:
                """Apparent AB magnitude, computed from ``phot_fnu``."""
                return fnu_to_ab_mag(self.phot_fnu)

            @property
            def mag_absolute(self) -> jnp.ndarray:
                """Absolute AB magnitude, computed from ``phot_rest_fnu``.

                The rest-frame photometry is the same-filter integral of
                the rest-frame SED at d_L=10 pc, physically correct
                without distance-modulus / K-correction debate.
                """
                return fnu_to_ab_mag(self.phot_rest_fnu)

        return Observables

    return base
