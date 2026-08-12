# SPDX-License-Identifier: BSD-3-Clause
"""Dale & Helou (2002) + Chary & Elbaz (2001) dust emission as SEDModelComponent.

Wraps the pure closure from :mod:`tengri.components.dust.emission`.
"""

from __future__ import annotations

from typing import ClassVar

import jax.numpy as jnp

from tengri.components.dust.emission._component_base import EmissionComponent

__all__ = ["DH02CE01IRSEDComponent"]


class DH02CE01IRSEDComponent(EmissionComponent):
    r"""Dale & Helou (2002) / Chary & Elbaz (2001) cold-dust IR emission template.

    Wraps the pure closure built from the tabulated DH02+CE01 template library.
    Its single grid axis is the total infrared luminosity
    :math:`\log_{10}(L_{\rm IR}/L_\odot)`, tabulated over [8.3, 14.3].

    Unlike its sibling templates this component declares **no free parameter**.
    ``components/grid_support.py`` records the reason:

        dh02_ce01 is deliberately absent: its only grid axis is L_TIR, derived
        from L_absorbed by energy balance rather than set by the user, so no
        prior can overhang it.

    So the template is selected from the absorbed luminosity the dust chain
    already supplies, via :math:`\log_{10}(L_{\rm IR}/L_\odot)` with
    :data:`~tengri.utils.physics_constants.L_SUN`. The closure clips that to
    the tabulated range.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable except at the grid boundaries,
    where the luminosity axis is clipped.

    **Template auto-loading**: the closure lazy-loads ``dh02_ce01_grid.h5`` on
    first call (at trace time). After lazy loading, all subsequent calls are
    pure JAX.

    This component existed only as a loader plus a grammar entry until #1738.
    ``_valid_dust_emission_types()`` accepted the name and
    :func:`~tengri.registry.list_dust_emission_models` advertised it as
    ``status='production'``, but no class was registered, so
    ``dust={'emission': {'type': 'dh02_ce01'}}`` raised at build. It was the
    only one of the nineteen dust-emission types with no component, and the
    registry emit census added for #1738 is what surfaced it.

    References
    ----------
    .. [1] Dale, D. A. & Helou, G., 2002, "The Infrared Spectral Energy
       Distribution of Normal Star-forming Galaxies: Calibration at Far-Infrared
       and Submillimeter Wavelengths", ApJ, 576, 159.
       https://doi.org/10.1086/341632
    .. [2] Chary, R. & Elbaz, D., 2001, "Interpreting the Cosmic Infrared
       Background: Constraints on the Evolution of the Dust-enshrouded Star
       Formation Rate", ApJ, 556, 562. https://doi.org/10.1086/321609

    """

    name: str = "dh02_ce01"

    #: Empty deliberately: neither Dale & Helou (2002) nor Chary & Elbaz (2001)
    #: has a key in ``tengri.citations.registry.REGISTRY`` yet, and a citation
    #: tuple naming absent keys would advertise provenance it cannot deliver.
    #: The papers are cited in the References section above.
    _citations_tuple: ClassVar[tuple[str, ...]] = ()

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Compute DH02+CE01 dust emission.

        Parameters
        ----------
        p : dict
            Parameters with prefix stripped. Unused: this template has no free
            parameter, its grid axis being derived from ``L_ir``.
        sed_in : ndarray, shape (n_wave,)
            Input SED [erg/s/Hz] (typically zeros for a dust emission component).
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid [Angstrom].
        L_ir : float
            Total absorbed luminosity [erg/s].

        Returns
        -------
        tuple[ndarray, dict]
            ``(sed_out, published)`` where ``sed_out`` is the updated SED
            [erg/s/Hz] and ``published`` carries ``"sed_dust_ir"`` [erg/s/Hz].

        Notes
        -----
        The template index is

        .. math:: \log_{10}(L_{\rm IR} / L_\odot)

        with :math:`L_{\rm IR}` the absorbed luminosity [erg/s]. The floor on
        the ratio keeps the logarithm finite when the dust chain absorbs
        nothing; the closure clips the result into the tabulated range anyway.
        """
        del p
        from tengri.components.dust.emission import DUST_EMISSION_MODELS
        from tengri.utils.physics_constants import L_SUN

        log_lir = jnp.log10(jnp.maximum(jnp.asarray(L_ir) / L_SUN, 1e-30))
        sed = DUST_EMISSION_MODELS["dh02_ce01"](wave, L_ir, dust_log_lir=log_lir)
        return sed_in + sed, {"sed_dust_ir": sed}
