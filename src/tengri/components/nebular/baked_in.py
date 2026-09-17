# SPDX-License-Identifier: BSD-3-Clause
"""Baked-in nebular emission backend.

When SSP files already include nebular emission (wNE files),
this backend is a no-op: it returns zero additional contribution.
The nebular emission is already part of the SSP flux arrays.
"""

import warnings

import jax.numpy as jnp

from tengri._cache_keys import KeyPolicy, content, derive_key


class BakedInNebularWarning(UserWarning):
    """Warning raised when BakedInBackend is used with SSPs.

    Indicates that nebular emission is baked into the SSP file at fixed logU
    and escape fraction. These are NOT free parameters; fitting nebular
    properties requires switching to a different backend (CueBackend or
    CloudyGridBackend).

    Notes
    -----
    The SSP file's nebular assumptions (typically logU = -3) are determined
    when the SSP grid was generated and cannot be changed by the model.

    """


class BakedInNebularGridWarning(UserWarning):
    """Warning raised when BakedInBackend receives an unstamped (unknown) SSP.

    Indicates that the SSP grid's nebular status could not be determined from
    metadata. The grid may or may not include nebular emission. This warning
    cannot be silenced by the neb={'type': 'ssp'} declaration because it is
    a question about the data, not about the model choice.

    Notes
    -----
    Use `tools/stamp_ssp_nebular_attrs.py` to resolve this ambiguity by
    reading the grid and stamping it with the nebular_included attribute.

    """


class BakedInNebularBareError(ValueError):
    """Raised when BakedInBackend receives a bare (no-nebular) SSP.

    The user has requested BakedInBackend on a bare-stellar SSP grid that
    carries no nebular emission whatsoever. The backend has no choice but to
    return zero nebular emission, resulting in a model with no emission lines
    and no nebular continuum — silently incorrect.

    """


class BakedInBackend:
    """Nebular backend for SSP files with pre-included emission.

    A no-op backend that returns zero additional nebular flux because the SSP
    templates already contain nebular emission at fixed ionization parameter
    and escape fraction. This is the default backend when no CLOUDY grid is
    specified.

    Parameters
    ----------
    ionizing_source_warning : str, optional
        Verbosity control for the fixed-nebular limitation warning. One of
        'raise' (raise ValueError), 'warn' (emit UserWarning), or 'suppress'
        (silent). Default: 'warn'.

    Attributes
    ----------
    has_continuum : bool
        True: continuum is baked into the SSP.
    has_free_params : bool
        False: ionization parameter and escape fraction are fixed.
    name : str
        Identifier "baked_in".

    Notes
    -----
    **JIT-compatible**: yes, predict_nebular_sed and
    predict_nebular_line_fluxes return zero arrays.

    To fit nebular properties, switch to CloudyGridBackend or CueBackend
    which provide free parameters for ionization parameter, escape fraction,
    and metallicity.

    """

    def __init__(self, ionizing_source_warning: str = "warn", ssp_data=None) -> None:
        """Initialize BakedInBackend.

        Parameters
        ----------
        ionizing_source_warning : str, optional
            Verbosity control for the fixed-parameter advisory. Default: "warn".
        ssp_data : SSPData, optional
            SSP data to check for nebular content. If provided, will raise
            on bare grids and warn on unknown grids.

        Raises
        ------
        BakedInNebularBareError
            If ssp_data.nebular == "bare" (unambiguously no nebular emission).

        """
        self.name = "baked_in"
        self.has_free_params = False

        # Check SSP nebular status and emit appropriate warnings/errors
        # Read directly: missing attribute is a bug to surface, not default
        if ssp_data is None:
            nebular_status = "unknown"
        else:
            nebular_status = ssp_data.nebular

        if nebular_status == "bare":
            # Bare grid: unambiguously wrong. Raise immediately.
            raise BakedInNebularBareError(
                "BakedInBackend received a bare-stellar SSP (no nebular emission): "
                "the grid carries zero nebular continuum and zero emission lines. "
                "BakedInBackend returns zero nebular contribution, which is correct "
                "only if the SSP file already includes the emission. "
                "\n"
                "Fix (one of): "
                "\n"
                "  1. Download an SSP with nebular emission included (wNE grid): "
                "\n"
                "     tengri.download_ssp('ssp_prsc_miles_chabrier_wNE_logGasU-3.0_logGasZ0.0') "
                "\n"
                "     Then rebuild the model with that grid. "
                "\n"
                "  2. Keep this bare-stellar grid and drop neb={'type': 'ssp'}: "
                "\n"
                "     use neb={'type': 'cloudy_grid'} or neb={'type': 'cue'} "
                "\n"
                "     to add a separate nebular component. "
                "\n"
                "  3. If the grid has been processed to remove nebular emission "
                "\n"
                "     by hand, re-run tools/stamp_ssp_nebular_attrs.py --bare "
                "\n"
                "     to stamp the metadata so other backends are also warned."
            )

        if nebular_status == "unknown":
            # Unknown grid: ambiguous, warn but allow. The warning is NOT
            # silenced by the explicit neb={'type':'ssp'} declaration.
            msg = (
                "BakedInBackend received an SSP whose nebular status is unknown: "
                "the metadata does not declare whether the grid includes nebular "
                "emission. If this grid is bare-stellar, you will silently get "
                "a model with zero nebular emission and zero emission lines. "
                "\n"
                "Resolve the ambiguity with tools/stamp_ssp_nebular_attrs.py: "
                "\n"
                "  python tools/stamp_ssp_nebular_attrs.py [--included|--bare] <path.h5> "
                "\n"
                "This warning cannot be suppressed by the explicit neb={'type': 'ssp'} "
                "declaration — it is a statement about the data, not your model choice."
            )
            warnings.warn(msg, BakedInNebularGridWarning, stacklevel=2)

        # has_continuum is True only for "included"; "unknown" is an assumption
        # stated by the warning above
        self.has_continuum = nebular_status in ("included", "unknown")

        if ionizing_source_warning not in ("raise", "warn", "suppress"):
            raise ValueError("ionizing_source_warning must be 'raise', 'warn', or 'suppress'")
        if ionizing_source_warning != "suppress":
            msg = (
                "BakedInBackend: nebular emission is baked into the SSP file at a "
                "fixed logU and fixed escape fraction determined when the SSP grid "
                "was generated (commonly logU = −3, but depends on the SSP file). "
                "The ionization parameter and escape fraction are NOT free parameters "
                ": varying neb_logU or neb_fesc in your Parameters will have no "
                "effect. Check your SSP file's nebular assumptions. Switch to "
                "CloudyGridBackend or CueBackend to vary nebular properties. "
                "State the choice explicitly when building via SEDModel.build to "
                "silence this advisory: neb={'type': 'ssp'} (or {'type': 'none'}). "
                "(ionizing_source_warning='suppress' also works when constructing "
                "BakedInBackend directly.)"
            )
            if ionizing_source_warning == "raise":
                raise ValueError(msg)
            warnings.warn(msg, BakedInNebularWarning, stacklevel=2)

    def cache_key(self) -> tuple:
        """Return a hashable cache key for this backend's structure.

        Returns
        -------
        tuple
            Cache key. A no-op backend has no data-carrying structure beyond
            its three identity flags.
        """
        return derive_key(self, _BAKED_IN_BACKEND_CACHE_KEY_POLICY)

    def predict_nebular_sed(
        self,
        ssp_weights: jnp.ndarray,
        ssp_wave: jnp.ndarray,
        log_z: float,
        **neb_params,
    ) -> jnp.ndarray:
        """Return zero nebular contribution (already in SSP).

        Parameters
        ----------
        ssp_weights : array, shape (n_age,) or (n_met, n_age)
            CSP mass weights (unused).
        ssp_wave : array, shape (n_wave,)
            Wavelength grid [Angstrom].
        log_z : float
            Stellar metallicity (unused) [log10(Z)].
        **neb_params
            Additional nebular parameters (all unused).

        Returns
        -------
        array, shape (n_wave,)
            Zero array: nebular emission is baked into the SSP [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes, returns jnp.zeros_like.

        """
        return jnp.zeros_like(ssp_wave)

    def predict_nebular_line_fluxes(
        self,
        ssp_weights: jnp.ndarray,
        log_z: float,
        **neb_params,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Return empty line arrays.

        Parameters
        ----------
        ssp_weights : array, shape (n_age,) or (n_met, n_age)
            CSP mass weights (unused).
        log_z : float
            Stellar metallicity (unused) [log10(Z)].
        **neb_params
            Additional nebular parameters (all unused).

        Returns
        -------
        wavelengths : array, shape (0,)
            Empty array [Angstrom].
        luminosities : array, shape (0,)
            Empty array [erg/s].

        References
        ----------
        Nebular emission is pre-calculated in the SSP templates when using
        this backend; see the SSP file documentation for assumptions about
        ionization parameter and escape fraction.

        Notes
        -----
        **JIT-compatible**: yes, returns empty jnp arrays.

        """
        return jnp.array([]), jnp.array([])


_BAKED_IN_BACKEND_CACHE_KEY_POLICY: KeyPolicy = {
    "name": content("backend identity string"),
    "has_free_params": content("whether ionization params are fittable"),
    "has_continuum": content("whether the backend publishes a continuum"),
}
