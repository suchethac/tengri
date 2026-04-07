"""Unified observation configuration for tengri SED fitting.

Bundles photometric and/or spectroscopic setup with noise configuration
into a single declarative object. Inspired by Synthesizer's Instrument
pattern, adapted for tengri's JAX/differentiable inference context.

The Observation class is a frozen configuration container — it never
enters JAX-traced code. It configures what the Model precomputes and
what the Fitter dispatches.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp

from tengri.distributions import Distribution
from tengri.models.observation.noise_model import NoiseConfig
from tengri.models.observation.photometry_config import Photometry
from tengri.models.observation.spectroscopy import SpectroscopyConfig


@dataclasses.dataclass(frozen=True)
class Observation:
    """Unified observation configuration.

    Composes optional photometric, spectroscopic, and noise model
    configurations. At least one of ``photometry`` or ``spectroscopy``
    must be provided.

    Parameters
    ----------
    photometry : Photometry or None
        Photometric filter configuration.
    spectroscopy : SpectroscopyConfig or None
        Spectroscopic instrument configuration.
    noise : NoiseConfig or None
        Noise model configuration (calibration floor, Student-t dof).

    Examples
    --------
    Photometry-only::

        obs = Observation(
            photometry=Photometry.from_names(["sdss_r", "sdss_i"]),
        )

    Joint photometry + spectroscopy::

        obs = Observation(
            photometry=Photometry.from_names(["jwst_f200w", "jwst_f356w"]),
            spectroscopy=SpectroscopyConfig.nirspec_prism(wave_obs),
            noise=NoiseConfig(calibration_floor=Uniform(0.01, 0.15)),
        )
    """

    photometry: Photometry | None = None
    spectroscopy: SpectroscopyConfig | None = None
    noise: NoiseConfig | None = None

    def __post_init__(self):
        if self.photometry is None and self.spectroscopy is None:
            raise ValueError("Observation requires at least one of photometry or spectroscopy.")

    # -------------------------------------------------------------------
    # Capability queries
    # -------------------------------------------------------------------

    @property
    def can_do_photometry(self) -> bool:
        """Whether photometric filters are configured."""
        return self.photometry is not None

    @property
    def can_do_spectroscopy(self) -> bool:
        """Whether a spectroscopic wavelength grid is configured."""
        return self.spectroscopy is not None

    @property
    def is_joint(self) -> bool:
        """Whether both photometry and spectroscopy are configured."""
        return self.can_do_photometry and self.can_do_spectroscopy

    @property
    def data_type(self) -> str:
        """Inferred data type string for backward compatibility.

        Returns
        -------
        str
            One of ``"photometry"``, ``"spectroscopy"``, or ``"joint"``.
        """
        if self.is_joint:
            return "joint"
        elif self.can_do_photometry:
            return "photometry"
        else:
            return "spectroscopy"

    # -------------------------------------------------------------------
    # Data dimensions
    # -------------------------------------------------------------------

    @property
    def n_data_phot(self) -> int:
        """Number of photometric data points (filters)."""
        return self.photometry.n_filters if self.photometry else 0

    @property
    def n_data_spec(self) -> int:
        """Number of spectroscopic data points (pixels)."""
        return self.spectroscopy.n_pixels if self.spectroscopy else 0

    @property
    def n_data(self) -> int:
        """Total number of data points."""
        return self.n_data_phot + self.n_data_spec

    # -------------------------------------------------------------------
    # Data packing / unpacking
    # -------------------------------------------------------------------

    def pack_data(
        self,
        phot: jnp.ndarray | None = None,
        spec: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """Concatenate photometry and spectroscopy data in canonical order.

        Validates array shapes against the observation configuration.
        Canonical order: ``[photometry, spectroscopy]``.

        Parameters
        ----------
        phot : array or None
            Photometric data, shape ``(n_filters,)``.
        spec : array or None
            Spectroscopic data, shape ``(n_pixels,)``.

        Returns
        -------
        jnp.ndarray
            Packed data array.

        Raises
        ------
        ValueError
            If array shapes don't match the observation configuration.
        """
        arrays = []

        if self.can_do_photometry:
            if phot is None:
                raise ValueError("Observation has photometry but phot= was not provided.")
            phot = jnp.asarray(phot)
            if phot.shape != (self.n_data_phot,):
                raise ValueError(
                    f"phot shape {phot.shape} doesn't match expected ({self.n_data_phot},)"
                )
            arrays.append(phot)

        if self.can_do_spectroscopy:
            if spec is None:
                raise ValueError("Observation has spectroscopy but spec= was not provided.")
            spec = jnp.asarray(spec)
            if spec.shape != (self.n_data_spec,):
                raise ValueError(
                    f"spec shape {spec.shape} doesn't match expected ({self.n_data_spec},)"
                )
            arrays.append(spec)

        if len(arrays) == 1:
            return arrays[0]
        return jnp.concatenate(arrays)

    def unpack_prediction(
        self,
        predicted: jnp.ndarray,
    ) -> dict[str, jnp.ndarray]:
        """Split a concatenated prediction into photometry and spectroscopy.

        Parameters
        ----------
        predicted : array
            Packed prediction array, shape ``(n_data,)``.

        Returns
        -------
        dict
            Keys are ``"photometry"`` and/or ``"spectroscopy"``, values
            are the corresponding sub-arrays.
        """
        result = {}
        idx = 0

        if self.can_do_photometry:
            result["photometry"] = predicted[idx : idx + self.n_data_phot]
            idx += self.n_data_phot

        if self.can_do_spectroscopy:
            result["spectroscopy"] = predicted[idx : idx + self.n_data_spec]

        return result

    # -------------------------------------------------------------------
    # Parameter generation
    # -------------------------------------------------------------------

    def get_all_params(self) -> dict[str, Distribution]:
        """Collect all observation-driven parameters.

        Merges calibration polynomial params from spectroscopy config
        and noise model params from noise config.

        Returns
        -------
        dict
            Parameter name → Distribution. Empty if no observation
            params are needed (e.g. photometry-only with no noise config).
        """
        params: dict[str, Distribution] = {}

        if self.spectroscopy is not None:
            params.update(self.spectroscopy.get_calibration_params())

        if self.noise is not None:
            params.update(self.noise.get_params())

        return params

    # -------------------------------------------------------------------
    # Display
    # -------------------------------------------------------------------

    def summary(self) -> str:
        """Return a human-readable summary of the observation.

        Returns
        -------
        str
            Multi-line summary string.
        """
        lines = ["Observation"]
        lines.append("-" * 50)

        if self.photometry is not None:
            lines.append(f"  Photometry : {self.photometry.summary()}")

        if self.spectroscopy is not None:
            lines.append(f"  Spectroscopy : {self.spectroscopy.summary()}")

        if self.noise is not None:
            lines.append(f"  Noise      : {self.noise.summary()}")

        lines.append(f"  Data type  : {self.data_type}")
        lines.append(f"  N data     : {self.n_data}")
        if self.is_joint:
            lines.append(f"               ({self.n_data_phot} phot + {self.n_data_spec} spec)")

        obs_params = self.get_all_params()
        if obs_params:
            lines.append(f"  Auto params: {', '.join(sorted(obs_params.keys()))}")

        return "\n".join(lines)
