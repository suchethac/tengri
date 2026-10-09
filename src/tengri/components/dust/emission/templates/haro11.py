# SPDX-License-Identifier: BSD-3-Clause
"""Lyu, Rieke & Alberts (2016) Haro 11 dust emission template."""

from __future__ import annotations

from typing import Any, ClassVar

import h5py
import jax.numpy as jnp

from tengri._data_setup import find_data_str
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.dust.emission._physics import integrate_lnu_over_nu

__all__ = ["Haro11IRSEDComponent"]


class Haro11IRSEDComponent(EmissionComponent):
    """Emit the fixed public Haro 11 relative F_nu template above 5 μm."""

    name: str = "haro11"
    declares_no_parameters: ClassVar[bool] = True
    _citations_tuple: ClassVar[tuple[str, ...]] = ("lyu_rieke_alberts2016",)
    accepts_threaded_templates: ClassVar[bool] = True
    energy_balanced: ClassVar[bool] = True
    factors_l_ir: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None) -> dict[str, jnp.ndarray]:
        """Load the packaged Haro 11 spectrum before tracing.

        Parameters
        ----------
        wave : array_like, shape (n_wave,), optional
            Rest-frame wavelength grid [Å]. Unused.

        Returns
        -------
        dict[str, ndarray]
            ``wavelength_aa`` and ``flux_nu_relative``, each with shape
            ``(n_template,)``. Wavelengths are in [Å]; flux values are relative
            :math:`F_\nu` values.

        Raises
        ------
        FileNotFoundError
            The packaged ``haro11.h5`` data file is unavailable.

        Notes
        -----
        **JIT-compatible**: no. This method reads HDF5 data at model-build time.
        """
        del wave
        path = find_data_str("haro11.h5")
        if path is None:
            raise FileNotFoundError("Haro 11 dust template 'haro11.h5' was not found.")
        with h5py.File(path, "r") as source:
            return {
                "wavelength_aa": jnp.asarray(source["wavelength_aa"][:]),
                "flux_nu_relative": jnp.asarray(source["flux_nu_relative"][:]),
            }

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: float,
        templates: Any = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Interpolate and normalize the Haro 11 template on the model grid.

        Parameters
        ----------
        p : mapping[str, array_like]
            No shape parameters are read.
        sed_in : array_like, shape (n_wave,)
            Input spectral luminosity density [erg/s/Hz].
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_ir : array_like, scalar
            Total IR luminosity [erg/s].
        templates : mapping[str, array_like], optional
            Threaded ``wavelength_aa`` and ``flux_nu_relative`` arrays, each
            with shape ``(n_template,)``. The relative flux is :math:`F_\nu`.

        Returns
        -------
        tuple[ndarray, dict[str, ndarray]]
            Updated SED and ``sed_dust_ir`` [erg/s/Hz], each with shape
            ``(n_wave,)``.

        Notes
        -----
        **JIT-compatible**: yes — interpolation, integration, and normalization
        use JAX array operations. **Gradient-safe**: yes, except at the template
        endpoints and the 5 μm cutoff.

        The spectrum is normalized as

        .. math::

            L_\nu(\lambda) = L_{\rm IR}\,
            \frac{H_G(\lambda)}{\int_G H_G\,d\nu},

        where :math:`H_G` is the source-relative :math:`F_\nu` template
        interpolated onto the model grid :math:`G`; :math:`L_\nu` is the emitted
        spectral luminosity density [erg/s/Hz], :math:`L_{\rm IR}` is the total
        dust luminosity [erg/s], and :math:`d\nu` is frequency [Hz]. Values at
        :math:`\lambda \leq 5\,\mu\mathrm{m}` and outside the tabulated range are
        zero. The source table is the public Haro 11 template (Lyu, Rieke &
        Alberts 2016 [1]_).

        References
        ----------
        .. [1] J. Lyu, G. H. Rieke, and S. Alberts, "The Contribution of Host
           Galaxies to the Infrared Energy Output of z\gtrsim5.0 Quasars," ApJ, 816,
           85 (2016). arXiv:1511.05938. https://doi.org/10.3847/0004-637X/816/2/85
        """
        del p
        if templates is None:
            templates = self.data

        wavelength_aa = jnp.asarray(templates["wavelength_aa"])
        flux_nu_relative = jnp.asarray(templates["flux_nu_relative"])
        shape = jnp.interp(wave, wavelength_aa, flux_nu_relative, left=0.0, right=0.0)
        integral = integrate_lnu_over_nu(shape, wave)
        sed_ir = shape * (jnp.asarray(L_ir) / integral)
        return sed_in + sed_ir, {"sed_dust_ir": sed_ir}
