# SPDX-License-Identifier: BSD-3-Clause
"""Public Rieke et al. normal star-forming galaxy dust templates."""

from __future__ import annotations

from typing import Any, ClassVar

import h5py
import jax.numpy as jnp

from tengri._data_setup import find_data_str
from tengri.components.dust._params import (
    RIEKE2009_LOG_L_IR_TEMPLATE_DEFAULT,
    RIEKE2009_LOG_L_IR_TEMPLATE_RANGE,
)
from tengri.components.dust.emission._component_base import EmissionComponent
from tengri.components.dust.emission._physics import integrate_lnu_over_nu
from tengri.parameters.priors import Uniform

__all__ = ["Rieke2009IRSEDComponent"]

_RELATIVE_FLUX_FLOOR = 1e-30


class Rieke2009IRSEDComponent(EmissionComponent):
    r"""Emit a Rieke normal-SFG template selected by its IR luminosity."""

    name: str = "rieke2009"
    log_L_ir_template = Uniform(
        *RIEKE2009_LOG_L_IR_TEMPLATE_RANGE,
        description="Template IR luminosity selecting the normal-SFG dust shape",
        units="log10(L_IR/L_sun)",
        default=RIEKE2009_LOG_L_IR_TEMPLATE_DEFAULT,
    )
    _citations_tuple: ClassVar[tuple[str, ...]] = ("rieke2009", "lyu2022")
    accepts_threaded_templates: ClassVar[bool] = True
    energy_balanced: ClassVar[bool] = True
    factors_l_ir: ClassVar[bool] = True

    def load(self, wave: jnp.ndarray | None = None) -> dict[str, jnp.ndarray]:
        """Load the common-grid SFG library before JAX tracing.

        Parameters
        ----------
        wave : array_like, shape (n_wave,), optional
            Rest-frame wavelength grid [Å]. Unused.

        Returns
        -------
        dict[str, ndarray]
            ``log_L_ir_template`` with shape ``(n_template,)``,
            ``wavelength_aa`` with shape ``(n_wave,)``, and
            ``flux_nu_relative`` with shape ``(n_template, n_wave)``.

        Raises
        ------
        FileNotFoundError
            If the packaged ``rieke2009.h5`` file is unavailable.

        Notes
        -----
        **JIT-compatible**: no. This method reads HDF5 data during model build.
        """
        del wave
        path = find_data_str("rieke2009.h5")
        if path is None:
            raise FileNotFoundError("Rieke 2009 dust template 'rieke2009.h5' was not found.")
        with h5py.File(path, "r") as source:
            return {
                "log_L_ir_template": jnp.asarray(source["log_L_ir_template"][:]),
                "wavelength_aa": jnp.asarray(source["wavelength_aa"][:]),
                "flux_nu_relative": jnp.asarray(source["flux_nu_relative"][:]),
            }

    def predict(
        self,
        p: dict[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        L_ir: jnp.ndarray,
        templates: Any = None,
    ) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
        r"""Interpolate, normalize, and scale a public normal-SFG template.

        Parameters
        ----------
        p : mapping[str, array_like]
            Parameter mapping with ``log_L_ir_template`` in
            :math:`\log_{10}(L_{\rm IR,template}/L_\odot)`.
        sed_in : array_like, shape (n_wave,)
            Input spectral luminosity density [erg/s/Hz].
        wave : array_like, shape (n_wave,)
            Rest-frame wavelength grid [Å].
        L_ir : array_like, scalar
            Total dust IR luminosity supplied by the model [erg/s].
        templates : mapping[str, array_like], optional
            Threaded template arrays returned by :meth:`load`.

        Returns
        -------
        tuple[ndarray, dict[str, ndarray]]
            Updated SED and ``sed_dust_ir`` [erg/s/Hz], each with shape
            ``(n_wave,)``.

        Notes
        -----
        **JIT-compatible**: yes when template arrays are passed. If they are
        omitted, the packaged data are loaded during the call. **Gradient-safe**:
        yes except at template luminosity nodes and wavelength interpolation
        knots, where the piecewise-linear interpolation has a derivative change.

        The source-shape interpolation is geometric between adjacent luminosity
        templates,

        .. math::

            H(\lambda) = H_k(\lambda)^{1-t} H_{k+1}(\lambda)^t,

        where :math:`t` is the fractional position between adjacent values of
        :math:`\log_{10}(L_{\rm IR,template}/L_\odot)`. The resulting curve is
        linearly interpolated in wavelength and normalized on the model grid:

        .. math::

            L_\nu(\lambda) = L_{\rm IR}\,
            \frac{H_G(\lambda)}{\int_G H_G\,d\nu}.

        Here :math:`H_G` is the interpolated table shape, :math:`L_{\rm IR}` is
        the total dust luminosity supplied by the upstream model budget [erg/s],
        and :math:`d\nu` is frequency [Hz]. The component adds no stellar
        attenuation screen. When ``dust_log_L_ir`` is explicitly declared, it
        supplies an independent total IR amplitude; otherwise the upstream
        energy-balance budget is used. For adjacent templates, entries that are
        zero in both rows remain zero; the established :math:`10^{-30}`
        relative-flux floor is used inside the logarithm where one row is zero.
        The converter retains the original source rows and clips only the 18
        finite negative shortwave subtraction residuals in the six templates
        with the lowest luminosities for runtime interpolation.

        References
        ----------
        .. [1] G. H. Rieke et al., "Determining Star Formation Rates for
           Infrared Galaxies," ApJ, 692, 556–573 (2009). arXiv:0810.4150.
           https://doi.org/10.1088/0004-637X/692/1/556
        .. [2] J. Lyu, S. Alberts, G. H. Rieke, and W. Rujopakarn, "AGN Selection
           and Demographics in GOODS-S/HUDF from X-ray to Radio," ApJ, 941, 191
           (2022). arXiv:2209.06219. https://doi.org/10.3847/1538-4357/ac9e5d
        """
        if templates is None:
            templates = self.load()

        luminosity_axis = jnp.asarray(templates["log_L_ir_template"])
        wavelength_aa = jnp.asarray(templates["wavelength_aa"])
        flux_nu_relative = jnp.asarray(templates["flux_nu_relative"])

        # Bound interpolation coordinates; zero is a valid endpoint, not a flux floor.
        position = jnp.clip(
            (p["log_L_ir_template"] - luminosity_axis[0])
            / (luminosity_axis[1] - luminosity_axis[0]),
            0.0,
            luminosity_axis.size - 1.0,
        )
        lower = jnp.minimum(jnp.floor(position).astype(jnp.int32), luminosity_axis.size - 2)
        fraction = jnp.clip(position - lower, 0.0, 1.0)
        flux_lower = flux_nu_relative[lower]
        flux_upper = flux_nu_relative[lower + 1]
        shape = jnp.exp(
            (1.0 - fraction) * jnp.log(jnp.maximum(flux_lower, _RELATIVE_FLUX_FLOOR))
            + fraction * jnp.log(jnp.maximum(flux_upper, _RELATIVE_FLUX_FLOOR))
        )
        shape = jnp.where((flux_lower == 0.0) & (flux_upper == 0.0), 0.0, shape)
        shape = jnp.interp(wave, wavelength_aa, shape, left=0.0, right=0.0)

        integral = integrate_lnu_over_nu(shape, wave)
        sed_ir = shape * (jnp.asarray(L_ir) / integral)
        return sed_in + sed_ir, {"sed_dust_ir": sed_ir}
