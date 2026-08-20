# SPDX-License-Identifier: BSD-3-Clause
"""SKIRTOR clumpy torus (Stalevski et al. 2012, 2016) SEDModelComponent.

Implements the SKIRTOR clumpy torus model on the SEDModelComponent contract,
enabling use of radiative-transfer templates in the model-building API.

This is an opt-in adapter — the existing AGNSEDComponent continues to
support SKIRTOR through the unified AGN registry.

References
----------
.. [1] M. Stalevski et al., "3D radiative transfer modeling of the dusty
   torus around AGN — the influence of clumping," MNRAS, 420, 2756 (2012).
   arXiv:1109.1286. https://doi.org/10.1111/j.1365-2966.2011.19775.x
.. [2] M. Stalevski et al., "The dust covering factor in AGN — combining the
   IR torus emission with polar dust component," MNRAS, 458, 2288 (2016).
   arXiv:1602.01954. https://doi.org/10.1093/mnras/stw444
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, ClassVar

import jax.numpy as jnp

from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Uniform
from tengri.protocols.component import SEDComponentConfig, SEDComponentState

__all__ = ["SKIRTORTorus"]


@dataclass(frozen=True)
class SKIRTORTorusConfig(SEDComponentConfig):
    """Configuration for SKIRTOR torus templates.

    Parameters
    ----------
    grid_path : str or None
        Path to SKIRTOR template grid (.npz or .h5). If None, templates
        are not pre-loaded (deferred to first use in predict).
    disk_type : int
        Disc spectrum model selector (CIGALE ``skirtor2016.py`` ``disk_type``).
        Options:

        - 0: SKIRTOR intrinsic disc (Stalevski et al. 2012)
        - 1: Schartmann et al. (2005) disc
        - 2: ADAF -> thin-disc blend (Lopez et al. 2024)

        Default: 0. tengri's SKIRTOR template grid bundles the SKIRTOR
        *intrinsic* disc, so ``disk_type=0`` with ``delta=0`` reproduces the
        tabulated disc bit-for-bit. ``disk_type`` re-tilts the disc spectral
        shape relative to that intrinsic disc (CIGALE's module default is 1
        (Schartmann); set ``disk_type=1`` to match it).
    """

    grid_path: str | None = None
    disk_type: int = 0  # 0 reproduces the SKIRTOR-intrinsic template disc bit-exactly


@dataclass(frozen=True)
class SKIRTORTorusState(SEDComponentState):
    """Cached SKIRTOR template data.

    Attributes
    ----------
    name : str
        Component identifier.
    skirtor_fn : callable or None
        Compiled interpolation function from create_skirtor_from_grid,
        or None if templates are not available.
    """

    name: str = "skirtor"
    skirtor_fn: Any | None = None


@dataclass(frozen=True)
class SKIRTORTorus(SEDModelComponent):
    """Clumpy torus SED from SKIRTOR radiative-transfer models.

    A pure-JAX implementation with C²-continuous gradients via triweight
    kernel interpolation in the 5D parameter space (tau, p, q, opening angle,
    inclination). Publishes separate disc and torus contributions, with
    polar dust wire-in for Type 1 sightlines.

    Attributes
    ----------
    name : str
        Component registry key: ``"skirtor"``.
    parameter_prefix : str
        Parameter namespace: ``"agn_"``.
    config : SKIRTORTorusConfig
        Frozen configuration (grid path).

    Free parameters (class-level declarations, auto-discovered)
    -----------------------------------------------------------
    log_lbol : Uniform
        log₁₀(L_bol / L_sun). [dex, 8–14]
    tau_skirtor : Uniform
        Edge-on optical depth at 9.7 μm. [dimensionless, 3–11]
    p_skirtor : Uniform
        Radial dust density power-law gradient. [dimensionless, 0–1.5]
    q_skirtor : Uniform
        Polar dust density power-law gradient. [dimensionless, 0–1.5]
    oa_skirtor : Uniform
        Torus half-opening angle. [degrees, 20–60]
    cos_inc : Uniform
        Cosine of inclination (1 = face-on, 0 = edge-on). [dimensionless, 0–1]
    frac_agn : Uniform
        AGN fraction in a configurable band (CIGALE convention).
        [dimensionless, 0–1]
    polar_ebv : Uniform
        Polar dust E(B-V) (Type-1 sightline only). [mag, 0–0.5]
    polar_temperature : Uniform
        Polar dust graybody temperature. [K, 50–200]
    polar_beta : Uniform
        Polar dust emissivity index (Casey 2012 modified blackbody).
        [dimensionless, 1–2.5]
    delta : Uniform
        Disc spectral slope modulation (CIGALE ``skirtor2016`` delta).
        [dimensionless, -1.0–1.0]. For ``disk_type`` 0/1 it tilts the disc
        power-law index at 100–5000 Å via α_mid = -1.5 + delta; for
        ``disk_type=2`` it is the ADAF->thin-disc blend weight (clipped to
        [0, 1]). Default 0.0 (no modulation).

    Cross-component outputs
    -----------------------
    L_agn_disc : erg/s
        Bolometric luminosity from accretion disc (intrinsic, at θ=30°).
    L_agn_torus : erg/s
        Bolometric luminosity from torus dust thermal emission.
    L_agn_polar_dust : erg/s
        Bolometric luminosity from polar dust reemission (Type 1 only).
    L_2500_30deg : erg/s/Hz
        Specific luminosity at 2500 Å, θ=30°; feeds X-ray normalization.
    L_6um : erg/s/Hz
        Specific luminosity at 6 μm for mid-IR diagnostics.
    L_12um : erg/s/Hz
        Specific luminosity at 12 μm for mid-IR diagnostics.

    Notes
    -----
    **JIT-compatible**: yes — predict() is pure JAX.

    **Gradient-safe**: yes — triweight interpolation is fully differentiable.

    **Requires template grid**: The SKIRTOR template library (~1 GB) must be
    downloaded separately and pointed to via ``grid_path`` in config. The
    predict method gracefully returns zero emission if templates are unavailable.

    **Polar dust**: Applied to Type 1 sightlines (cos_inc ≥ cos(90° - oa))
    via the smooth sigmoid from polar_dust.py. Energy-conserving reemission
    as Casey-2012 modified blackbody.

    **Citation**: Stalevski et al. 2016 (SKIRTOR); Yang et al. 2020, §2.2.2
    (polar dust + anisotropy).

    Examples
    --------
    Minimal model with SKIRTOR torus::

        from tengri import SEDModel, Fixed, Uniform, builders
        from tengri.components.agn.skirtor_model import SKIRTORTorus

        # Register and use
        model = SEDModel.build(
            ssp_data=ssp,
            observation=obs,
            sfh=builders.sfh.dpl(_=Fixed(1.5), beta=Fixed(1.0)),
            dust={"type": "two_component", "*": Fixed},
            agn=SKIRTORTorus(config=SKIRTORTorusConfig(grid_path="path/to/grid.h5")),
        )

    See Also
    --------
    tengri.components.agn.skirtor : template loader and interpolation.
    """

    name = "skirtor"
    parameter_prefix = "agn_"
    config: SKIRTORTorusConfig = field(default_factory=SKIRTORTorusConfig)

    # Free parameters — auto-discovered
    log_lbol = Uniform(
        8.0,
        14.0,
        description="AGN bolometric luminosity",
        units="dex (L_sun)",
        default=11.0,
    )
    tau_skirtor = Uniform(
        3.0,
        11.0,
        description="9.7 µm optical depth (Stalevski et al.)",
        units="dimensionless",
        default=7.0,
    )
    p_skirtor = Uniform(
        0.0,
        1.5,
        description="Radial dust density gradient",
        units="dimensionless",
        default=1.0,
    )
    q_skirtor = Uniform(
        0.0,
        1.5,
        description="Polar dust density gradient",
        units="dimensionless",
        default=1.0,
    )
    oa_skirtor = Uniform(
        20.0,
        60.0,
        description="Torus half-opening angle",
        units="deg",
        default=40.0,
    )
    cos_inc = Uniform(
        0.0,
        1.0,
        description="Cosine of inclination",
        units="dimensionless",
        default=0.45,
    )
    band_frac = Uniform(
        0.0,
        1.0,
        description="AGN fraction (L_AGN / L_total, CIGALE convention)",
        units="dimensionless",
        default=0.2,
    )
    polar_ebv = Uniform(
        0.0,
        0.5,
        description="Polar dust E(B-V) (Type-1 sightline)",
        units="mag",
        default=0.1,
    )
    polar_temperature = Uniform(
        50.0,
        200.0,
        description="Polar dust graybody temperature",
        units="K",
        default=100.0,
    )
    polar_beta = Uniform(
        1.0,
        2.5,
        description="Polar dust emissivity index",
        units="dimensionless",
        default=1.6,
    )
    delta = Uniform(
        -1.0,
        1.0,
        description="Disc spectral slope modulation delta (CIGALE skirtor2016). "
        "For disk_type 0/1 it tilts the optical-MIR disc slope; for disk_type 2 "
        "it is the ADAF->thin-disc blend weight (clipped to [0, 1]).",
        units="dimensionless",
        default=0.0,
    )
    # NOTE: CIGALE's ``lambda_fracAGN`` (band over which the AGN fraction is
    # normalized) is intentionally NOT exposed here. tengri normalizes frac_agn
    # bolometrically, which is exactly CIGALE's default ("0/0"). A band-restricted
    # normalization is a documented follow-up (umbrella audit issue) rather than a
    # dead Fixed knob.

    # Cross-component outputs
    outputs: ClassVar[dict[str, str]] = {
        "L_agn_disc": "erg/s",
        "L_agn_torus": "erg/s",
        "L_agn_polar_dust": "erg/s",
        "L_2500_30deg": "erg/s/Hz",
        "L_6um": "erg/s/Hz",
        "L_12um": "erg/s/Hz",
    }

    def load(self, wave: jnp.ndarray | None = None) -> Any | None:
        """Load SKIRTOR v3 template grid with separate components.

        Parameters
        ----------
        wave : ndarray, optional
            Rest-frame wavelength grid (not used by SKIRTOR; templates
            interpolate to any target grid).

        Returns
        -------
        callable or None
            Interpolation function from create_skirtor_components_from_grid
            (returns SKIRTORComponents), or None if ``grid_path`` is not set --
            i.e. no torus library was requested, so the component contributes
            zero emission by design.

        Raises
        ------
        FileNotFoundError
            If ``grid_path`` is set but the grid cannot be loaded. A named grid
            that fails to load is a user error, not a reason to fall back to
            zero emission: every torus parameter would become a silent no-op and
            the fit would report an AGN torus contributing exactly nothing.
        """
        from tengri.components.agn.skirtor import create_skirtor_components_from_grid

        if not self.config.grid_path:
            return None

        try:
            return create_skirtor_components_from_grid(self.config.grid_path)
        except (FileNotFoundError, OSError, KeyError) as exc:
            raise FileNotFoundError(
                f"SKIRTOR torus grid could not be loaded from {self.config.grid_path!r} "
                f"({type(exc).__name__}: {exc}). A grid was requested via grid_path, so "
                "tengri will not silently fall back to zero AGN emission. Fix the path, "
                "or leave grid_path unset to run without a torus. Templates are available "
                "from https://sites.google.com/site/skirtorus/sed-library"
            ) from exc

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        **inputs: Any,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX SKIRTOR prediction with separate components and polar dust.

        Interpolates the SKIRTOR template grid, applies polar dust extinction
        to Type 1 sightlines, and publishes all derived luminosities. Supports
        multiple disc spectrum models via the config.disk_type parameter.

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix already stripped:

            - log_lbol: log₁₀(L_bol / L_sun)
            - tau_skirtor: optical depth at 9.7 µm
            - p_skirtor: radial density gradient
            - q_skirtor: polar density gradient
            - oa_skirtor: opening angle (degrees)
            - cos_inc: cosine of inclination
            - frac_agn: AGN luminosity fraction
            - delta: disc spectral slope modulation (-1.0 to 1.0)
            - polar_ebv: polar dust E(B-V)
            - polar_temperature: polar dust graybody temperature (K)
            - polar_beta: polar dust emissivity index

        sed_in : ndarray, shape (n_wave,)
            Input SED in erg/s/Hz.
        wave : ndarray, shape (n_wave,)
            Rest-frame wavelength grid in Angstrom.
        **inputs : ndarray
            Unused (AGN torus is self-contained).

        Returns
        -------
        tuple[ndarray, dict]
            (sed_out, published) where:

            - sed_out: Updated SED (sed_in + attenuated disc + torus + polar dust reemission).
            - published: dict with keys L_agn_disc, L_agn_torus, L_agn_polar_dust,
              L_2500_30deg, L_6um, L_12um [erg/s] or [erg/s/Hz].

        Notes
        -----
        **JIT-compatible**: yes — uses static disk_type (not traced).

        **Disc selection**: config.disk_type selects the disc spectrum model:

        - 0: SKIRTOR intrinsic disc (Stalevski et al. 2012)
        - 1: Schartmann et al. (2005) torus model (CIGALE default)
        - 2: ADAF + thin disc blend (Lopez et al. 2024)

        **Polar dust model**: The X-CIGALE polar dust model (Yang et al. 2020,
        §2.2.2) applies a Type-1/Type-2 mask to the observer-frame disc attenuation.
        However, the absorbed luminosity driving the graybody FIR reemission is
        viewing-angle-independent (bi-conical geometry). This means both Type 1
        (face-on) and Type 2 (edge-on) sightlines see the FIR bump in the combined SED.
        """
        from tengri.components.agn._phys import bolometric_integral_nu, wavelength_to_nu
        from tengri.components.agn.disc_cigale import (
            adaf_disk_spectrum,
            schartmann2005_disk_spectrum,
            skirtor_disk_spectrum,
        )
        from tengri.components.agn.polar_dust import (
            polar_dust_emission,
            polar_dust_extinction,
        )

        # If templates are not loaded, return zero emission
        if not hasattr(self, "data") or self.data is None:
            zero_dict = {
                "L_agn_disc": jnp.array(0.0),
                "L_agn_torus": jnp.array(0.0),
                "L_agn_polar_dust": jnp.array(0.0),
                "L_2500_30deg": jnp.array(0.0),
                "L_6um": jnp.array(0.0),
                "L_12um": jnp.array(0.0),
            }
            return sed_in, zero_dict

        skirtor_fn = self.data

        # Call SKIRTOR interpolator to get separate components
        components = skirtor_fn(
            wavelength=wave,
            agn_log_lbol=p["log_lbol"],
            agn_tau_skirtor=p["tau_skirtor"],
            agn_p_skirtor=p["p_skirtor"],
            agn_q_skirtor=p["q_skirtor"],
            agn_oa_skirtor=p["oa_skirtor"],
            agn_cos_inc=p["cos_inc"],
            frac_agn=p["band_frac"],
        )

        # Unpack components
        sed_disc_template = components.disk
        sed_torus_dust = components.dust

        # Compute derived quantities from disc
        nu = wavelength_to_nu(wave)

        # L_agn_disc: bolometric luminosity of the intrinsic disc. Preserved
        # across the disc-shape selection below so frac_agn / energy balance is
        # unaffected by disk_type / delta (only the SED *shape* changes).
        L_agn_disc = bolometric_integral_nu(sed_disc_template, nu)

        # --- CIGALE disc-shape selection (skirtor2016.py:324-339) -----------
        # CIGALE builds the disc analytically and selects its shape via
        # ``disk_type`` and re-tilts it via ``delta``. tengri's grid carries the
        # SKIRTOR *intrinsic* disc, so we re-tilt the tabulated disc by the
        # ratio  (selected analytic disc) / (SKIRTOR analytic disc, delta=0).
        # At disk_type=0, delta=0 this ratio is identically 1 -> the tabulated
        # disc is reproduced bit-for-bit. The disc bolometric luminosity is then
        # restored to L_agn_disc so only the spectral shape is modified.
        #
        # ``disk_type`` is a STATIC structural choice (resolved at trace time, no
        # branch on a traced value); ``delta`` is a differentiable free param.
        disk_type = int(self.config.disk_type)  # static
        delta = p["delta"]
        wave_nm = wave / 10.0  # disc_cigale functions take nm
        if disk_type == 0:
            shape_sel = skirtor_disk_spectrum(wave_nm, delta=delta)
        elif disk_type == 1:
            shape_sel = schartmann2005_disk_spectrum(wave_nm, delta=delta)
        elif disk_type == 2:
            shape_sel = adaf_disk_spectrum(wave_nm, delta=delta)
        else:
            raise ValueError(
                f"disk_type must be 0 (SKIRTOR), 1 (Schartmann2005), or 2 "
                f"(ADAF/Lopez24); got {disk_type!r}."
            )
        shape_ref = skirtor_disk_spectrum(wave_nm, delta=0.0)
        # Re-tilt factor (unit-area / lambda-vs-nu normalizations cancel in the
        # ratio). Floor the denominator to stay finite where the disc is ~0.
        retilt = shape_sel / jnp.maximum(shape_ref, 1e-100)
        sed_disc = sed_disc_template * retilt
        # Restore the disc bolometric luminosity (shape-only change).
        L_retilt_safe = bolometric_integral_nu(sed_disc, nu, floor=1e-100)
        sed_disc = sed_disc * (L_agn_disc / L_retilt_safe)

        # L_2500_30deg: specific luminosity at 2500 Å (for α_OX)
        L_2500 = jnp.interp(2500.0, wave, sed_disc)

        # L_6um and L_12um: mid-IR diagnostics
        L_6um = jnp.interp(60000.0, wave, sed_disc + sed_torus_dust)  # 6 um = 60000 A
        L_12um = jnp.interp(120000.0, wave, sed_disc + sed_torus_dust)  # 12 um = 120000 A

        # L_agn_torus: bolometric torus dust luminosity
        L_agn_torus = bolometric_integral_nu(sed_torus_dust, nu)

        # Apply polar dust (Type 1 only): extinction of disc, reemission
        sed_disc_polar, l_abs = polar_dust_extinction(
            sed_disc,
            wave,
            p["cos_inc"],
            p["oa_skirtor"],
            p["polar_ebv"],
            law="smc",
        )
        sed_polar_reemit = polar_dust_emission(
            bolometric_integral_nu(l_abs, nu),
            wave,
            temperature=p["polar_temperature"],
            beta=p["polar_beta"],
            lambda_0=2e6,
        )
        L_agn_polar_dust = bolometric_integral_nu(sed_polar_reemit, nu)

        # Total SED: attenuated disc + torus + polar reemission
        sed_out = sed_in + sed_disc_polar + sed_torus_dust + sed_polar_reemit

        published = {
            "L_agn_disc": L_agn_disc,
            "L_agn_torus": L_agn_torus,
            "L_agn_polar_dust": L_agn_polar_dust,
            "L_2500_30deg": L_2500,
            "L_6um": L_6um,
            "L_12um": L_12um,
        }

        return sed_out, published
