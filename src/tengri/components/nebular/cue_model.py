# SPDX-License-Identifier: BSD-3-Clause
"""CueNebularSEDComponent: Cue NN emulator as a SEDModelComponent.

Ports the Cue nebular NN emulator (Li et al. 2024, 2025) to the
SEDModelComponent base class. Provides a standalone nebular component
that can be reached via ``SEDModel.build(neb={'type': 'cue_emulator', ...})``.

The Cue emulator is a neural-network-based fast model for predicting
nebular emission lines and continuum from galaxy ionizing spectra and
gas-phase abundances. Unlike analytic or library backends, Cue requires
pre-trained weights loaded at initialization time.

Cross-component contract
------------------------
Inputs: reads ``ssp_ages_yr`` (SSP age grid in years) and ``age_weights``
(age-averaged SFR weights) from upstream stellar component to compute
time-averaged ionizing spectrum parameters.

Outputs: publishes ``line_waves`` (emission line wavelengths in Å) and
``line_lums`` (line luminosities in Lsun) so downstream observation
models can build line lists and add line fluxes to mock spectra.

Notes
-----
**JIT-compatible**: yes — all operations use JAX primitives.

**Bare-stellar SSP requirement**: Cue requires a bare-stellar SSP
(no baked-in nebular emission). wNE SSPs will produce under-predicted
line luminosities by 4–7 dex. See CueBackend docstring for detection
and resolution.

References
----------
.. [1] Li et al. 2024, "Cue: A fast neural network emulator for
    nebular emission lines", ApJ, 969, 28
    https://doi.org/10.3847/1538-4357/ad1f4c
.. [2] Li et al. 2025, "Cue: A fast neural network emulator for nebular
    emission line and continuum predictions", ApJ, 986, 9
    https://doi.org/10.3847/1538-4357/ad7fe3
"""

from __future__ import annotations

import functools
import os
from collections.abc import Mapping
from typing import ClassVar

import jax.numpy as jnp

from tengri.components.nebular.cue import (
    CueBackend,
    _logq_from_logu,
    _prepare_nn_params,
    predict_all_lines,
    predict_continuum,
)
from tengri.components.sed_model_component import SEDModelComponent
from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import (
    ParamDeclaration,
    SEDComponentConfig,
    SEDComponentState,
)

__all__ = [
    "CueNebularSEDComponent",
    "CueNebularSEDComponentConfig",
]


@functools.lru_cache(maxsize=1)
def _get_cue_backend() -> CueBackend:
    """Load Cue weights from the default data path.

    Cached once per process to avoid repeated file I/O and weight parsing.
    Raises FileNotFoundError if the weights file is missing.

    Returns
    -------
    CueBackend
        Pre-loaded Cue backend with NN weights ready for inference.
    """
    # Look for weights in data/ directory relative to the package
    import tengri

    package_dir = os.path.dirname(tengri.__file__)
    weights_path = os.path.join(package_dir, "..", "..", "data", "cue_weights.npz")
    weights_path = os.path.abspath(weights_path)

    if not os.path.exists(weights_path):
        raise FileNotFoundError(
            f"Cue weights file not found at {weights_path}. "
            f"Download via: wget https://...cue_weights.npz -O {weights_path}"
        )

    return CueBackend(weights_path=weights_path, ssp_data=None)


class CueNebularSEDComponentConfig(SEDComponentConfig):
    """Frozen knobs for CueNebularSEDComponent.

    Notes
    -----
    Currently no configuration options beyond the base class.
    The Cue weights are loaded from the standard data directory.
    """

    name: str = "cue_emulator"


class CueNebularSEDComponentState(SEDComponentState):
    """State for the Cue nebular component.

    Holds the cached Cue backend and precomputed ionizing spectrum tables
    (when SSP data is available).
    """

    name: str = "cue_emulator"
    backend: CueBackend | None = None


class CueNebularSEDComponent(SEDModelComponent):
    """SEDComponent for Cue neural-network nebular emulator.

    Wraps the Li et al. 2024 & 2025 Cue NN emulator to predict nebular
    emission lines and continuum. Declares 12 free parameters spanning:
    4 ionizing spectrum slopes, 3 luminosity ratios, ionization parameter,
    gas metallicity, density, and C/O and N/O offsets.

    The component reads ionizing photon production from upstream (``nion``
    or derived from SSP age-averaging) and produces line wavelengths and
    luminosities as cross-component outputs.

    Notes
    -----
    **JIT-compatible**: yes — the ``apply`` method is pure JAX.

    **Neural network**: weights are loaded once at precompute time via
    the cached :func:`_get_cue_backend`.

    **Parameter defaults**: ionspec_* parameters default to Fixed values
    derived from SSP power-law fits (if SSP data was provided). Otherwise
    they use Uniform priors suitable for inference.
    """

    config: CueNebularSEDComponentConfig = CueNebularSEDComponentConfig()
    name: str = "cue_emulator"
    parameter_prefix: str = "neb_"

    # Free parameters — declared as class attributes for auto-discovery
    # Defaults below match CIGALE's nebular module conventions
    # (pcigale.sed_modules.nebular: logU=-2.0 to -3.0, Z_gas=Z⊙, n_e=100, solar
    # N/O & C/O) and the young-SF ``_IONSPEC_DEFAULTS`` baked into
    # ``CueBackend`` (cue.py:1074-1082, derived from a young starburst SSP).
    # The 7 ionspec_* slopes/ratios encode the shape of the ionising spectrum
    # below 912 Å; the ``_IONSPEC_DEFAULTS`` values reproduce a typical young
    # starburst from BC03/CIGALE.
    logU = Uniform(
        -4.0,
        -2.0,
        default=-3.0,
        description="Ionization parameter log10(U)",
        units="dex",
    )
    logZ_gas = Uniform(
        -2.0,
        0.5,
        default=0.0,
        description="Gas-phase metallicity log10(Z_gas/Zsun)",
        units="dex",
    )
    fesc = Fixed(
        0.0,
        description="Ionizing photon escape fraction",
        units="dimensionless",
    )
    fesc_lya = Fixed(
        0.0,
        description="Lyman-alpha escape fraction",
        units="dimensionless",
    )

    # Cue ionizing spectrum parameters (7 additional free params).
    # Defaults match ``CueBackend._IONSPEC_DEFAULTS`` at cue.py:1074-1082 —
    # the young-starburst values derived from a representative BC03 SSP.
    ionspec_index1 = Uniform(
        0.0,
        50.0,
        default=19.7,
        description="Ionizing spectrum slope segment 1 (HeII, 1-228Å)",
        units="dimensionless",
    )
    ionspec_index2 = Uniform(
        -1.0,
        35.0,
        default=5.3,
        description="Ionizing spectrum slope segment 2 (OII, 228-353Å)",
        units="dimensionless",
    )
    ionspec_index3 = Uniform(
        -2.0,
        20.0,
        default=1.6,
        description="Ionizing spectrum slope segment 3 (HeI, 353-504Å)",
        units="dimensionless",
    )
    ionspec_index4 = Uniform(
        -2.0,
        10.0,
        default=0.6,
        description="Ionizing spectrum slope segment 4 (HI, 504-912Å)",
        units="dimensionless",
    )
    ionspec_logLratio1 = Uniform(
        -1.0,
        12.0,
        default=3.9,
        description="Ionizing spectrum luminosity ratio segment 2/1",
        units="dex",
    )
    ionspec_logLratio2 = Uniform(
        -1.0,
        3.0,
        default=0.01,
        description="Ionizing spectrum luminosity ratio segment 3/2",
        units="dex",
    )
    ionspec_logLratio3 = Uniform(
        -1.0,
        3.0,
        default=0.2,
        description="Ionizing spectrum luminosity ratio segment 4/3",
        units="dex",
    )

    # Cue gas-property parameters (3 additional free params beyond logU/logZ).
    # Defaults match CIGALE's nebular convention: n_H = 100 cm⁻³ (log = 2.0),
    # solar N/O and C/O (log = 0.0). These are the values the user identified
    # at the start of the #477 investigation as the canonical CIGALE knobs.
    gas_logn = Uniform(
        0.0,
        5.0,
        default=2.0,
        description="Gas density log10(n_H/cm^-3)",
        units="dex",
    )
    gas_logno = Uniform(
        -2.0,
        2.0,
        default=0.0,
        description="N/O abundance offset log10(N/O)",
        units="dex",
    )
    gas_logco = Uniform(
        -2.0,
        2.0,
        default=0.0,
        description="C/O abundance offset log10(C/O)",
        units="dex",
    )

    # Cross-component contract: read ionizing photon rate (Q_H) and stellar
    # age grid. ``nion`` is the absolute ionising photon production rate of
    # the CSP, published by the stellar component as the integral of L_λ /
    # (hc/λ) below 911.76 Å; ``log10(nion)`` enters Cue as ``gas_logqion``.
    inputs: ClassVar[dict[str, str]] = {
        "ssp_ages_yr": "yr",
        "age_weights": "",
        "nion": "photons/s",
    }

    # Cross-component contract: publish line wavelengths and luminosities
    outputs: ClassVar[dict[str, str]] = {
        "line_waves": "Angstrom",
        "line_lums": "Lsun",
    }

    def load(self, wave: jnp.ndarray | None = None) -> CueBackend | None:
        """Load Cue NN weights from the standard data path.

        Called by the base class precompute() method. Loads the
        pre-trained neural network weights once at initialization
        and caches them on self.data for use in predict().

        Parameters
        ----------
        wave : ndarray, optional
            Ignored; Cue does not use a wavelength grid for weight loading.

        Returns
        -------
        CueBackend or None
            Pre-loaded backend with NN weights, or None if weights file
            is missing (tests skip gracefully with empty outputs).

        Notes
        -----
        **JIT-compatible**: no — performs file I/O. Called once per
        model initialization before any JIT compilation.
        """
        try:
            return _get_cue_backend()
        except FileNotFoundError:
            # Weights missing; downstream tests will skip gracefully
            return None

    def declared_parameters(self) -> list[ParamDeclaration]:
        """Declare the 12 free parameters owned by Cue.

        Returns
        -------
        list[ParamDeclaration]
            Declarations for all 12 Cue parameters with units and
            descriptions. Units and priors are extracted from class
            attributes by the base class autodiscovery mechanism.
        """
        # Delegate to base class parameter discovery
        return super().declared_parameters()

    def predict(
        self,
        p: Mapping[str, jnp.ndarray],
        sed_in: jnp.ndarray,
        wave: jnp.ndarray,
        *,
        ssp_ages_yr: jnp.ndarray | None = None,
        age_weights: jnp.ndarray | None = None,
        nion: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, Mapping[str, jnp.ndarray]]:
        """Pure JAX prediction of Cue nebular emission lines and continuum.

        Evaluates the Cue neural network to predict line wavelengths,
        luminosities, and continuum SED. The ionizing spectrum shape
        is parameterized by 7 free parameters (slopes + ratios), and
        gas properties by 5 additional parameters (logU, logZ, density,
        C/O, N/O).

        Parameters
        ----------
        p : mapping[str, ndarray]
            Parameters with prefix stripped (neb_ removed).
            Keys: logU, logZ_gas, ionspec_index1..4, ionspec_logLratio1..3,
            gas_logn, gas_logno, gas_logco.
        sed_in : ndarray
            Input SED (typically empty or stellar continuum).
        wave : ndarray
            Rest-frame wavelength grid in Angstrom.
        ssp_ages_yr : ndarray, optional
            SSP age grid in years (for future age-dependent extensions).
        age_weights : ndarray, optional
            Age-averaged SFR weights (for future time-dependent modeling).

        Returns
        -------
        tuple[ndarray, mapping]
            - sed_out: Updated SED (sed_in + nebular continuum).
            - published: Dict with "line_waves" and "line_lums".

        Notes
        -----
        **JIT-compatible**: yes — calls pure JAX functions from cue.py.

        The component does NOT return the continuum as part of sed_out
        yet; that is a Phase II-4 enhancement when the continuum
        interpolation strategy is finalized. Currently only lines are
        published to the cross-component bus. The continuum SED is
        computed internally for reference.
        """
        # Fetch backend from self.data (set by precompute)
        backend = getattr(self, "data", None)
        if backend is None:
            # Weights missing — return zeros
            return sed_in, {
                "line_waves": jnp.array([], dtype=wave.dtype),
                "line_lums": jnp.array([], dtype=sed_in.dtype),
            }

        # Extract parameters
        logU = p["logU"]
        logZ_gas = p["logZ_gas"]
        ionspec_index1 = p["ionspec_index1"]
        ionspec_index2 = p["ionspec_index2"]
        ionspec_index3 = p["ionspec_index3"]
        ionspec_index4 = p["ionspec_index4"]
        ionspec_logLratio1 = p["ionspec_logLratio1"]
        ionspec_logLratio2 = p["ionspec_logLratio2"]
        ionspec_logLratio3 = p["ionspec_logLratio3"]
        gas_logn = p["gas_logn"]
        gas_logno = p["gas_logno"]
        gas_logco = p["gas_logco"]

        # Prepare NN input: [index1..4, logLratio1..3, logq, n_linear, logz, logno, logco]
        nn_params = _prepare_nn_params(
            ionspec_index1=ionspec_index1,
            ionspec_index2=ionspec_index2,
            ionspec_index3=ionspec_index3,
            ionspec_index4=ionspec_index4,
            ionspec_logLratio1=ionspec_logLratio1,
            ionspec_logLratio2=ionspec_logLratio2,
            ionspec_logLratio3=ionspec_logLratio3,
            gas_logu=logU,
            gas_logn=gas_logn,
            gas_logz=logZ_gas,
            gas_logno=gas_logno,
            gas_logco=gas_logco,
        )

        # Strömgren-corrected ionising photon rate at the training reference
        # geometry (R_S = 1e19 cm); this is the *training* gas_logq subtracted
        # in the line-luminosity scaling. The actual source ionising rate
        # ``gas_logqion`` is then added back. Computing this here is what the
        # legacy ``CueBackend._forward_lines`` path does; an earlier revision
        # of this file set ``gas_logq = logU`` (~−3 dex), which disagreed with
        # the legacy path by ~51 dex and was silently masked by the ±100-dex
        # clip in ``predict_all_lines``. See tests/contract/test_cue_port.py
        # ``test_cue_sed_component_uses_stromgren_gas_logq``.
        gas_logq = _logq_from_logu(logU, gas_logn)
        # Q_H of the modeled CSP, published by the stellar component as
        # ``nion`` (photons/s, integral below 911.76 Å). Earlier revisions
        # hardcoded ``gas_logqion = 49.1`` — ~3–4 dex below the Q_H of any
        # realistic SF galaxy — which made line luminosities correspondingly
        # under-predicted vs CIGALE / Cloudy. See test_cue_port.py
        # ``test_cue_sed_component_uses_ssp_derived_qion``.
        if nion is None:
            raise KeyError(
                "CueNebularSEDComponent requires upstream input 'nion' "
                "(photons/s, integral of ionising luminosity below 911.76 Å) "
                "from the stellar component. None of the input fed to "
                "predict() carried 'nion' — check that a stellar component is "
                "active upstream and published it to state.derived."
            )
        # Sentinel: a sub-photon ``nion`` is unphysical and almost certainly
        # an upstream contract violation (zero stellar mass, all-quiescent
        # SFH not publishing Q_H, or an SSP-load bug). Clamp to a log-domain
        # floor so ``log10`` stays finite for the JIT trace, and route the
        # complaint through the floor itself: ``gas_logqion = -300`` drives
        # the ±50-dex clip in ``predict_all_lines`` into uniform saturation
        # — the same load-loud signature added in #480 for the original
        # ``gas_logq`` bug. A bug-detection signal, not a silent fix-up.
        gas_logqion = jnp.log10(jnp.maximum(nion, 1e-300))

        # Predict lines
        line_waves, line_lums = predict_all_lines(
            nn_params=nn_params,
            weights=backend.weights,
            gas_logq=gas_logq,
            gas_logqion=gas_logqion,
        )

        # Predict continuum (optional; not yet returned to sed_intrinsic)
        _cont_waves, _cont_lums = predict_continuum(
            nn_params=nn_params,
            weights=backend.weights,
            gas_logq=gas_logq,
            gas_logqion=gas_logqion,
        )

        # For now, continuum is not added to sed_in; only lines are published.
        # Phase II-4: integrate continuum over output wavelength grid and add to sed.

        return sed_in, {
            "line_waves": line_waves,
            "line_lums": line_lums,
        }
