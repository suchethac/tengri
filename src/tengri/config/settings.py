# SPDX-License-Identifier: BSD-3-Clause
"""Frozen configuration dataclasses for sub-model selection.

These objects encode *which* physics modules are active: structural choices that
do NOT appear in the gradient tape.  They are distinct from fittable Parameters.

Usage
-----
::

    from tengri.config.settings import DustConfig, NebularConfig, SEDModelConfig

    cfg = SEDModelConfig(
        nebular=NebularConfig(backend="cloudy", grid_path="/data/cloudy.h5"),
        dust=DustConfig(law_bc="calzetti"),
    )

All fields have defaults so a bare ``SEDModelConfig()`` is equivalent to the
standard smooth parametric model.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any


def _validate_enum(value: Any, valid: Iterable, where: str) -> None:
    """Raise ``ValueError`` if ``value`` is outside ``valid``.

    ``where`` is the human label used in the error message (e.g.
    ``"AGNConfig.disc"``). ``None`` entries in ``valid`` are formatted
    as the string ``"None"`` in the suggestion list.
    """
    valid = frozenset(valid)
    if value not in valid:
        choices = sorted(str(v) for v in valid)
        raise ValueError(f"{where}={value!r} is not valid. Choose from: {choices}")


@dataclass(frozen=True)
class AGNConfig:
    """Static configuration for AGN sub-model selection.

    Parameters
    ----------
    disc: str
        AGN accretion disc model.
        ``"powerlaw"``; simple power-law SED.
        ``"multicolor"``: multi-color blackbody disc (default).
        ``"kubota_done"``; Kubota & Done (2018) 3-zone model.
        ``"adaf"``; ADAF (low-luminosity AGN).
    torus: str
        AGN torus/obscuration model.
        ``"simple"``; single-temperature MBB (toy).
        ``"two_temperature"``; two-temperature MBB (toy).
        ``"skirtor"``; SKIRTOR clumpy torus (default, science-grade).
    nlr: str
        Narrow Line Region emission model.
        ``"analytic"``; analytic Gaussian line profiles (default, fast).
        ``"cue"``; Cue neural emulator (physically consistent).
    blr: bool
        Include Broad Line Region emission (Type 1 AGN). Default True.
    polar_dust: bool
        Include SMC polar dust reddening. Default False.
    fe2: bool
        Include Fe II pseudo-continuum. Default False.
    agn_blr_enabled: bool
        Enable BLR Gaussian emitter (additive to disc SED). Default False.
        **Reserved**: declared for an upcoming AGN-nebular PR; no effect in
        the current version.
    agn_nlr_gaussian_enabled: bool
        Enable NLR Gaussian emitter (additive to disc SED). Default False.
        **Reserved**: declared for an upcoming AGN-nebular PR; no effect in
        the current version.
    agn_nlr_backend: str or None
        Enable Feltre NLR backend. Options: None (disabled), "feltre".
        Default None. Mutually exclusive with Cue path.
        **Reserved**: declared for an upcoming AGN-nebular PR; no effect in
        the current version.

    Attributes
    ----------
    disc: str
        AGN accretion disc model choice.
    torus: str
        AGN torus/obscuration model choice.
    nlr: str
        Narrow Line Region emission model choice.
    blr: bool
        Whether to include Broad Line Region emission.
    polar_dust: bool
        Whether to include SMC polar dust reddening.
    fe2: bool
        Whether to include Fe II pseudo-continuum.
    agn_blr_enabled: bool
        Whether to enable BLR Gaussian emitter.
    agn_nlr_gaussian_enabled: bool
        Whether to enable NLR Gaussian emitter.
    agn_nlr_backend: str or None
        Feltre backend choice for AGN NLR.

    Notes
    -----
    **JIT-compatible**: no, configuration object, frozen dataclass.

    Examples
    --------
    >>> from tengri.config import AGNConfig
    >>> cfg = AGNConfig()
    >>> cfg.disc, cfg.torus
    ('multicolor', 'skirtor')
    >>> cfg_kd = AGNConfig(disc="kubota_done", nlr="cue")
    >>> cfg_kd.disc
    'kubota_done'
    """

    disc: str = "multicolor"
    torus: str = "skirtor"
    nlr: str = "analytic"
    blr: bool = True
    polar_dust: bool = False
    fe2: bool = False
    agn_blr_enabled: bool = False
    agn_nlr_gaussian_enabled: bool = False
    agn_nlr_backend: str | None = None

    def __post_init__(self) -> None:
        _validate_enum(
            self.disc, {"powerlaw", "multicolor", "kubota_done", "adaf"}, "AGNConfig.disc"
        )
        _validate_enum(self.torus, {"simple", "two_temperature", "skirtor"}, "AGNConfig.torus")
        _validate_enum(self.nlr, {"analytic", "cue"}, "AGNConfig.nlr")
        _validate_enum(self.agn_nlr_backend, {None, "feltre"}, "AGNConfig.agn_nlr_backend")


@dataclass(frozen=True)
class SFHConfig:
    """SFH structural settings (non-parametric shape choices).

    Parameters
    ----------
    mean_type: list[str]
        Analytic mean-SFH components, e.g. ``["tsnorm"]``, ``["dpl"]``,
        ``["dpl", "field"]``.
    n_grid: int
        GP latent grid size (only relevant when ``"field"`` is in ``mean_type``).
        Default: 64.
    evolving_metallicity: bool
        Replace ``met_logzsol`` with a two-endpoint ramp.  Default: ``False``.
    alpha_fe_evolving: bool
        Enable [α/Fe] evolution with lookback time.  Default: ``False``.
    chem_evol: bool
        Derive Z(t) from SFH via gas-regulator model.  Default: ``False``.
    met_interp: str
        Metallicity interpolation method: ``"smooth"`` (triweight, default)
        or ``"linear"`` (FSPS/Prospector-compatible).
    lgmet_scatter: float
        Triweight kernel bandwidth in dex (``met_interp="smooth"``).
        Default: 0.1.

    Attributes
    ----------
    mean_type, n_grid, evolving_metallicity, alpha_fe_evolving, chem_evol,
    met_interp, lgmet_scatter
        All constructor parameters are read-only frozen attributes.

    Notes
    -----
    Frozen dataclass; all fields are immutable after construction. Pass to
    :class:`~tengri.forward.sed_model.SEDModel` via the ``sfh`` field of
    :class:`SEDModelConfig`. Changes require constructing a new instance.

    Examples
    --------
    >>> from tengri.config import SFHConfig
    >>> cfg = SFHConfig(mean_type=["dpl", "field"], n_grid=128)
    >>> cfg.mean_type
    ('dpl', 'field')
    """

    mean_type: tuple[str, ...] = ("dpl",)
    n_grid: int = 256
    evolving_metallicity: bool = False
    alpha_fe_evolving: bool = False
    chem_evol: bool = False
    met_interp: str = "smooth"
    lgmet_scatter: float = 0.1

    def __post_init__(self) -> None:
        _validate_enum(self.met_interp, {"smooth", "linear"}, "SFHConfig.met_interp")


@dataclass(frozen=True)
class DustConfig:
    """Dust attenuation and emission structural settings.

    Parameters
    ----------
    model: str
        Geometry model: ``"two_component"`` (Charlot & Fall, default) or
        ``"single_component"`` (uniform screen).
    law_bc: str
        Birth cloud attenuation law.  Default: ``"power_law"``.
        Options: ``power_law``, ``calzetti``, ``kriek_conroy``, ``smc``,
        ``cardelli``, ``salim``, ``li08``.
    law_diff: str or None
        Diffuse ISM attenuation law.  ``None`` = same as ``law_bc``.
    emission: str or None
        IR dust emission model.  ``None`` disables IR emission (default).
        Options: ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``.

    Attributes
    ----------
    model, law_bc, law_diff, emission
        All constructor parameters are read-only frozen attributes.

    Notes
    -----
    Frozen dataclass; all fields are immutable after construction. Validated
    in ``__post_init__``: invalid ``model``, ``law_bc``, ``law_diff``, or
    ``emission`` strings raise :exc:`ValueError` immediately. Pass to
    :class:`SEDModelConfig` via the ``dust`` field.

    Examples
    --------
    >>> from tengri.config import DustConfig
    >>> cfg = DustConfig(law_bc="calzetti", emission="dale2014")
    >>> cfg.law_bc
    'calzetti'
    """

    model: str = "two_component"
    law_bc: str = "power_law"
    law_diff: str | None = None
    emission: str | None = None

    def __post_init__(self) -> None:
        valid_laws = {
            "power_law",
            "calzetti",
            "kriek_conroy",
            "smc",
            "cardelli",
            "salim",
            "li08",
            "vw07_bc",
            "vw07_diff",
        }
        _validate_enum(self.model, {"two_component", "single_component"}, "DustConfig.model")
        _validate_enum(self.law_bc, valid_laws, "DustConfig.law_bc")
        if self.law_diff is not None:
            _validate_enum(self.law_diff, valid_laws, "DustConfig.law_diff")
        valid_emission = {
            None,
            "modified_blackbody",
            "casey2012",
            "dale2014",
            "draine_li2007",
            "draine_li2014",
        }
        _validate_enum(self.emission, valid_emission, "DustConfig.emission")


@dataclass(frozen=True)
class NebularConfig:
    """Nebular emission structural settings.

    Parameters
    ----------
    backend: str
        Nebular emission backend.
        ``"off"``; disabled (default).
        ``"baked_in"``; lines from SSP grid (no free nebular params).
        ``"cloudy"``; CLOUDY grid interpolation.
        ``"cue"``: Cue neural emulator.
    grid_path: str or None
        Path to CLOUDY HDF5 grid (required when ``backend="cloudy"``).
    weights_path: str or None
        Override default Cue weights path (only for ``backend="cue"``).
    ionization: str
        Ionization source for Cue: ``"ssp"`` (default).
    eline_mode: str
        Emission line fitting mode.
        ``"off"``; no line treatment (default).
        ``"fixed"``; fixed profiles.
        ``"marginalized"``: analytic marginalization.
    eline_broad: bool
        Enable broad AGN emission line component.  Default: ``False``.

    Attributes
    ----------
    backend, grid_path, weights_path, ionization, eline_mode, eline_broad
        All constructor parameters are read-only frozen attributes.

    Notes
    -----
    Frozen dataclass: all fields are immutable after construction. Validated
    in ``__post_init__``: unsupported ``backend`` or ``eline_mode`` strings raise
    :exc:`ValueError`; ``grid_path`` is required when ``backend="cloudy"``.
    Pass to :class:`SEDModelConfig` via the ``nebular`` field.

    Examples
    --------
    >>> from tengri.config import NebularConfig
    >>> cfg = NebularConfig(backend="cue")
    >>> cfg.backend
    'cue'
    """

    backend: str = "off"
    grid_path: str | None = None
    weights_path: str | None = None
    ionization: str = "ssp"
    eline_mode: str = "off"
    eline_broad: bool = False

    def __post_init__(self) -> None:
        _validate_enum(self.backend, {"off", "baked_in", "cloudy", "cue"}, "NebularConfig.backend")
        _validate_enum(
            self.eline_mode, {"off", "fixed", "marginalized"}, "NebularConfig.eline_mode"
        )
        if self.backend == "cloudy" and self.grid_path is None:
            raise ValueError("NebularConfig: grid_path is required when backend='cloudy'.")


@dataclass(frozen=True)
class MultiwavelengthConfig:
    """Multi-wavelength extension settings (radio, X-ray, shock).

    Parameters
    ----------
    radio: bool
        Enable radio synchrotron + AGN jet emission.  Default: ``False``.
    xray: bool
        Enable X-ray (XRB + AGN corona) emission.  Default: ``False``.
    shock: bool
        Enable shock emission (MAPPINGS III).  Default: ``False``.
    apply_igm: bool
        Apply IGM absorption.  Default: ``True``.
    igm_model: str
        IGM absorption model: ``"inoue"`` (Inoue+2014, default) or
        ``"madau"`` (Madau+1995, 17 absorption lines).

    Attributes
    ----------
    radio, xray, shock, apply_igm, igm_model
        All constructor parameters are read-only frozen attributes.

    Notes
    -----
    Frozen dataclass; all fields are immutable after construction.
    ``apply_igm=True`` applies IGM absorption to the full SED at the galaxy
    redshift.  The model is selected via ``igm_model``.  Pass to
    :class:`SEDModelConfig` via the ``multiwavelength`` field.

    Examples
    --------
    >>> from tengri.config import MultiwavelengthConfig
    >>> cfg = MultiwavelengthConfig(radio=True, xray=True)
    >>> cfg.radio
    True
    """

    radio: bool = False
    xray: bool = False
    shock: bool = False
    apply_igm: bool = True
    igm_model: str = "inoue"


@dataclass(frozen=True)
class SEDModelConfig:
    """Top-level frozen configuration collecting all sub-model settings.

    Groups structural choices (which physics modules are active) separately
    from fittable ``Parameters`` (scalars with priors).

    Parameters
    ----------
    sfh: SFHConfig
        SFH structural settings.
    dust: DustConfig
        Dust attenuation and emission settings.
    nebular: NebularConfig
        Nebular emission settings.
    multiwavelength: MultiwavelengthConfig
        Radio, X-ray, shock, and IGM settings.
    agn_model: str or None
        AGN SED model name.  ``None`` disables AGN (default).
        See ``tengri.components.agn.unified`` for valid names.
    agn_config: AGNConfig or None
        Detailed AGN sub-model choices.  ``None`` = use defaults when
        ``agn_model`` is set.

    Attributes
    ----------
    sfh, dust, nebular, multiwavelength, agn_model, agn_config
        All constructor parameters are read-only frozen attributes.

    Notes
    -----
    Frozen dataclass; all fields are immutable after construction.
    ``SEDModelConfig()`` with no arguments produces the default smooth parametric
    SED model (double power-law SFH, power-law dust, no nebular emission, no
    AGN). Pass a ``SEDModelConfig`` instance as the ``config`` argument to
    :class:`~tengri.forward.sed_model.SEDModel`.

    Examples
    --------
    Standard photometric fit::

        cfg = SEDModelConfig()

    With cloudy nebular emission and radio extension::

        cfg = SEDModelConfig(
            nebular=NebularConfig(backend="cloudy", grid_path="/data/cloudy.h5"),
            multiwavelength=MultiwavelengthConfig(radio=True),
        )
    """

    sfh: SFHConfig = field(default_factory=SFHConfig)
    dust: DustConfig = field(default_factory=DustConfig)
    nebular: NebularConfig = field(default_factory=NebularConfig)
    multiwavelength: MultiwavelengthConfig = field(default_factory=MultiwavelengthConfig)
    agn_model: str | None = None
    agn_config: AGNConfig | None = None


# Deprecated alias: removed in v1.0 per docs/dev/NAMING_CONTRACT.md
def __getattr__(name: str):
    if name == "ModelConfig":
        import warnings

        warnings.warn(
            "ModelConfig is deprecated; use SEDModelConfig instead. "
            "ModelConfig will be removed in v1.0.",
            DeprecationWarning,
            stacklevel=2,
        )
        return SEDModelConfig
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
