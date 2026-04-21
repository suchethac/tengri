"""Frozen configuration dataclasses for sub-model selection.

These objects encode *which* physics modules are active — structural choices that
do NOT appear in the gradient tape.  They are distinct from fittable Parameters.

Usage
-----
::

    from tengri.config.settings import DustConfig, NebularConfig, ModelConfig

    cfg = ModelConfig(
        nebular=NebularConfig(backend="cloudy", grid_path="/data/cloudy.h5"),
        dust=DustConfig(law_bc="calzetti"),
    )

All fields have defaults so a bare ``ModelConfig()`` is equivalent to the
standard smooth parametric model.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class SFHConfig:
    """SFH structural settings (non-parametric shape choices).

    Parameters
    ----------
    mean_type : list[str]
        Analytic mean-SFH components, e.g. ``["tsnorm"]``, ``["dpl"]``,
        ``["dpl", "field"]``.
    n_grid : int
        GP latent grid size (only relevant when ``"field"`` is in ``mean_type``).
        Default: 64.
    evolving_metallicity : bool
        Replace ``met_logzsol`` with a two-endpoint ramp.  Default: ``False``.
    alpha_fe_evolving : bool
        Enable [α/Fe] evolution with lookback time.  Default: ``False``.
    chem_evol : bool
        Derive Z(t) from SFH via gas-regulator model.  Default: ``False``.
    met_interp : str
        Metallicity interpolation method: ``"smooth"`` (triweight, default)
        or ``"linear"`` (FSPS/Prospector-compatible).
    lgmet_scatter : float
        Triweight kernel bandwidth in dex (``met_interp="smooth"``).
        Default: 0.1.
    """

    mean_type: tuple[str, ...] = ("dpl",)
    n_grid: int = 64
    evolving_metallicity: bool = False
    alpha_fe_evolving: bool = False
    chem_evol: bool = False
    met_interp: str = "smooth"
    lgmet_scatter: float = 0.1

    def __post_init__(self) -> None:
        valid_interp = frozenset({"smooth", "linear"})
        if self.met_interp not in valid_interp:
            raise ValueError(
                f"SFHConfig.met_interp={self.met_interp!r} is not valid."
                f" Choose from: {sorted(valid_interp)}"
            )


@dataclass(frozen=True)
class DustConfig:
    """Dust attenuation and emission structural settings.

    Parameters
    ----------
    model : str
        Geometry model: ``"two_component"`` (Charlot & Fall, default) or
        ``"single_component"`` (uniform screen).
    law_bc : str
        Birth cloud attenuation law.  Default: ``"power_law"``.
        Options: ``power_law``, ``calzetti``, ``kriek_conroy``, ``smc``,
        ``cardelli``, ``salim``, ``li08``.
    law_diff : str or None
        Diffuse ISM attenuation law.  ``None`` = same as ``law_bc``.
    emission : str or None
        IR dust emission model.  ``None`` disables IR emission (default).
        Options: ``"modified_blackbody"``, ``"casey2012"``, ``"dale2014"``,
        ``"draine_li2007"``, ``"draine_li2014"``.
    approx : bool
        Use approximate (fused) photometry path.  Default: ``True``.
    """

    model: str = "two_component"
    law_bc: str = "power_law"
    law_diff: str | None = None
    emission: str | None = None
    approx: bool = True

    def __post_init__(self) -> None:
        valid_models = frozenset({"two_component", "single_component"})
        valid_laws = frozenset(
            {
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
        )
        valid_emission = frozenset(
            {None, "modified_blackbody", "casey2012", "dale2014", "draine_li2007", "draine_li2014"}
        )
        if self.model not in valid_models:
            raise ValueError(
                f"DustConfig.model={self.model!r} is not valid."
                f" Choose from: {sorted(valid_models)}"
            )
        if self.law_bc not in valid_laws:
            raise ValueError(
                f"DustConfig.law_bc={self.law_bc!r} is not valid."
                f" Choose from: {sorted(valid_laws)}"
            )
        if self.law_diff is not None and self.law_diff not in valid_laws:
            raise ValueError(
                f"DustConfig.law_diff={self.law_diff!r} is not valid."
                f" Choose from: {sorted(valid_laws)}"
            )
        if self.emission not in valid_emission:
            raise ValueError(
                f"DustConfig.emission={self.emission!r} is not valid."
                f" Choose from: {sorted(str(v) for v in valid_emission)}"
            )


@dataclass(frozen=True)
class NebularConfig:
    """Nebular emission structural settings.

    Parameters
    ----------
    backend : str
        Nebular emission backend.
        ``"off"`` — disabled (default).
        ``"baked_in"`` — lines from SSP grid (no free nebular params).
        ``"cloudy"`` — CLOUDY grid interpolation.
        ``"cue"`` — Cue neural emulator.
    grid_path : str or None
        Path to CLOUDY HDF5 grid (required when ``backend="cloudy"``).
    weights_path : str or None
        Override default Cue weights path (only for ``backend="cue"``).
    ionization : str
        Ionization source for Cue: ``"ssp"`` (default).
    eline_mode : str
        Emission line fitting mode.
        ``"off"`` — no line treatment (default).
        ``"fixed"`` — fixed profiles.
        ``"marginalized"`` — analytic marginalization.
    eline_broad : bool
        Enable broad AGN emission line component.  Default: ``False``.
    """

    backend: str = "off"
    grid_path: str | None = None
    weights_path: str | None = None
    ionization: str = "ssp"
    eline_mode: str = "off"
    eline_broad: bool = False

    def __post_init__(self) -> None:
        valid_backends = frozenset({"off", "baked_in", "cloudy", "cue"})
        valid_eline_modes = frozenset({"off", "fixed", "marginalized"})
        if self.backend not in valid_backends:
            raise ValueError(
                f"NebularConfig.backend={self.backend!r} is not valid."
                f" Choose from: {sorted(valid_backends)}"
            )
        if self.eline_mode not in valid_eline_modes:
            raise ValueError(
                f"NebularConfig.eline_mode={self.eline_mode!r} is not valid."
                f" Choose from: {sorted(valid_eline_modes)}"
            )
        if self.backend == "cloudy" and self.grid_path is None:
            raise ValueError("NebularConfig: grid_path is required when backend='cloudy'.")


@dataclass(frozen=True)
class MultiwavelengthConfig:
    """Multi-wavelength extension settings (radio, X-ray, shock).

    Parameters
    ----------
    radio : bool
        Enable radio synchrotron + AGN jet emission.  Default: ``False``.
    xray : bool
        Enable X-ray (XRB + AGN corona) emission.  Default: ``False``.
    shock : bool
        Enable shock emission (MAPPINGS III).  Default: ``False``.
    apply_igm : bool
        Apply Inoue+2014 IGM absorption.  Default: ``True``.
    """

    radio: bool = False
    xray: bool = False
    shock: bool = False
    apply_igm: bool = True


@dataclass(frozen=True)
class ModelConfig:
    """Top-level frozen configuration collecting all sub-model settings.

    Groups structural choices (which physics modules are active) separately
    from fittable ``Parameters`` (scalars with priors).

    Parameters
    ----------
    sfh : SFHConfig
        SFH structural settings.
    dust : DustConfig
        Dust attenuation and emission settings.
    nebular : NebularConfig
        Nebular emission settings.
    multiwavelength : MultiwavelengthConfig
        Radio, X-ray, shock, and IGM settings.
    agn_model : str or None
        AGN SED model name.  ``None`` disables AGN (default).
        See ``tengri.components.agn.unified`` for valid names.
    agn_config : AGNConfig or None
        Detailed AGN sub-model choices.  ``None`` = use defaults when
        ``agn_model`` is set.

    Examples
    --------
    Standard photometric fit::

        cfg = ModelConfig()

    With cloudy nebular emission and radio extension::

        cfg = ModelConfig(
            nebular=NebularConfig(backend="cloudy", grid_path="/data/cloudy.h5"),
            multiwavelength=MultiwavelengthConfig(radio=True),
        )
    """

    sfh: SFHConfig = field(default_factory=SFHConfig)
    dust: DustConfig = field(default_factory=DustConfig)
    nebular: NebularConfig = field(default_factory=NebularConfig)
    multiwavelength: MultiwavelengthConfig = field(default_factory=MultiwavelengthConfig)
    agn_model: str | None = None
    agn_config: object | None = None  # AGNConfig | None — avoid circular import
