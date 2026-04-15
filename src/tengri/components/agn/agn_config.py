"""AGNConfig: static configuration for AGN sub-model selection."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class AGNConfig:
    """Static configuration for AGN sub-model selection.

    Parameters
    ----------
    disc : str
        AGN accretion disc model.
        ``"powerlaw"`` — simple power-law SED.
        ``"multicolor"`` — multi-colour blackbody disc (default).
        ``"kubota_done"`` — Kubota & Done (2018) 3-zone model.
        ``"adaf"`` — ADAF (low-luminosity AGN).
    torus : str
        AGN torus/obscuration model.
        ``"simple"`` — single-temperature MBB (toy).
        ``"two_temperature"`` — two-temperature MBB (toy).
        ``"skirtor"`` — SKIRTOR clumpy torus (default, science-grade).
    nlr : str
        Narrow Line Region emission model.
        ``"analytic"`` — analytic Gaussian line profiles (default, fast).
        ``"cue"`` — Cue neural emulator (physically consistent).
    blr : bool
        Include Broad Line Region emission (Type 1 AGN). Default True.
    polar_dust : bool
        Include SMC polar dust reddening. Default False.
    fe2 : bool
        Include Fe II pseudo-continuum. Default False.
    """

    disc: str = "multicolor"
    torus: str = "skirtor"
    nlr: str = "analytic"
    blr: bool = True
    polar_dust: bool = False
    fe2: bool = False

    def __post_init__(self) -> None:
        valid_disc = frozenset({"powerlaw", "multicolor", "kubota_done", "adaf"})
        valid_torus = frozenset({"simple", "two_temperature", "skirtor"})
        valid_nlr = frozenset({"analytic", "cue"})
        if self.disc not in valid_disc:
            raise ValueError(
                f"AGNConfig.disc={self.disc!r} is not valid. Choose from: {sorted(valid_disc)}"
            )
        if self.torus not in valid_torus:
            raise ValueError(
                f"AGNConfig.torus={self.torus!r} is not valid. Choose from: {sorted(valid_torus)}"
            )
        if self.nlr not in valid_nlr:
            raise ValueError(
                f"AGNConfig.nlr={self.nlr!r} is not valid. Choose from: {sorted(valid_nlr)}"
            )
