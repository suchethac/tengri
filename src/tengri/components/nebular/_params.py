"""Free-parameter declarations owned by the nebular component.

Each tuple in this module is the canonical source for one legacy
bucket in ``tengri.parameters._param_defs``:

- :data:`PARAMS` → ``_NEBULAR_PARAMS`` (standard CLOUDY / Cue / CB_19
  ionisation knobs registered when ``nebular in {cloudy, cue, cb19}``).
- :data:`CB19_PARAMS` → ``_CB19_PARAMS`` (CB_19 grid-only continuous
  axes: density, C/O, ΔN/O, HbFrac).
- :data:`ELINE_PARAMS` → ``_ELINE_PARAMS`` (line velocity dispersion +
  offset, registered when ``eline_mode`` is active).
- :data:`ELINE_BROAD_PARAMS` → ``_ELINE_BROAD_PARAMS`` (AGN broad-line
  component velocity dispersion).
- :data:`CUE_IONSPEC_PARAMS` → ``_CUE_IONSPEC_PARAMS`` (Cue's
  broken-power-law ionising spectrum: 4 slopes + 3 ratios).
- :data:`CUE_GAS_EXTRA_PARAMS` → ``_CUE_GAS_EXTRA_PARAMS`` (Cue gas
  density + N/O + C/O knobs beyond logU/logZ).
- :data:`SHOCK_PARAMS` → ``_SHOCK_PARAMS`` (MAPPINGS shock-emission
  backend: velocity, density, B/√n, abundance, component).

The CUE and SHOCK tuples carry ``None`` priors deliberately — the
upstream registry only registers them when the user provides them
explicitly. ``_bucket_from_declarations`` preserves the ``None``.

Why not also share with `declared_parameters`
---------------------------------------------
:meth:`NebularSEDComponent.declared_parameters` performs backend
dispatch (``cloudy_grid`` vs ``cue`` vs ``shock`` vs ``baked_in``) and
intentionally uses ``Uniform`` defaults for the SEDComponent /
nested-dict-recipe path so users sampling those parameters get a
plausible range out of the box. The flat-builder bucket here uses
``Fixed`` defaults so legacy notebooks keep behaving like
"everything fixed unless overridden". The two priors differ **by
design** — not drift. Unifying them is deferred to a dedicated nebular
PR; this file is currently only the flat-builder source of truth.
"""

from __future__ import annotations

from tengri.protocols.component import ParamDeclaration
from tengri.parameters.priors import Fixed, Uniform

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "neb_logU",
        Fixed(-3.0),
        "Ionization parameter log10(U)",
        lambda lo, hi: lo >= -5 and hi <= 0,
        "must be in [-5, 0]",
    ),
    ParamDeclaration(
        "neb_logZ_gas",
        Fixed(-0.3),  # will be overridden to match met_logzsol if not set
        "Gas-phase metallicity log10(Z_gas/Zsun)",
    ),
    ParamDeclaration(
        "neb_fesc",
        Fixed(0.0),
        "Ionizing photon escape fraction",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_fesc_lya",
        Fixed(0.0),
        "Ly-alpha escape fraction (resonant scattering)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_dig_frac",
        Fixed(0.0),
        "DIG fraction of nebular emission (Tacchella+2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "neb_dig_delta_logU",
        Fixed(-1.0),
        "DIG ionization parameter offset (dex, negative)",
        lambda lo, hi: lo >= -4 and hi <= 0,
        "must be in [-4, 0]",
    ),
)

# CB_19 extra continuous axes (nebular == "cb19").
# Unit convention: CB_19 stores L_line/L_Hβ; the CB19Backend converts to
# L_sun/Q_H using L_Hβ/Q_H = 4.78e-13 erg/photon (Case B, T_e=10^4 K;
# Osterbrock & Ferland 2006, Table 4.4).
CB19_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "neb_log_nH",
        Fixed(2.0),  # n_H = 100 cm⁻³, typical HII region
        "Log hydrogen density log10(n_H / cm⁻³) for CB_19 grid [grid range: 1–4]",
        lambda lo, hi: lo >= 0 and hi <= 6,
        "must be in [0, 6] (CB_19 grid: 1–4; extrapolated outside)",
    ),
    ParamDeclaration(
        "neb_co",
        Fixed(-0.36),  # near-solar C/O (CLOUDY c17 default)
        "Log C/O abundance ratio log10(C/O) for CB_19 grid [grid range: −1 to 0.15]",
        lambda lo, hi: lo >= -3 and hi <= 2,
        "must be in [−3, 2]",
    ),
    ParamDeclaration(
        "neb_dno",
        Fixed(0.0),  # solar N/O scaling (Nicholls+2017)
        "ΔN/O offset (log10) from default N/O–O/H scaling [grid range: −0.25 to 0.25]",
        lambda lo, hi: lo >= -1 and hi <= 1,
        "must be in [−1, 1]",
    ),
    ParamDeclaration(
        "neb_hbfrac",
        Fixed(1.0),  # radiation-bounded (default)
        "HbFrac: L_Hβ(matter-bounded)/L_Hβ(radiation-bounded) for CB_19 [0–1]. "
        "HbFrac=1 = fully radiation-bounded; escape fraction ≈ 1 − HbFrac",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
)

# Emission line velocity parameters (eline_mode in {marginalized, fitted}).
ELINE_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "eline_sigma_kms",
        Fixed(0.0),  # Default: instrument resolution only
        "Emission line velocity dispersion in km/s (added in quadrature to instrument resolution)",
        lambda lo, hi: lo >= 0,
        "must have lo >= 0",
    ),
    ParamDeclaration(
        "eline_delta_v_kms",
        Fixed(0.0),  # Default: no velocity offset
        "Emission line velocity offset from systemic redshift in km/s",
    ),
)

# Broad emission line component (AGN), eline_broad=True.
ELINE_BROAD_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "eline_broad_sigma_kms",
        Uniform(500.0, 5000.0),
        "Broad emission line velocity dispersion in km/s",
        lambda lo, hi: lo >= 200,
        "must have lo >= 200 km/s (broad component)",
    ),
)

# Cue-specific optional params — only registered if the user provides
# them explicitly. The ``None`` prior is intentional: the registry
# treats absence-of-default as "must come from user kwargs".
CUE_IONSPEC_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "ionspec_index1",
        None,
        "Cue ionizing spectrum slope segment 1 (HeII, 1-228A)",
        lambda lo, hi: lo >= 0 and hi <= 50,
        "must be in [0, 50]",
    ),
    ParamDeclaration(
        "ionspec_index2",
        None,
        "Cue ionizing spectrum slope segment 2 (OII, 228-353A)",
        lambda lo, hi: lo >= -1 and hi <= 35,
        "must be in [-1, 35]",
    ),
    ParamDeclaration(
        "ionspec_index3",
        None,
        "Cue ionizing spectrum slope segment 3 (HeI, 353-504A)",
        lambda lo, hi: lo >= -2 and hi <= 20,
        "must be in [-2, 20]",
    ),
    ParamDeclaration(
        "ionspec_index4",
        None,
        "Cue ionizing spectrum slope segment 4 (HI, 504-912A)",
        lambda lo, hi: lo >= -2 and hi <= 10,
        "must be in [-2, 10]",
    ),
    ParamDeclaration(
        "ionspec_logLratio1",
        None,
        "Cue log luminosity ratio seg2/seg1",
        lambda lo, hi: lo >= -1 and hi <= 12,
        "must be in [-1, 12]",
    ),
    ParamDeclaration(
        "ionspec_logLratio2",
        None,
        "Cue log luminosity ratio seg3/seg2",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
    ),
    ParamDeclaration(
        "ionspec_logLratio3",
        None,
        "Cue log luminosity ratio seg4/seg3",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
    ),
)

CUE_GAS_EXTRA_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "gas_logn",
        None,
        "Cue gas density log10(n_H/cm^-3)",
        lambda lo, hi: lo >= 0 and hi <= 5,
        "must be in [0, 5]",
    ),
    ParamDeclaration(
        "gas_logno",
        None,
        "Cue [N/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
    ),
    ParamDeclaration(
        "gas_logco",
        None,
        "Cue [C/O] abundance ratio (dex)",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
    ),
)

# MAPPINGS shock-emission backend (shock=True).
# ``shock_abundance`` and ``shock_component`` carry string Fixed defaults
# — registered for completeness but configured on the backend instance,
# not as JAX-traced free parameters.
SHOCK_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "shock_frac",
        Fixed(0.0),
        "Fraction of nebular Halpha replaced by shock emission [0, 1]",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
    ),
    ParamDeclaration(
        "shock_velocity",
        Fixed(300.0),
        "Shock velocity in km/s (100-1000 for MAPPINGS III; 200-1000 for MAPPINGS V)",
        lambda lo, hi: lo >= 100 and hi <= 1000,
        "must be in [100, 1000]",
    ),
    ParamDeclaration(
        "shock_log_density",
        Fixed(0.0),
        "Log10 pre-shock density in cm^-3; snapped to nearest grid point",
    ),
    ParamDeclaration(
        "shock_b_over_sqrt_n",
        Fixed(1.0),
        "B/sqrt(n) in uG cm^(3/2) (MAPPINGS III) or absolute B in uG (MAPPINGS V); "
        "snapped to nearest grid point",
    ),
    ParamDeclaration(
        "shock_abundance",
        Fixed("solar"),
        "Abundance set: solar | 2xsolar | dopita2005 | lmc | smc",
    ),
    ParamDeclaration(
        "shock_component",
        Fixed("combined"),
        "Emission component: shock | precursor | combined",
    ),
)

__all__ = [
    "CB19_PARAMS",
    "CUE_GAS_EXTRA_PARAMS",
    "CUE_IONSPEC_PARAMS",
    "ELINE_BROAD_PARAMS",
    "ELINE_PARAMS",
    "PARAMS",
    "SHOCK_PARAMS",
]
