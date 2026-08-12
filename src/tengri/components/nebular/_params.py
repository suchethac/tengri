# SPDX-License-Identifier: BSD-3-Clause
"""Free-parameter declarations owned by the nebular component.

Each tuple in this module is the canonical source for one legacy
bucket in ``tengri.parameters._builders``:

- :data:`PARAMS` → ``_NEBULAR_PARAMS`` (standard CLOUDY / Cue / CB_19
  ionization knobs registered when ``nebular in {cloudy, cue, cb19}``).
- :data:`CB19_PARAMS` → ``_CB19_PARAMS`` (CB_19 grid-only continuous
  axes: density, C/O, ΔN/O, HbFrac).
- :data:`ELINE_PARAMS` → ``_ELINE_PARAMS`` (line velocity dispersion +
  offset, registered when ``eline_mode`` is active).
- :data:`ELINE_BROAD_PARAMS` → ``_ELINE_BROAD_PARAMS`` (AGN broad-line
  component velocity dispersion).
- :data:`CUE_IONSPEC_PARAMS` → ``_CUE_IONSPEC_PARAMS`` (Cue's
  broken-power-law ionizing spectrum: 4 slopes + 3 ratios).
- :data:`CUE_GAS_EXTRA_PARAMS` → ``_CUE_GAS_EXTRA_PARAMS`` (Cue gas
  density + N/O + C/O knobs beyond logU/logZ).
- :data:`SHOCK_PARAMS` → ``_SHOCK_PARAMS`` (MAPPINGS shock-emission
  backend: velocity, density, B/√n, abundance, component).

Single source of truth (#887)
-----------------------------
:data:`CUE_IONSPEC_PARAMS` and :data:`CUE_GAS_EXTRA_PARAMS` are now the ONE
canonical declaration of the Cue ionizing-spectrum + gas-extra priors,
defaults, bounds, and descriptions. Both the flat-builder bucket (via
``_bucket_from_declarations``) and
:meth:`NebularSEDComponent.declared_parameters` (which returns these tuples
verbatim) consume them, so the two paths cannot drift. Previously the bucket
carried ``None`` priors while the component re-declared the same params inline
with ``Uniform`` priors + physical defaults — the divergence #887 removed.

The flat-builder still registers these Cue params only when the user provides
them explicitly (an opt-in policy on the ``Parameters`` path, unchanged); the
canonical prior/default is used when a param IS registered.
"""

from __future__ import annotations

from tengri.parameters.priors import Fixed, Uniform
from tengri.protocols.component import ParamDeclaration

PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "neb_logU",
        Fixed(-3.0),
        "Ionization parameter log10(U)",
        lambda lo, hi: lo >= -5 and hi <= 0,
        "must be in [-5, 0]",
        free_prior=Uniform(-5.0, 0.0, "Ionization parameter log10(U)", default=-3.0),
    ),
    ParamDeclaration(
        "neb_logZ_gas",
        Fixed(-0.3),  # will be overridden to match met_logzsol if not set
        "Gas-phase metallicity log10(Z_gas/Zsun)",
        # Deliberately NO free_prior, for the same reason as ``neb_xid`` (which
        # lives in ``parameters/_builders.py::_AGN_EXTRAS``):
        # its admissible range is the selected nebular backend's grid, and those
        # differ (Cue, the Cloudy grids and the baked-in SSP tables do not share
        # an extent). The ``neb`` group wildcard is not backend-scoped the way
        # ``dust.emission`` is since #1482, so one declared range would be right
        # for one backend and clipped or unreachable for the others.
        #
        # It is also tied: absent an explicit setting it tracks ``met_logzsol``,
        # so freeing it silently decouples gas-phase from stellar metallicity
        # and adds a near-degenerate dimension. Free it explicitly when you mean
        # to fit that decoupling, with a range drawn from your backend's grid.
    ),
    ParamDeclaration(
        "neb_fesc",
        Fixed(0.0),
        "Ionizing photon escape fraction",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Ionizing photon escape fraction", default=0.0),
    ),
    ParamDeclaration(
        "neb_fesc_lya",
        Fixed(0.0),
        "Ly-alpha escape fraction (resonant scattering)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Ly-alpha escape fraction", default=0.0),
    ),
    ParamDeclaration(
        "neb_fdust",
        Fixed(0.0),
        "Dust-absorption fraction of ionizing photons in HII regions",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Ionizing-photon dust-absorption fraction", default=0.0),
    ),
    ParamDeclaration(
        "neb_dig_frac",
        Fixed(0.0),
        "DIG fraction of nebular emission (Tacchella+2022)",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "DIG fraction of nebular emission", default=0.0),
    ),
    ParamDeclaration(
        "neb_dig_delta_logU",
        Fixed(-1.0),
        "DIG ionization parameter offset (dex, negative)",
        lambda lo, hi: lo >= -4 and hi <= 0,
        "must be in [-4, 0]",
        free_prior=Uniform(-4.0, 0.0, "DIG log10(U) offset", units="dex", default=-1.0),
        units="dex",
    ),
    ParamDeclaration(
        "neb_eline_sigma_kms",
        Fixed(100.0),
        "Intrinsic nebular emission-line velocity dispersion [km/s] — sets the "
        "triweight line profile width in the forward SED. Distinct from "
        "sigma_v_kms (stellar LOSVD) and eline_sigma_kms (line-fitting template).",
        lambda lo, hi: lo >= 0 and hi <= 2000,
        "must be in [0, 2000] km/s",
        free_prior=Uniform(
            0.0, 2000.0, "Nebular line velocity dispersion", units="km/s", default=100.0
        ),
        units="km/s",
    ),
)

# CB_19 extra continuous axes (nebular == "cb19").
# Unit convention: CB_19 stores L_line/L_Hβ; the CB19Backend converts to
# L_sun/Q_H using L_Hβ/Q_H = 4.78e-13 erg/photon (Case B, T_e=10^4 K;
# Osterbrock & Ferland 2006, Table 4.4).
CB19_PARAMS: tuple[ParamDeclaration, ...] = (
    # ``free_prior`` on the three grid axes below spans the **CB_19 grid**, not
    # the wider admissible bound. The bound is what the interpolator tolerates
    # (it extrapolates outside the grid); the grid is where the templates carry
    # information. Freeing over the bound would spend most of the prior mass on
    # extrapolated values, so FREE opens the grid and a caller who genuinely
    # wants to extrapolate passes an explicit wider prior.
    ParamDeclaration(
        "neb_log_nH",
        Fixed(2.0),  # n_H = 100 cm⁻³, typical HII region
        "Log hydrogen density log10(n_H / cm⁻³) for CB_19 grid [grid range: 1–4]",
        lambda lo, hi: lo >= 0 and hi <= 6,
        "must be in [0, 6] (CB_19 grid: 1–4; extrapolated outside)",
        free_prior=Uniform(1.0, 4.0, "Log hydrogen density", units="log10(cm^-3)", default=2.0),
        units="log10(cm^-3)",
    ),
    ParamDeclaration(
        "neb_co",
        Fixed(-0.36),  # near-solar C/O (CLOUDY c17 default)
        "Log C/O abundance ratio log10(C/O) for CB_19 grid [grid range: −1 to 0.15]",
        lambda lo, hi: lo >= -3 and hi <= 2,
        "must be in [−3, 2]",
        free_prior=Uniform(-1.0, 0.15, "Log C/O abundance ratio", default=-0.36),
        units="dex",
    ),
    ParamDeclaration(
        "neb_dno",
        Fixed(0.0),  # solar N/O scaling (Nicholls+2017)
        "ΔN/O offset (log10) from default N/O–O/H scaling [grid range: −0.25 to 0.25]",
        lambda lo, hi: lo >= -1 and hi <= 1,
        "must be in [−1, 1]",
        free_prior=Uniform(-0.25, 0.25, "Delta N/O offset", units="dex", default=0.0),
        units="dex",
    ),
    ParamDeclaration(
        "neb_hbfrac",
        Fixed(1.0),  # radiation-bounded (default)
        "HbFrac: L_Hβ(matter-bounded)/L_Hβ(radiation-bounded) for CB_19 [0–1]. "
        "HbFrac=1 = fully radiation-bounded; escape fraction ≈ 1 − HbFrac",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        # Deliberately NO free_prior. Unlike the CB19 axes beside it, HbFrac is
        # not an interpolation axis at runtime: ``load_cb19_grid`` snaps it to
        # the nearest of the grid's two HbFrac values and collapses the axis at
        # load time, and the collapsed grid is what every prediction reads. The
        # snap target comes from the ``CB19Backend(hbfrac=...)`` constructor
        # argument, which the build grammar does not forward, so the parameter
        # has no runtime consumer at all -- sweeping it across [0, 1] moves the
        # SED by exactly 0.0. A continuous prior over two reachable values would
        # be wrong even if it were wired.
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
        units="km/s",
        # 0 (the default) means "instrument resolution only" and is a genuine
        # value here, not a sentinel. The ceiling is narrow-line: broad AGN
        # components have their own ELINE_BROAD_PARAMS block, and the sibling
        # ``neb_eline_sigma_kms`` carries the wider [0, 2000] precisely because
        # it must also describe those.
        free_prior=Uniform(0.0, 500.0, "Line velocity dispersion", units="km/s", default=0.0),
    ),
    ParamDeclaration(
        "eline_delta_v_kms",
        Fixed(0.0),  # Default: no velocity offset
        "Emission line velocity offset from systemic redshift in km/s",
        units="km/s",
        # Symmetric about zero because the offset is signed: blueshifted for
        # outflows, redshifted for inflows. +/-1000 km/s spans the velocity
        # offsets seen in starburst winds without letting the line wander into
        # a neighboring feature.
        free_prior=Uniform(-1000.0, 1000.0, "Line velocity offset", units="km/s", default=0.0),
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
        units="km/s",
    ),
)

# Cue-specific optional params — only registered if the user provides
# them explicitly. The ``None`` prior is intentional: the registry
# treats absence-of-default as "must come from user kwargs".
# Canonical single source (#887) for the 7 Cue ionizing-spectrum shape
# parameters — consumed BOTH by the flat-builder bucket (via
# ``_bucket_from_declarations``) AND by
# ``NebularSEDComponent.declared_parameters`` (which returns these verbatim).
# Priors + physical defaults live here only; previously the component
# re-declared them inline with the same bounds but a separate default, and the
# bucket carried ``None``, so the two could drift (that was the #887 target).
# Defaults = a fiducial young-starburst ionizing spectrum: the 1-Myr solar-Z
# BPASS SSP fit with ``fit_ionizing_spectrum`` (#845), so ``'*': FIXED`` yields a
# physical ionizing SED instead of the prior midpoint (which, for these
# correlated slopes, would be unphysical — #477 / #478).
#
# NOTE: these prior BOUNDS are the user-settable range and are DISTINCT from
# ``ionizing_spectrum.py::_CLIP_RANGES`` — the tighter Cue-emulator training
# grid used to clip the auto-derived (SSP-fit) coefficients against
# extrapolation. They are different quantities and are intentionally NOT
# unified.
CUE_IONSPEC_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "ionspec_index1",
        Uniform(0.0, 50.0, default=22.21),
        "Cue ionizing slope segment 1 (HeII, 1-228 Å) [dimensionless]",
        lambda lo, hi: lo >= 0 and hi <= 50,
        "must be in [0, 50]",
    ),
    ParamDeclaration(
        "ionspec_index2",
        Uniform(-1.0, 35.0, default=10.52),
        "Cue ionizing slope segment 2 (OII, 228-353 Å) [dimensionless]",
        lambda lo, hi: lo >= -1 and hi <= 35,
        "must be in [-1, 35]",
    ),
    ParamDeclaration(
        "ionspec_index3",
        Uniform(-2.0, 20.0, default=5.69),
        "Cue ionizing slope segment 3 (HeI, 353-504 Å) [dimensionless]",
        lambda lo, hi: lo >= -2 and hi <= 20,
        "must be in [-2, 20]",
    ),
    ParamDeclaration(
        "ionspec_index4",
        Uniform(-2.0, 10.0, default=2.15),
        "Cue ionizing slope segment 4 (HI, 504-912 Å) [dimensionless]",
        lambda lo, hi: lo >= -2 and hi <= 10,
        "must be in [-2, 10]",
    ),
    ParamDeclaration(
        "ionspec_logLratio1",
        Uniform(-1.0, 12.0, default=2.78),
        "Cue log luminosity ratio seg2/seg1 [dimensionless]",
        lambda lo, hi: lo >= -1 and hi <= 12,
        "must be in [-1, 12]",
    ),
    ParamDeclaration(
        "ionspec_logLratio2",
        Uniform(-1.0, 3.0, default=0.47),
        "Cue log luminosity ratio seg3/seg2 [dimensionless]",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
    ),
    ParamDeclaration(
        "ionspec_logLratio3",
        Uniform(-1.0, 3.0, default=0.56),
        "Cue log luminosity ratio seg4/seg3 [dimensionless]",
        lambda lo, hi: lo >= -1 and hi <= 3,
        "must be in [-1, 3]",
    ),
)

# Canonical single source (#887) for the 3 Cue gas-property knobs beyond
# logU / logZ_gas. Defaults: n_H = 100 cm^-3 (typical HII region), solar [N/O]
# and [C/O] (log ratios = 0.0). See :data:`CUE_IONSPEC_PARAMS` for the
# consumed-by-both-paths rationale.
CUE_GAS_EXTRA_PARAMS: tuple[ParamDeclaration, ...] = (
    ParamDeclaration(
        "gas_logn",
        Uniform(0.0, 5.0, default=2.0),
        "Cue gas density log10(n_H/cm^-3) [dimensionless]",
        lambda lo, hi: lo >= 0 and hi <= 5,
        "must be in [0, 5]",
        units="log10(cm^-3)",
    ),
    ParamDeclaration(
        "gas_logno",
        Uniform(-2.0, 2.0, default=0.0),
        "Cue [N/O] abundance ratio [dex]",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        units="dex",
    ),
    ParamDeclaration(
        "gas_logco",
        Uniform(-2.0, 2.0, default=0.0),
        "Cue [C/O] abundance ratio [dex]",
        lambda lo, hi: lo >= -2 and hi <= 2,
        "must be in [-2, 2]",
        units="dex",
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
        "Fraction of nebular Halpha replaced by shock emission [0, 1] "
        "(used when shock norm='frac')",
        lambda lo, hi: lo >= 0 and hi <= 1,
        "must be in [0, 1]",
        free_prior=Uniform(0.0, 1.0, "Shock fraction of Halpha", default=0.0),
    ),
    ParamDeclaration(
        "shock_log_lhalpha",
        Fixed(41.0),
        "log10(shock Halpha luminosity / [erg/s]) — absolute normalization "
        "(used when shock norm='lhalpha')",
        lambda lo, hi: lo >= 30 and hi <= 46,
        "must be in [30, 46]",
        free_prior=Uniform(
            30.0, 46.0, "log10 shock Halpha luminosity", units="log10(erg/s)", default=41.0
        ),
        units="log10(erg/s)",
    ),
    ParamDeclaration(
        "shock_velocity",
        Fixed(300.0),
        "Shock velocity in km/s (100-1000 for MAPPINGS III; 200-1000 for MAPPINGS V)",
        lambda lo, hi: lo >= 100 and hi <= 1000,
        "must be in [100, 1000]",
        # MAPPINGS III tabulation spans 100-1000 km/s; V starts at 200. Use the
        # union bound so the prior is valid for either backend.
        free_prior=Uniform(100.0, 1000.0, "Shock velocity", units="km/s", default=300.0),
        units="km/s",
    ),
    ParamDeclaration(
        # Neither shock grid axis gets a free_prior, and the reason is in their
        # own descriptions: both are *snapped to the nearest grid point*. A
        # continuous prior over a snapped axis is piecewise constant, so its
        # gradient is exactly zero almost everywhere and a gradient-based
        # sampler cannot move along it -- the parameter would look free and
        # behave frozen, which is the failure mode #887 exists to remove rather
        # than one to introduce. Fitting these needs either a grid-interpolating
        # kernel or an explicitly discrete sampler; until then set them
        # structurally.
        "shock_log_density",
        Fixed(0.0),
        "Log10 pre-shock density in cm^-3; snapped to nearest grid point",
        units="log10(cm^-3)",
    ),
    ParamDeclaration(
        "shock_b_over_sqrt_n",
        Fixed(1.0),
        "B/sqrt(n) in uG cm^(3/2) (MAPPINGS III) or absolute B in uG (MAPPINGS V); "
        "snapped to nearest grid point",
    ),
    # NOTE: the categorical ``shock_abundance`` / ``shock_component`` knobs are
    # NOT free parameters — they are static structural config on Parameters
    # (``shock_abundance`` / ``shock_component``, like ``radio_sfr_mode``),
    # surfaced via the ``shock={...}`` grammar group (#851).
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
