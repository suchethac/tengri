# Known Bugs

All physics bugs in this file have been fixed. See `docs/known_bugs.md` for
the full audit history.

---

## Open implementation gaps surfaced by paper validation (2026-04-29)

These are features described or planned in the methods paper
(*(private paper draft)*) that are not yet implemented in
the code. The paper currently flags each of them explicitly as
"planned" so the manuscript and the code stay consistent. When you
implement one, drop the corresponding `--- planned` qualifier in the
appendix and remove the entry from this section.

### MAGPHYS dust emission backend — TODO

**Where the paper describes it:** `999-appendix.tex` §F.4
(`\subsubsection{MAGPHYS (da Cunha et al.\ 2008) --- planned}`,
label `app:magphys`).

**Status:** Not registered. The four-component model (PAH Drudes per
Smith+2007, hot MIR, warm + cold modified blackbodies with
$\beta = 1.5\,/\,2.0$, energy fractions $\xi_{\rm PAH}$,
$\xi_{\rm MIR}$, $\xi_W$) is fully specified in the appendix but no
entry exists in `DUST_EMISSION_MODELS` in
`src/tengri/components/dust/emission.py`. Add it via the
`@register_dust_emission` decorator. Cite da Cunha+2008 and
Smith+2007 in `bibliography.py`.

### Standalone differentiable Kerr ray-tracing AGN backend — TODO

**Where the paper describes it:** `999-appendix.tex` §H.5
(`\subsection{Relativistic Outer Disc and Warm Comptonisation}`,
paragraph "Planned: standalone differentiable Kerr ray-tracing
backend").

**Status:** KYCONV-style relativistic effects are currently folded
into the `relagn` disc grid in `src/tengri/components/agn/disc.py`,
sampled by triweight trilinear interpolation in
$(\log M_{\rm BH},\,\log\dot{m},\,a_\star)$ with inclination applied
analytically as a $2\cos i$ projection. The paper's "planned"
paragraph commits us to a separate `kyconv` backend that exposes
inclination as an explicit grid axis, so that
$\partial F_\nu / \partial i$ and $\partial F_\nu / \partial a_\star$
can be propagated through autodiff and used as fit parameters. This
needs a new pre-computed transfer-function grid (e.g.\ via the
`kyconv` ray-tracing code) and a registered backend distinct from
`relagn`.

### `powerlaw` SFH model registration — TODO (decide)

**Where the paper described it:** the appendix used to list a
`powerlaw` SFH ($\dot{M}^{\rm pk}\,(t/t_0)^{\beta}$) in
`Table~\ref{tab:sfh_models}`. The function exists at
`src/tengri/components/sfh/mean_sfh.py:889` (`powerlaw_sfh`) but is
not registered in the SFH registry, so users cannot select it via
the public API. The table entry has been removed from the paper for
now; if you add the missing `@register_sfh("powerlaw")` decorator,
restore the table row.

### Documentation: spectroscopic calibration default — RESOLVED IN PAPER

The paper's spectroscopy appendix used to claim "default $K = 3$" for
the Chebyshev calibration polynomial. The actual default in
`src/tengri/observation/spectroscopy.py:135` is
`calibration_order = 0` (calibration disabled). The paper text has
been corrected; this discrepancy was tracked as documentation drift,
not a bug, and is listed here so a future reader does not
re-introduce the wrong "K = 3 default" claim from older drafts.

---

## ADAF synchrotron self-absorption spectral index — FIXED

**File:** `src/tengri/components/agn/disc.py`, `adaf_disc`

**Was:** `sync_thick = nu_ratio_sa**2.5` — incorrect exponent for non-thermal
power-law electrons, not the thermal ADAF case.

**Fix (applied):** Changed to `nu_ratio_sa**2.0` per Mahadevan (1997, ApJ
477, 585) Eq. 19/23: Rayleigh-Jeans self-absorbed regime of thermal electrons
gives `ν²`.

---

## Bell 2003 synchrotron suppression formula — FIXED

**File:** `src/tengri/components/radio/radio.py`, `_synchrotron_suppression`

**Was:** `L / (1 + (L₀/L)²)` — quadratic suppression, not matching Bell.

**Fix (applied):** Replaced with Bell (2003, ApJ 586, 794) Eq. 3 piecewise
non-thermal fraction: `n = 0.9` for `L > L*`; `n = 0.9 * (L/L*)^{0.3}` for
`L ≤ L*`, where `L*` is the 1.4 GHz luminosity corresponding to M_V = −21.

---

## ADAF electron temperature approximation — NOTE (not a bug)

**File:** `src/tengri/components/agn/disc.py`, `adaf_disc`

`T_e = 5e9 * sqrt(delta / mdot)` is a rough scaling approximation.
Mahadevan (1997) Section 5.1 solves T_e self-consistently from ion-electron
energy balance. The approximation is acceptable for broadband SED fitting.
