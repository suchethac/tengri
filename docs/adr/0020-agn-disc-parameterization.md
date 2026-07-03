# ADR-0020: AGN disc parameterization — luminosity-first (L_bol drives the shape)

**Status**: Accepted
**Date**: 2026-07-02
**Supersedes / clarifies**: the disc parameter contract implied by ADR-0018 (composable AGN grammar)

## Context

tengri's composable AGN menu contains two kinds of disc block:

- **Physical accretion discs** — `multicolor` (Shakura & Sunyaev 1973),
  `kubota_done` (Kubota & Done 2018 three-zone), `relagn` (Hagen & Done 2023),
  `adaf` (Mahadevan 1997). These are specified by black-hole mass, accretion
  rate, and spin.
- **Template / empirical discs** — `qsogen`, `richards2006`, the CIGALE
  `skirtor`/`schartmann` discs. These have a *fixed* spectral shape scaled by an
  overall luminosity.

A geometrically-thin accretion disc has **three physical degrees of freedom**:
black-hole mass `M_BH`, accretion rate `Ṁ`, and spin `a`. Everything else is
derived by fixed physics:

- `L_Edd = 1.26e38 (M/M_sun) erg/s` (depends on M only),
- `η(a) = 1 − √(1 − 2/(3 r_isco(a)))` (Novikov–Thorne efficiency; spin only),
- `L_bol = η Ṁ c²` (radiated bolometric),
- `λ_Edd ≡ L_bol / L_Edd = Ṁ / Ṁ_Edd` (dimensionless accretion rate;
  `Ṁ_Edd = L_Edd/(η c²)`),
- `T_in ∝ (M Ṁ / r_in³)^{1/4} ∝ (λ_Edd / (η r_isco³ M))^{1/4}`.

The binding constraint is `L_bol = λ_Edd · L_Edd(M)` — **only two of
{L_bol, λ_Edd, M} are independent**. The disc *shape* (peak wavelength, colors)
is set by `λ_Edd` and `M`; the *amplitude* is `L_bol`. Specifying `L_bol`, `λ_Edd`
and `M` independently over-determines the disc and is unphysical.

### The bug this ADR closes (#846)

The physical discs took **both** `agn_log_lbol` (amplitude) **and**
`agn_log_ledd` (shape, `λ_Edd`) as *independent* free parameters: they built the
zone structure (`T_in`, `r_out`, corona zones) from `agn_log_ledd`, then rescaled
the spectrum to `agn_log_lbol`. This is the over-determined, unphysical case —
the disc *shape* corresponded to `agn_log_ledd` while the *luminosity* was a
disconnected rescale (`kubota_done` delivered only ~76% of the requested `L_bol`
over an X-ray-inclusive grid). Meanwhile the template discs already used
`agn_log_lbol` as their single knob, so the two halves of the menu spoke
different languages.

### How other codes parameterize the disc (upstream-verified)

- **RELAGN / qsosed** (the source `multicolor`/`kubota_done` cite): *accretion-
  first* `(M, λ_Edd, a)`. `relagn.py:288` `self.mdot = 10**log_mdot`
  (Eddington-normalized), `Mdot_edd = L_edd/(η c²)`. `L_bol` is a **derived
  output**; the knob is `λ_Edd`.
- **CIGALE** `skirtor2016`, **AGNfitter**, **ProSpect/Prospector**: a fixed disc
  **template** scaled by an overall luminosity (`agn_power` / `fagn`). Shape is
  frozen; luminosity is the only knob.

So physical-disc codes are accretion-first `(M, λ_Edd)`; template / SED-fitting
codes are luminosity-scaled. tengri must serve one composable menu spanning
both.

## Decision

**Luminosity-first.** `agn_log_lbol` (log₁₀ L_bol / L_sun) is the single,
universal AGN luminosity knob for **every** disc block.

- **Template discs** scale their fixed shape by `agn_log_lbol` (unchanged).
- **Physical discs** additionally take `agn_log_mbh` and `agn_a_spin`; the
  Eddington ratio is **derived**, `λ_Edd = L_bol / L_Edd(M_BH)`, and the zone
  structure (`Ṁ = L_bol/(η c²)`, `T_in`, `r_out`, corona split) is built from it.
  `(L_bol, M_BH, spin)` fully determines the disc — no over-determination, no
  free `λ_Edd`.
- **`agn_log_ledd` is retired** as a shape driver. It remains a declared
  parameter for backward compatibility but has **no effect** on the physical
  discs; setting or freeing it raises a build-time `UserWarning`
  (`Parameters.__init__`, Python-side / JIT-safe). It is removed from those
  blocks' `AGN_BLOCK_CONSUMES`, so a scoped `'*': FREE` no longer frees it.
- **`adaf`** is exempt (deprecated, #898); it still reads `agn_log_ledd` pending
  its rewrite.

### Rationale

1. **Physically self-consistent** — the disc shape now corresponds to the
   requested luminosity (`L_bol` delivered to ≤0.1% on an X-ray-inclusive grid).
2. **One language across the menu** — physical and template discs share the same
   luminosity knob, so a fit (or a recipe) treats them uniformly.
3. **Right coordinate for SED fitting** — `L_bol` is what the photometry/spectrum
   constrains directly; `(L_bol, M_BH)` → `λ_Edd` is the natural fit basis
   (equivalent 2-D manifold to RELAGN's `(M, λ_Edd)`, just re-coordinated).

The one cost is that we depart from RELAGN's `log_mdot` input for the physical
discs; a future RELAGN-parity mode could expose `(M, log_mdot)` and derive
`L_bol` if bit-parity with RELAGN's own driver is ever required, but the SED
values on the shared `(L_bol, M, spin)` manifold are identical.

## Consequences

- `disc.py` (`multicolor_disc`, `_compute_bh_params`, `_compute_zone_radii`) and
  the parallel `kd_precompute.py` derive `λ_Edd`/`Ṁ` from `agn_log_lbol`.
- Tests that asserted the old `agn_log_ledd`-driven shape (peak ∝ `λ_Edd`) are
  migrated to the equivalent luminosity-first statement (peak ∝ `L_bol` at fixed
  `M_BH`); tests that used super-Eddington `agn_log_lbol` values (which now clip
  at the Eddington limit) use physical luminosities.
- The `ss_disc` precompute grid axis (`agn_log_mdot`) is now degenerate and is
  tracked for migration to `agn_log_lbol` in #902.
