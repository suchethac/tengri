# Unified AGN Model for tengri

See full implementation plan at `~/.claude/plans/cuddly-purring-sketch.md`.

## What is inherited vs novel

**Reimplemented in pure JAX (with credit to original codes/papers):**
- Emission tree (incident/transmitted/nebular/escaped): following Synthesizer UnifiedAGN (Lovell+2025, Appendix F.3.1)
- K&D 3-zone disc + self-consistent corona: reimplemented in JAX following qsosed (Quera-Bofarull) and RELAGN (Hagen & Done 2023). Original packages use numpy/scipy; our JAX implementation enables JIT compilation and autodiff.
- alpha_ox, X-ray anisotropy, polar dust, delta_AGN: reimplemented in JAX from X-CIGALE (Yang+2020, 2022)
- CAT3D-Wind torus, DPL radio: template interpolation reimplemented in JAX from AGNfitter-rx (Martinez-Ramirez+2024)
- Feltre NLR grids: grid loading + JAX interpolation from BEAGLE-AGN (Vidal-Garcia+2024)
- Cue emulator: already reimplemented in pure JAX in tengri (Li+2025)

**Novel to tengri:**
1. Chain 2: disc EUV -> Cue ionspec -> NLR lines (no other code does this)
2. Free N/O, C/O in NLR via Cue in a differentiable SED fitting code
3. Unified preset system (xcigale, beagle_agn, agnfitter, physical configs)
4. Full JAX differentiability + gradient-based inference (geoVI, NUTS, RT) for 20+ AGN params

## Parameter naming: `agn_{subsystem}_{name}`

All AGN params use hierarchical prefixes to avoid collisions:
- `agn_disc_*`: disc parameters
- `agn_torus_*`: torus parameters
- `agn_nlr_*`: narrow-line region
- `agn_blr_*`: broad-line region
- `agn_xray_*`: X-ray/corona
- `agn_radio_*`: radio
- `agn_polar_*`: polar dust
- Engine params: `agn_log_mbh`, `agn_log_ledd`, `agn_spin`, `agn_cos_inc`
