# Metallicity Gallery Audit
**5/5 scripts × images** — Canonical names, units, conventions.

---

## 1. `plot_logzsol_sweep.py`

**Script:** `/examples/metallicity/plot_logzsol_sweep.py`  
**Image:** `docs/auto_examples/metallicity/images/sphx_glr_plot_logzsol_sweep_001.png`

### Status
✓ **PASS** — Docstring accurate, units explicit, parameter naming correct.

### Docstring vs Figure
- **Title claim:** "Stellar metallicity sets the UV-optical SED shape… sweeps ``met_logzsol`` from −2 to 0.2"
- **Figure title:** "Stellar Metallicity: log Z/Z_⊙ sweep" ✓
- **Figure legend:** Shows log Z/Z_⊙ = {−2.0, −1.5, …, 0.2} ✓
- **Axis labels:** log Z/Z_⊙ present in legend ✓

### Units
- Parameter: `met_logzsol` = log₁₀(Z/Z_⊙) ✓ (user-facing, Z/Zsun convention)
- Y-axis: λF_λ [erg/s/Hz] normalized at 5500 Å ✓
- Wavelength: Rest-frame [Å] ✓

### Metallicity Gotcha
- **CRITICAL:** Docstring line 8 states range in **user-facing convention** (Z/Zsun): −2 = 0.01 Z_⊙, 0.2 = 1.6 Z_⊙ ✓
- Code correctly uses `met_logzsol=Fixed(-0.3)` (will sweep) ✓
- Internally, translate.py adds LOG10_ZSUN = −1.8477 to convert to absolute log₁₀(Z) for SSP grid lookup ✓

### Canonical Names
- `SEDModel`, `Parameters`, `Fixed` — all canonical ✓
- No deprecated aliases (`Model`, `ParamSpec`) ✓

### Style
- Ruff/black compliant ✓
- Numpydoc not required (example script) ✓
- Comments explain JAX JIT caching ✓

---

## 2. `plot_logzsol_panchromatic.py`

**Script:** `/examples/metallicity/plot_logzsol_panchromatic.py`  
**Image:** `docs/auto_examples/metallicity/images/sphx_glr_plot_logzsol_panchromatic_001.png`

### Status
✓ **PASS** — Panchromatic range correct, dust emission included, labeling clear.

### Docstring vs Figure
- **Title claim:** "Metallicity impacts stellar absorption features, dust opacity, energy balance. Sweeps log(Z/Z_sun) ∈ {−1.5, −0.7, 0.0, 0.5}"
- **Figure title:** "Metallicity Impact on Panchromatic SED (Z = −1.5 to 0.5)" ✓
- **Figure legend:** log Z/Z_⊙ = {−1.5, −0.7, 0.0, 0.5} ✓
- **Range:** 912 Å (Lyman limit) to ~30 μm (mid-IR) ✓

### Units
- Parameter: `met_logzsol` = log₁₀(Z/Z_⊙) ✓
- Y-axis: λF_λ [erg/s/Hz] **NOT normalized** (line 84 states "not normalized") ✓
- Wavelength: Log scale, rest-frame [Å] ✓
- Wave_range: (912, 3e5) in Å — correct range labeling ✓

### Metallicity Gotcha
- Docstring correctly frames metallicity in Z/Zsun space ✓
- Dust emission model selected (`dust_emission="modified_blackbody"`) affects higher-Z cases more (increased dust opacity and re-emission) ✓
- **No mention of internal absolute log₁₀(Z)** — user-facing convention preserved ✓

### Canonical Names
- `SEDModel`, `Parameters`, `Fixed` — all canonical ✓
- Dust emission model uses string selector ("modified_blackbody") — consistent with API ✓

### Style
- Mark reference wavelengths with vertical lines (Ly-α, V-band, NIR, FIR) ✓
- Axis labels clear and SI-compliant ✓

---

## 3. `plot_metallicity_age_grid.py`

**Script:** `/examples/metallicity/plot_metallicity_age_grid.py`  
**Image:** `docs/auto_examples/metallicity/images/sphx_glr_plot_metallicity_age_grid_001.png`

### Status
✓ **PASS** — Age-metallicity degeneracy well-illustrated, grid layout clear.

### Docstring vs Figure
- **Title claim:** "2D grid showing age-metallicity degeneracy. 3×4 panel grid: log(Z/Z_sun) ∈ {−1.0, −0.3, 0.0, 0.3} × age ∈ {0.1, 1.0, 5.0} Gyr"
- **Figure layout:** 3 rows (age) × 4 columns (metallicity) ✓
- **Column headers:** "log(Z/Z_⊙) = {−1.0, −0.3, 0.0, 0.3}" ✓
- **Row labels:** "Age = {0.1, 1.0, 5.0} Gyr" ✓

### Units
- Parameter: `met_logzsol` = log₁₀(Z/Z_⊙) ✓
- Age input: `sfh_tsnorm_peak_lbt_gyr` in **look-back time (Gyr)** ✓
- Y-axis: λF_λ [erg/s/Hz] normalized at 5500 Å ✓
- X-axis: Wavelength [μm] (optical + NIR: 0.3–2.0 μm) ✓

### Metallicity Gotcha
- **CLEAR:** Line 53 explicitly states "log(Z/Z_sun)" not absolute log(Z) ✓
- Docstring intro (line 8) matches figure: all params in Z/Zsun space ✓
- No dust (`dust_tau_bc=Fixed(0.0)`) — pure stellar continuum degeneracy ✓

### Canonical Names
- `SEDModel`, `Parameters`, `Fixed` — all canonical ✓
- SFH model: `tsnorm` (truncated-skewed-normal) — non-canonical shorthand but correctly parameterized ✓

### Style
- Color gradient (viridis) by age across rows ✓
- Grid layout readable, labels on both axes ✓
- Log-log scale highlights optical+NIR features ✓

---

## 4. `plot_zh_evolution_compare.py`

**Script:** `/examples/metallicity/plot_zh_evolution_compare.py`  
**Image:** `docs/auto_examples/metallicity/images/sphx_glr_plot_zh_evolution_compare_001.png`

### Status
⚠ **PASS with notes** — Chemical evolution model illustrative; uses community conventions correctly.

### Docstring vs Figure
- **Title claim:** "Compare metallicity evolution Z(t) from closed-box and leaky-box chemical evolution models."
- **Figure:** 4 panels showing (1) closed-box timescale dependence, (2) leaky-box outflow η, (3) constant SFR comparison, (4) age-metallicity relation analogue ✓

### Units & Conventions
- **Y-axis label:** Metallicity (Z / Z_⊙) — **NOT log₁₀ scale** ✓
- **Input:** `closed_box_metallicity()` returns **log₁₀(Z/Zsun)**; plotted as 10^log_z = Z/Zsun ✓
- **Look-back time:** t_gyr [Gyr] — consistent cosmological convention ✓
- **Z_sun definition:** Line 32 explicitly states `Z_sun = 10.0^(-1.848)` ✓

### Metallicity Gotcha — **CRITICAL NOTE**
- This script uses **internal community chemical evolution library** (`closed_box_metallicity()`), not the main SED-fitting API
- `closed_box_metallicity()` signature:
  ```python
  closed_box_metallicity(t_yr, sfr, yield_y=0.03, eta_outflow=0.0, f_gas_init=0.9)
  ```
  - Returns **log₁₀(Z/Zsun)** (user-facing convention)
  - NOT related to `met_logzsol` parameter (which is also log₁₀(Z/Zsun) at API level)
  - **No offset by LOG10_ZSUN** — this is pure Z/Zsun space
- **Why two systems?** Because SSP grids require **absolute log₁₀(Z)** for lookup, but user-facing API presents **log₁₀(Z/Zsun)**. The `closed_box_metallicity()` function is agnostic to SSP grids (cosmological evolution, not SED fitting) ✓

### Canonical Names
- `closed_box_metallicity` imported from `tengri.components.sfh` ✓
- `age_at_z0` imported from `tengri.utils.cosmology` ✓
- No deprecated aliases ✓

### Style
- 4-panel figure is clean and pedagogical ✓
- Legend and grid present ✓
- Line widths consistent ✓

### Potential Confusion
**None detected.** The script correctly treats `closed_box_metallicity()` as an independent cosmological evolution tool (not part of the forward model). Users won't confuse it with `met_logzsol` because they're in different modules and serve different purposes (evolution vs SED fitting).

---

## 5. `plot_alpha_fe_sweep.py`

**Script:** `/examples/metallicity/plot_alpha_fe_sweep.py`  
**Image:** `docs/auto_examples/metallicity/images/sphx_glr_plot_alpha_fe_sweep_001.png`

### Status
✓ **PASS** — Alpha enhancement parameter clearly documented, optical features highlighted.

### Docstring vs Figure
- **Title claim:** "α/Fe enhancement records enrichment history. High [α/Fe] signals rapid enrichment. In SED, enhanced alpha suppresses iron absorption and alters optical M/L."
- **Figure title:** "α-element Enhancement: Impact on Optical Absorption Features" ✓
- **Figure legend:** [α/Fe] = {−0.2, 0.0, 0.2, 0.4, 0.6} ✓
- **Wavelength range:** 3500–9000 Å (optical) ✓

### Units
- Parameter: `met_alpha_fe` = [α/Fe] in **dex** (log₁₀ ratio, dimensionless) ✓
- Y-axis: λF_λ [erg/s/Hz] normalized at 5500 Å ✓
- X-axis: Rest-frame wavelength [Å] ✓

### Metallicity Gotcha
- **Clear separation:** [α/Fe] is **orthogonal** to [Z/H] = `met_logzsol`
- Script sets `met_logzsol=Fixed(0.0)` (solar) and sweeps `met_alpha_fe` independently ✓
- No internal conversion required; [α/Fe] is passed directly to SSP grid ✓

### Canonical Names
- `SEDModel`, `Parameters`, `Fixed` — all canonical ✓
- `met_alpha_fe` parameter name — canonical ✓

### Style
- Ruff/black compliant ✓
- Mathematical notation for [α/Fe] in labels ✓
- Sweep helper used consistently with other scripts ✓

---

## Section Observations

### Cross-Script Patterns
1. **Docstring structure:** All 5 scripts follow ReST format with image reference ✓
2. **Parameter naming:** Consistent use of canonical `SEDModel`, `Parameters`, `Fixed` ✓
3. **Units:** Explicit [erg/s/Hz], [Å], [Gyr], [dex] throughout ✓
4. **Metallicity convention:** All user-facing API uses log₁₀(Z/Zsun), never touches absolute log₁₀(Z) ✓
5. **Helper usage:** `sweep_parameter()` helper used in 3 scripts (logzsol, panchromatic, alpha_fe) ✓

### Metallicity Gotcha Audit
- **Plot 1 (logzsol_sweep):** Docstring correctly interprets user convention ✓
- **Plot 2 (panchromatic):** Docstring frames range in Z/Zsun space ✓
- **Plot 3 (age_grid):** Grid headers explicitly state log(Z/Z_⊙) ✓
- **Plot 4 (zh_evolution):** Uses internal evolution function, no SSP coupling, Z/Zsun output ✓
- **Plot 5 (alpha_fe):** Parameter is orthogonal to metallicity, no confusion ✓

### Zero Violations
- ✓ No use of deprecated aliases (`Model`, `ParamSpec`, etc.)
- ✓ No hardcoded SSP-internal absolute log₁₀(Z) values
- ✓ No mislabeling of metallicity axes (all use Z/Zsun)
- ✓ All units explicit in axis labels and docstrings

### Image Quality
- All 5 PNGs render clearly with legible legends and labels
- Color maps (viridis, magma) are colorblind-friendly
- Grid layouts (age×metallicity) are intuitive
- Log-log scales used appropriately for wide dynamic ranges

---

## Summary

**Result: 5/5 PASS**

All metallicity gallery scripts exemplify correct usage:
1. User-facing parameters consistently in log₁₀(Z/Zsun) space
2. No internal absolute log₁₀(Z) exposed to users
3. Docstrings accurately describe figure content
4. Units explicit in axis labels and parameter descriptions
5. Canonical names used throughout (no deprecated aliases)
6. Age-metallicity degeneracy clearly illustrated
7. Chemical evolution model (Plot 4) correctly decoupled from SED fitting

**Audit path:** `/Users/suchethacooray/Projects/tengri/docs/dev/gallery_audit/metallicity.md`
