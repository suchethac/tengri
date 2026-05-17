# Radio Gallery Audit

## Summary

Counter: **4/4** scripts audited with PNG visual analysis.

All four radio example scripts are **PASS** with no critical issues. Physics are correctly implemented (synchrotron spectral index, FIR-radio correlation, AGN loudness). Units documented, canonical names used. State dependencies properly documented. Style is consistent; no linting violations.

---

## Detailed Audit

### 1. plot_alpha_sf_sweep.py

**Script path:** `/Users/suchethacooray/Projects/tengri/examples/radio/plot_alpha_sf_sweep.py`

**Status:** ✓ PASS

**Docstring & Physics:**
- Title: "Radio Spectral Index (α_sf)"
- Correctly explains synchrotron spectral index (0.7–0.8 typical for star-forming galaxies)
- Notes flat-spectrum free-free (α ≈ 0) and steeper aged spectra (α > 1)
- Links to synchrotron physics; well-motivated

**Visual (PNG: `sphx_glr_plot_alpha_sf_sweep_001.png`):**
- **Looks good:** Clean loglog plot, 6 alpha values [0.3, 0.5, 0.7, 0.8, 1.0, 1.2]
- **SED shape:** Correct inverted-axis GHz scale (200–0.1 GHz, decreasing left-to-right)
- **Spectral slopes:** All curves show power-law behavior; steeper lines at higher α, as expected
- **Reference:** Vertical dashed line at 1.4 GHz normalization point (standard radio anchor)
- **Normalisation:** Y-axis normalized at 1.4 GHz; allows clean spectral-index comparison
- **Color gradient:** Cyan→magenta span; visually clear separation

**Code:**
- Line 29: `from tengri.radio import radio_star_forming` — correct alias (backward compat for `radio_sfr_bell2003`)
- Line 33: `wave = jnp.logspace(7, 11, 600)` — radio wavelengths 1 mm to 10 m (Å)
- Line 34: `L_ir = 1e11 L_sun` — fixed IR luminosity; alpha sweep orthogonal to luminosity
- Lines 44–46: Calls `radio_star_forming()` (alias to `radio_sfr_bell2003`) with q_ir=2.64 (default)
- **L_ir dependency:** ✓ `radio_star_forming()` does NOT read `state.derived["L_ir"]`; passes it as explicit arg (line 45: `L_ir=L_ir`). This is correct for the examples (they use simple functions, not the full forward model).
- Line 47: `L_nu_norm = ... / float(L_nu_ref[0])` — safe normalization; handles JAX arrays
- Line 48: `nu_ghz = (3e18 / np.array(wave)) / 1e9` — wavelength to GHz conversion correct
- Line 54: `ax.invert_xaxis()` — standard radio presentation (higher freq left)

**Units:**
- ✓ Y-axis labeled "L_ν (normalized at 1.4 GHz)" — correct dimensionless
- ✓ X-axis labeled "Frequency [GHz]" — clear

**Style:**
- ✓ No ruff violations detected
- ✓ Canonical names used (radio_star_forming)
- ✓ Comments explain wavelength range and normalization

---

### 2. plot_q_ir_sweep.py

**Script path:** `/Users/suchethacooray/Projects/tengri/examples/radio/plot_q_ir_sweep.py`

**Status:** ✓ PASS

**Docstring & Physics:**
- Title: "FIR-Radio Correlation (q_IR)"
- Correctly defines q_IR = log10(L_IR / 3.75×10^12 L_1.4GHz) per Bell 2003
- Note: Higher q_IR → less radio per unit star formation (correct inverse relationship)
- Canonical value 2.64 cited with authority (Bell 2003)

**Visual (PNG: `sphx_glr_plot_q_ir_sweep_001.png`):**
- **Looks good:** Loglog plot, 5 q_ir values [2.0, 2.3, 2.64, 3.0, 3.3]
- **L_IR constant:** All curves use same IR luminosity (10^11 L_sun, ULIRG-like)
- **Spectral shape:** All curves have same synchrotron slope (α=0.8); q_ir changes absolute normalization only
- **Visible trend:** Cyan (q=2.0, more radio) to magenta (q=3.3, less radio) — correct inverse effect
- **Axis range:** 200–0.1 GHz; standard radio band
- **Luminosity range:** 1e-8 to 1e2 erg/s/Hz — appropriate dynamic range

**Code:**
- Line 29: `from tengri.radio import radio_star_forming` — correct
- Line 34: `wave = jnp.logspace(7, 11, 600)` — radio wavelengths (Å)
- Line 36: `L_ir = 1e11 L_sun` — fixed; allows pure q_ir sweep
- Lines 44–46: Calls `radio_star_forming()` with varying q_ir, fixed alpha_sf=0.8
- **L_ir dependency:** ✓ Passed as explicit arg; not read from state
- Line 47: `nu_ghz` conversion correct
- Line 54: `ax.set_ylim(1e-8, 1e2)` — appropriate range for ULIRG

**Units:**
- ✓ Y-axis labeled "L_ν [erg s^{-1} Hz^{-1}]" — explicit erg/s/Hz
- ✓ X-axis labeled "Frequency [GHz]"

**Style:**
- ✓ No ruff violations
- ✓ Legible legend with q_ir values
- ✓ Title includes fixed L_IR value for context

---

### 3. plot_radio_lir_relation.py

**Script path:** `/Users/suchethacooray/Projects/tengri/examples/radio/plot_radio_lir_relation.py`

**Status:** ✓ PASS

**Docstring & Physics:**
- Title: "FIR-Radio Correlation: L_IR Luminosity Sweep"
- Correctly explains far-infrared → radio link via FIRRC
- Sweeps log_L_IR ∈ {10, 11, 12, 13} L_sun (starburst → ULIRG range)
- Uses canonical q_IR = 2.64 (Bell 2003), allowing pure L_IR effect
- Clear pedagogical intent: "more luminous starbursts → stronger radio"

**Visual (PNG: `sphx_glr_plot_radio_lir_relation_001.png`):**
- **Looks good:** Loglog plot, 4 L_IR values [10^10, 10^11, 10^12, 10^13 L_sun]
- **Strong vertical spread:** L_IR=10^13 (yellow) ~10^4× brighter than 10^10 (purple) at fixed frequency
- **Parallel slopes:** All curves have same synchrotron slope (α=0.8); vertical shifts only
- **Axis range:** 200–0.1 GHz; 1e-5 to 1e7 erg/s/Hz
- **Trend clear:** Dark → bright color span matches starburst → ULIRG luminosity increase

**Code:**
- Line 39: `wave = jnp.logspace(7, 11, 600)` — radio wavelengths (Å)
- Lines 42–44: Sweeps log_L_IR; properly formats as exponent labels
- Line 47: `q_ir = 2.64` — fixed canonical value
- Lines 54–57: Calls `radio_star_forming()` with varying L_ir, q_ir and alpha_sf fixed
- **L_ir dependency:** ✓ Passed as explicit arg; not read from state
- Line 59: `nu_ghz` conversion correct
- Line 73: `ax.set_ylim(1e-5, 1e7)` — wide span covers 10^10–10^13 L_sun range
- Line 79: `ax.grid(True, alpha=0.3, which="both")` — light grid aids loglog readability
- Lines 83–84: Safely saves to script directory (fallback to "." if `__file__` unavailable)

**Units:**
- ✓ Y-axis labeled "L_ν [erg s^{-1} Hz^{-1}]"
- ✓ X-axis labeled "Frequency [GHz]"
- ✓ Title includes fixed q_IR value (2.64) for context

**Style:**
- ✓ No ruff violations
- ✓ Uses `Path` for cross-platform path handling (good practice)
- ✓ Handles matplotlib backend explicitly (`matplotlib.use("Agg")`)
- ✓ Uses `plt.close()` to release figure (memory-conscious)

---

### 4. plot_radio_loudness_sweep.py

**Script path:** `/Users/suchethacooray/Projects/tengri/examples/radio/plot_radio_loudness_sweep.py`

**Status:** ✓ PASS

**Docstring & Physics:**
- Title: "AGN Radio Loudness (R)"
- Correctly defines R = log10(L_5GHz / L_B)
- Correctly states radio-quiet range (R ≲ 1) and radio-loud range (R ~ 3–5)
- Correctly notes "each decade in R adds order of magnitude to jet luminosity"
- AGN context is clear; distinct from star-forming radio

**Visual (PNG: `sphx_glr_plot_radio_loudness_sweep_001.png`):**
- **Looks good:** Loglog plot, 5 radio_loudness values [0.0, 1.0, 2.0, 3.0, 4.0]
- **Spectral shape:** All curves have same synchrotron slope (α=0.7); vertical shifts reflect loudness
- **Loudness trend:** Light red (quiet) → dark red (R=4.0, loud) — clear visual hierarchy
- **Vertical span:** Each decade in R adds ~1 dex to L_nu (power-law shift)
- **Axis range:** 200–0.1 GHz; 1e-8 to 1e2 erg/s/Hz (appropriate for AGN jets)

**Code:**
- Line 29: `from tengri.radio import radio_agn` — correct import
- Line 33: `wave = jnp.logspace(7, 11, 600)` — radio wavelengths (Å)
- Line 34: `L_agn_bol = 1e44 erg/s` — Seyfert-1-like luminosity
- Line 36: `L_agn_bol_lsun = L_agn_bol / 3.828e33` — converts erg/s to L_sun (correct factor)
- Line 39: `cmap = plt.get_cmap("Reds")` — appropriate for AGN (distinct from star-forming blue/cyan)
- Lines 47–51: Calls `radio_agn()` with varying radio_loudness, fixed alpha_agn=0.7
- **L_agn_bol dependency:** ✓ Passed as explicit arg; not read from state
- Line 50: Handles R=0 with special label "radio-quiet" (good UX)
- Line 56–57: Appropriate y-limits for AGN range (1e-8 to 1e2)

**Units:**
- ✓ Y-axis labeled "L_ν [erg s^{-1} Hz^{-1}]"
- ✓ X-axis labeled "Frequency [GHz]"
- ✓ Title uses standard R definition notation

**Style:**
- ✓ No ruff violations
- ✓ Color scheme (red) visually distinct from SF examples (cyan–magenta)
- ✓ Legible legend; distinguishes "radio-quiet" from numbered R values

---

## Cross-Cutting Observations

### State Dependency Handling
All four scripts use **explicit function arguments** rather than reading from `state.derived["L_ir"]` or similar:

- **plot_alpha_sf_sweep.py:** `L_ir=L_ir` (line 45)
- **plot_q_ir_sweep.py:** `L_ir=L_ir` (line 46)
- **plot_radio_lir_relation.py:** `L_ir=L_ir` (line 56)
- **plot_radio_loudness_sweep.py:** `L_agn_bol=L_agn_bol_lsun` (line 48)

This is **correct** for stand-alone examples. The radio component (in `component.py`, lines 261–264) properly reads upstream dependencies with fallbacks:
```python
L_ir = jnp.asarray(state.derived.get("L_ir", 0.0))
L_agn_bol = jnp.asarray(state.derived.get("L_agn_bol", 0.0))
log_mstar = jnp.asarray(state.derived.get("log_mstar", 10.0))
```

No flags needed; examples and component are in consistent patterns.

### Units & Constants
- All wavelength conversions use c = 3e18 Å·Hz (correct)
- L_sun conversions use 3.828e33 erg/s (standard IAU 2015)
- Electron temperature T_e = 1e4 K (Murphy+2011 default, not overridden in examples)
- All outputs correctly documented as erg/s/Hz

### Canonical Names
- ✓ `radio_star_forming` (alias to `radio_sfr_bell2003`) used correctly
- ✓ `radio_agn` (single power-law) used in loudness sweep
- ✓ No deprecated names (ParamSpec, LineCatalog, etc.)

### Visual QA
All four PNGs pass visual inspection:
- ✓ Axes labeled clearly (GHz, erg/s/Hz)
- ✓ Frequency axis inverted (standard radio convention)
- ✓ Frequency range 200–0.1 GHz (mm–dm wavelengths, appropriate radio band)
- ✓ Legends readable; parameter values clearly marked
- ✓ Titles match docstring summary
- ✓ No broken curves, axis artifacts, or label overflow

---

## Conclusion

**Radio section: 4/4 PASS**

All four example scripts are production-ready. Physics correct, units documented, canonical names used, state dependencies properly factored (examples use function args; component reads `state.derived` with fallbacks). No linting violations. Visual output clean and pedagogically clear.

No action items.
