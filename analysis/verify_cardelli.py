"""Verify tengri Cardelli et al. (1989) against reference implementation.

Compares the tengri cardelli() function against a direct transcription of
the CCM89 paper equations (matching the extinction.pyx reference by Barbary).

Usage:
    python analysis/verify_cardelli.py
"""

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np


# ── Reference implementation (from extinction.pyx / CCM89 paper) ─────────


def cardelli_reference(wavelength_aa: np.ndarray, r_v: float = 3.1) -> np.ndarray:
    """Reference CCM89 A(lambda)/A(V), valid 0.3 <= x <= 10 um^-1.

    Transcribed from kbarbary/extinction (extinction.pyx) which follows
    the original Cardelli, Clayton & Mathis (1989) Table 3 + Equations 1-4.
    Includes the far-UV extension (8 < x < 10).
    """
    wave_um = wavelength_aa / 1e4
    x = 1.0 / wave_um
    a = np.zeros_like(x)
    b = np.zeros_like(x)

    # IR: 0.3 <= x < 1.1
    ir = (x >= 0.3) & (x < 1.1)
    if np.any(ir):
        y = x[ir] ** 1.61
        a[ir] = 0.574 * y
        b[ir] = -0.527 * y

    # Optical: 1.1 <= x < 3.3
    opt = (x >= 1.1) & (x < 3.3)
    if np.any(opt):
        y = x[opt] - 1.82
        a[opt] = (1.0 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3
                  + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6
                  + 0.32999 * y**7)
        b[opt] = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3
                  - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6
                  - 2.09002 * y**7)

    # UV: 3.3 <= x < 8.0
    uv = (x >= 3.3) & (x < 8.0)
    if np.any(uv):
        xu = x[uv]
        a[uv] = 1.752 - 0.316 * xu - 0.104 / ((xu - 4.67)**2 + 0.341)
        b[uv] = -3.090 + 1.825 * xu + 1.206 / ((xu - 4.62)**2 + 0.263)
        fuv = xu >= 5.9
        if np.any(fuv):
            y = xu[fuv] - 5.9
            a[uv] = np.where(xu >= 5.9,
                             a[uv] + (-0.04473 * (xu - 5.9)**2
                                       - 0.009779 * (xu - 5.9)**3),
                             a[uv])
            b[uv] = np.where(xu >= 5.9,
                             b[uv] + (0.2130 * (xu - 5.9)**2
                                       + 0.1207 * (xu - 5.9)**3),
                             b[uv])

    # Far-UV: 8.0 <= x <= 10.0  (CCM89 Table 4 / extinction.pyx)
    fuv = (x >= 8.0) & (x <= 10.0)
    if np.any(fuv):
        y = x[fuv] - 8.0
        a[fuv] = -0.070 * y**3 + 0.137 * y**2 - 0.628 * y - 1.073
        b[fuv] = 0.374 * y**3 - 0.420 * y**2 + 4.257 * y + 13.670

    k = a + b / r_v
    return k  # A(lambda)/A(V), NOT clipped


# ── tengri implementation ───────────────────────────────────────────────

from tengri.models.dust.attenuation import cardelli as tengri_cardelli  # noqa: E402


# ── Comparison ───────────────────────────────────────────────────────────

def main() -> None:
    r_v = 3.1

    # Wavelength grid: 900 to 30000 Angstrom
    wave_aa = np.linspace(900.0, 30000.0, 5000)
    x = 1.0 / (wave_aa / 1e4)  # um^-1

    # Reference
    ref = cardelli_reference(wave_aa, r_v=r_v)

    # tengri (returns jax array)
    ds = np.asarray(tengri_cardelli(jnp.array(wave_aa), dust_Rv=r_v))

    # Reference clipped (what tengri does)
    ref_clipped = np.clip(ref, 0.0, None)

    # ── Print diagnostics ────────────────────────────────────────────
    print("=" * 72)
    print("Cardelli CCM89 verification: tengri vs reference")
    print("=" * 72)

    # 1. Check V-band normalization (5500 A => x = 1.818)
    v_idx = np.argmin(np.abs(wave_aa - 5500.0))
    print(f"\nV-band (5500 A): tengri={ds[v_idx]:.6f}  ref={ref[v_idx]:.6f}"
          f"  (should be ~1.0)")

    # 2. Max absolute difference across valid range (x <= 8)
    valid = x <= 8.0
    diff_valid = np.abs(ds[valid] - ref_clipped[valid])
    print(f"\nMax |diff| for x <= 8.0 (1250-30000 A): {diff_valid.max():.2e}")
    if diff_valid.max() < 1e-10:
        print("  => PASS: IR + Optical + UV regimes match exactly.")
    else:
        print("  => FAIL: mismatch in IR/Optical/UV regimes!")
        worst = np.argmax(diff_valid)
        ww = wave_aa[valid][worst]
        print(f"     Worst at {ww:.0f} A (x={1e4/ww:.2f}): "
              f"tengri={ds[valid][worst]:.6f} ref={ref_clipped[valid][worst]:.6f}")

    # 3. Far-UV regime (x > 8, wavelength < 1250 A)
    fuv = x > 8.0
    if np.any(fuv):
        print(f"\n--- Far-UV (x > 8.0, lambda < 1250 A) ---")
        print(f"  Number of wavelength points: {fuv.sum()}")

        # tengri uses UV formula beyond x=8 (no far-UV branch)
        diff_fuv = ds[fuv] - ref_clipped[fuv]
        print(f"  Max |diff|: {np.abs(diff_fuv).max():.4f}")
        print(f"  tengri range: [{ds[fuv].min():.4f}, {ds[fuv].max():.4f}]")
        print(f"  reference range: [{ref[fuv].min():.4f}, {ref[fuv].max():.4f}]")
        print(f"  ref (clipped) range: [{ref_clipped[fuv].min():.4f}, "
              f"{ref_clipped[fuv].max():.4f}]")

        # Show a few sample points
        sample_waves = [900, 1000, 1100, 1200]
        print(f"\n  {'Wave(A)':>8s} {'x(um^-1)':>8s} {'tengri':>10s} "
              f"{'ref':>10s} {'ref_clip':>10s} {'diff':>10s}")
        for ww in sample_waves:
            idx = np.argmin(np.abs(wave_aa - ww))
            print(f"  {wave_aa[idx]:8.0f} {x[idx]:8.3f} {ds[idx]:10.4f} "
                  f"{ref[idx]:10.4f} {ref_clipped[idx]:10.4f} "
                  f"{ds[idx]-ref_clipped[idx]:10.4f}")

    # 4. Check effect of clipping
    neg = ref < 0
    if np.any(neg):
        print(f"\n--- Clipping analysis ---")
        print(f"  Reference goes negative at {neg.sum()} points")
        print(f"  Wavelength range: {wave_aa[neg].min():.0f} - "
              f"{wave_aa[neg].max():.0f} A")
        print(f"  Min reference value (before clip): {ref[neg].min():.4f}")
    else:
        print(f"\n--- Clipping analysis ---")
        print(f"  Reference never goes negative in 900-30000 A range.")

    # 5. Summary of issues
    print("\n" + "=" * 72)
    print("SUMMARY OF ISSUES")
    print("=" * 72)

    issues_found = False

    # Issue 1: Missing far-UV
    if np.any(fuv):
        fuv_diff = np.abs(ds[fuv] - ref_clipped[fuv]).max()
        if fuv_diff > 0.01:
            issues_found = True
            print(f"\n[ISSUE 1] MISSING FAR-UV REGIME (x > 8, lambda < 1250 A)")
            print(f"  tengri applies the UV formula (valid for 3.3 < x < 8.0)")
            print(f"  beyond its domain. CCM89 Table 4 defines separate")
            print(f"  polynomials for 8 < x < 10:")
            print(f"    y = x - 8.0")
            print(f"    a(x) = -0.070*y^3 + 0.137*y^2 - 0.628*y - 1.073")
            print(f"    b(x) =  0.374*y^3 - 0.420*y^2 + 4.257*y + 13.670")
            print(f"  Max error: {fuv_diff:.4f} in A(lambda)/A(V)")

    # Issue 2: Clipping hiding problems
    ds_noclip = np.asarray(
        _cardelli_noclip(jnp.array(wave_aa), r_v)
    )
    goes_neg = ds_noclip < 0
    if np.any(goes_neg):
        issues_found = True
        print(f"\n[ISSUE 2] CLIPPING HIDES UNPHYSICAL VALUES")
        print(f"  Without clip, tengri cardelli goes negative at "
              f"{goes_neg.sum()} points")
        print(f"  Wavelength range: {wave_aa[goes_neg].min():.0f} - "
              f"{wave_aa[goes_neg].max():.0f} A")
        print(f"  Min value (no clip): {ds_noclip[goes_neg].min():.4f}")
        print(f"  This is because the UV formula extrapolated beyond x=8")
        print(f"  produces unphysical negative extinction.")

    # Issue 3: Normalization check
    v_err = abs(ds[v_idx] - 1.0)
    if v_err > 0.01:
        issues_found = True
        print(f"\n[ISSUE 3] V-BAND NORMALIZATION ERROR")
        print(f"  A(V)/A(V) should be 1.0, got {ds[v_idx]:.6f}")

    if not issues_found:
        print("\n  No issues found. Implementation matches reference.")

    print()


def _cardelli_noclip(wavelength: jnp.ndarray, r_v: float) -> jnp.ndarray:
    """tengri cardelli WITHOUT the final clip, for diagnostics."""
    wave_um = wavelength / 1e4
    x = 1.0 / wave_um

    a_ir = 0.574 * x**1.61
    b_ir = -0.527 * x**1.61

    y = x - 1.82
    a_opt = (1.0 + 0.17699 * y - 0.50447 * y**2 - 0.02427 * y**3
             + 0.72085 * y**4 + 0.01979 * y**5 - 0.77530 * y**6
             + 0.32999 * y**7)
    b_opt = (1.41338 * y + 2.28305 * y**2 + 1.07233 * y**3
             - 5.38434 * y**4 - 0.62251 * y**5 + 5.30260 * y**6
             - 2.09002 * y**7)

    f_a = jnp.where(x >= 5.9,
                     -0.04473 * (x - 5.9)**2 - 0.009779 * (x - 5.9)**3, 0.0)
    f_b = jnp.where(x >= 5.9,
                     0.2130 * (x - 5.9)**2 + 0.1207 * (x - 5.9)**3, 0.0)
    a_uv = 1.752 - 0.316 * x - 0.104 / ((x - 4.67)**2 + 0.341) + f_a
    b_uv = -3.090 + 1.825 * x + 1.206 / ((x - 4.62)**2 + 0.263) + f_b

    a = jnp.where(x < 1.1, a_ir, jnp.where(x < 3.3, a_opt, a_uv))
    b = jnp.where(x < 1.1, b_ir, jnp.where(x < 3.3, b_opt, b_uv))

    return a + b / r_v


if __name__ == "__main__":
    main()
