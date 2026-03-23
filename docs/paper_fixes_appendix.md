# Paper Appendix Fix Suggestions

Pass this to the paper-writing agent for corrections.

---

## Fix 1: Dust attenuation sigmoid (Eq. 17)

**Current paper:** Eq. 17 shows a sharp step function at `t_age = 10^7 yr`:

```latex
\tau_\lambda(t_{\rm age}) = \begin{cases}
  (\tau_{\rm BC} + \tau_{\rm diff}) \, k(\lambda) & t_{\rm age} < 10^7\,\text{yr} \\
  \tau_{\rm diff} \, k(\lambda) & t_{\rm age} \geq 10^7\,\text{yr}
\end{cases}
```

**What the code actually does:** A differentiable sigmoid transition with width 0.3 dex in log-age space:

```latex
w(t_{\rm age}) = \sigma\!\left(-\frac{\log_{10}(t_{\rm age}) - 7.0}{0.3}\right)
```

where `\sigma(x) = 1/(1+e^{-x})` is the logistic sigmoid, and the total optical depth is:

```latex
\tau_\lambda(t_{\rm age}) = w(t_{\rm age}) \, \tau_{\rm BC} \, k_{\rm BC}(\lambda)
                           + \tau_{\rm diff} \, k_{\rm diff}(\lambda)
```

**Why this matters:** The sigmoid spans ~5-20 Myr, which is physically meaningful (stars don't instantly escape their birth clouds). It also enables JAX autodiff — a hard step function has zero gradient everywhere except at the discontinuity, which breaks gradient-based inference.

**Suggested fix:** Replace Eq. 17 with the sigmoid form, and add a sentence:

> "Following Charlot \& Fall (2000), young stellar populations experience additional
> attenuation from their birth clouds. We implement this as a differentiable sigmoid
> transition centred at $10^7$~yr with a width of 0.3~dex in $\log_{10}(t_{\rm age})$,
> enabling exact gradient computation via automatic differentiation. The transition
> spans approximately 5--20~Myr, consistent with the typical timescale for molecular
> cloud dispersal (Chevance et al.~2020)."

---

## Fix 2: Jacobian convention in geoVI vs MGVI (Appendix C)

**Current paper:**
- geoVI: `M = J^T J + I` where `J = \partial x / \partial \xi`
- MGVI: `M = J^T N^{-1} J + I`

These are inconsistent unless J means different things in the two sections.

**What the code does:**
- The geoVI metric uses the **whitened** Jacobian: `J_w = N^{-1/2} \cdot (\partial m / \partial \xi)`, so `M = J_w^T J_w + I`
- The MGVI metric uses the **raw** Jacobian: `J = \partial m / \partial \xi`, with explicit noise: `M = J^T N^{-1} J + I`

Both are mathematically equivalent (`J_w^T J_w = J^T N^{-1} J`), but the notation should be consistent.

**Suggested fix (option A — define J consistently):**

> In both sections, define $J \equiv \partial \boldsymbol{m} / \partial \boldsymbol{\xi}$
> as the raw Jacobian. Then write:
> - geoVI: $M = J^T N^{-1} J + \mathbb{I}$
> - MGVI: $M = J^T N^{-1} J + \mathbb{I}$
>
> Note: these are the same Fisher metric. The difference between geoVI and MGVI lies
> in the **sampling strategy** (geometric vs mean-field), not in the metric definition.

**Suggested fix (option B — note the whitening):**

> "In the geoVI implementation, we absorb the noise covariance into the Jacobian
> via whitening: $\tilde{J} = N^{-1/2} J$, so that $M = \tilde{J}^T \tilde{J} + \mathbb{I}$."

---

## Fix 3: Metallicity upper bound (Appendix A parameter table)

**Current paper:** `log(Z/Z_\odot) \in [-2.0, 0.19]`

**Code:** `Uniform(-2.0, 0.2)`

**Suggested fix:** Change paper to 0.2 (matches code and is a rounder number).

---

## Fix 4: Calibration polynomial (Appendix B)

**Current paper:** States K=10 Chebyshev coefficients with priors N(0, 0.01).

**Code default:** 3 coefficients with priors of width 0.1 and 0.05.

**Suggested fix:** Either:
- Change to "K=3 with priors N(0, 0.1)" to match defaults, or
- Add: "K and the prior width are configurable; we use K=3 with $\sigma_c = 0.1$
  for the demonstrations in this paper."
