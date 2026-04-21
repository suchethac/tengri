# tengri Docstring Standard

This document defines the canonical docstring format for all tengri source code. All new
functions, classes, and methods must follow this standard. Existing code should be updated
toward this standard over time.

The format is **numpydoc**, rendered by Sphinx + Napoleon. See also:
- [numpydoc format guide](https://numpydoc.readthedocs.io/en/stable/format.html)
- [LSST DM numpydoc guide](https://developer.lsst.io/python/numpydoc.html)
- [contributing.md](../developer/contributing.md) — section on code style

---

## Tier system

Apply different levels of documentation detail based on who reads the function:

| Tier | Who | Mandatory sections |
|------|-----|--------------------|
| **1 — Public API** | Symbols re-exported from `tengri.__init__` (SEDModel, Fitter, Parameters, Posterior, …) | Summary, Parameters, Returns, Raises, Notes (JIT flag), References (if physical formula), Examples |
| **2 — Scientific functions** | Pure JAX functions in `components/`, `forward/`, `observation/` | Summary, Parameters, Returns, Notes (JIT flag + equations), References |
| **3 — Utilities** | `utils/`, `config/`, `analysis/` helpers | Summary, Parameters, Returns |
| **4 — Private helpers** | `_`-prefixed functions | Single-sentence summary; Parameters if non-obvious |

---

## Canonical section order

Sections must appear in this order (omit sections that do not apply):

1. Short summary
2. Extended summary *(optional)*
3. `Parameters`
4. `Returns` — or `Yields` for generators, `Attributes` for classes/dataclasses
5. `Other Parameters` *(rarely-used kwargs)*
6. `Raises`
7. `Warns` *(if applicable)*
8. `See Also` *(optional)*
9. `Notes`
10. `References`
11. `Examples`

---

## Full template

```python
def my_function(param: jnp.ndarray, tau_yr: float, *, flag: bool = True) -> jnp.ndarray:
    """One-sentence summary ending with a period.

    Extended summary (optional). Explain *what* the function does and *why* it exists.
    May span multiple paragraphs. Do NOT describe implementation details here — those go
    in the Notes section. Do NOT start with "This function..." — start with the verb.

    Parameters
    ----------
    param : array_like, shape (n_wave,)
        Spectral luminosity density at each wavelength. [erg/s/Hz]
    tau_yr : float
        Damping timescale. Must be positive. [yr]
    flag : bool, optional
        Whether to apply the normalization correction. Default: True.

    Returns
    -------
    result : ndarray, shape (n_wave,)
        Attenuated spectral luminosity density. [erg/s/Hz]

    Raises
    ------
    ParameterError
        If ``tau_yr <= 0``.
    ValueError
        If ``param`` and the wavelength grid have incompatible shapes.

    See Also
    --------
    related_function : Brief description of what it does differently.

    Notes
    -----
    **JIT-compatible**: yes — all operations are ``jnp`` primitives.

    **Gradient-safe**: yes — differentiable everywhere; gradient w.r.t. ``tau_yr``
    is well-defined for ``tau_yr > 0``.

    The damped random walk power spectral density is:

    .. math::

        P(\omega) = \frac{\sigma^2 \,\tau}{1 + (\tau\,\omega)^2}

    where :math:`\sigma` is the PSD amplitude [dimensionless], :math:`\tau` is the
    damping timescale [yr], and :math:`\omega` is angular frequency [rad/yr].
    This is Eq. 3 of Author+2026 [1]_.

    **Approximation**: When ``tau_yr`` is much smaller than the grid spacing ``d_grid``,
    the function underestimates the true power by :math:`\mathcal{O}(\Delta t / \tau)`.
    This approximation is valid for :math:`\tau \gtrsim 10 \,\Delta t`.

    **Upstream**: Follows the Prospector ``transforms.py`` implementation
    (Johnson et al. 2021 [2]_), adapted for JAX.

    References
    ----------
    .. [1] A. Author et al., "Title of the Paper," ApJ, 900, 1 (2026).
       arXiv:2601.07912. https://doi.org/10.3847/1538-4357/abcdef
    .. [2] B. D. Johnson et al., "Prospector: Stellar Population Inference from
       Spectra and SEDs," ApJS, 254, 22 (2021).
       https://doi.org/10.3847/1538-4365/abef67

    Examples
    --------
    >>> import jax.numpy as jnp
    >>> from tengri.components.sfh.psd_models import my_function
    >>> omega = jnp.linspace(0, 1, 100)
    >>> result = my_function(omega, tau_yr=1e8)
    >>> result.shape
    (100,)
    """
```

---

## Rules by section

### Short summary

- One sentence, ends with `.`
- No variable names, no function names
- Starts with an active verb: *"Compute..."*, *"Return..."*, *"Build..."*
- Not: *"This function computes..."*

### Parameters

- Format: `name : type, shape_or_constraint`
- **`array_like`** for inputs that accept numpy/jax/list; **`ndarray`** for outputs
- **Always include shape**: `array_like, shape (n_wave,)` or `ndarray, shape (n_age, n_wave)`
- **Always include units** for physical quantities: `[erg/s/Hz]`, `[yr]`, `[Msun/yr]`, `[Angstrom]`
- For optional kwargs: add `optional` after type; state the default

### Returns

- Format: `name : type, shape`
- Include units on every physical quantity
- For multiple return values, use named entries

### Raises

- Document every exception the function can raise
- Include the condition that triggers it
- Required for Tier 1 functions; include in Tier 2/3 when the condition is non-obvious

### Notes

**For all Tier 1 and Tier 2 functions, the Notes section is mandatory and must include:**

1. **JIT-compatibility statement** (for `components/` and `forward/` functions):
   - `**JIT-compatible**: yes` — if all ops are ``jnp`` primitives with no Python-level branching on traced values
   - `"This function is not compatible with :func:\`jax.jit\`"` — if it has Python-side effects, concrete value requirements, or host callbacks
   - `**Gradient-safe**: yes` — if the function is in a gradient path and is differentiable everywhere
   - `"Not differentiable at X"` — if there is a discontinuity or non-differentiable point

2. **Equations** (for physics functions):
   - Use `.. math::` directive for block equations (renders with MathJax in Sphinx)
   - Use `:math:`\omega`` for inline math
   - After every equation, define *all* variables with units
   - Cite the source equation: *"This is Eq. 3 of Author+Year [N]_."*

3. **Approximation flags** (mandatory when applicable):
   - Always document if the implementation is a simplified or approximate form
   - State the validity range: *"valid for :math:`\tau \gtrsim 10\,\Delta t`"*
   - State where it breaks down: *"underestimates power by :math:`\mathcal{O}(\Delta t/\tau)` at small :math:`\tau`"*

4. **Upstream code credit** (mandatory when applicable):
   - If ported from, inspired by, or validated against another codebase, say so
   - Examples: *"Ported from bagpipes (Carnall et al. 2018 [N]_)"*, *"Follows Prospector transforms.py (Johnson et al. 2021 [N]_), adapted for JAX"*
   - Include the source in References

### References

Use the numbered footnote format from numpydoc:

```
References
----------
.. [1] First Author et al., "Exact Paper Title," Journal Abbrev., Vol., Pages (Year).
   arXiv:XXXX.XXXXX. https://doi.org/10.XXXX/XXXXX
```

Rules:
- **Exact titles** — do not paraphrase or abbreviate the paper title
- **Always include arXiv ID** when the paper has one: `arXiv:2601.07912`
- **Always include DOI** as a full URL: `https://doi.org/...`
- Use standard journal abbreviations: ApJ, ApJS, MNRAS, A&A, AJ, PhRvD, JCAP
- Inline citations in Notes must use the `[N]_` syntax to link to the References entry
- **Verify all citations before writing them** — use authoritative sources, not memory

### Examples

- Tier 1 (Public API): mandatory, must be executable
- Tiers 2–3: include when the function has non-obvious usage or multiple modes
- Start from `from tengri import ...` or `from tengri.components... import ...`
- Use realistic parameter values (not `1.0`, `0.0` everywhere)
- Show output shape or value for at least one call
- Use the doctest `>>>` format

---

## Equation verification rule

Before writing any equation in a docstring:

1. Find the original paper
2. Read the exact equation (do not rely on memory or other code)
3. Confirm the variable definitions match the code's parameter names
4. Confirm the formula is the same as the code (not a different approximation)
5. If the code differs from the paper, document *why*

If the code implements an approximation of the paper equation, this **must** be flagged in Notes.
Undocumented approximations are a correctness failure.

---

## Citation format

Full example of a correct References entry:

```rst
References
----------
.. [1] S. Charlot and S. M. Fall, "A Simple Model for the Absorption of Starlight by
   Dust in Star-forming Galaxies," ApJ, 539, 718 (2000).
   https://doi.org/10.1086/309250

.. [2] J. Leja et al., "Deriving Physical Properties from Broadband Photometry with
   Prospector: Description of the Code and Case Studies," ApJ, 876, 3 (2019).
   arXiv:1905.11997. https://doi.org/10.3847/1538-4357/ab133c

.. [3] B. D. Johnson et al., "Prospector: Stellar Population Inference from Spectra
   and SEDs," ApJS, 254, 22 (2021).
   arXiv:2012.01426. https://doi.org/10.3847/1538-4365/abef67
```

---

## Common tengri-specific patterns

### JAX JIT note (compatible)

```
Notes
-----
**JIT-compatible**: yes — all operations use ``jnp`` primitives. Safe to call
inside :func:`jax.jit`, :func:`jax.vmap`, and :func:`jax.grad`.
```

### JAX JIT note (not compatible)

```
Notes
-----
This function is not compatible with :func:`jax.jit` because it uses Python-level
branching on the value of ``method`` to select a backend. Call it outside of JIT
boundaries; the returned callable is JIT-compatible.
```

### Approximation note

```
Notes
-----
**Approximation**: This implements the two-screen approximation of Charlot & Fall
(2000) [1]_, which separates birth-cloud and diffuse ISM attenuation into two
independent power-law screens. The approximation is valid when the dust geometry
can be factored into these two components. It breaks down for complex, clumpy
geometries where the birth-cloud fraction is spatially variable.
```

### Upstream credit note

```
Notes
-----
**Upstream**: Ported from the Prospector ``dust_attenuation`` module
(Johnson et al. 2021 [2]_), with the birth-cloud attenuation parameterized
by ``dust_tau_bc`` instead of ``dust2_bc`` for consistency with tengri naming.
```

### Physical units reminder

For the most common units in tengri:

| Quantity | Unit | Parameter name pattern |
|----------|------|------------------------|
| Time / age | yr (internal), Gyr/Myr (user-facing) | `*_yr`, `*_gyr`, `*_myr` |
| Wavelength | Angstrom (Å) | `wave_aa`, `wavelength_aa` |
| SFR | Msun/yr | `sfr` |
| SED luminosity | erg/s/Hz (L_nu) | `lnu`, `sed` |
| SED flux | erg/s/cm²/Hz (f_nu) | `fnu`, `flux` |
| PSD timescale | yr (internal), Myr (user API) | `psd_tau_yr`, `psd_tau_myr` |
| Metallicity (SSP grid) | log10(Z) absolute | `log_z` |
| Metallicity (user-facing) | log10(Z/Zsun) | `met_logzsol`, `neb_logZ_gas` |
| AGN bolometric luminosity | log10(L_bol/Lsun) at API | `agn_log_lbol` |
