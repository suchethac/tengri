# tengri Style and Voice

This complements [`docstring-standard.md`](docstring-standard.md) (which governs *what*
a docstring contains) and [`NAMING_CONTRACT.md`](NAMING_CONTRACT.md) (which governs
identifier names). This document governs **voice** — how the code reads to an astronomer.

The north-star file is [`src/tengri/components/dust/attenuation.py`](../../src/tengri/components/dust/attenuation.py).
When in doubt, open it and copy its rhythm: physics-first prose, citations inline, math
in `.. math::`, variables named like the paper, no narration.

---

## 1. Tier-4 helpers get one-line docstrings

The tier system in `docstring-standard.md` already says this. We enforce it here.
`_`-prefixed helpers, internal translators, and one-call utilities take a single sentence —
no Parameters block, no Returns block, no Notes.

```python
# BAD — parameters/groups.py
def _translate_structural(group, prefix):
    """Translate a group's structural choice into Parameters kwargs.

    Parameters
    ----------
    group : dict
        The group dictionary with a 'type' key.
    prefix : str
        Parameter-name prefix (e.g. 'sfh_').

    Returns
    -------
    kwargs : dict
        Keyword arguments suitable for Parameters(...).

    Raises
    ------
    ValueError
        If 'type' is missing or unknown.
    """
    ...

# GOOD
def _translate_structural(group, prefix):
    """Resolve a group's `type` choice into the matching Parameters kwargs."""
    ...
```

If a helper's signature is non-obvious, add a one-line Parameters note — not the full
numpydoc machinery.

---

## 2. No narration

A comment must add information the code does not already convey. Labels above expressions
that compute exactly what the label says are noise.

```python
# BAD — observation/eline_marginalization.py
# Inverse noise variance
N_inv = 1.0 / sigma**2
# G^T N^{-1} G
GtNinvG = G.T @ (N_inv[:, None] * G)
# Prior precision
P = jnp.eye(n) / prior_variance

# GOOD — let the math block in the docstring carry the labels
N_inv = 1.0 / sigma**2
GtNinvG = G.T @ (N_inv[:, None] * G)
P = jnp.eye(n) / prior_variance
```

If the line is genuinely opaque, fix the variable names — don't tag it with a comment.

The same rule kills `# Step 1:`, `# Now compute X`, `# Handle the case where`, and section
headers inside function bodies.

---

## 3. No commented-out algebra

Derivations belong in the function's docstring (as a `.. math::` block) or in the PR
description. They do not belong as 20 lines of `# tau_k = ...` above the loop that
actually computes the recurrence. If a reader needs the derivation, they will read the
cited paper; if they need the equation, it goes in Notes.

```python
# BAD — observation/calibration.py
# Clenshaw recurrence for S = sum_k a_k T_k(x):
#   b_{N+1} = b_{N+2} = 0
#   b_k = 2x * b_{k+1} - b_{k+2} + a_k
#   S = a_0 + x * b_1 - b_2
# ... 15 more lines of derivation ...

# GOOD — put the derivation in the docstring once, then write the loop cleanly.
```

---

## 4. No `# TODO` in source

Use the issue tracker, or — if the project has one — a single `docs/dev/TODO.md`. TODO
comments scattered through files rot; they describe a state of mind from a particular
commit and never get updated.

---

## 5. No defensive re-asserts after a helper that already guarantees the type

If `_split()` returns `int`, callers should not write `assert n is not None` ten times
afterwards. If the helper returns `Optional[int]`, resolve the option **once** at the top
of the consumer, then propagate the resolved value.

```python
# BAD — inference/fitter.py (repeated ten times)
n_phot, n_spec = self._n_phot_split(...)   # signature says Optional[int]
assert n_phot is not None
... use n_phot ...
# (later in the same function)
assert n_phot is not None
... use n_phot ...

# GOOD — fix the helper's contract, or resolve once
n_phot, n_spec = self._n_phot_split(...)   # now returns int (raises on misuse)
... use n_phot ...
... use n_phot ...
```

---

## 6. No `if x is None: x = default` chains in function bodies

Resolve at the signature with a default, or use `or` / a conditional expression once.
Don't scatter null-coalesce branches through the body.

```python
# BAD
def kernel(x, x2=None):
    if x2 is None:
        x2 = x        # explanatory comment about why x2 defaults to x
    ...

# GOOD
def kernel(x, x2=None):
    x2 = x if x2 is None else x2
    ...
```

For configuration-shaped objects that are conceptually never-None after construction,
enforce that in `__post_init__` once — not in every accessor.

---

## 7. No single-call wrappers

A function whose body is one line of "call the real thing with these arguments" should not
exist. Either the call site uses the real thing directly, or the wrapper has a clear
abstraction it provides (a different signature, a default, a side effect).

```python
# BAD — inference/jit_engine.py
def get_or_build_loss_fn_cached(key, builder):
    return _shared_get_or_build(LOSS_CACHE, LOSS_LOCK, key, builder)

def get_or_build_grad_fn_cached(key, builder):
    return _shared_get_or_build(GRAD_CACHE, GRAD_LOCK, key, builder)
# ... two more identical wrappers ...

# GOOD — one function, callers pass the (cache, lock) pair
def get_or_build_cached(cache, lock, key, builder):
    ...
```

Three near-identical wrappers is the threshold: collapse them into one parameterised
function.

---

## 8. Variable names track the paper

This is already strong across the tree — keep it that way. `tau_bc`, `tau_diff`, `k_lambda`,
`L_nu`, `M_star`, `sigma_v`, `xi_PAH`, `f_obscuration`, `log_peak_sfr`. Not
`stellar_mass_total`, not `dust_optical_depth_v_band`. When the math uses Greek, the code
uses the Greek-letter transliteration. Reviewers should compare the code against the cited
paper and find the same symbols.

---

## 9. Examples are physics, not API tours

Docstring `Examples` sections should show a realistic call with realistic numbers
(`sfh_dpl_tau_gyr=2.5`, `tau_bc=0.5`), not `obj = Class(); obj.method()` placeholders. If
the example doesn't teach the astronomer something about the physics or workflow, delete
it — the type signature already shows the API.

---

## How to use this guide

When writing new code: re-read sections 1, 2, 5, 6, 7 before opening a PR.

When reviewing: a violation of any of these nine rules blocks the review the same way a
ruff violation blocks CI. They are not stylistic preferences — they are the difference
between code that an astronomer reads as physics and code that reads as boilerplate.
