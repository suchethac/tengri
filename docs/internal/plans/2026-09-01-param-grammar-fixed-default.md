# Parameter-freedom grammar redesign: `Fixed(DEFAULT)` + wildcard synonyms

**Status:** approved (owner, 2026-09-01) · **Scope:** one PR, whole sweep ·
**Labels:** `area:api`, `breaking-change`, `enhancement`

## Problem

The build grammar spells parameter freedom as a hidden 2×2 — (free vs fixed) ×
(explicit vs registry-default) — with unrelated conventions per cell:

|           | explicit           | registry default |
|-----------|--------------------|------------------|
| **fixed** | `Fixed(0.3)`       | `FIXED` sentinel |
| **free**  | any `Distribution` | `FREE` sentinel  |

Owner-confirmed pain points: the `FIXED`/`Fixed(x)` case collision (with a
third bare-`Fixed()` meaning contemplated in
`docs/dev/cross-compile-fixed-z-design.md`); `all_params` over-promising (it is
a fallback for the *unnamed rest*, and `FREE` frees only params with a registry
`free_prior`); docs teaching a "state `all_params: FIXED`, then overrule it"
cascade.

## Decision

1. Remove the `FIXED` sentinel — clean pre-1.0 break, no shim
   (`ImportError` on import).
2. New `DEFAULT` sentinel, legal only inside `Fixed(...)`:
   `Fixed(0.3)` pins at your value; `Fixed(DEFAULT)` pins at the registry
   default; `Fixed()` stays `TypeError` (reserved for per-call supply);
   bare `DEFAULT` in a group dict raises `ParameterError`.
3. `FREE` survives unchanged.
4. Single resolution path: `Fixed(DEFAULT)` resolves through
   `_default_fixed_value` (the #412 canonical table) — never a second path.
5. `all_params` and `other_params` are exact synonyms in every group and every
   nested sub-block that takes a wildcard; `'*'` stays the internal canonical
   key only (already refused on input).
6. Wildcard values: `FREE` or `Fixed(DEFAULT)` only; a concrete `Fixed(v)` in
   the wildcard slot errors (one value cannot be smeared across parameters).
7. Style convention (taught by docs and the emitter, never enforced by the
   parser): all-free → `all_params` alone; mixed → explicit entries first,
   `other_params` last. `to_groups()` emits by this rule.
8. `WildcardPartialFreeWarning` semantics unchanged.
9. The `DefaultFixedParametersWarning` suppression idiom (#1995) survives,
   spelled `all_params: Fixed(DEFAULT)`; docs stop showcasing the all-fixed
   baseline only in examples where the group frees something anyway.
10. Builders gain `other_params=` as an exact synonym kwarg (both given →
    error); the four duplicated wildcard-validation gates collapse into one
    shared helper in `builders/_factory.py`.
11. New loud errors closing silent holes: bare group-level sentinel
    (`sfh=FREE` is a silent no-op today); `foreground` enters
    `valid_top_groups` and a wildcard there says "declares no parameters"
    instead of doing nothing.

Target end state:

```python
# before                                   # after
sfh={"type": "dpl",                        sfh={"type": "dpl",
     "all_params": FIXED,                       "log_total_mass": FREE}
     "log_total_mass": FREE}
sfh={"type": "dpl",                        sfh={"type": "dpl",
     "all_params": FREE,                        "met_logzsol": Fixed(DEFAULT),
     "met_logzsol": FIXED}                      "other_params": FREE}
```

## Implementation invariants (pinned so parallel work cannot diverge)

- `_is_default_fixed(x)` = `isinstance(x, Fixed) and x._value is DEFAULT`,
  defined in `parameters/priors.py` beside `Fixed`. Raw `_value`, never the
  raising `.value`. The token is unhashable — never a set member or dict key.
- Unresolved-token readers `value`/`default`/`bounds`/`sample`/`unstandardize`
  raise with guidance; `__eq__`/`__repr__` are exempt (repr `Fixed(DEFAULT)`);
  `is_fixed` stays `True`. The `Parameters` declaration path rejects the
  unresolved token.
- Resolver ordering: the `_is_default_fixed` check precedes every
  `isinstance(val, Distribution)` arm (per-param, wildcard, Cue optional
  knobs — which must reorder — and the `_TOP_LEVEL_SETTINGS` filter, or
  `redshift=Fixed(DEFAULT)` leaks into the structural pre-pass).
- Disposition arms in `_warn_silently_fixed_parameters`: the `"*"` arm is the
  live one (the warner receives normalized kwargs); the dust-none injections
  switch to `"*"` spelling and the `"all_params"` arm is then deleted.
- Serialization: new wire form `{"__fixed_default__": true}` (not
  `{"__fixed__": "DEFAULT"}` — collides with categorical `Fixed("DEFAULT")`);
  the arm precedes the `.value`-reading `is_fixed` branch; the legacy
  `"FIXED"` wire string raises `ConfigError` case-insensitively.
- Summary tags: `[all_params Fixed(DEFAULT)]` for `wildcard_fixed`; a
  `wildcard_fixed_inactive` entry is added (renders empty today); tags stay
  group-agnostic. Partial-free messages use neutral wording naming both
  spellings.
- `_analyze_wildcard_intent` returns `Fixed(DEFAULT)`; every former
  `is FIXED` comparison switches to the helper (identity does not survive an
  instance token); `is FREE` comparisons are all untouched.

## Rollout

Waves on one branch: (0) stale-reference repairs; (1) `other_params` synonym,
additive; (2) `DEFAULT` + `Fixed(DEFAULT)` accepted alongside `FIXED`
(branch-internal dual window — the released surface has exactly one grammar);
(2.5) mechanical AST codemod of ~2000 idiom sites on a green tree; (3) the
removal break, small because of 2.5 — includes every src docstring naming
`FIXED` (the `check_doc_examples` guard enforces this at the same commit);
(4) emitter contextual spelling + ordering; (5) published prose, notebooks
(via jupytext sources + re-execution + spine sync), gallery regeneration,
CI guard tools, and the `api_migration_v0.x.md` entry — which must answer the
existing anti-dual-spelling precedent: a key alias normalized at one choke
point is one grammar, and the emitter's spelling is a deterministic function
of group content, so round-trips stay single-valued.
