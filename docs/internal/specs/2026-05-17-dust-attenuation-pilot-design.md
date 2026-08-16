# dust_attenuation pilot — readable, pedagogical gallery scripts

## Context

The 130+ scripts under `examples/` exist to teach a working scientist
(Bagpipes/Prospector-experienced astronomer) what tengri's idiomatic
API looks like. Today they're cluttered with defensive boilerplate that
hides the physics:

- **91 of 130 scripts** open with a 12-line `_find_ssp()` four-level
  parent-directory path walker plus an `if SSP_PATH is None: raise`
  defensive guard. This is library-level concern leaking into every
  example.
- **71 scripts** use the legacy flat-kwarg `Parameters(...)` constructor
  with 10–20 lines of `Fixed(x.x)` boilerplate. **Zero** scripts use the
  recommended `SEDModel.from_groups(...) + tengri.recipes.*` path, even
  though CLAUDE.md flags it as the preferred user-facing API since 2026-05.
- 9 scripts exceed 200 lines. Average script: 137 lines.

This pilot rewrites the 8 scripts under `examples/dust_attenuation/`
(619 lines total) as the proving ground for an audit-and-rewrite of the
whole gallery. Working-scientist style: terse, domain-fluent, every line
earns its place.

## Library additions

Three small additions move boilerplate from "every script" to "the
library once". Each replaces a pattern that currently lives in dozens of
scripts.

### 1. `tengri.load_ssp(name=None)`

```python
def load_ssp(name: str | None = None) -> SSPData:
    """Load an SSP grid by short name, walking parent dirs for ``data/``.

    Parameters
    ----------
    name : str, optional
        Short name resolving to a bundled SSP file
        (e.g. ``"prsc_miles_chabrier_wNE"``). Defaults to the most
        commonly used SSP for tutorial/demo scripts.
    """
```

The full long-form `load_ssp_data(path)` stays — `load_ssp` is the
short user-facing helper. Looks up by short name through a small
registry, then walks `Path.cwd()` upward until it finds a `data/`
directory containing the resolved filename. Raises `FileNotFoundError`
with a helpful message if not found.

### 2. `tengri.recipes.dust_demo(galaxy="star_forming")`

Recipe matching the dust_attenuation scripts' shared parameter block:
tsnorm SFH peaked at 0.5 Gyr, solar-ish metallicity, modest dust,
redshift 0.1. Multiple existing scripts use nearly identical 12-line
`Parameters(...)` blocks that differ only in which one dust knob they
sweep. The recipe captures the shared part so each script can show
only what it's sweeping.

Existing recipes (`star_forming_photometry`, `quiescent_z0`, etc.) are
aimed at photometric *fitting* and pull in observation/filter setup
that the dust demos don't need. `dust_demo` is the minimal forward-only
model suitable for sweep demos.

### 3. `tengri.dust.list_laws()`

```python
def list_laws() -> dict[str, Callable]:
    """Return all attenuation laws keyed by display label.

    Each value is a one-arg callable ``fn(wave_aa) -> k(wave)`` evaluated
    at τ_V = 1.0 with the law's canonical parameters baked in. For finer
    control, use ``resolve_dust_law(name)`` directly.
    """
```

Replaces the hand-rolled 6-tuple `[(name, kwargs, label), ...]` list at
the top of `plot_attenuation_law_compare.py`. Self-describing API per
the project's existing preference (memory: `feedback_self_describing_apis`).

## Two templates for the rewritten scripts

Every rewritten script collapses to one of these.

### Template A — curve comparison (no SED)

```python
"""
<Topic Title>
=============
<One-line physics summary.>

.. sphx-glr-precomputed-img:
.. image:: images/sphx_glr_<basename>_001.png
"""
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tengri.analysis.plotting import setup_style
from tengri.dust import list_laws  # or resolve_dust_law for single-param sweep

setup_style()
wave = jnp.linspace(1000.0, 10000.0, 2000)

fig, ax = plt.subplots(figsize=(10, 6))
for label, fn in list_laws().items():
    ax.plot(wave / 1e4, fn(wave), label=label, lw=2.0)
# ... domain-specific annotations ...
ax.set(xlabel=..., ylabel=..., xlim=..., ylim=...)
ax.legend(frameon=False, loc="upper left")
fig.tight_layout()
plt.savefig("<basename>.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Template B — SED sweep

```python
"""
<Topic Title>
=============
<One-line physics summary.>

.. sphx-glr-precomputed-img:
.. image:: images/sphx_glr_<basename>_001.png
"""
import matplotlib.pyplot as plt
from tengri import SEDModel, load_ssp, recipes
from tengri.analysis.plotting import setup_style, sweep_parameter, SWEEP_CMAPS

setup_style()
model = SEDModel.from_groups(ssp_data=load_ssp(), **recipes.dust_demo())

fig, ax = sweep_parameter(
    model, "<param_name>", [<values>],
    cmap=SWEEP_CMAPS["dust"], label_fmt=r"...",
    wave_range=(1000, 10000),
)
ax.set(yscale="log", ylim=..., title=..., ylabel=...)
fig.tight_layout()
plt.savefig("<basename>.png", dpi=150, bbox_inches="tight")
plt.show()
```

## Per-script line budget

| Script | Current | Target | Template | Notes |
|---|---:|---:|---|---|
| plot_attenuation_law_compare | 72 | 25 | A | uses `list_laws()` |
| plot_dust_curves | 65 | 25 | A | |
| plot_uv_bump_sweep | 61 | 25 | A | kriek_conroy bump param sweep |
| plot_dust_geometry_sweep | 64 | 30 | B | |
| plot_dust_slope_sweep | 88 | 30 | B | also fix UV-spike artifact (audit flagged: τ_bc>0 to prevent unattenuated nebular spike) |
| plot_tau_bc_sweep | 88 | 30 | B | |
| plot_tau_diff_sweep | 87 | 30 | B | |
| plot_two_component | 94 | 40 | B (bespoke) | shows both dust components side by side |
| **TOTAL** | **619** | **~235** | | ~62% reduction |

## Things deliberately dropped

- 4-level `_find_ssp` path walker + `if SSP_PATH is None: raise` guard — moves into `tengri.load_ssp()` once.
- Hand-defined color hex arrays — `tab10` mpl default cycle is fine for ≤8 curves. (User confirmed.)
- Long inline comments explaining JIT/caching/sweep_parameter internals — the working-scientist audience knows JAX exists.
- `# sphinx_gallery_thumbnail_number = 1` directive — only meaningful when a script generates multiple figures; ours all generate one.
- `jax.config.update("jax_enable_x64", True)` lines — `import tengri` already enables x64 globally via `__init__.py`.

## Things kept

- `setup_style()` call — one-line unified BAGPIPES-inspired look.
- `.. sphx-glr-precomputed-img:` docstring marker + inline `.. image::` directive — required for the sphinx-gallery "use committed PNG" code path.
- `plt.savefig(...) + plt.show()` at end — reproducibility outside sphinx-gallery.

## Out of scope for this pilot

- Scripts outside `examples/dust_attenuation/`.
- Renaming any existing canonical API.
- Changing the figure style (`setup_style()` already establishes that).
- Any change to the inference / NUTS / SVI scripts.

## Verification

- `.venv/bin/ruff check examples/dust_attenuation/ src/tengri/{__init__.py,dust/__init__.py,recipes/__init__.py}` → clean.
- `cd docs && make html` succeeds; `docs/auto_examples/dust_attenuation/sg_execution_times.rst` shows nonzero time for all 8 scripts.
- Each rendered PNG is visually equivalent to the pre-rewrite version (spot check: same curves, same axes, same legend entries — modulo the τ_bc>0 fix in slope_sweep).
- Line-count check: `wc -l examples/dust_attenuation/*.py` reports total ≤ 260.

## Critical files

- `src/tengri/__init__.py` — re-export `load_ssp`.
- `src/tengri/components/stellar/sps/dsps_wrapper.py` — define `load_ssp`.
- `src/tengri/dust/__init__.py` — define `list_laws`.
- `src/tengri/recipes/__init__.py` — define `dust_demo`.
- `examples/dust_attenuation/plot_*.py` — 8 rewrites.

## Acceptance

Single PR against `main`. Reviewer evaluates: (1) every script is shorter and reads top-to-bottom like a recipe card, (2) the library additions are tasteful and reusable beyond dust_attenuation, (3) rendered figures still convey the same physics. If accepted, the same templates roll out to `dust_emission/` next, then physics-component sections, then onboarding sections.
