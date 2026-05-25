# Contributing

Guidelines for setting up a development environment, writing code, running
tests, and submitting changes.

## Development environment

```bash
git clone https://github.com/suchethac/tengri.git
cd tengri
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

SSP template data (`data/ssp_*.h5`) is not tracked in git. Integration tests
skip gracefully if the data files are missing.

## Code style

tengri uses **Ruff** for both linting and formatting. Configuration lives in
`pyproject.toml` under `[tool.ruff]`.

```bash
ruff check src/ tests/              # lint -- must pass with zero errors
ruff format --check src/ tests/     # format check -- must pass
ruff check --fix src/ tests/        # auto-fix safe violations
ruff format src/ tests/             # auto-format all files
```

```{important}
Run both `ruff check` and `ruff format --check` before every commit.
Zero violations required.
```

### Naming and formatting

- **snake_case** everywhere
- **99-character** line length limit
- **Numpydoc** docstrings — see [docs/dev/docstring-standard.md](../dev/docstring-standard.md) for the full standard
- **Greek letters** (sigma, xi, theta) are allowed in docstrings and comments
- **Type hints**: use `X | None` (PEP 604), not `Optional[X]`

#### Docstring tier system

Apply documentation depth based on the function's audience:

| Tier | Location | Required sections |
|------|----------|-------------------|
| 1 — Public API | `__init__.py` exports | Parameters, Returns, Raises, Notes (JIT flag), References, Examples |
| 2 — Scientific functions | `components/`, `forward/`, `observation/` | Parameters, Returns, Notes (JIT + equations), References |
| 3 — Utilities | `utils/`, `config/`, `analysis/` | Parameters, Returns |
| 4 — Private helpers | `_`-prefixed | One-sentence summary |

Key rules for all docstrings:
- **Units** in brackets for every physical parameter: `[erg/s/Hz]`, `[yr]`
- **Array shapes** always annotated: `array_like, shape (n_wave,)` for inputs
- **Equations** use the RST `.. math::` directive; define all variables with units; cite source equation by number
- **Approximations** must be flagged explicitly with validity range
- **Citations** use `.. [N]` numbered References with exact title, arXiv ID, and DOI
- **Upstream code** must be credited in Notes

### Ruff rules enforced

| Rule set | Purpose |
|----------|---------|
| `F` | Unused imports/variables |
| `E/W` | pycodestyle basics (99-char line limit) |
| `I` | Import sorting (stdlib, third-party, first-party) |
| `UP` | Python 3.10+ syntax |
| `B` | Bugbear patterns |
| `SIM` | Simplifiable constructs |
| `RUF` | Ruff-specific (sorted `__all__`, no unused unpacked vars) |

**Allowed exceptions** (configured in `pyproject.toml`):

- `E402` ignored: `jax.config.update()` must run before JAX imports
- `E741` ignored: single-letter variables (`l`, `I`) common in scientific code
- `__init__.py` files: `F401` ignored for re-exports
- `tests/`: `F841` ignored for fixtures
- `notebooks/`, `analysis/`: relaxed rules for exploratory code

### Immutability

Never mutate arrays. Use `jnp.ndarray.at[].set()` for updates. All model
components must be stateless pure JAX functions with no side effects.

### Lazy imports

Optional dependencies (`nifty8.re`, `blackjax`, `optax`, `arviz`) must be
imported inside the methods that use them, never at module level.

### Units

| Quantity | Internal unit | User-facing unit |
|----------|--------------|-----------------|
| Time | years | Gyr or Myr |
| Wavelength | Angstrom | Angstrom |
| SFR | Msun/yr | Msun/yr |
| PSD timescale | years (`psd_tau_yr`) | Myr (`psd_tau_myr`) |
| Metallicity (SSP grid) | `log10(Z)` absolute | `log10(Z/Zsun)` (`met_logzsol`) |

Metallicity offset: `LOG10_ZSUN = -1.848`.

## Testing

### Test organization

```
tests/
├── unit/            # Fast, no SSP data needed
├── integration/     # Needs data/ssp_*.h5, skips gracefully if missing
└── crossval/        # Against bagpipes/FSPS, excluded from default pytest runs
```

### Running tests

```bash
# Full suite (~1221 tests, ~105s)
pytest tests/ -q

# Specific module
pytest tests/unit/test_distributions.py -v
pytest tests/unit/test_fitter.py -v
pytest tests/unit/test_raytrace.py -v

# With coverage
pytest tests/ --cov=src/tengri

# Cross-validation tests (not run by default)
pytest -m crossval tests/crossval/

# FSPS cross-validation (needs SPS_HOME env var)
SPS_HOME=~/Projects/fsps pytest -m crossval
```

```{note}
All tests use `jax.config.update("jax_enable_x64", True)` for numerical
precision. JAX Metal (Apple GPU) is experimental and causes test failures --
use `JAX_PLATFORMS=cpu` for reliable results.
```

### Cross-validation details

- `python-fsps` needs the `SPS_HOME` environment variable and cannot coexist
  with JAX (numpy version conflict)
- Use `/tmp/tf_env` venv for TensorFlow/CUE reference generation
- CUE reference outputs: `data/cue_reference_outputs.npz`
- DL07 tabulated templates: `data/dl07_templates.npz`

### Verifying gradient cleanliness

When adding new distributions or model components, verify that gradients are
finite:

```python
import jax, jax.numpy as jnp
from tengri.distributions import MyNewDist

d = MyNewDist(...)
g = jax.grad(d.unstandardize)(jnp.array(0.5))
assert jnp.isfinite(g)
```

## Notebook workflow

Notebooks are **jupytext percent-format `.py` files** in `notebooks/`.
These are the source of truth.

### Editing

1. Open the `.py` file -- it is plain Python with `# %%` cell markers
2. Make changes directly in the `.py` file
3. Sync to `.ipynb`: `cd notebooks && jupytext --sync *.py`

### Cell format

```python
# %% [markdown]
# # Section Title
#
# Some explanation with $\LaTeX$ math.

# %%
import jax
import jax.numpy as jnp
result = jnp.array([1, 2, 3])
```

```{warning}
Never edit `.ipynb` files directly -- they are generated from the `.py`
source files. Never create new notebooks as `.ipynb`; always create them
as `.py` in percent format.
```

## Commit conventions

```
<type>: <description>
```

| Type | Purpose |
|------|---------|
| `feat` | New feature |
| `fix` | Bug fix |
| `refactor` | Code restructure (no behavior change) |
| `docs` | Documentation only |
| `test` | Adding or updating tests |
| `chore` | Maintenance (re-execute notebooks, etc.) |
| `perf` | Performance improvement |
| `ci` | CI/CD changes |

## Pull request workflow

1. Create a feature branch from `main`
2. Make changes
3. Run linting: `ruff check src/ tests/ && ruff format --check src/ tests/`
4. Run tests: `pytest tests/ -q`
5. Commit with a descriptive message following the conventions above
6. Push and open a pull request

## Common contribution tasks

### Adding a new prior distribution

1. Subclass `Distribution` in `src/tengri/distributions.py`
2. Implement: `bounds`, `sample()`, `log_prob()`, `unstandardize()`, `standardize()`
3. `unstandardize()` must be JAX-differentiable -- test with `jax.grad`
4. Add to `__init__.py` exports
5. Add tests in `tests/unit/test_distributions.py`

### Adding a new inference method

1. Add `_run_METHOD()` to `Fitter` in `src/tengri/inference/fitter.py`
2. Route in `Fitter.run()` method
3. Return a `Posterior` object
4. Use lazy imports for optional dependencies
5. Add tests in `tests/unit/test_fitter.py`

### Adding a new PSD model

1. Write a function: `def my_psd(sigma, tau_yr, n_grid, log_ages) -> sqrt_power`
2. Pass to `StandardizedForwardModel(model, psd_model=my_psd)`
3. The function must be JAX-differentiable
4. Add tests verifying the integral equals the expected variance

### Adding a new dust model

1. Create `src/tengri/models/dust/my_model.py`
2. Implement: `(wavelength, age_grid, **params) -> attenuation_factor`
3. Must be pure JAX (`jnp` operations only, no side effects)
4. Add tests in `tests/unit/test_dust.py`
