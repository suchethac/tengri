# Contributing to tengri

Thank you for your interest in contributing to tengri! We welcome contributions of all kinds: bug reports, feature requests, code improvements, documentation, examples, and code review.

## Welcome

Tengri is a scientific software project for differentiable SED fitting. We aim to be inclusive and supportive of contributors at all experience levels. If you're interested in astronomy, Bayesian inference, JAX, or scientific software in general, we'd love your input.

## Ways to contribute

**Non-code contributions:**
- File bug reports if you find issues
- Propose new features or improvements via GitHub issues
- Review pull requests and share feedback
- Improve documentation and examples
- Help others in GitHub Discussions
- Spread the word about tengri in your community

**Code contributions:**
- Fix bugs
- Implement planned features (see ROADMAP.md)
- Add tests
- Improve documentation
- Optimize performance
- Port or integrate new physics models

## Quick start for developers

1. **Clone the repository:**
   ```bash
   git clone https://github.com/suchethac/tengri.git
   cd tengri
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # macOS/Linux
   # or
   .venv\Scripts\activate  # Windows
   ```

3. **Install in editable mode with dev dependencies:**
   ```bash
   .venv/bin/pip install -e ".[dev]"
   ```

4. **Run the test suite:**
   ```bash
   .venv/bin/pytest tests/ -q
   ```

5. **Check code style:**
   ```bash
   .venv/bin/ruff check src/ tests/
   .venv/bin/ruff format --check src/ tests/
   ```

6. **Auto-fix style issues:**
   ```bash
   .venv/bin/ruff check --fix src/ tests/
   .venv/bin/ruff format src/ tests/
   ```

## Filing a good bug report

When opening a bug report, please include:
- **What happened:** A clear description of the problem
- **Minimum reproducer:** The smallest code snippet that triggers the bug
- **Expected behavior:** What you thought should happen
- **Environment:**
  - tengri version (`pip show tengri`)
  - Python version (`python --version`)
  - Operating system
  - JAX backend (CPU, GPU) and version
  - JAX device (`JAX_PLATFORMS` env var)
- **Log output:** Full traceback or error message

See `.github/ISSUE_TEMPLATE/bug_report.md` for a template.

## Proposing a feature

To propose a new feature:
1. Open a GitHub issue and describe what you want and why
2. Discuss the scope and approach with maintainers
3. Once consensus is reached, you're welcome to implement it
4. See "Development workflow" below

See `.github/ISSUE_TEMPLATE/feature_request.md` for a template.

## Adding a new alternative within a component

Tengri's physics blocks (AGN models, dust attenuation laws, SFH variants, nebular backends) each have a registry of swappable alternatives. To add yours:

1. **Pick the component.** Run `tengri.list_agn_models()` (or `list_dust_laws()`, `list_sfh_models()`, etc.) to see what's already there and what status each is at. Avoid duplicating an existing alternative.
2. **Copy the worked example.** Start from [`examples/contrib/example_new_agn_torus.py`](examples/contrib/example_new_agn_torus.py). It registers a model, declares metadata (citation + status), and exercises introspection end-to-end.
3. **Register with metadata.** Use the relevant decorator with `citation`, `status`, `short_doc` filled in:
   ```python
   @register_agn_model(
       "my_model",
       citation="Author+Year, ADS bibcode",
       status="production",   # or "experimental", "demo", "deprecated"
       short_doc="One-line description",
   )
   def my_model(...): ...
   ```
   `citation` and `short_doc` are recommended but not required — fill what you have.
4. **Declare new free parameters via the component's `declared_parameters()` method.** Translation entries are auto-derived from there for identity (no-unit-conversion) cases — you typically don't need to edit `parameters/translate.py`.
5. **Open a PR.** GitHub Actions runs `tools/check_param_prefixes.py` (parameter naming guard) and a 30-second smoke test before the full suite.

Status conventions: `production` (validated), `experimental` (works but not yet validated against observations or another code), `demo` (toy model, not for science), `deprecated` (slated for removal).

## Development workflow

**Branch naming:**
- Feature work: `feat/description` (e.g., `feat/agn-xray-emission`)
- Bug fixes: `fix/description` (e.g., `fix/dust-attenuation-edge-case`)
- Documentation: `docs/description`
- Testing: `test/description`

**Commit messages:**
Follow conventional commits format:
```
<type>: <description>

<optional body with details>
```

Types: `feat`, `fix`, `refactor`, `docs`, `test`, `chore`, `perf`, `ci`

Example:
```
feat: add SKIRTOR AGN torus model with anisotropic scattering

Implements disk-torus geometry from Stalevski et al. (2016).
Anisotropy parameter theta_torus controls opening angle.
```

**Pre-commit hooks (planned):**
- Style checking will be automated in CI

## Coding standards

- **Style guide:** Follow ruff (configured in `pyproject.toml`)
- **Naming:** See `docs/dev/NAMING_CONTRACT.md` — canonical names are non-negotiable
- **Docstrings:** See `docs/dev/docstring-standard.md` — numpydoc format with units, shapes, equations, and references
- **Immutability:** Never mutate arrays in-place; use `.at[].set()`
- **JAX compatibility:** Ensure all functions are JIT-safe unless explicitly marked otherwise

## Testing standards

Every code change must include tests.

- **Coverage goal:** 80%+ across the codebase
- **Unit tests** (`tests/unit/`): Fast, no large data files
- **Integration tests** (`tests/integration/`): Require SSP data from `data/`, skip gracefully if missing
- **Crossval tests** (`tests/crossval/`): Validation against bagpipes/FSPS; run manually with `-m crossval` flag

Run locally before pushing:
```bash
.venv/bin/pytest tests/ -q
.venv/bin/pytest --cov=src tests/  # Check coverage
```

## Scientific standards

Every new physics module must:

1. **Cite a primary source:** Every formula or algorithm must reference a peer-reviewed paper, arXiv preprint, or well-documented code repository
2. **Pin a known value:** Implement a regression test that validates one key prediction to 1% against the paper or upstream reference (e.g., SED shape, line flux, attenuation curve)
3. **Register a Citation:** Add an entry to `src/tengri/citations/registry.py` with paper title, authors, year, and arXiv/DOI

Example registration in `registry.py`:
```python
CITATIONS["dust_charlot_fall_2000"] = Citation(
    title="Dust attenuation laws for galaxy models",
    authors="Charlot & Fall",
    year=2000,
    arxiv="N/A",
    doi="10.1086/308936"
)
```

Example regression test in `tests/integration/test_dust_attenuation.py`:
```python
def test_charlot_fall_attenuation_calibration():
    """Verify Charlot & Fall (2000) attenuation law at known wavelength."""
    tau_v = 1.0
    wave = 5500  # V-band, Angstrom
    atten = dust_attenuation_charlot_fall(wave, tau_v)
    # Known value from Charlot & Fall (2000), Table 1
    expected = 1.0  # magnitude of attenuation
    assert abs(atten - expected) < 0.01  # <1% error
```

See VERIFICATION.md for a status table of all physics modules.

## Docstring expectations

Follow numpydoc format. All parameter descriptions must include units in brackets:

Bad:
```python
def sfr_to_mass(sfr, duration):
    """Convert SFR to stellar mass."""
```

Good:
```python
def sfr_to_mass(sfr, duration):
    """
    Integrate star formation rate to total stellar mass.

    Parameters
    ----------
    sfr : array_like, shape (n_time,)
        Star formation rate [Msun/yr].
    duration : float
        Integration time [yr].

    Returns
    -------
    ndarray, shape ()
        Integrated stellar mass [Msun].
    """
```

For equations, use `.. math::` directive:
```python
def planck(nu, T):
    """
    Planck function.

    .. math::

        B_\\nu(T) = \\frac{2 h \\nu^3}{c^2} \\frac{1}{e^{h\\nu/k_B T} - 1}

    where:
    - $h$ = Planck constant [erg·s]
    - $\\nu$ = frequency [Hz]
    - $T$ = temperature [K]
    - $k_B$ = Boltzmann constant [erg/K]
    - $c$ = speed of light [cm/s]

    Parameters
    ----------
    nu : float
        Frequency [Hz].
    T : float
        Temperature [K].

    Returns
    -------
    float
        Specific intensity [erg/s/Hz/sr/cm^2].
    """
```

## PR review expectations

When you open a PR, a maintainer will review it within ~2 weeks. Reviews focus on:

- **Correctness:** Does the code do what it claims?
- **Tests:** Are there adequate tests? Do they pass?
- **Style:** Does it follow ruff and naming conventions?
- **Documentation:** Are docstrings complete? Are new functions in docs?
- **Citations:** For new physics, are primary sources verified and cited?
- **Performance:** Could this be slow? Any obvious bottlenecks?

We may request changes. This is normal and collaborative — we're trying to improve code quality.

## PR checklist

Before submitting, ensure:
- [ ] `ruff check src/ tests/` passes with zero violations
- [ ] `pytest tests/ -q` passes locally
- [ ] New or modified functions have docstrings (numpydoc format)
- [ ] New physics modules have VERIFICATION entries
- [ ] All new exports are added to `__init__.py`
- [ ] CHANGELOG.md is updated (for features/fixes)
- [ ] AI-assistance checkbox is filled in (if applicable)

## AI-use disclosure

Tengri was initially drafted with AI assistance. We welcome AI-assisted contributions, but with accountability:

1. **Review every line personally** — AI can make mistakes; you are responsible for correctness
2. **Verify every equation and citation against a primary source** — Don't trust AI's citations; check them yourself
3. **Run tests locally** — Execute `pytest tests/ -q` and `ruff check src/ tests/` before pushing
4. **Check the AI-assistance box in the PR template** — Transparency helps maintainers understand the review context

AI is a tool, not an author. Maintainers review AI-assisted and human-only PRs to the same standard.

## Release process

Releases are cut by the maintainer (see GOVERNANCE.md). The process:
1. Update version in `pyproject.toml` and `CITATION.cff`
2. Update CHANGELOG.md with release notes
3. Tag on main: `git tag v0.2.0`
4. Push tag: `git push origin v0.2.0`
5. GitHub Actions builds and uploads to PyPI

## Governance and recognition

- **Governance:** See GOVERNANCE.md for decision-making and maintainer responsibilities
- **Contributors:** Major contributors are listed in CONTRIBUTORS.md
- **Code of Conduct:** See CODE_OF_CONDUCT.md

## Questions?

- **How-to questions:** Use GitHub Discussions
- **Bug reports:** Use GitHub Issues
- **Ideas and feedback:** Open an issue or discussion
- **Don't email the maintainer directly for support** — use public channels so others can learn too

Thank you for contributing!
