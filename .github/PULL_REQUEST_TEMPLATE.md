## Summary

<!-- One paragraph describing the changes and their purpose. -->

## Related issues

<!-- Link to related GitHub issues: #123, #456 -->

## Type of change

- [ ] Bug fix (non-breaking change which fixes an issue)
- [ ] New feature (non-breaking change which adds functionality)
- [ ] Documentation (docstrings, README, guides)
- [ ] Refactoring (code restructure with no functional change)
- [ ] Tests (new tests or test improvements)
- [ ] Chore (build config, dependencies, CI/CD)

## Testing

- [ ] Unit tests added
- [ ] Integration tests added
- [ ] Regression test added (if physics changed)
- [ ] Manual verification completed

## Checklist

- [ ] `ruff check src/ tests/` and `ruff format src/ tests/` pass
- [ ] `pytest tests/ -q` passes
- [ ] New functions/classes have numpydoc docstrings with Parameters, Returns, Raises sections
- [ ] Citations registered in docstrings if a new physics component was added
- [ ] CHANGELOG entry added (if user-facing change)
- [ ] Array shapes annotated in docstrings (e.g., `shape (n_wave,)`)
- [ ] Physical units included in docstring brackets (e.g., `[erg/s/Hz]`, `[yr]`)
- [ ] Equations cited against primary sources in docstrings (if physics formula added)

## AI-use disclosure

- [ ] This PR was drafted with AI assistance

If yes, I have reviewed every line, verified all equations and citations against the primary sources, and run the regression tests.
