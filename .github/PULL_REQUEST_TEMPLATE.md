## Summary

<!-- One paragraph: what changed and why. -->

## Related issues

<!-- #123, #456 -->

## Type of change

- [ ] Bug fix
- [ ] New feature
- [ ] Documentation
- [ ] Refactor (no behavior change)
- [ ] Tests
- [ ] Chore (build, dependencies, CI)

## Labels

- [ ] At least one `area:*` label applied (see the label table in `CLAUDE.md`)

## Before you push

CI runs two gate jobs, `lint` and `smoke`, and between them about forty steps.
`ruff` and `pytest` are only the first two, so passing those locally does not
mean CI is green. To run the real list rather than guessing at it:

```bash
# every step CI runs, extracted from the workflow itself
sed -n '/^  lint:/,/^  [a-z-]*:$/p'  .github/workflows/tests.yml | grep -oE '^      - run: .*' | sed 's/^      - run: //'
sed -n '/^  smoke:/,/^  [a-z-]*:$/p' .github/workflows/tests.yml | grep -oE '^      - run: .*' | sed 's/^      - run: //'
```

A `tools/check_*.py` glob is **not** the same list: it misses
`add_spdx_headers.py --check` and `gen_property_table.py --check`, both of which
have failed PRs that looked green locally.

- [ ] `ruff check src/ tests/` and `ruff format --check src/ tests/` pass
- [ ] `pytest tests/ -q` passes
- [ ] The `lint` and `smoke` steps above pass

## Physics and documentation

- [ ] New functions have numpydoc docstrings: Parameters, Returns, Raises
- [ ] Array shapes annotated, e.g. `shape (n_wave,)`
- [ ] Units in brackets, e.g. `[erg/s/Hz]`, `[yr]`
- [ ] Equations checked against the primary source, and cited
- [ ] Regression test added if physics changed
- [ ] CHANGELOG entry if user-facing

## AI-use disclosure

- [ ] This PR was drafted with AI assistance

If so: every line reviewed, every equation and citation checked against the
primary source, regression tests run.
