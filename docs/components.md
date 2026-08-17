# Component reference

Every model registered in tengri, as the package itself reports it. The tables
below are generated from the live registries at build time, so they cannot drift
from what `SEDModel.build` will actually accept.

Read the **status** column before choosing. It is not decoration:

- `production` — validated, and what you want unless you have a reason.
- `unvalidated` — registered, but not yet checked against the DSPS forward path.
  Selecting one raises rather than returning a number you should not trust.
- `experimental` — runs, not validated for science.
- `broken` — known to fail; asking for it by name raises, deliberately.
- `deprecated` — an old spelling kept working for existing code.
- `comparison` / `demo` — present for cross-code parity or teaching, not for
  science.

The **use** column is the exact call. Copy it.

Anything here can also be reached from Python — `tengri.list_all()` returns every
registry at once, and `tengri.describe("calzetti")` explains one entry including
its parameters. {doc}`spine/03_discovering_the_menu` walks through that
interactively; this page is the same information for reading rather than running.

```{eval-rst}
.. include:: _generated/component_tables.rst
```
