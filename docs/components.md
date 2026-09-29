# Component reference

Every model registered in tengri, as the package itself reports it. The tables below are generated from the live registries at build time, so they cannot drift from what `SEDModel.build` will accept.

Read the **status** column before choosing a model:

- `production`: validated, and the right choice unless you have a reason to pick another.
- `unvalidated`: registered, but not yet verified against the DSPS forward path. Selecting one raises an error rather than returning an untrusted number.
- `experimental`: runs, but not validated for science.
- `broken`: known to fail. Asking for it by name raises an error deliberately.
- `deprecated`: an old spelling kept working for backward compatibility.
- `comparison` / `demo`: present for cross-code parity or teaching, not for science.

The **use** column is the exact call. Copy it.

Anything here can also be reached from Python. `tengri.list_all()` returns every registry at once, and `tengri.describe("calzetti")` explains one entry including its parameters. {doc}`spine/03_discovering_the_menu` walks through that interactively, with this page providing the same information for reading rather than running.

```{eval-rst}
.. include:: _generated/component_tables.rst
```
