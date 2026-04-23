# Examples

The **Sphinx gallery** (`examples/*.py`) builds short scripts that each produce **one figure**
(thumbnail in the built docs). Use it for quick **single-parameter sweeps** and visual spot checks.

**Primary learning path:** the **reader spine** — Jupytext notebooks at repository root
[`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) (`00`–`16`, plus `02b`,
`03b`, `05b` for advanced topics), linked from the [project home](index.md#reader-spine-notebooks).
The spine is written as forward-facing documentation for astronomers new to `tengri`: each
notebook opens with a What / What-you'll-see / Why-tengri frame and ends with a standalone
summary figure. Advanced follow-ups (`02b_sfh_advanced`, `03b_dust_emission`,
`05b_agn_advanced`, `16_quickstart_stochastic`) are optional deep dives — not required reading
for first-time users. The Sphinx gallery below does not replace the spine.

For the long-term division of labor between gallery scripts and notebooks, see
[DOCS_REFACTOR_REFINED.md](https://github.com/suchethac/tengri/blob/main/docs/dev/DOCS_REFACTOR_REFINED.md) (maintainer-facing).

```{raw} html
<script>window.location.href = "auto_examples/index.html";</script>
```

If not redirected, go to the [examples gallery](auto_examples/index.rst).
