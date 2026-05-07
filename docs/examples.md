# Examples gallery

Short, self-contained scripts that each produce one figure. Use these when you
want to see a specific physics variation quickly without reading a full tutorial.
Every example under [`examples/`](https://github.com/suchethac/tengri/tree/main/examples)
is rendered into the thumbnailed gallery below by Sphinx-Gallery.

**Primary learning path:** the tutorial spine in
[`notebooks/`](https://github.com/suchethac/tengri/tree/main/notebooks) —
progressive, astronomer-facing chapters (00–08, with more coming) that cover framework,
forward model, fitting workflows, and physics deep dives.
Start there if you are new to tengri; come here when you need a targeted
reference.

## How to run an example locally

```bash
# Any script in examples/ is a standalone program.
python examples/agn/plot_skirtor_variants.py
```

Each script imports only from the public tengri API, loads an SSP grid from
`data/` (see the [SSP grids section](index.md#ssp-grids) on the home page),
and writes a single PNG next to the script.

## Gallery

```{toctree}
:hidden:

auto_examples/index
```

The browseable gallery with thumbnails is at
[`auto_examples/`](auto_examples/index.rst) once the docs have been built.
