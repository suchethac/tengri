# Examples gallery

Short, self-contained scripts that each produce one figure. Useful when
you want to see a specific physics variation quickly without reading a
full tutorial. Each script under
[`examples/`](https://github.com/suchethac/tengri/tree/main/examples) is
rendered into the thumbnailed gallery by Sphinx-Gallery.

The main learning path is the spine notebooks
(`00_quickstart` through `08_emission_lines`); start there if you are
new and come here for a targeted reference.

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
