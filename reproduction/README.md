# Reproduction studies

Component-by-component reproductions of tengri against the reference
SED-fitting codes the community already relies on. Each subfolder
holds one comparison.

```
reproduction/
├── cigale/                     # Boquien et al. 2019, A&A 622, A103
├── bagpipes/                   # planned — Carnall et al. 2018
├── prospector/                 # planned — Johnson et al. 2021
└── synthesizer/                # planned — Vijayan et al. 2024
```

Each comparison ships a notebook (`01_<code>.py`, jupytext percent
format), thin code-specific driver modules, and the rendered figures.
The CIGALE folder is the first.

## Running a notebook

```bash
cd reproduction/<code>
jupytext --to ipynb 01_<code>.py
jupyter nbconvert --to html --execute 01_<code>.ipynb
```

See the per-code README inside each subfolder for prerequisites and
data setup specific to that comparison.
