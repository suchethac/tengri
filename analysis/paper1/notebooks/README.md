# Figure notebooks

One notebook per figure in the tengri framework paper. Each runs the figure's script
under `analysis/paper1/` as a module, in the environment the paper's figures README
prescribes, then checks inputs, records provenance, shows the rendered figure, and
optionally copies the PDF into the paper. The plotting code lives only in the scripts;
the notebooks are the reproducible, inspectable driver for them.

| Notebook | Paper label | File | Script | Inputs (under `analysis/paper1/`) |
|---|---|---|---|---|
| `fig_gpu.ipynb` | `fig:gpu` | `fig04_gpu_datacenter.pdf` | `fig04_gpu_datacenter.py` | `results/sherlock_h100_batch.json` |
| `fig_gpu_consumer.ipynb` | `fig:gpu_consumer` | `figD1_gpu_consumer.pdf` | `fig04_gpu_datacenter.py` | `results/consumer_gpu_batch.json` |
| `fig_backends.ipynb` | `fig:backends` | `fig07_backends.pdf` | `fig07_backends.py` | `results/backend_sweep_pin` |
| `fig_precomp_accuracy.ipynb` | `fig:precomp_accuracy` | `figB1_lut_accuracy.pdf` | `fig03_precompute.py` | `results/fig03_precompute_data.json`, `results/fig03_bench_forward_2026-08-30.json` |
| `fig_sample.ipynb` | `fig:sample` | `fig09_sample_level.pdf` | `fig09_sample_level.py` | `results/fits` |
| `fig_candels.ipynb` | `fig:candels` | `fig05_candels_galaxies.pdf` | `fig05_candels_galaxies.py` | `results/fits` |
| `fig_code_overlay.ipynb` | `fig:code_overlay` | `fig06_code_overlay.pdf` | `fig06_code_overlay.py` | `results/fits`, `results/art_sedfitting_z1.csv` |
| `fig_multiwavelength.ipynb` | `fig:multiwavelength` | `fig01_mock_joint_infer.pdf` | `fig01_mock_joint_infer.py` | `results/mock_joint_mcmc_nuts.npz` |

## Running

From this directory, with the interpreter that has tengri's dependencies:

```bash
cd analysis/paper1/notebooks
jupyter nbconvert --to notebook --execute --inplace fig_gpu.ipynb
```

Environment variables, all optional:

| Variable | Effect |
|---|---|
| `PAPER1_CHECKOUT` | render from another checkout, such as the paper's pinned branch, instead of the one holding the notebook |
| `PAPER1_PYTHON` | interpreter used to run the script (default: the notebook kernel's) |
| `PAPER_FIGURES_DIR` | copy the rendered PDF into this directory, e.g. the paper's `figures/` |
| `JAX_PLATFORMS` | defaults to `cpu` |

Each notebook fails before rendering if an input is missing or empty, and names it.
The three grid figures (`fig_sample`, `fig_candels`, `fig_code_overlay`) need the
production grid's cells in `results/fits`; `fig_multiwavelength` needs the mock's
NUTS posterior. The others render from committed measurement files.

`_driver.py` holds the shared helpers; the notebooks contain no plotting code.
