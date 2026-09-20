# Superseded: old configuration suite

These cells were fit under the configuration suite that predates the locked
`tab:configs` (paper, 2026-09-17). Every row of that table changed -- SFH,
stellar library, attenuation law, dust IR model, and nebular backend -- so
these results describe models the manuscript no longer contains.

They are kept for provenance only. Do **not** merge them with the new grid and
do **not** leave them in `results/fits/`: the driver's `--only-missing` keys on
the `{galaxy}_{config}.json` filename, which is identical across both suites,
so a resumed run would skip cells that were fit with different physics.
