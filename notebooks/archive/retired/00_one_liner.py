# %% [markdown]
# # One-liner SED fitting with tengri
#
# This notebook demonstrates the minimal API for SED fitting with tengri. Start here to understand the core workflow: load data, choose a model preset, fit, and explain results.
#
# **Note:** This notebook requires access to SSP (Single Stellar Population) data. Set the environment variable `TENGRI_SSP_PATH` to point to your SSP data directory, or pass `ssp_path=...` explicitly to `Galaxy.from_arrays()`. If SSP data is not available, fitting will be skipped.

# %% [code]
import tengri as tg

tg.print_logo()
print(f"tengri {tg.__version__}")

tg.doctor()

# %% [markdown]
# ## See which models are available

# %% [code]
tg.presets.list_presets()
print(tg.presets.describe("starforming"))

# %% [markdown]
# ## Inspect the citations you'll pick up by using tengri

# %% [code]
for c in tg.cite_all():
    print(c)

# %% [markdown]
# ## Build a Galaxy from arrays (demo data; replace with your own)

# %% [code]
import os

ssp_path = os.environ.get("TENGRI_SSP_PATH")
if ssp_path and os.path.exists(ssp_path):
    g = tg.Galaxy.from_arrays(
        filters=["sdss_u", "sdss_g", "sdss_r", "sdss_i", "sdss_z"],
        flux=[1.0e-28, 2.0e-28, 3.0e-28, 2.5e-28, 2.0e-28],
        flux_err=[1.0e-29] * 5,
        flux_unit="erg/s/cm2/Hz",
        redshift=0.1,
        ssp_path=ssp_path,
        preset="starforming",
    )
    g.fit(backend="map")
    print(g.summary())
    print(g.explain())
else:
    print("Set TENGRI_SSP_PATH to run a real fit. Skipping.")

# %% [markdown]
# ## That's the whole flow: load, fit, summarise, cite.
#
# For more details, see the documentation and other notebooks in this directory.
