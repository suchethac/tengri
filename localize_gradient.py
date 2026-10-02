"""Localize gradient divergence in kubota_done disc model."""
import sys
import warnings
warnings.filterwarnings("ignore")
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
import numpy as np

# Test the core gradient of kubota_done on agn_log_mbh
# which is where the big discrepancy is

import tengri
from tengri import Fixed, FREE, DEFAULT, SEDModel

SSP = "/Users/suchethacooray/Projects/tengri/data/bc03_pdva_stelib_chabrier.h5"
NONE = {"type": "none"}
disc = {"type": "kubota_done", "all_params": Fixed(DEFAULT)}
disc.update({"agn_log_mbh": FREE, "agn_log_ledd": FREE})
agn = {"type": "composable", "disc": disc, "torus": NONE, "nlr": NONE, "blr": NONE, "atten": NONE,
       "agn_log_lbol": FREE, "all_params": Fixed(DEFAULT), "norm": "independent"}
sfh = {"type": "const", "log_total_mass": -10.0, "all_params": Fixed(DEFAULT)}
m = SEDModel.build(tengri.load_ssp(SSP), sfh=sfh, agn=agn, redshift=Fixed(0.5))
base = {k: jnp.asarray(v, dtype=jnp.float64) for k, v in m.spec.sample(jax.random.PRNGKey(0)).items()}
point = {"agn_log_lbol": 11.5, "agn_log_mbh": 8.5, "agn_log_ledd": -1.0}
for n in ["agn_log_lbol", "agn_log_mbh", "agn_log_ledd"]:
    base[n] = jnp.asarray(point[n], dtype=jnp.float64)

def f(sub):
    p = dict(base); p.update(sub)
    s = m.predict(p).sed.components["sed_agn"]
    return jnp.sum(jnp.log10(jnp.maximum(s, 1e-300)))

# Test gradient of agn_log_mbh
print("=" * 80)
print("Testing gradient with respect to agn_log_mbh")
print("=" * 80)

# Objective at base point
sub_base = {"agn_log_lbol": base["agn_log_lbol"], "agn_log_mbh": base["agn_log_mbh"], "agn_log_ledd": base["agn_log_ledd"]}
val_base = f(sub_base)
print(f"Value at base: {float(val_base)}")

# Gradient via JAX
g = jax.grad(f)(sub_base)
print(f"JAX gradient wrt agn_log_mbh: {float(g['agn_log_mbh'])}")

# Finite difference
h = 1e-4
sub_up = dict(sub_base); sub_up["agn_log_mbh"] = sub_base["agn_log_mbh"] + h
sub_dn = dict(sub_base); sub_dn["agn_log_mbh"] = sub_base["agn_log_mbh"] - h
fd = (f(sub_up) - f(sub_dn)) / (2 * h)
print(f"FD gradient wrt agn_log_mbh: {float(fd)}")
print(f"Discrepancy: {float(g['agn_log_mbh']) / float(fd) if float(fd) != 0 else 'inf'}")

# Now try testing with disc separately
print("\n" + "=" * 80)
print("Testing disc function directly")
print("=" * 80)

# Get the disc block directly and test its gradient
from tengri.components.agn.disc import kubota_done_disc

# Create simple test
wavelength_test = jnp.logspace(-2, 4, 100)

def f_disc(mbh):
    # Create sub dict with current parameters
    p_test = {
        'agn_log_lbol': point["agn_log_lbol"],
        'agn_log_mbh': mbh,
        'agn_log_ledd': point["agn_log_ledd"],
        'agn_a_spin': 0.0,
        'agn_cos_inc': 0.5,
        'agn_f_hard': 0.02,
        'agn_gamma_warm': 2.5,
        'agn_kt_warm': 0.2,
        'agn_gamma_hard': 1.8,
        'agn_kt_hot': 100.0,
        'agn_r_warm_ratio': 2.0,
        'n_radii': 50,
    }
    sed = kubota_done_disc(wavelength_test, **p_test)
    return jnp.sum(jnp.log10(jnp.maximum(sed, 1e-300)))

# Test gradient
g_disc = jax.grad(f_disc)(jnp.asarray(point["agn_log_mbh"], dtype=jnp.float64))
print(f"Direct disc gradient wrt agn_log_mbh: {float(g_disc)}")

# Finite difference
h = 1e-4
fd_up = f_disc(jnp.asarray(point["agn_log_mbh"] + h, dtype=jnp.float64))
fd_dn = f_disc(jnp.asarray(point["agn_log_mbh"] - h, dtype=jnp.float64))
fd_disc = (fd_up - fd_dn) / (2 * h)
print(f"Direct disc FD gradient wrt agn_log_mbh: {float(fd_disc)}")
print(f"Discrepancy: {float(g_disc) / float(fd_disc) if float(fd_disc) != 0 else 'inf'}")
