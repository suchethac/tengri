import h5py
import numpy as np
from tengri import data_path

with h5py.File(data_path("themis_templates.h5"), "r") as f:
    wave_aa = np.asarray(f["wavelength_aa"][:])
    qhac_grid = np.asarray(f["qhac_grid"][:])
    umin_grid = np.asarray(f["umin_grid"][:])
    single_u = np.asarray(f["single_u"][:])

print(f"Shape of single_u: {single_u.shape}")
print(f"qhac_grid: {qhac_grid}")
print(f"umin_grid: {umin_grid}")

i_umin = int(np.argmin(np.abs(umin_grid - 1.0)))
print(f"Selected umin index: {i_umin}, value: {umin_grid[i_umin]}")

c_aa_per_s = 2.99792458e18
nu = c_aa_per_s / wave_aa

for k, qhac in enumerate(qhac_grid):
    L_nu = single_u[k, i_umin]
    nu_Lnu = nu * L_nu
    print(f"\nq_HAC={qhac:.2f}:")
    print(f"  L_nu min/max: {L_nu.min():.4e} / {L_nu.max():.4e}")
    print(f"  nu*L_nu min/max: {nu_Lnu.min():.4e} / {nu_Lnu.max():.4e}")
