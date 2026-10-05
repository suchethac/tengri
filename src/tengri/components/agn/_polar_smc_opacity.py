# SPDX-License-Identifier: BSD-3-Clause
r"""Tabulated SMC-mixture dust opacity below 100 nm, for the polar-dust Bongiorno law.

CIGALE's ``skirtor2016`` and ``fritz2006`` modules (Boquien et al. 2019 [1]_) evaluate the
Bongiorno et al. (2012) power law :math:`k = 1.39\,\lambda_{\mu{\rm m}}^{-1.2}` above 100 nm
and, below it, replace the power law by the *shape* of a tabulated SMC dust-mixture mass
extinction coefficient, rescaled to match the power law at 100 nm. The table is repackaged here
from CIGALE's ``pcigale/sed_modules/curves/extFun_SMC.dat`` (pcigale 2025.1, distributed under
the CeCILL licence), as numerical data with attribution. The file's header names only its
columns (wavelength, total, absorption and scattering mass opacity in m^2/kg, mean asymmetry
parameter) and no paper; CIGALE's history records it as the correction "of the extinction
curves of the polar dust below 100 nm using computation from Marko Stalevski" (commit
2d2c3874, 2021-04-30), a dust-mixture calculation that CIGALE credits to M. Stalevski and
that is not published as a table in a paper.

Only the wavelengths up to the first node above 100 nm are kept (the splice never reads
more), and only the total extinction column, which sets the shape.

References
----------
.. [1] M. Boquien et al., "CIGALE: a python Code Investigating GALaxy Emission," A&A, 622,
   A103 (2019). arXiv:1811.03094. https://doi.org/10.1051/0004-6361/201834156
"""

import numpy as np

#: Wavelengths of the tabulated opacity. [um]
SMC_OPACITY_WAVE_UM = np.array(
    [
        1.000000e-03,
        1.071891e-03,
        1.148951e-03,
        1.231551e-03,
        1.320088e-03,
        1.414991e-03,
        1.516717e-03,
        1.625756e-03,
        1.742633e-03,
        1.867914e-03,
        2.002200e-03,
        2.146141e-03,
        2.300430e-03,
        2.465811e-03,
        2.643081e-03,
        2.833096e-03,
        3.036771e-03,
        3.255089e-03,
        3.489101e-03,
        3.739937e-03,
        4.008806e-03,
        4.297005e-03,
        4.605922e-03,
        4.937048e-03,
        5.291979e-03,
        5.672426e-03,
        6.080224e-03,
        6.517340e-03,
        6.985880e-03,
        7.488104e-03,
        8.026434e-03,
        8.603464e-03,
        9.221979e-03,
        9.884959e-03,
        1.059560e-02,
        1.135733e-02,
        1.217383e-02,
        1.304902e-02,
        1.398713e-02,
        1.499268e-02,
        1.607053e-02,
        1.722586e-02,
        1.846425e-02,
        1.979167e-02,
        2.121452e-02,
        2.273966e-02,
        2.437444e-02,
        2.612675e-02,
        2.800504e-02,
        3.001836e-02,
        3.217642e-02,
        3.448962e-02,
        3.696913e-02,
        3.962689e-02,
        4.247572e-02,
        4.552935e-02,
        4.880252e-02,
        5.231099e-02,
        5.607170e-02,
        6.010277e-02,
        6.442364e-02,
        6.905514e-02,
        7.401960e-02,
        7.934097e-02,
        8.504489e-02,
        9.115888e-02,
        9.771242e-02,
        1.047371e-01,
    ]
)

#: Total mass extinction coefficient at SMC_OPACITY_WAVE_UM. [m^2/kg]
SMC_OPACITY_EXT = np.array(
    [
        6.777918e02,
        7.814090e02,
        8.844537e02,
        9.929317e02,
        1.106923e03,
        1.214646e03,
        1.255719e03,
        1.155390e03,
        1.100148e03,
        1.296353e03,
        1.447162e03,
        1.557418e03,
        1.300308e03,
        1.252727e03,
        1.519165e03,
        1.720445e03,
        1.911619e03,
        2.099654e03,
        2.294184e03,
        2.492154e03,
        2.689912e03,
        2.799381e03,
        2.948647e03,
        3.200891e03,
        3.454348e03,
        3.690129e03,
        3.919954e03,
        4.143377e03,
        4.357731e03,
        4.421038e03,
        4.684831e03,
        4.917728e03,
        5.132084e03,
        5.307470e03,
        5.313827e03,
        5.128535e03,
        5.393154e03,
        5.521861e03,
        5.638576e03,
        5.852750e03,
        6.054693e03,
        6.251103e03,
        6.423006e03,
        6.424145e03,
        6.634528e03,
        6.838926e03,
        7.057516e03,
        7.345270e03,
        7.616558e03,
        7.877817e03,
        8.128195e03,
        8.367591e03,
        8.586756e03,
        8.798448e03,
        9.022539e03,
        9.267598e03,
        9.540743e03,
        9.886940e03,
        1.040316e04,
        1.235623e04,
        1.629332e04,
        1.826094e04,
        1.855779e04,
        1.768315e04,
        1.644099e04,
        1.583636e04,
        1.512779e04,
        1.385008e04,
    ]
)
