"""Why the DPL row is the worst-conditioned in the suite, and the delayed-tau row is not.

79/II (DPL, D=7) scored 381/1200 divergences at ess_min 1.9; 79/III
(delayed-tau) scored 22 at higher D. If DPL's tau has a sensitivity cliff where
delayed-tau's does not, that explains the inversion without invoking dimension.

For a DPL, tau is the TURNOVER between two power laws: push it past the
galaxy's age and the shape is set by alpha alone. For a delayed-tau, tau is an
e-folding time that keeps shaping the history at any value, so there should be
no cliff.

Same library and same redshift for both, so only the SFH functional form
differs. Sensitivity is max |dF/F| over the CANDELS bands.

Measured (z=1.065, cosmic age 5.61 Gyr, tau prior 0.5-13 Gyr in both):

    sfh       tau inside     tau outside     ratio
    dpl       293-2149%      0.018-0.63%     467 - 121000
    delayed   245-345%       3.9-4.4%        64 - 79

Delayed-tau's tau is an e-folding time that keeps shaping the history at any
value, so it stays a well-conditioned direction (~4% even when large) and its
ratio is tight across draws. The DPL's tau is a TURNOVER: once it exceeds the
galaxy's age the shape is set by alpha alone and tau flattens by ~200x, with
the ratio itself swinging 260x across prior draws -- so the geometry changes
character within the prior, which one step size cannot serve.

This is a prior-geometry defect, not a sampler-tuning one, and not a
dimensional one: the DPL row is the LOWEST-dimensional in the suite (D=7) and
scored 381/1200 divergences at ess_min 1.9 against 22 for delayed-tau at
higher D. The indicated fix is to cap tau_gyr near the cosmic age, as
age_gyr already is, so the turnover stays reachable.

NOTE ON THE LIBRARY: Configuration II's own grid (fsps_prsc_c3k_a_chabrier)
is not present on every machine; this script substitutes a C3K grid that is.
The question is a property of the SFH parametrization, not of the library.
"""

import sys
import warnings

warnings.filterwarnings("ignore")
import jax
import numpy as np

sys.path.insert(0, "analysis")
from paper1 import configs
from paper1.candels_io import CANDELS_TO_TENGRI

import tengri
from tengri import DEFAULT, Fixed, SEDModel, Uniform, WavePrecomp

Z = 1.065
AGE = configs.age_at_z(Z)
ssp = tengri.load_ssp("fsps_mist_c3k_a_chabrier")
names = sorted(set(CANDELS_TO_TENGRI.values()))
obs = tengri.Photometry.from_names(names)


def build(sfh_type):
    sfh = {
        "type": sfh_type,
        "all_params": Fixed(DEFAULT),
        "tau_gyr": Uniform(0.5, 13.0),
        "age_gyr": Uniform(1.0, AGE),
        "log_total_mass": Uniform(8.0, 12.5),
        "met_logzsol": configs.met_prior_for(ssp),
    }
    if sfh_type == "dpl":
        sfh["alpha"] = Uniform(0.5, 5.0)
        sfh["beta"] = Uniform(0.3, 3.0)
    return SEDModel.build(
        ssp_data=ssp,
        observation=obs,
        sfh=sfh,
        dust_attenuation={
            "type": "single_component",
            "law": "calzetti",
            "all_params": Fixed(DEFAULT),
            "tau_v": Uniform(0.0, 3.0),
        },
        dust_emission={"type": "dale2014", "all_params": Fixed(DEFAULT)},
        redshift=Fixed(Z),
        igm={"type": "inoue"},
        approx=WavePrecomp(),
    )


print(f"cosmic age at z={Z:.3f} is {AGE:.2f} Gyr; tau prior runs to 13 Gyr in both")
print(f"{'sfh':<12} {'draw':<6} {'tau inside':>14} {'tau outside':>14}   ratio")
for sfh_type in ("dpl", "delayed"):
    model = build(sfh_type)
    pre = f"sfh_{sfh_type}_"
    for seed in (0, 1, 2):
        p = {
            k: float(np.asarray(v)) for k, v in model.spec.sample(jax.random.PRNGKey(seed)).items()
        }
        p[pre + "age_gyr"] = min(
            p[pre + "age_gyr"], 0.35 * AGE
        )  # young: turnover clearly escapable
        out = []
        for lo, hi in ((0.5, 3.0), (7.0, 13.0)):
            a = dict(p)
            a[pre + "tau_gyr"] = lo
            b = dict(p)
            b[pre + "tau_gyr"] = hi
            fa = np.asarray(model.predict_photometry(a))
            fb = np.asarray(model.predict_photometry(b))
            out.append(100.0 * float((np.abs(fb - fa) / np.abs(fa)).max()))
        ratio = out[0] / out[1] if out[1] > 0 else float("inf")
        print(f"{sfh_type:<12} {seed:<6d} {out[0]:13.4f}% {out[1]:13.4f}%   {ratio:.3g}")
