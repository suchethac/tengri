"""Power-law AGN radio jet: radio_log_nu_cut is accepted and ignored."""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, tengri
from tengri import DEFAULT, Fixed, SEDModel
ssp = tengri.load_ssp()
def radio_lnu(agn_radio):
    m = SEDModel.build(ssp_data=ssp, met={"logzsol": Fixed(0.0), "all_params": Fixed(DEFAULT)},
        sfh={"type": "delayed", "tau_gyr": Fixed(1.0), "age_gyr": Fixed(3.0), "log_total_mass": Fixed(10.0), "all_params": Fixed(DEFAULT)},
        dust_attenuation={"type": "single_component", "law": "calzetti", "all_params": Fixed(DEFAULT)},
        dust_emission={"type": "none"}, neb={"type": "none"},
        agn={"type": "composable", "all_params": Fixed(DEFAULT)},
        radio={"all_params": Fixed(DEFAULT), "agn": {"all_params": Fixed(DEFAULT), **agn_radio}}, redshift=Fixed(0.0))
    st = m.predict_state({})
    wave = np.asarray(m.wave if hasattr(m, "wave") else st.wave); sed = np.asarray(st.derived["sed_radio"])
    return wave, sed
C = 2.99792458e18
w, base = radio_lnu({})
nu = C / w
def at(sed, ghz):
    return float(np.interp(np.log10(ghz * 1e9), np.log10(nu[::-1]), sed[::-1]))
print("sed_radio L_nu [erg/s/Hz] at 5, 100, 300, 1000 GHz (default power-law AGN radio model)")
rows = (("default (cutoff exp(-nu/1e13 Hz))", {}),
        ("radio_log_nu_cut = 40 (no cutoff)", {"radio_log_nu_cut": Fixed(40.0)}),
        ("radio_log_nu_cut = 11 (cut at 100 GHz)", {"radio_log_nu_cut": Fixed(11.0)}),
        ("radio_loudness = 3 (control: the jet is in the sum)", {"radio_loudness": Fixed(3.0)}))
for label, extra in rows:
    _, sed = radio_lnu(extra)
    print(f"  {label:52s}", " ".join(f"{at(sed, g):.6e}" for g in (5, 100, 300, 1000)), "| identical to default:", bool(np.array_equal(sed, base)))
print("expected with the key honored, relative to the default at 100/300/1000 GHz: cut = 40 -> x1.0101, x1.0305, x1.1052; cut = 11 -> x0.372, x0.0513, x5.0e-05")
print("turnover keys of the 'dpl' model given to the power-law model:")
for key, val in (("radio_alpha_thin", 1.5), ("radio_alpha_thick", -1.0), ("radio_log_nu_t", 10.5)):
    try:
        _, sed = radio_lnu({key: Fixed(val)})
        print(f"  {key} = {val}: accepted | identical to default:", bool(np.array_equal(sed, base)))
    except Exception as e:
        print(f"  {key} = {val}: {type(e).__name__}: {str(e).splitlines()[0][:140]}")
