# Lyu and Rieke infrared templates

Tengri provides three Lyu 2018 AGN template families and two public host-galaxy
infrared templates. These model types are experimental; the recorded
comparison covers one fixed example only.

## Model choices

| Group | Type | Template |
| --- | --- | --- |
| `agn` | `lyu2018` | NORMAL |
| `agn` | `lyu2018_wdd` | WDD |
| `agn` | `lyu2018_hdd` | HDD |
| `dust_emission` | `haro11` | Haro 11 host-galaxy template, with source data above 5 μm |
| `dust_emission` | `rieke2009` | Luminosity-dependent normal star-forming-galaxy templates |

The AGN families use the public template data discussed by [Lyu, Rieke & Shi
(2017)](https://doi.org/10.3847/1538-4357/835/2/257) and [Lyu & Rieke
(2018)](https://doi.org/10.3847/1538-4357/aae075). Haro 11 comes from [Lyu,
Rieke & Alberts (2016)](https://doi.org/10.3847/0004-637X/816/2/85). The Rieke
library is from [Rieke et al. (2009)](https://doi.org/10.1088/0004-637X/692/1/556);
[Lyu et al. (2022)](https://doi.org/10.3847/1538-4357/ac9e5d) discuss a
modified version and its use in SED fitting.

## Amplitudes and dust attenuation

For the Lyu AGN families, `agn_lyu2018_tau_v` selects a value from 0 to 10.
`agn_log_lbol` scales the finite 0.01–1000 μm integral of the unreddened
(`tau_v=0`) reference spectrum. Tengri uses that family's tau-zero integral
for every optical depth. This parameter is a reference-template normalization,
not a measurement of the central engine's physical bolometric luminosity.
For these three types, the derived fields `L_agn_bol` and `log_L_agn_bol` carry
this same finite reference-template normalization despite their generic names.

For the dust templates, `dust_log_L_ir` sets the total infrared amplitude in
log10 solar luminosities when explicitly declared. For `rieke2009`,
`dust_log_L_ir_template` selects the template shape over the library's
9.75–13 range; it does not set the emitted amplitude. The table summarizes how
attenuation and emission work together:

| `dust_attenuation` | Explicit `dust_log_L_ir` | Emission normalization |
| --- | --- | --- |
| Active | Omitted | `dust_eta_balance × L_absorbed` (default `dust_eta_balance=1`) |
| Active | Set | The explicit value scales the IR template; attenuation remains active and `L_absorbed` is still computed |
| `{"type": "none"}` | Set | The explicit value is required because there is no absorbed-light budget |

An independent IR amplitude can therefore be used with an active stellar dust
screen. Declaring it changes the emission normalization; it does not turn off
attenuation. Leaving it undeclared retains energy balance. If attenuation is
disabled while a dust-emission type is selected, construction requires an
explicit `dust_log_L_ir`.

Both emitters normalize the template shape over the model wavelength grid.
Haro 11 contributes above 5 μm; the Rieke tables cover their source wavelength
ranges and contribute zero outside those ranges. `L_ir` is this full-grid
normalization. `L_TIR` is a separate integral of the resulting SED over
8–1000 μm, so the two values need not be equal.

## Example

This configuration keeps the attenuation screen active while giving the
Rieke template an independent amplitude and shape selector:

```python
from tengri import DEFAULT, Fixed, SEDModel, Uniform

model = SEDModel.build(
    ssp_data=ssp,
    observation=obs,
    sfh={"type": "delayed", "all_params": Fixed(DEFAULT)},
    dust_attenuation={
        "type": "two_component",
        "law": "calzetti",
        "tau_bc": Uniform(0.0, 1.0),
        "tau_diff": Uniform(0.0, 1.0),
        "other_params": Fixed(DEFAULT),
    },
    dust_emission={
        "type": "rieke2009",
        "dust_log_L_ir": Uniform(9.0, 13.0),
        "dust_log_L_ir_template": Uniform(9.75, 13.0),
        "other_params": Fixed(DEFAULT),
    },
)
```

To keep the attenuation-based energy budget, remove the `dust_log_L_ir` entry
and leave `dust_attenuation` active. To use `haro11`, select that type; it has
no shape parameter. With `dust_attenuation={"type": "none"}`, either dust
emission type needs an explicit `dust_log_L_ir`.
