# ITO light response

Characterizes the measured electrical response of ITO under illumination. This module analyzes voltage sweeps, illumination-intensity sweeps, and changes in illuminated spot size. It is separate from the nanotube trajectory analysis.

## Folder layout

| Folder or file | Purpose |
| --- | --- |
| [analysis_scripts/](analysis_scripts/README.md) | Measurement loading, current conversion, response fits, and illumination-profile plots. |
| `data/ito_characterization_1/voltage_sweep/` | Local oscilloscope CSV measurements for illuminated and dark voltage sweeps. |
| `data/ito_characterization_1/intensity_sweep/` | Local oscilloscope measurements, camera images, and the shared dark camera image. |
| `data/ito_characterization_1/spot_sizes/` | Local oscilloscope measurements and camera images for different spot sizes. |
| `output/` | Generated response and cross-section figures. |
| `HANDOFF.md` | Local experimental interpretation and results, excluded from Git. |

The existing acquisition subfolder name `current_measurments` is used literally by the scripts. Within intensity and spot-size sweeps, a measurement CSV stem identifies the corresponding image subfolder.

## What is calculated

- **Intensity response:** background-subtracted integrated camera counts versus the light-induced current, fitted with a saturating response.
- **Spot-size response:** integrated camera counts versus light-induced current for changing spot geometry, fitted with an unweighted line through the origin.
- **Voltage response:** measured current versus applied voltage for illuminated and dark conditions, plus a separate on-minus-off comparison for rising and falling branches.
- **Spot profiles:** horizontal and vertical camera-intensity cross-sections through representative illumination spots.

CSV channel means are interpreted as applied voltage on CH1 and amplifier output on CH2. The scripts use the configured shunt resistance and amplifier gain to express current in microamperes. Camera intensity remains in arbitrary camera-count units; the cross-section positions are in pixels.

## Running an analysis

Use the `tube-tracking-clean` environment and run scripts by file path from the repository root. For example, in this Windows checkout:

```powershell
& 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe' .\ito_light_response\analysis_scripts\plot_intensity_response.py
```

The [script README](analysis_scripts/README.md) lists all entry points, output filenames, input requirements, and fitting choices. Dataset paths, electrical conversion factors, intensity selections, and plot settings are set in each script's executable block. Dependencies are NumPy, SciPy, matplotlib, and tifffile.

Data, generated figures, and handoff documents remain local. The READMEs describe the processing, without reporting fitted experimental results.
