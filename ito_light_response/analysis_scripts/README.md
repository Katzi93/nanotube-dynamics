# ITO analysis scripts

Processes the oscilloscope and camera measurements used to characterize ITO light response. The local acquisition folders are described in the [parent README](../README.md).

## Entry points and outputs

All output paths below are relative to `ito_light_response/output/`.

| Script | Calculation | Saved figure |
| --- | --- | --- |
| [plot_intensity_response.py](plot_intensity_response.py) | Light-induced current versus integrated illumination, with a saturation fit. Also provides shared loading and conversion helpers. | `intensity_response.png` |
| [plot_spot_size_response.py](plot_spot_size_response.py) | Light-induced current versus integrated illumination across spot sizes, with a through-origin linear fit. | `spot_size_response.png` |
| [plot_spot_cross_sections.py](plot_spot_cross_sections.py) | Horizontal and vertical intensity profiles through the selected camera spots. | `spot_cross_section_small_spot.png`, `spot_cross_section_large_spot.png` |
| [plot_voltage_response.py](plot_voltage_response.py) | Illuminated and dark current-voltage scatter plots. | `voltage_response.png` |
| [plot_voltage_difference.py](plot_voltage_difference.py) | On-minus-off current at matched voltages, keeping sweep directions separate. | `voltage_response_difference.png` |

Scripts print measurements and fit summaries to the console. They create output directories as needed and replace their named PNG files on rerun; they do not export fitted tables.

## Measurement loading and conversion

`load_channel_means()` reads CSV columns named `Channel` and `Mean`, returning separate CH1 and CH2 arrays in recorded order. The scripts assume the two channel sequences correspond measurement by measurement. CH1 supplies the applied voltage. CH2 is converted as `current_uA = CH2 / (shunt_resistance_ohm * amplifier_gain) * 1e6`.

For intensity and spot-size sweeps, CH2 entries are paired as off/on measurements: entries `0, 2, 4, ...` are subtracted from entries `1, 3, 5, ...`. Each plotted current change is the pair mean, with standard error calculated from the sample standard deviation (`ddof=1`).

Camera images are converted to floating point before dark-image subtraction. Integrated illumination is the sum of all corrected pixels divided by `intensity_scale`. Negative corrected pixels contribute to this sum. Measurement CSV filenames are sorted by their numeric component and matched to camera subfolders by filename stem.

## Response fits and selections

**Intensity sweep:** each measurement requires exactly one matching TIFF image. Measurements above `maximum_intensity` are excluded. The fitted function is `I_sat * I_int / (I_half + I_int)`. SciPy `curve_fit` uses the current SEM values as absolute uncertainties and constrains both fit parameters to be nonnegative. Parameter uncertainties are taken from the covariance diagonal; the printed R-squared uses mean-centered measured currents.

**Spot-size sweep:** measurements without a matching camera image are explicitly excluded, as are measurements above the intensity cutoff. The fit is `current_change = c * integrated_intensity`, constrained through zero and unweighted. Displayed error bars are SEM values, but they are not used as fit weights. The coefficient uncertainty comes from residual variance. Both response scripts use the shared dark camera image from the intensity-sweep folder.

**Voltage sweeps:** `plot_voltage_response.py` reads the selected illuminated and dark CSVs and plots absolute converted current, without averaging or fitting a response model. `plot_voltage_difference.py` selects a rising branch from the pre-maximum minimum to the global maximum and a falling branch from that maximum to the following minimum. For each direction, it interpolates dark current onto illuminated-sweep voltages only inside their shared voltage range, then subtracts it. It does not extrapolate.

## Illumination cross-sections

`find_spot_center()` clips corrected intensities to nonnegative values for center estimation and selects pixels satisfying `positive_intensity > threshold_fraction * positive_intensity.max()`. Their intensity-weighted center is rounded to integer image coordinates.

`extract_cross_sections()` averages horizontal and vertical strips centered on that location. For an odd `profile_band_width = 2*h + 1`, it averages rows `center_y - h` through `center_y + h` for the horizontal profile and the corresponding columns for the vertical profile. Profiles use the original signed background-corrected intensities. The selected strip must fit inside the image. Positions are relative to the center in pixels; there is no spatial calibration in this script.

## Run

Run from the repository root using Python from `tube-tracking-clean`. Use file-path execution, which also allows the scripts to import their sibling `plot_intensity_response` helpers:

```powershell
$pythonExe = 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe'
& $pythonExe .\ito_light_response\analysis_scripts\plot_intensity_response.py
& $pythonExe .\ito_light_response\analysis_scripts\plot_spot_size_response.py
& $pythonExe .\ito_light_response\analysis_scripts\plot_spot_cross_sections.py
& $pythonExe .\ito_light_response\analysis_scripts\plot_voltage_response.py
& $pythonExe .\ito_light_response\analysis_scripts\plot_voltage_difference.py
```

The scripts require NumPy, SciPy, matplotlib, and tifffile; CSV loading uses Python's standard library. Edit the named parameters in each executable block for different acquisition paths, resistance, gain, cutoffs, or representative spots. Current paths are relative to the repository root. Matching image dimensions and consistent camera settings are assumed; the scripts do not register or rescale images.
