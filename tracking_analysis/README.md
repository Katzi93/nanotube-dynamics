# Tracking analysis

Converts saved ridge fits into tube geometry and motion measurements, then compares motion with an illumination-gradient estimate. This is the current analysis implementation and remains under scientific review. Tracking and NPZ export are described in [tube_tracker](../tube_tracker/README.md).

## Files and outputs

| File | Responsibility | Output relative to the repository root |
| --- | --- | --- |
| [tube_tracks.py](tube_tracks.py) | Loads and reduces track data; adds velocity, illumination gradient, and velocity projections. | Console summary when run as a module; no saved table. |
| [drift_vs_gradient.py](drift_vs_gradient.py) | Fits parallel, perpendicular, and angular motion against the projected gradient. | `analysis_outputs/drift_vs_gradient.png` and console fit summaries. |
| [mobility_vs_length.py](mobility_vs_length.py) | Fits one parallel mobility per track and compares it with median tube length. | `analysis_outputs/mobility_vs_length.png` and console fit/correlation summaries. |

## Inputs

The scripts currently pool `test_1`, `test_2`, and `test_3` from `tracking_data/<dataset>.npz`, using `test_data/background.tif` as the shared illumination map. These names and paths are selected in each script's executable block.

`load_datasets()` expects the ridge-column arrays exported by the tracker, including fit diagnostics. It adds a dataset label and combines them in a pandas DataFrame. `simplify_dataframe()` subsequently keeps the geometry columns; it does not apply additional fit-quality filtering.

Track IDs are local to a movie. Records are grouped by `(dataset, track, frame)` so equal track IDs in different datasets remain distinct. These scripts read canonical NPZ filenames, rather than selecting the latest timestamped GUI export automatically.

## Processing steps

1. Sort each tube's ridge centers by ridge index. Store their arithmetic mean as the tube center, their summed consecutive distances as contour length, and their first and last positions as the two tips. The tips therefore refer to fitted ridge centers.
2. Calculate the mean undirected orientation from doubled ridge angles. The stored orientation lies in `[0, pi)`.
3. Add a central-difference velocity only when frames `f - 1` and `f + 1` exist in the same track: `v(f) = (center(f + 1) - center(f - 1)) / 2`. Other rows retain NaN velocities. Angular velocity uses the corresponding axial angle difference, wrapped to `[-pi/2, pi/2)`, divided by two.
4. Bilinearly sample the illumination map at both tips. With `d = tip2 - tip1`, use `g = (I(tip2) - I(tip1)) * d / dot(d, d)` as the gradient vector. Its denominator uses the Euclidean tip distance, rather than contour length.
5. Orient the mean tube axis `t` so `g dot t >= 0`. Calculate `v_parallel = v dot t`, `v_perp = v dot (-t_y, t_x)`, and `projected_gradient = g dot t`.

`fill_missing_frames()` is an optional helper that interpolates bounded internal gaps and labels inserted records with `interpolated=True`. The current executable workflows do not call it, so their central differences use the recorded neighboring frames.

The in-memory table uses pixels, frames, and radians. The plotting scripts convert to micrometres and seconds with their module-level calibration constants. Calibration is not read from TIFF metadata or NPZ acquisition metadata.

## Fits

`drift_vs_gradient.py` keeps rows where both velocity projections and angular velocity are defined, then performs separate through-origin least-squares fits. Its reported fit statistic is `1 - sum(residual^2) / sum(measured^2)`, rather than the mean-centered coefficient of determination.

`mobility_vs_length.py` keeps tracks meeting `minimum_measured_frames`, currently five eligible velocity rows. It fits parallel velocity against projected gradient through the origin for each track, pairs the result with that track's median contour length, and reports a Pearson correlation and a straight-line trend with intercept.

## Run

Run from the repository root with the `tube-tracking-clean` environment. For this Windows checkout:

```powershell
$pythonExe = 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe'
& $pythonExe -m tracking_analysis.tube_tracks
& $pythonExe -m tracking_analysis.drift_vs_gradient
& $pythonExe -m tracking_analysis.mobility_vs_length
```

Dependencies are NumPy, pandas, SciPy, matplotlib, and tifffile. Review the input paths, calibration constants, and selection parameters before applying the scripts to different movies. Plotting scripts create the output directory and replace their named PNG on rerun.

## Assumptions and scope

- The illumination map must share the coordinate system of every movie; this code does not register or transform it.
- Frame-index differences are interpreted with one fixed frame interval. Acquisition timestamps are not used for differentiation.
- The gradient calculation uses the supplied illumination image without smoothing and requires distinct tip positions.
- The analysis center is an arithmetic mean of ridge centers; the tracker's candidate-comparison center is length-weighted. These definitions should not be treated as interchangeable.

Inputs and generated figures remain local and are excluded from Git. This README documents implemented calculations; it does not establish their scientific validity.
