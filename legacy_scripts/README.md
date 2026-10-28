# Legacy scripts

Earlier exploratory tube-motion analyses and sensitivity checks, formerly stored in `test_analysis/`. They are retained for reference. The current analysis is documented in [tracking_analysis](../tracking_analysis/README.md); the calculations here use different geometry and gradient conventions.

## Scripts

| File | What it does | Figure filenames under `test_outputs/` |
| --- | --- | --- |
| [analyze_tube_motion.py](analyze_tube_motion.py) | Combines trajectory geometry, central-difference drift, gradient comparisons, mobility versus length, lagged MSD fits, and direct finite-lag displacement analysis. Provides shared helpers for the diagnostics. | `test_combined_motion_analysis.png`, `test_combined_noise_analysis.png`, `test_combined_direct_drift_analysis.png` |
| [diagnose_central_difference.py](diagnose_central_difference.py) | Fits three-frame central-difference velocities against a length-weighted, along-ridge gradient; displays parallel and perpendicular velocities. | `test_combined_central_difference_diagnosis.png` |
| [diagnose_drift_lag.py](diagnose_drift_lag.py) | Compares fitted mobility and binned drift across displacement lags; also compares start/end gradient magnitudes in selected intervals. | `test_2_drift_lag_diagnosis.png` |
| [diagnose_gradient_definition.py](diagnose_gradient_definition.py) | Compares center-point, endpoint-per-contour-length, and whole-tube gradient estimates over uninterrupted frame intervals. | `test_2_gradient_definition_diagnosis.png` |
| [diagnose_mobility_length.py](diagnose_mobility_length.py) | Compares per-track mobilities across gradient definitions, including an unnormalized endpoint difference; reports Pearson, Spearman, and leave-one-track-out correlation checks. | `test_combined_mobility_length_diagnosis.png` |
| [diagnose_velocity_window.py](diagnose_velocity_window.py) | Compares Savitzky-Golay velocity windows while holding the orientation window fixed. | `test_combined_velocity_window_diagnosis.png` |

The scripts print diagnostic summaries and replace their named figures on rerun. They do not modify the saved input tracks.

## Input selection

Each executable block specifies one or more `test_outputs/*tube_tracks*.npz` patterns. `newest_track_file()` sorts matching paths and selects the lexicographically last one, relying on the timestamp in the filename rather than filesystem modification time. Different diagnostics currently select different movie combinations, so check `track_glob` or `track_globs` before comparing their outputs.

All scripts use `test_data/background.tif` as an illumination image. The geometry loader reconstructs the ordered ridge positions and angles saved by the tracker. When pooling files, `load_and_merge_tube_geometry()` offsets track IDs to avoid collisions between movies.

## Implemented conventions

- Tube centers are segment-length-weighted polyline centers imported from `tube_tracker`; whole-tube axes are principal axes of the ridge positions.
- The illumination map is Gaussian-smoothed before taking `numpy.gradient`. Local gradients are sampled by bilinear interpolation.
- The whole-tube gradient averages along-ridge projections with length weights. The legacy endpoint gradient divides the tip intensity difference by contour length.
- The combined analysis and central-difference diagnostic use recorded frames `f - 1` and `f + 1` for velocity. Finite-lag diagnostics require every frame in an interval to be present.
- The velocity-window diagnostic first interpolates geometry over all internal gaps, then smooths centers and undirected orientations before differentiating. This interpolation is specific to that diagnostic.
- Reported lengths and velocities generally remain in pixels and frames. Through-origin fit statistics and the combined MSD fit statistics use different denominators, as implemented in their helper functions.

Some introductory comments still refer to an earlier smoothed main workflow. The executable combined analysis currently uses bare central differences. These scripts are diagnostic implementations, and their printed interpretation text is not a validation of a physical conclusion.

## Run

Run from the repository root with the `tube-tracking-clean` environment. For example, in this Windows checkout:

```powershell
$pythonExe = 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe'
& $pythonExe -m legacy_scripts.analyze_tube_motion
& $pythonExe -m legacy_scripts.diagnose_gradient_definition
```

Use the same `-m legacy_scripts.<script_name>` form for the other diagnostics. Dependencies are NumPy, SciPy, matplotlib, tifffile, and the local `tube_tracker` package. Input patterns, smoothing widths, frame lags, windows, exclusions, and plotting settings live in the individual executable blocks. No command-line argument parser is provided.

The local input archives, illumination image, and diagnostic figures are excluded from Git. This folder contains the exploratory source code only.
