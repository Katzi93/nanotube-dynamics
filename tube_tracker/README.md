# Tube tracker

Detects fluorescent DNA nanotubes in microscopy images, constructs their center lines, and follows manually initialized tubes forward through a movie. This folder contains the numerical tracker and its Tkinter interface; motion analysis is documented in [tracking_analysis](../tracking_analysis/README.md).

## Files

| File | Responsibility |
| --- | --- |
| [configuration.py](configuration.py) | Stores the tracking parameters. `set_parameters()` updates known parameter names before a run. |
| [ridge_detection.py](ridge_detection.py) | Builds oriented, zero-mean, normalized convolution kernels and finds the strongest local position and angle. |
| [tube_construction.py](tube_construction.py) | Fits finite ridges, checks fit acceptance, and extends a seed in both directions to construct a tube. |
| [tube_tracking.py](tube_tracking.py) | Manages track records, calculates polyline geometry, and searches for the next accepted tube. |
| [interface.py](interface.py) | Displays frames and overlays, handles tube selection and tracking, and exports fitted tracks. |

`configuration` is shared mutable state. Set its parameters before building kernels or starting tracking. The calling script supplies a NumPy random generator so the trial order can be reproduced.

## Open the interface

Use Python from the `tube-tracking-clean` environment. In this Windows checkout, run the following from the repository root:

```powershell
& 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe' .\run_tracking_interface.py
```

The launcher runs [tests/test_interface.py](../tests/test_interface.py). That script selects the input TIFF, output base path, display settings, algorithm parameters, and random seed. Its current input is `test_data/test_3.tif`; it loads the complete stack into memory as `(frame, row, column)`. The interface does not provide a file-selection dialog. Dependencies are NumPy, SciPy, Pillow, tifffile, and Tkinter.

## Interface workflow

1. Navigate with the frame slider, Previous/Next buttons, or left/right arrow keys. Adjust the display with Black level, White level, Auto, and Reset.
2. Click a tube to construct it in the displayed frame. Each accepted initialization gets a track ID. Multiple tubes may start in different frames.
3. Control-click a displayed center line to remove that entire track, subject to the configured screen-pixel distance. Selection and removal are disabled during tracking.
4. Click **Track tubes** to track the newly initialized tubes forward. Previously started tracks are retained and are not restarted by this button.
5. Click **Save tracks** to export all accumulated tracks. Closing the window returns the tracks to the caller but does not automatically save them.

Tracking runs one frame at a time and yields to Tkinter between frames. There is currently no dedicated stop, backward-tracking, or saved-session loading control.

## Tracking method

- The convolution search proposes a ridge center and angle around the click or previous-frame seed.
- Each fit uses a fixed circular pixel selection satisfying `(x - x0)^2 + (y - y0)^2 <= fitting_radius^2`. The fitted intensity model is Gaussian across the ridge and super-Gaussian along it. Its center can move perpendicular to the ridge; there is no fitted parallel shift.
- A fitted ridge must pass convergence, width, amplitude/error, angle-change, and perpendicular-displacement criteria. Extension tries the configured normal, longer, and shorter steps, and stops a branch when all attempts are rejected.
- For a previous tube with `n` ridges and `q = n // 4`, the tracker shuffles seeds with indices `q <= i < n - q` first, followed by the remaining indices. It returns the first complete candidate satisfying length, center-displacement, and end-to-end-angle limits.
- Candidate length is compared with the track's running mean contour length. The center used for candidate comparison is the segment-length-weighted polyline center.

The interface can retry later frames after a missed detection, using the last accepted tube and `maximum_rescue_frames`. Missing frames remain absent from the saved track. Fitting requires the complete fitting region to lie inside the image; an out-of-bounds region raises an error.

## Track representation and export

Coordinates are `(x, y) = (column, row)` in source-image pixels; frame indices are zero-based. Ridge angles are undirected and stored in radians in `[0, pi)`.

In memory, a tube is an ordered list of ridge dictionaries. A track holds `track_id`, `initial_frame`, `tracking_started`, a running `average_length`, and a `tubes` dictionary keyed by accepted frame index.

**Save tracks** writes a compressed NPZ file using the configured base name plus a timestamp with one-second resolution. The archive contains:

| Fields | Contents |
| --- | --- |
| `format_version`, `source_tiff_path`, `analysis_parameters` | Format identifier, source path, and JSON snapshot of the tracking configuration. |
| `track_ids`, `initial_frames`, `tracking_started`, `average_lengths` | One entry per track. |
| `ridge_track_ids`, `ridge_frame_indices`, `ridge_indices` | Track, frame, and within-tube index for every fitted ridge. |
| `ridge_centers_x`, `ridge_centers_y`, `ridge_angles` | Fitted geometry, one entry per ridge. |
| Remaining `ridge_*` arrays | Perpendicular shift and width, amplitude, background, error, convergence, and fit-evaluation count. |

The GUI currently saves under `test_outputs/`. Source TIFFs are read-only, and saved tracking data are excluded from Git.
