# Tracking development scripts

Interactive checks of ridge detection, tube construction, frame-to-frame tracking, and the graphical interface. These scripts require local TIFF inputs and usually a mouse click or GUI interaction. They are development entry points rather than an unattended unit-test suite.

## Scripts using the current tracker

| File | Default input and interaction | Output |
| --- | --- | --- |
| [test_interface.py](test_interface.py) | Loads the complete `test_data/test_3.tif` stack and opens the tracking GUI. Supports multiple manual tube initializations. | Clicking Save tracks writes timestamped `test_outputs/test_3_tube_tracks_*.npz` files. |
| [test_local_convolution.py](test_local_convolution.py) | Reads frame 80 of `test_data/test_1.tif`; asks for one click and displays the predicted ridge center and angle. | Console details and an interactive matplotlib view; no file is saved. |
| [test_tube_construction.py](test_tube_construction.py) | Reads frame 80 of `test_data/test_1.tif`; constructs a complete tube from one click and shows its fitted centers. | Console fit diagnostics and an interactive matplotlib view; no file is saved. |
| [test_tube_tracking.py](test_tube_tracking.py) | Initializes one tube in frame 80 of `test_data/test_1.tif` and attempts frames 81 through 89. | `test_outputs/new_tracker_frames_080_089.tif`, containing the successfully tracked prefix with RGB overlays. |

The single-track test stops on its first failed next-frame search; it does not use the GUI's later-frame rescue loop. Its TIFF is a visualization, not the numerical NPZ export used by tracking analysis.

`test_interface.py` supplies the configuration used by the repository-root `run_tracking_interface.py`. It selects input/output paths, display parameters, tracking acceptance criteria, and the NumPy random seed. The GUI behavior and export fields are described in [tube_tracker](../tube_tracker/README.md).

## Earlier standalone prototypes

| File | Purpose | Default output |
| --- | --- | --- |
| [test_raw_image_ridge_fit.py](test_raw_image_ridge_fit.py) | Traces one clicked tube in frame 80 of `test_data/test_1.tif` using its own convolution and iterative ridge-fitting implementation. | `test_outputs/raw_image_ridge_fit.png` |
| [test_candidate_tube_tracking.py](test_candidate_tube_tracking.py) | Uses the standalone fitter to trace candidates from five positions along the previous tube and selects the accepted candidate with the lowest geometric disagreement score. Its current demonstration uses frames 70 through 99 of `test_data/test_2.tif`. | `test_outputs/legacy_tracker_test2_frames_070_099.tif` |

These two files do not call the current `tube_tracker` construction/tracking implementation. The candidate prototype imports its sibling `test_raw_image_ridge_fit`, and writes an updated partial TIFF after each completed frame. It raises an assertion if no candidate is accepted. Both prototypes replace their named output files.

## Run

Use the `tube-tracking-clean` environment from the repository root. A desktop display and an interactive matplotlib backend are needed for the click-based scripts. In this Windows checkout:

```powershell
$pythonExe = 'C:\Users\Flori\anaconda3\envs\tube-tracking-clean\python.exe'
& $pythonExe .\run_tracking_interface.py
& $pythonExe -m tests.test_local_convolution
& $pythonExe -m tests.test_tube_construction
& $pythonExe -m tests.test_tube_tracking
```

Run the earlier prototypes by file path so their sibling imports resolve:

```powershell
& $pythonExe .\tests\test_raw_image_ridge_fit.py
& $pythonExe .\tests\test_candidate_tube_tracking.py
```

Dependencies include NumPy, SciPy, matplotlib, tifffile, Pillow, and Tkinter for the GUI. Script parameters are defined in each executable block. The current-tracker scripts call `configuration.set_parameters()` before creating kernels or fitting tubes.

## Reading the output

Inspect whether the displayed ridge or full center line follows the selected tube, then inspect continuity in the generated tracking overlays. Console output records fit rejections, tracking trials, and timing. The result depends on the selected input and manual initialization; there is no fixed numerical reference result in this folder.

All input paths are relative to the repository root and use read-only `test_data/`. Generated NPZ, TIFF, and PNG files belong under the ignored `test_outputs/` directory. These scripts do not download test inputs.
