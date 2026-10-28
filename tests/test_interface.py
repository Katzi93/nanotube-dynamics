"""Interactively test complete-stack loading and TIFF frame navigation."""

from pathlib import Path

import numpy as np
import tifffile

from tube_tracker import configuration
from tube_tracker.interface import show_tiff_stack
from tube_tracker.ridge_detection import make_convolution_kernels


if __name__ == "__main__":
    input_path = Path("test_data/test_3.tif")  # Read-only microscopy TIFF stack displayed by this test.
    output_path = Path("test_outputs/test_3_tube_tracks.npz")  # Base path for saved tube tracks; each Save button click writes a timestamped copy.
    display_low_percentile = 1.0  # First-frame intensity percentile displayed as black in every frame.
    display_high_percentile = 99.8  # First-frame intensity percentile displayed as white in every frame.
    window_title = "Tube tracking TIFF viewer"  # Text shown in the interface title bar.
    display_width = 1100  # Width in screen pixels reserved for the displayed microscopy frame.
    display_height = 850  # Height in screen pixels reserved for the displayed microscopy frame.
    initialized_tube_color = "#ffffff"  # Color used for tubes that have been initialized but not tracked.
    tracked_tube_colors = (  # Repeating high-contrast colors assigned when tracking starts.
        "#00ffff",
        "#ff4dff",
        "#ffff00",
        "#ff8000",
        "#ff4d4d",
        "#66a3ff",
        "#bf80ff",
        "#ff80bf",
        "#ffbf00",
        "#80d4ff",
        "#ff8066",
    )
    tube_line_width = 3  # Width in screen pixels of each constructed tube line.
    tube_label_offset = 7.0  # Horizontal and upward screen-pixel offset of each track label from one tube endpoint.
    tube_removal_distance = 8.0  # Largest screen-pixel distance at which a control-click removes a tube.
    random_seed = 0  # Seed making the randomized tracking trials reproducible.

    configuration.set_parameters(
        convolution_search_radius=12,  # Half-width in pixels of the square searched around each click.
        convolution_perpendicular_fwhm=6.0,  # Width in pixels across each convolution kernel.
        convolution_parallel_fwhm=30.0,  # Length in pixels along each convolution kernel.
        convolution_angle_step=np.pi / 90.0,  # Angle spacing in radians between convolution kernels.
        convolution_parallel_power=4.0,  # Power controlling the convolution kernel's longitudinal profile.
        fitting_radius=21.0,  # Radius in pixels of the fixed circular ridge-fitting region.
        ridge_parallel_fwhm=30.0,  # Fixed longitudinal FWHM in pixels of every fitted ridge.
        ridge_parallel_power=12.0,  # Power controlling the sharpness of the fitted ridge ends.
        initial_perpendicular_fwhm=6.2,  # Starting estimate of the fitted tube width in pixels.
        extension_step_length=6.0,  # Distance in pixels between consecutive predicted ridge centers.
        long_retry_step_multiplier=2.0,  # Multiplier used to jump past a rejected normal prediction.
        short_retry_step_multiplier=0.5,  # Multiplier used for the final shorter prediction attempt.
        maximum_points_per_direction=100,  # Maximum number of ridges constructed toward each tube end.
        minimum_accepted_fwhm=2.0,  # Smallest fitted tube width accepted as a ridge.
        maximum_accepted_fwhm=12.0,  # Largest fitted tube width accepted as a ridge.
        minimum_amplitude_to_error=1.2,  # Smallest accepted ratio between ridge amplitude and fitting RMSE.
        maximum_accepted_angle_change=np.pi / 9.0,  # Largest accepted angle change from the predicted ridge.
        maximum_accepted_perpendicular_shift=8.0,  # Largest accepted sideways displacement in pixels.
        fitting_verbose=False,  # Suppress individual ridge-fitting and retry messages after a click.
        tracking_verbose=False,  # Suppress individual tracking trials and show only interface-level progress lines.
        maximum_relative_length_difference=0.15,  # Largest accepted fractional difference from the running mean length.
        maximum_center_position_difference=25.0,  # Largest accepted tube-center displacement between frames in pixels.
        maximum_overall_angle_difference=np.pi / 6.0,  # Largest accepted end-to-end tube-angle change between frames.
        maximum_rescue_frames=3,  # Number of additional future frames searched after a tube is first lost.
    )

    print(f"Loading complete TIFF stack from {input_path}...")
    tiff_stack = tifffile.imread(input_path)
    frame_count, image_height, image_width = tiff_stack.shape
    memory_gibibytes = tiff_stack.nbytes / 1024**3
    print(
        f"Loaded {frame_count} frames of {image_width} x {image_height} pixels, dtype={tiff_stack.dtype}, "
        f"memory={memory_gibibytes:.2f} GiB."
    )

    convolution_kernels = make_convolution_kernels()
    random_generator = np.random.default_rng(random_seed)
    tube_tracks = show_tiff_stack(
        input_path,
        output_path,
        tiff_stack,
        convolution_kernels,
        random_generator,
        display_low_percentile,
        display_high_percentile,
        window_title,
        display_width,
        display_height,
        initialized_tube_color,
        tracked_tube_colors,
        tube_line_width,
        tube_label_offset,
        tube_removal_distance,
    )
    print(f"The closed interface contains {len(tube_tracks)} tube tracks.")
