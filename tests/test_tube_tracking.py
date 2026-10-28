"""Interactively test one tube track across consecutive microscopy frames."""

from pathlib import Path
from time import perf_counter

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw
import tifffile

from tube_tracker import configuration
from tube_tracker.ridge_detection import find_likely_ridge, make_convolution_kernels
from tube_tracker.tube_construction import construct_tube
from tube_tracker.tube_tracking import add_tube_to_track, create_tube_track, track_tube, tube_positions


def draw_tube(frame, tube, display_low_percentile, display_high_percentile, outline_width, line_width):
    """Return one RGB frame with the fitted tube drawn over the microscopy image."""
    display_minimum, display_maximum = np.percentile(frame, [display_low_percentile, display_high_percentile])
    scaled_frame = np.clip((frame - display_minimum) / (display_maximum - display_minimum), 0.0, 1.0)
    grayscale_frame = np.round(255.0 * scaled_frame).astype(np.uint8)
    rgb_frame = np.repeat(grayscale_frame[:, :, np.newaxis], 3, axis=2)
    labeled_frame = Image.fromarray(rgb_frame)
    drawing = ImageDraw.Draw(labeled_frame)
    tube_center_line = [tuple(position) for position in tube_positions(tube)]
    drawing.line(tube_center_line, fill=(0, 0, 0), width=outline_width, joint="curve")
    drawing.line(tube_center_line, fill=(0, 255, 255), width=line_width, joint="curve")
    return np.asarray(labeled_frame)


if __name__ == "__main__":
    input_path = Path("test_data/test_1.tif")  # Read-only microscopy TIFF stack used for this tracking test.
    output_path = Path("test_outputs/new_tracker_frames_080_089.tif")  # Single labeled TIFF written by this test.
    frame_indices = tuple(range(80, 90))  # Consecutive zero-based frames tracked in this test.
    track_id = 1  # Unique identifier assigned to the clicked physical tube.
    display_low_percentile = 1.0  # Raw-image percentile displayed as black.
    display_high_percentile = 99.8  # Raw-image percentile displayed as white.
    trace_outline_width = 7  # Width in pixels of the black outline around the tracked tube.
    trace_line_width = 3  # Width in pixels of the cyan tracked-tube line.
    playback_frames_per_second = 2.0  # Playback speed stored in the output TIFF metadata.
    random_seed = 0  # Seed making the random ridge-trial order reproducible.

    configuration.set_parameters(
        convolution_search_radius=12,  # Half-width in pixels of the square searched around clicks and tracking seeds.
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
        fitting_verbose=False,  # Suppress individual ridge-fitting and retry messages during frame tracking.
        tracking_verbose=True,  # Print individual trials and rejection reasons in this algorithm test.
        maximum_relative_length_difference=0.15,  # Largest accepted fractional difference from the running mean length.
        maximum_center_position_difference=25.0,  # Largest accepted tube-center displacement between frames in pixels.
        maximum_overall_angle_difference=np.pi / 6.0,  # Largest accepted end-to-end angle change between frames.
    )

    kernel_start_time = perf_counter()
    convolution_kernels = make_convolution_kernels()
    random_generator = np.random.default_rng(random_seed)
    kernel_time = perf_counter() - kernel_start_time
    print(f"Created convolution kernels in {kernel_time:.2f}s.")
    first_frame_index = frame_indices[0]
    print(f"Reading initial frame {first_frame_index} from {input_path}...")
    frame = tifffile.imread(input_path, key=first_frame_index).astype(np.float32)
    display_minimum, display_maximum = np.percentile(frame, [display_low_percentile, display_high_percentile])
    figure, axis = plt.subplots(figsize=(11, 10))
    axis.imshow(frame, cmap="gray", vmin=display_minimum, vmax=display_maximum, interpolation="nearest")
    axis.set_title(f"Click one tube in frame {first_frame_index}")
    axis.set_axis_off()
    print(f"Click one tube in frame {first_frame_index}...")
    clicked_center = np.asarray(plt.ginput(1, timeout=-1)[0])
    plt.close(figure)

    predicted_center, predicted_angle, convolution_score = find_likely_ridge(
        frame,
        clicked_center,
        convolution_kernels,
        configuration.convolution_search_radius,
    )
    predicted_angle_degrees = np.rad2deg(predicted_angle)
    print(
        f"Initial convolution center=({predicted_center[0]:.2f}, {predicted_center[1]:.2f}), "
        f"angle={predicted_angle_degrees:.2f} degrees, score={convolution_score:.2f}"
    )
    initial_construction_start_time = perf_counter()
    previous_tube = construct_tube(frame, predicted_center, predicted_angle)
    initial_construction_time = perf_counter() - initial_construction_start_time
    print(f"Constructed the initial tube in {initial_construction_time:.2f}s.")
    if not previous_tube:
        raise RuntimeError("No initial tube was constructed from the clicked position.")

    tube_track = create_tube_track(track_id, first_frame_index, previous_tube)
    labeled_frames = [
        draw_tube(frame, previous_tube, display_low_percentile, display_high_percentile, trace_outline_width, trace_line_width)
    ]

    automatic_tracking_start_time = perf_counter()
    for frame_index in frame_indices[1:]:
        print(f"Tracking frame {frame_index}...")
        frame = tifffile.imread(input_path, key=frame_index).astype(np.float32)
        new_tube, trial_count = track_tube(frame, previous_tube, tube_track["average_length"], convolution_kernels, random_generator)
        if new_tube is None:
            break
        add_tube_to_track(tube_track, frame_index, new_tube)
        previous_tube = new_tube
        labeled_frame = draw_tube(
            frame,
            new_tube,
            display_low_percentile,
            display_high_percentile,
            trace_outline_width,
            trace_line_width,
        )
        labeled_frames.append(labeled_frame)
        print(f"Frame {frame_index} retained on trial {trial_count}; running mean length={tube_track['average_length']:.2f} pixels.")

    automatic_tracking_time = perf_counter() - automatic_tracking_start_time
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_start_time = perf_counter()
    tifffile.imwrite(
        output_path,
        np.asarray(labeled_frames),
        imagej=True,
        photometric="rgb",
        metadata={"axes": "TYXS", "fps": playback_frames_per_second},
    )
    output_time = perf_counter() - output_start_time
    print(f"Saved {len(labeled_frames)} tracked frames to {output_path}.")
    print(f"Automatic frame tracking took {automatic_tracking_time:.2f}s; output writing took {output_time:.2f}s.")
