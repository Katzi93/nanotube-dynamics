"""Interactively test tube construction initialized by local convolution."""

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tube_tracker import configuration
from tube_tracker.ridge_detection import find_likely_ridge, make_convolution_kernels
from tube_tracker.tube_construction import construct_tube


if __name__ == "__main__":
    input_path = "test_data/test_1.tif"  # Read-only microscopy TIFF stack used for this test.
    frame_index = 80  # Zero-based frame containing the crossing-tube example.
    display_low_percentile = 1.0  # Lower image percentile displayed as black.
    display_high_percentile = 99.8  # Upper image percentile displayed as white.

    configuration.set_parameters(
        convolution_search_radius=12,  # Half-width in pixels of the square searched around the click.
        convolution_perpendicular_fwhm=6.0,  # Width in pixels across the convolution kernel.
        convolution_parallel_fwhm=30.0,  # Length in pixels along the convolution kernel.
        convolution_angle_step=np.pi / 90.0,  # Angle spacing in radians between convolution kernels.
        convolution_parallel_power=4.0,  # Power controlling the convolution kernel's longitudinal profile.
        fitting_radius=21.0,  # Radius in pixels of the fixed circular fitting region.
        ridge_parallel_fwhm=30.0,  # Fixed longitudinal FWHM in pixels of the fitted ridge.
        ridge_parallel_power=12.0,  # Power controlling the sharpness of the fitted ridge ends.
        initial_perpendicular_fwhm=6.2,  # Initial fitted tube width in pixels.
        extension_step_length=6.0,  # Distance in pixels between consecutive predicted ridge centers.
        long_retry_step_multiplier=2.0,  # Multiplier used to jump past a rejected normal prediction.
        short_retry_step_multiplier=0.5,  # Multiplier used for the final shorter prediction attempt.
        maximum_points_per_direction=100,  # Maximum number of ridges constructed from each side of the initial ridge.
        minimum_accepted_fwhm=2.0,  # Smallest fitted tube width accepted as a ridge.
        maximum_accepted_fwhm=12.0,  # Largest fitted tube width accepted as a ridge.
        minimum_amplitude_to_error=1.2,  # Smallest accepted ratio between ridge amplitude and fitting RMSE.
        maximum_accepted_angle_change=np.pi / 9.0,  # Largest accepted angle change from the predicted ridge.
        maximum_accepted_perpendicular_shift=8.0,  # Largest accepted sideways displacement in pixels.
        fitting_verbose=True,  # Print detailed ridge fitting, retry, and rejection messages.
    )

    kernels = make_convolution_kernels()
    print(f"Reading frame {frame_index} from {input_path}...")
    image = tifffile.imread(input_path, key=frame_index).astype(np.float32)
    display_minimum, display_maximum = np.percentile(image, [display_low_percentile, display_high_percentile])

    plt.imshow(image, cmap="gray", vmin=display_minimum, vmax=display_maximum, interpolation="nearest")
    plt.title(f"Click one tube in frame {frame_index}")
    plt.axis("off")
    print("Click one tube in the image window...")
    clicked_center = np.asarray(plt.ginput(1, timeout=-1)[0])

    predicted_center, predicted_angle, convolution_score = find_likely_ridge(
        image,
        clicked_center,
        kernels,
        configuration.convolution_search_radius,
    )
    predicted_angle_degrees = np.rad2deg(predicted_angle)
    print(
        f"Convolution center=({predicted_center[0]:.2f}, {predicted_center[1]:.2f}), "
        f"angle={predicted_angle_degrees:.2f} degrees"
    )
    tube_ridges = construct_tube(image, predicted_center, predicted_angle)

    if tube_ridges:
        tube_centers = np.asarray([ridge["center"] for ridge in tube_ridges])
        plt.plot(tube_centers[:, 0], tube_centers[:, 1], "c-", linewidth=2)
        plt.plot(tube_centers[:, 0], tube_centers[:, 1], "co", markersize=2)
        plt.title(f"Constructed tube: {len(tube_ridges)} ridges")
    else:
        plt.title("No tube constructed")
    plt.show()
