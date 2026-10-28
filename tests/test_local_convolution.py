"""Interactively test local convolution around one click."""

import matplotlib.pyplot as plt
import numpy as np
import tifffile

from tube_tracker import configuration
from tube_tracker.ridge_detection import find_likely_ridge, make_convolution_kernels


if __name__ == "__main__":
    input_path = "test_data/test_1.tif"  # Read-only microscopy TIFF stack used for this interactive test.
    frame_index = 80  # Zero-based frame containing the previously verified crossing-tube example.
    display_low_percentile = 1.0  # Lower image percentile displayed as black.
    display_high_percentile = 99.8  # Upper image percentile displayed as white.

    configuration.set_parameters(
        convolution_search_radius=12,  # Half-width in pixels of the square searched around the manual click.
        convolution_perpendicular_fwhm=6.0,  # Full width at half maximum in pixels across the tube.
        convolution_parallel_fwhm=30.0,  # Full width at half maximum in pixels along the tube.
        convolution_angle_step=np.pi / 90.0,  # Angular spacing in radians between consecutive convolution kernels.
        convolution_parallel_power=4.0,  # Power controlling the flatness of the profile along the tube.
    )

    convolution_kernels = make_convolution_kernels()
    print(f"Reading frame {frame_index} from {input_path}...")
    image = tifffile.imread(input_path, key=frame_index).astype(np.float32)
    display_minimum, display_maximum = np.percentile(image, [display_low_percentile, display_high_percentile])

    plt.imshow(image, cmap="gray", vmin=display_minimum, vmax=display_maximum, interpolation="nearest")
    plt.title(f"Click one tube in frame {frame_index}")
    plt.axis("off")
    print("Click one tube in the image window...")
    clicked_center = np.asarray(plt.ginput(1, timeout=-1)[0])

    predicted_center, predicted_angle, convolution_score = find_likely_ridge(
        image, clicked_center, convolution_kernels, configuration.convolution_search_radius
    )
    clicked_column, clicked_row = clicked_center
    predicted_column, predicted_row = predicted_center
    predicted_angle_degrees = np.rad2deg(predicted_angle)
    print(f"Clicked center=({clicked_column:.2f}, {clicked_row:.2f})")
    print(
        f"Predicted center=({predicted_column:.2f}, {predicted_row:.2f}), "
        f"angle={predicted_angle_degrees:.2f} degrees, score={convolution_score:.2f}"
    )

    ridge_direction = np.array([np.cos(predicted_angle), np.sin(predicted_angle)])
    displayed_start = predicted_center - configuration.convolution_parallel_fwhm * ridge_direction / 2.0
    displayed_end = predicted_center + configuration.convolution_parallel_fwhm * ridge_direction / 2.0
    plt.plot(clicked_column, clicked_row, "yx")
    plt.plot([displayed_start[0], displayed_end[0]], [displayed_start[1], displayed_end[1]], "c-", linewidth=2)
    plt.plot(predicted_column, predicted_row, "co", markersize=4)
    plt.title(f"Local convolution: {predicted_angle_degrees:.0f} degrees")
    plt.show()
