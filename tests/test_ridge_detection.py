"""Find ridge candidates in the angle-response stack."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates
from scipy.signal import fftconvolve
import tifffile


def make_long_range_normal_derivative_kernel(
    angle_degrees,
    kernel_half_size,
    tangent_sigma,
    normal_sigma,
):
    """Build a spatial derivative kernel long along and odd across a tube."""
    coordinates = np.arange(-kernel_half_size, kernel_half_size + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    angle_radians = np.deg2rad(angle_degrees)
    tangent = horizontal * np.cos(angle_radians) + vertical * np.sin(
        angle_radians
    )
    normal = -horizontal * np.sin(angle_radians) + vertical * np.cos(
        angle_radians
    )

    tangent_weight = np.exp(-0.5 * (tangent / tangent_sigma) ** 2)
    normal_derivative = (
        -normal
        * np.exp(-0.5 * (normal / normal_sigma) ** 2)
        / normal_sigma**2
    )
    kernel = tangent_weight * normal_derivative
    kernel -= kernel.mean()
    kernel /= np.sum(np.abs(kernel))
    return kernel.astype(np.float32)


def find_ridge_candidates(
    response_stack,
    angle_step,
    angular_derivative_sigma_degrees,
    angular_block_height,
    spatial_kernel_half_size,
    tangent_sigma,
    normal_sigma,
    derivative_sign_sample_distance,
    boundary_mode,
    progress_angle_interval,
):
    """Find zero crossings of wide angular and spatial derivatives."""
    angle_count, image_height, image_width = response_stack.shape
    expected_angle_count = 180 // angle_step
    assert angle_count == expected_angle_count

    angular_sigma_layers = angular_derivative_sigma_degrees / angle_step
    angular_crossing_strength = np.empty(response_stack.shape, dtype=np.float16)
    print(
        "Calculating the wide angular derivative in "
        f"{angular_block_height}-row blocks..."
    )
    for row_start in range(0, image_height, angular_block_height):
        row_stop = min(row_start + angular_block_height, image_height)
        response_block = np.asarray(
            response_stack[:, row_start:row_stop, :]
        )
        angular_derivative = gaussian_filter1d(
            response_block,
            sigma=angular_sigma_layers,
            axis=0,
            order=1,
            mode="wrap",
        )
        derivative_before = np.roll(angular_derivative, 1, axis=0)
        derivative_after = np.roll(angular_derivative, -1, axis=0)
        block_crossing_strength = np.minimum(
            derivative_before,
            -derivative_after,
        )
        np.maximum(block_crossing_strength, 0.0, out=block_crossing_strength)
        angular_crossing_strength[:, row_start:row_stop, :] = (
            block_crossing_strength
        )

    row_coordinates, column_coordinates = np.indices(
        (image_height, image_width),
        dtype=np.float32,
    )
    ridge_derivative_strength = np.zeros(
        (image_height, image_width),
        dtype=np.float32,
    )
    ridge_angle = np.zeros((image_height, image_width), dtype=np.float32)

    for angle_index in range(angle_count):
        angle_degrees = angle_index * angle_step
        if angle_degrees % progress_angle_interval == 0:
            print(f"Testing ridges at {angle_degrees} degrees...")

        current_response = response_stack[angle_index]
        spatial_derivative_kernel = make_long_range_normal_derivative_kernel(
            angle_degrees,
            spatial_kernel_half_size,
            tangent_sigma,
            normal_sigma,
        )
        padded_response = np.pad(
            current_response,
            pad_width=spatial_kernel_half_size,
            mode=boundary_mode,
        )
        padded_spatial_derivative = fftconvolve(
            padded_response,
            spatial_derivative_kernel,
            mode="same",
        )
        spatial_derivative = padded_spatial_derivative[
            spatial_kernel_half_size:-spatial_kernel_half_size,
            spatial_kernel_half_size:-spatial_kernel_half_size,
        ]

        angle_radians = np.deg2rad(angle_degrees)
        normal_row_offset = derivative_sign_sample_distance * np.cos(
            angle_radians
        )
        normal_column_offset = -derivative_sign_sample_distance * np.sin(
            angle_radians
        )
        derivative_on_negative_side = map_coordinates(
            spatial_derivative,
            (
                row_coordinates - normal_row_offset,
                column_coordinates - normal_column_offset,
            ),
            order=1,
            mode="reflect",
        )
        derivative_on_positive_side = map_coordinates(
            spatial_derivative,
            (
                row_coordinates + normal_row_offset,
                column_coordinates + normal_column_offset,
            ),
            order=1,
            mode="reflect",
        )

        spatial_crossing_strength = np.minimum(
            derivative_on_negative_side,
            -derivative_on_positive_side,
        )
        np.maximum(spatial_crossing_strength, 0.0, out=spatial_crossing_strength)
        combined_derivative_strength = np.sqrt(
            angular_crossing_strength[angle_index]
            * spatial_crossing_strength
        )

        is_stronger_ridge = (
            combined_derivative_strength > ridge_derivative_strength
        )
        ridge_derivative_strength[is_stronger_ridge] = (
            combined_derivative_strength[is_stronger_ridge]
        )
        ridge_angle[is_stronger_ridge] = angle_degrees

    return ridge_derivative_strength, ridge_angle


if __name__ == "__main__":
    # Read-only stack containing one response image per tested angle.
    response_stack_path = Path("test_outputs/angle_responses.tif")
    # Single ridge visualization, overwritten on every run.
    output_path = Path("test_outputs/ridge_candidates.png")
    # Angle difference in degrees between consecutive response layers.
    angle_step = 2
    # Scale in degrees over which the angular derivative is calculated.
    angular_derivative_sigma_degrees = 6.0
    # Number of image rows processed together for the angular derivative.
    angular_block_height = 32
    # Half-size in pixels of the long-range spatial derivative kernel.
    spatial_kernel_half_size = 36
    # Spatial scale in pixels over which evidence is combined along the tube.
    tangent_sigma = 12.0
    # Spatial scale in pixels over which the derivative is calculated across the tube.
    normal_sigma = 2.0
    # Distance in pixels on either side used to detect derivative sign changes.
    derivative_sign_sample_distance = 1.0
    # Response-image edge handling used during spatial convolution.
    boundary_mode = "reflect"
    # Minimum combined angular and spatial derivative-crossing strength.
    derivative_strength_threshold = 6.0
    # Angular interval between console progress messages.
    progress_angle_interval = 30
    # Size of the ridge-candidate figure in inches.
    figure_size = (10, 9)
    # Resolution of the saved ridge-candidate figure.
    figure_dpi = 150

    assert response_stack_path.is_file(), (
        f"Missing angle-response stack: {response_stack_path}"
    )

    print(f"Reading angle responses from {response_stack_path}...")
    response_stack = tifffile.memmap(response_stack_path, mode="r")
    assert response_stack.ndim == 3

    ridge_derivative_strength, ridge_angle = find_ridge_candidates(
        response_stack,
        angle_step,
        angular_derivative_sigma_degrees,
        angular_block_height,
        spatial_kernel_half_size,
        tangent_sigma,
        normal_sigma,
        derivative_sign_sample_distance,
        boundary_mode,
        progress_angle_interval,
    )

    positive_strengths = ridge_derivative_strength[
        ridge_derivative_strength > 0.0
    ]
    assert positive_strengths.size > 0
    print("Derivative-strength percentiles:")
    for percentile in (50.0, 75.0, 90.0, 95.0, 97.0, 99.0, 99.5, 99.9):
        value = np.percentile(positive_strengths, percentile)
        print(f"  {percentile:4.1f}%: {value:.4f}")

    ridge_positions = ridge_derivative_strength > derivative_strength_threshold
    assert np.any(ridge_positions)

    figure, axis = plt.subplots(figsize=figure_size)
    axis.imshow(
        ridge_positions,
        cmap="gray",
        vmin=0.0,
        vmax=1.0,
        interpolation="nearest",
    )
    axis.set_title(
        "Ridge positions with derivative strength > "
        f"{derivative_strength_threshold:g}"
    )
    axis.set_axis_off()
    figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    ridge_pixel_count = positive_strengths.size
    thresholded_ridge_pixel_count = np.count_nonzero(ridge_positions)
    print(
        f"Found {ridge_pixel_count} ridge pixels before thresholding and "
        f"retained {thresholded_ridge_pixel_count} at derivative strength > "
        f"{derivative_strength_threshold:g}."
    )
    print(f"Saved ridge visualization: {output_path}")
