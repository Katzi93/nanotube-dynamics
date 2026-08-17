"""Trace one clicked tube using consecutive local raw-image ridge fits."""

from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.optimize import least_squares
from scipy.signal import fftconvolve
import tifffile


class RidgeFit(NamedTuple):
    center: np.ndarray
    angle_degrees: float
    sigma: float
    amplitude: float
    background: float
    root_mean_square_error: float
    flat_error_reduction: float


def make_oriented_line_template(
    tube_width,
    angle_degrees,
    template_half_size,
    along_half_height_length,
    super_gaussian_power,
    side_width_multiplier,
    gaussian_fwhm_factor,
):
    """Build one normalized bright-center, dark-sides line template."""
    coordinates = np.arange(-template_half_size, template_half_size + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    angle_radians = np.deg2rad(angle_degrees)
    along = horizontal * np.cos(angle_radians) + vertical * np.sin(
        angle_radians
    )
    across = -horizontal * np.sin(angle_radians) + vertical * np.cos(
        angle_radians
    )

    center_sigma = tube_width / gaussian_fwhm_factor
    side_sigma = center_sigma * side_width_multiplier
    along_scale = along_half_height_length / (
        2.0 * (2.0 * np.log(2.0)) ** (1.0 / super_gaussian_power)
    )
    along_weight = np.exp(
        -0.5 * (np.abs(along) / along_scale) ** super_gaussian_power
    )
    bright_center = np.exp(-0.5 * (across / center_sigma) ** 2) / center_sigma
    broad_sides = np.exp(-0.5 * (across / side_sigma) ** 2) / side_sigma
    template = along_weight * (bright_center - broad_sides)
    template -= template.mean()
    template /= np.sqrt(np.sum(template**2))
    return template.astype(np.float32)


def find_local_initial_guess(
    image,
    clicked_center,
    search_radius,
    median_filter_size,
    tube_width,
    angle_step_degrees,
    template_half_size,
    along_half_height_length,
    super_gaussian_power,
    side_width_multiplier,
    gaussian_fwhm_factor,
):
    """Find the strongest nearby line position and angle by local convolution."""
    clicked_column = int(round(clicked_center[0]))
    clicked_row = int(round(clicked_center[1]))
    crop_margin = template_half_size + search_radius
    row_minimum = clicked_row - crop_margin
    row_maximum = clicked_row + crop_margin + 1
    column_minimum = clicked_column - crop_margin
    column_maximum = clicked_column + crop_margin + 1
    assert row_minimum >= 0
    assert column_minimum >= 0
    assert row_maximum <= image.shape[0]
    assert column_maximum <= image.shape[1]

    local_image = image[row_minimum:row_maximum, column_minimum:column_maximum]
    filtered_local_image = median_filter(local_image, size=median_filter_size)
    local_rows, local_columns = np.mgrid[
        template_half_size : template_half_size + 2 * search_radius + 1,
        template_half_size : template_half_size + 2 * search_radius + 1,
    ]
    global_rows = local_rows + row_minimum
    global_columns = local_columns + column_minimum
    inside_search_circle = (
        (global_columns - clicked_center[0]) ** 2
        + (global_rows - clicked_center[1]) ** 2
        <= search_radius**2
    )

    best_score = -np.inf
    best_center = None
    best_angle_degrees = None
    for angle_degrees in range(0, 180, angle_step_degrees):
        template = make_oriented_line_template(
            tube_width,
            angle_degrees,
            template_half_size,
            along_half_height_length,
            super_gaussian_power,
            side_width_multiplier,
            gaussian_fwhm_factor,
        )
        response = fftconvolve(filtered_local_image, template, mode="same")
        search_response = response[local_rows, local_columns]
        masked_response = np.where(inside_search_circle, search_response, -np.inf)
        maximum_index = np.unravel_index(
            np.argmax(masked_response),
            masked_response.shape,
        )
        maximum_score = masked_response[maximum_index]
        if maximum_score > best_score:
            best_score = maximum_score
            best_center = np.array(
                [
                    global_columns[maximum_index],
                    global_rows[maximum_index],
                ],
                dtype=np.float64,
            )
            best_angle_degrees = float(angle_degrees)

    return best_center, best_angle_degrees, best_score


def fit_local_ridge(
    image,
    starting_center,
    starting_angle_degrees,
    fitting_radius,
    center_search_range,
    angle_search_range_degrees,
    fitting_minimum_sigma,
    fitting_maximum_sigma,
    initial_sigma,
    fitting_iterations,
    maximum_function_evaluations,
):
    """Jointly fit ridge center, angle, width, brightness, and background."""
    center = np.array(starting_center, dtype=np.float64, copy=True)
    angle_degrees = float(starting_angle_degrees)
    sigma = float(initial_sigma)

    for _ in range(fitting_iterations):
        center_column, center_row = center
        row_minimum = int(np.floor(center_row - fitting_radius))
        row_maximum = int(np.ceil(center_row + fitting_radius)) + 1
        column_minimum = int(np.floor(center_column - fitting_radius))
        column_maximum = int(np.ceil(center_column + fitting_radius)) + 1
        assert row_minimum >= 0
        assert column_minimum >= 0
        assert row_maximum <= image.shape[0]
        assert column_maximum <= image.shape[1]

        rows, columns = np.mgrid[
            row_minimum:row_maximum,
            column_minimum:column_maximum,
        ]
        inside_fitting_circle = (
            (columns - center_column) ** 2 + (rows - center_row) ** 2
            <= fitting_radius**2
        )
        sample_columns = columns[inside_fitting_circle].astype(np.float64)
        sample_rows = rows[inside_fitting_circle].astype(np.float64)
        image_patch = image[
            row_minimum:row_maximum,
            column_minimum:column_maximum,
        ]
        measured_intensity = image_patch[inside_fitting_circle].astype(
            np.float64
        )

        initial_background = float(np.median(measured_intensity))
        initial_amplitude = float(measured_intensity.max() - initial_background)
        intensity_range = float(measured_intensity.max() - measured_intensity.min())
        assert intensity_range > 0.0

        def residuals(parameters):
            background, amplitude, normal_offset, angle_offset, fitted_sigma = (
                parameters
            )
            fitted_angle_radians = np.deg2rad(angle_degrees + angle_offset)
            normal_column = -np.sin(fitted_angle_radians)
            normal_row = np.cos(fitted_angle_radians)
            normal_coordinate = (
                (sample_columns - center_column) * normal_column
                + (sample_rows - center_row) * normal_row
            )
            predicted_intensity = background + amplitude * np.exp(
                -0.5 * ((normal_coordinate - normal_offset) / fitted_sigma) ** 2
            )
            return predicted_intensity - measured_intensity

        fit = least_squares(
            residuals,
            x0=(initial_background, initial_amplitude, 0.0, 0.0, sigma),
            bounds=(
                (
                    measured_intensity.min(),
                    0.0,
                    -center_search_range,
                    -angle_search_range_degrees,
                    fitting_minimum_sigma,
                ),
                (
                    measured_intensity.max(),
                    3.0 * intensity_range,
                    center_search_range,
                    angle_search_range_degrees,
                    fitting_maximum_sigma,
                ),
            ),
            max_nfev=maximum_function_evaluations,
        )
        assert fit.success, fit.message

        background, amplitude, normal_offset, angle_offset, sigma = fit.x
        ridge_sum_squared_error = np.sum(fit.fun**2)
        root_mean_square_error = np.sqrt(np.mean(fit.fun**2))
        flat_sum_squared_error = np.sum(
            (measured_intensity - measured_intensity.mean()) ** 2
        )
        flat_error_reduction = 1.0 - (
            ridge_sum_squared_error / flat_sum_squared_error
        )
        angle_degrees = (angle_degrees + angle_offset) % 180.0
        angle_radians = np.deg2rad(angle_degrees)
        normal = np.array([-np.sin(angle_radians), np.cos(angle_radians)])
        center += normal_offset * normal

    return RidgeFit(
        center,
        angle_degrees,
        sigma,
        amplitude,
        background,
        root_mean_square_error,
        flat_error_reduction,
    )


def undirected_angle_difference(first_angle, second_angle):
    """Return the smallest difference between two line angles in degrees."""
    return abs((first_angle - second_angle + 90.0) % 180.0 - 90.0)


def fit_rejection_reasons(
    fitted_result,
    previous_result,
    predicted_center,
    forward_direction,
    step_length,
    minimum_forward_progress,
    maximum_prediction_correction,
    maximum_angle_change_degrees,
    minimum_amplitude_to_error,
    minimum_flat_error_reduction,
    accepted_minimum_sigma,
    accepted_maximum_sigma,
):
    """List the explicit reasons an extension fit should terminate."""
    forward_progress = (
        fitted_result.center - previous_result.center
    ) @ forward_direction
    prediction_correction = np.linalg.norm(
        fitted_result.center - predicted_center
    )
    angle_change = undirected_angle_difference(
        fitted_result.angle_degrees,
        previous_result.angle_degrees,
    )
    amplitude_to_error = (
        fitted_result.amplitude / fitted_result.root_mean_square_error
    )

    reasons = []
    if forward_progress < minimum_forward_progress:
        reasons.append(
            f"forward progress {forward_progress:.2f} < "
            f"{minimum_forward_progress:.2f} pixels"
        )
    if forward_progress > step_length + maximum_prediction_correction:
        reasons.append(f"forward progress {forward_progress:.2f} is too large")
    if prediction_correction > maximum_prediction_correction:
        reasons.append(
            f"prediction correction {prediction_correction:.2f} > "
            f"{maximum_prediction_correction:.2f} pixels"
        )
    if angle_change > maximum_angle_change_degrees:
        reasons.append(
            f"angle change {angle_change:.2f} > "
            f"{maximum_angle_change_degrees:.2f} degrees"
        )
    if amplitude_to_error < minimum_amplitude_to_error:
        reasons.append(
            f"amplitude/RMSE {amplitude_to_error:.2f} < "
            f"{minimum_amplitude_to_error:.2f}"
        )
    if fitted_result.flat_error_reduction < minimum_flat_error_reduction:
        reasons.append(
            f"flat-error reduction {fitted_result.flat_error_reduction:.3f} < "
            f"{minimum_flat_error_reduction:.3f}"
        )
    if not accepted_minimum_sigma <= fitted_result.sigma <= accepted_maximum_sigma:
        fitted_fwhm = fitted_result.sigma * 2.35482
        reasons.append(f"FWHM {fitted_fwhm:.2f} pixels is outside measured range")
    return reasons


def extend_tube(
    image,
    initial_result,
    direction_sign,
    step_length,
    fitting_radius,
    center_search_range,
    angle_search_range_degrees,
    fitting_minimum_sigma,
    fitting_maximum_sigma,
    fitting_iterations,
    maximum_function_evaluations,
    maximum_extension_points,
    minimum_forward_progress,
    maximum_prediction_correction,
    maximum_angle_change_degrees,
    minimum_amplitude_to_error,
    minimum_flat_error_reduction,
    accepted_minimum_sigma,
    accepted_maximum_sigma,
    print_accepted_points=True,
):
    """Extend consecutive accepted ridge fits from one end of a seed fit."""
    accepted_results = []
    previous_result = initial_result
    required_edge_margin = fitting_radius + fitting_iterations * center_search_range
    initial_angle_radians = np.deg2rad(initial_result.angle_degrees)
    forward_direction = direction_sign * np.array(
        [np.cos(initial_angle_radians), np.sin(initial_angle_radians)]
    )

    for extension_index in range(1, maximum_extension_points + 1):
        predicted_center = previous_result.center + step_length * forward_direction
        predicted_column, predicted_row = predicted_center
        if (
            predicted_column < required_edge_margin
            or predicted_row < required_edge_margin
            or predicted_column >= image.shape[1] - required_edge_margin
            or predicted_row >= image.shape[0] - required_edge_margin
        ):
            print(f"Direction {direction_sign:+d} stopped at the image boundary.")
            break

        fitted_result = fit_local_ridge(
            image,
            predicted_center,
            previous_result.angle_degrees,
            fitting_radius,
            center_search_range,
            angle_search_range_degrees,
            fitting_minimum_sigma,
            fitting_maximum_sigma,
            previous_result.sigma,
            fitting_iterations,
            maximum_function_evaluations,
        )
        rejection_reasons = fit_rejection_reasons(
            fitted_result,
            previous_result,
            predicted_center,
            forward_direction,
            step_length,
            minimum_forward_progress,
            maximum_prediction_correction,
            maximum_angle_change_degrees,
            minimum_amplitude_to_error,
            minimum_flat_error_reduction,
            accepted_minimum_sigma,
            accepted_maximum_sigma,
        )
        if rejection_reasons:
            print(
                f"Direction {direction_sign:+d} stopped after "
                f"{len(accepted_results)} extensions: "
                + "; ".join(rejection_reasons)
            )
            break

        accepted_results.append(fitted_result)
        previous_result = fitted_result
        fitted_angle_radians = np.deg2rad(fitted_result.angle_degrees)
        fitted_direction = np.array(
            [np.cos(fitted_angle_radians), np.sin(fitted_angle_radians)]
        )
        if fitted_direction @ forward_direction < 0.0:
            fitted_direction *= -1.0
        forward_direction = fitted_direction
        fitted_fwhm = fitted_result.sigma * 2.35482
        if print_accepted_points:
            print(
                f"Direction {direction_sign:+d}, point {extension_index}: "
                f"center=({fitted_result.center[0]:.2f}, "
                f"{fitted_result.center[1]:.2f}), angle="
                f"{fitted_result.angle_degrees:.2f}, FWHM={fitted_fwhm:.2f}, "
                f"amplitude={fitted_result.amplitude:.2f}, flat-error reduction="
                f"{fitted_result.flat_error_reduction:.3f}"
            )

    return accepted_results


if __name__ == "__main__":
    # Read-only microscopy TIFF stack containing the video frames.
    input_path = Path("test_data/test_1.tif")
    # Single traced-tube visualization, overwritten on every run.
    output_path = Path("test_outputs/raw_image_ridge_fit.png")
    # Zero-based video frame displayed and traced.
    frame_index = 80
    # Radius in pixels searched by the local convolution around the click.
    initial_convolution_search_radius = 8
    # Width of the median-filter neighborhood used only by local convolution.
    initial_convolution_median_filter_size = 3
    # Cross-tube width in pixels of the local convolution template.
    initial_convolution_tube_width = 6.0
    # Angular spacing in degrees of the local convolution templates.
    initial_convolution_angle_step_degrees = 2
    # Half-size in pixels of the square local convolution template.
    initial_convolution_template_half_size = 36
    # Full along-tube half-height length of the local convolution template.
    initial_convolution_along_half_height_length = 30.0
    # Power controlling the flat center of the along-tube template.
    initial_convolution_super_gaussian_power = 4.0
    # Width of the template's negative sides relative to its bright center.
    initial_convolution_side_width_multiplier = 2.0
    # Conversion factor from Gaussian sigma to full width at half height.
    gaussian_fwhm_factor = 2.35482
    # Radius in pixels of each circular raw-image fitting region.
    fitting_radius = 15.0
    # Distance in pixels between consecutive predicted tube positions.
    extension_step_length = 6.0
    # Maximum perpendicular movement during one initial fitting iteration.
    initial_center_search_range = 8.0
    # Maximum angle movement during one initial fitting iteration.
    initial_angle_search_range_degrees = 30.0
    # Number of recentered fits used for the clicked initial position.
    initial_fitting_iterations = 3
    # Maximum perpendicular movement during one extension fitting iteration.
    extension_center_search_range = 3.0
    # Maximum angle movement during one extension fitting iteration.
    extension_angle_search_range_degrees = 15.0
    # Number of recentered fits used for each extension position.
    extension_fitting_iterations = 2
    # Smallest sigma allowed during fitting so bad fits can become too narrow.
    fitting_minimum_sigma = 0.5
    # Largest sigma allowed during fitting so bad fits can become too broad.
    fitting_maximum_sigma = 8.0
    # Smallest accepted tube FWHM, allowing modest variation beyond measurements.
    accepted_minimum_sigma = 2.0 / 2.35482
    # Largest accepted tube FWHM, allowing modest variation beyond measurements.
    accepted_maximum_sigma = 12.0 / 2.35482
    # Median measured tube FWHM converted to the initial Gaussian sigma.
    initial_sigma = 6.20 / 2.35482
    # Maximum residual evaluations allowed for each nonlinear fit.
    maximum_function_evaluations = 500
    # Maximum accepted points grown from each side of the clicked fit.
    maximum_extension_points = 100
    # Minimum required movement along the requested extension direction.
    minimum_forward_progress = 3.5
    # Maximum distance a fitted center may move from its predicted position.
    maximum_prediction_correction = 6.0
    # Maximum angle change allowed between consecutive accepted fits.
    maximum_angle_change_degrees = 20.0
    # Minimum fitted amplitude divided by the remaining fitting error.
    minimum_amplitude_to_error = 1.0
    # Minimum error reduction compared with fitting a flat image patch.
    minimum_flat_error_reduction = 0.05
    # Lower raw-intensity percentile displayed as black.
    display_low_percentile = 1.0
    # Upper raw-intensity percentile displayed as white.
    display_high_percentile = 99.8
    # Width of the final connected tube line in the saved image.
    traced_line_width = 0.75
    # Size of the click and result figures in inches.
    figure_size = (11, 10)
    # Resolution of the saved result figure.
    figure_dpi = 150

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"

    print(f"Reading raw frame {frame_index} from {input_path}...")
    frame = tifffile.imread(input_path, key=frame_index).astype(np.float32)
    assert frame.ndim == 2
    display_minimum, display_maximum = np.percentile(
        frame,
        [display_low_percentile, display_high_percentile],
    )

    selection_figure, selection_axis = plt.subplots(figsize=figure_size)
    selection_axis.imshow(
        frame,
        cmap="gray",
        vmin=display_minimum,
        vmax=display_maximum,
        interpolation="nearest",
    )
    selection_axis.set_title("Click once on the tube to trace")
    selection_axis.set_axis_off()
    print("Click once on the desired tube in the image window...")
    clicked_points = plt.ginput(1, timeout=-1)
    plt.close(selection_figure)
    assert len(clicked_points) == 1
    clicked_center = np.asarray(clicked_points[0], dtype=np.float64)

    print(
        f"Clicked center=({clicked_center[0]:.2f}, {clicked_center[1]:.2f})."
    )
    print(
        "Running local convolution around the click at "
        f"{initial_convolution_angle_step_degrees}-degree angle spacing..."
    )
    initial_center, initial_angle_degrees, initial_convolution_score = (
        find_local_initial_guess(
            frame,
            clicked_center,
            initial_convolution_search_radius,
            initial_convolution_median_filter_size,
            initial_convolution_tube_width,
            initial_convolution_angle_step_degrees,
            initial_convolution_template_half_size,
            initial_convolution_along_half_height_length,
            initial_convolution_super_gaussian_power,
            initial_convolution_side_width_multiplier,
            gaussian_fwhm_factor,
        )
    )
    print(
        f"Local convolution guess: center=({initial_center[0]:.2f}, "
        f"{initial_center[1]:.2f}), angle={initial_angle_degrees:.2f}, "
        f"score={initial_convolution_score:.2f}."
    )
    initial_result = fit_local_ridge(
        frame,
        initial_center,
        initial_angle_degrees,
        fitting_radius,
        initial_center_search_range,
        initial_angle_search_range_degrees,
        fitting_minimum_sigma,
        fitting_maximum_sigma,
        initial_sigma,
        initial_fitting_iterations,
        maximum_function_evaluations,
    )
    initial_amplitude_to_error = (
        initial_result.amplitude / initial_result.root_mean_square_error
    )
    assert initial_amplitude_to_error >= minimum_amplitude_to_error, (
        f"Initial fit amplitude/RMSE {initial_amplitude_to_error:.2f} is below "
        f"{minimum_amplitude_to_error:.2f}"
    )
    assert initial_result.flat_error_reduction >= minimum_flat_error_reduction, (
        f"Initial fit flat-error reduction "
        f"{initial_result.flat_error_reduction:.3f} is below "
        f"{minimum_flat_error_reduction:.3f}"
    )
    initial_fwhm = initial_result.sigma * 2.35482
    assert accepted_minimum_sigma <= initial_result.sigma <= accepted_maximum_sigma, (
        f"Initial fit FWHM {initial_fwhm:.2f} pixels is outside the accepted "
        f"range {accepted_minimum_sigma * 2.35482:.2f}-"
        f"{accepted_maximum_sigma * 2.35482:.2f} pixels"
    )
    print(
        f"Initial fit: center=({initial_result.center[0]:.2f}, "
        f"{initial_result.center[1]:.2f}), angle="
        f"{initial_result.angle_degrees:.2f}, FWHM="
        f"{initial_result.sigma * 2.35482:.2f}, amplitude="
        f"{initial_result.amplitude:.2f}, flat-error reduction="
        f"{initial_result.flat_error_reduction:.3f}"
    )

    backward_results = extend_tube(
        frame,
        initial_result,
        -1,
        extension_step_length,
        fitting_radius,
        extension_center_search_range,
        extension_angle_search_range_degrees,
        fitting_minimum_sigma,
        fitting_maximum_sigma,
        extension_fitting_iterations,
        maximum_function_evaluations,
        maximum_extension_points,
        minimum_forward_progress,
        maximum_prediction_correction,
        maximum_angle_change_degrees,
        minimum_amplitude_to_error,
        minimum_flat_error_reduction,
        accepted_minimum_sigma,
        accepted_maximum_sigma,
    )
    forward_results = extend_tube(
        frame,
        initial_result,
        1,
        extension_step_length,
        fitting_radius,
        extension_center_search_range,
        extension_angle_search_range_degrees,
        fitting_minimum_sigma,
        fitting_maximum_sigma,
        extension_fitting_iterations,
        maximum_function_evaluations,
        maximum_extension_points,
        minimum_forward_progress,
        maximum_prediction_correction,
        maximum_angle_change_degrees,
        minimum_amplitude_to_error,
        minimum_flat_error_reduction,
        accepted_minimum_sigma,
        accepted_maximum_sigma,
    )
    ordered_results = (
        list(reversed(backward_results))
        + [initial_result]
        + forward_results
    )
    traced_centers = np.asarray([result.center for result in ordered_results])

    result_figure, result_axis = plt.subplots(figsize=figure_size)
    result_axis.imshow(
        frame,
        cmap="gray",
        vmin=display_minimum,
        vmax=display_maximum,
        interpolation="nearest",
    )
    result_axis.plot(
        traced_centers[:, 0],
        traced_centers[:, 1],
        color="cyan",
        linewidth=traced_line_width,
        marker="o",
        markersize=0.75,
        label="Accepted fitted centers",
    )
    result_axis.scatter(
        clicked_center[0],
        clicked_center[1],
        color="red",
        marker="x",
        s=100,
        linewidths=2,
        label="Click",
    )
    result_axis.set_title(f"Connected tube trace: {len(ordered_results)} fits")
    result_axis.legend(loc="upper right")
    result_axis.set_axis_off()
    result_figure.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    result_figure.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(result_figure)
    print(f"Saved connected tube trace: {output_path}")
