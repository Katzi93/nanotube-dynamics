"""Test whether finer angle steps improve a long oriented tube template."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve
import tifffile


def scale_for_display(image, low_percentile, high_percentile):
    """Scale one image to the range zero to one for display."""
    low_value, high_value = np.percentile(
        image, [low_percentile, high_percentile]
    )
    assert high_value > low_value
    return np.clip((image - low_value) / (high_value - low_value), 0.0, 1.0)


def make_gaussian_line_template(
    tube_width,
    angle_degrees,
    template_half_size,
    along_half_height_length,
    side_width_multiplier,
    gaussian_fwhm_factor,
):
    """Build a rotated Gaussian bright-center, dark-sides line template."""
    coordinates = np.arange(-template_half_size, template_half_size + 1)
    horizontal, vertical = np.meshgrid(coordinates, coordinates)
    angle_radians = np.deg2rad(angle_degrees)
    along = horizontal * np.cos(angle_radians) + vertical * np.sin(
        angle_radians
    )
    across = -horizontal * np.sin(angle_radians) + vertical * np.cos(
        angle_radians
    )

    across_center_sigma = tube_width / gaussian_fwhm_factor
    across_side_sigma = across_center_sigma * side_width_multiplier
    along_sigma = along_half_height_length / gaussian_fwhm_factor
    along_weight = np.exp(-0.5 * (along / along_sigma) ** 2)
    bright_center = (
        np.exp(-0.5 * (across / across_center_sigma) ** 2)
        / across_center_sigma
    )
    broad_sides = (
        np.exp(-0.5 * (across / across_side_sigma) ** 2)
        / across_side_sigma
    )

    template = along_weight * (bright_center - broad_sides)
    template -= template.mean()
    template /= np.sqrt(np.sum(template**2))
    return template.astype(np.float32)


if __name__ == "__main__":
    # Read-only microscopy TIFF stack used for this comparison.
    input_path = Path("test_data/test_1.tif")
    # Directory for separate results from each angular step size.
    output_directory = Path("test_outputs/angle_resolution_comparison")
    # Zero-based TIFF frame tested by this script.
    frame_index = 0
    # Width and height of the square median-filter neighborhood in pixels.
    median_filter_size = 3
    # Number of median-filter passes selected from the manual measurements.
    median_filter_passes = 1
    # Fixed cross-tube width selected from the measured typical width.
    tube_width = 6.0
    # Fixed long template length used to expose errors from coarse angle sampling.
    along_half_height_length = 47.0
    # Angular increments compared while all other parameters remain fixed.
    angle_steps = (10, 5, 2)
    # Half-size supporting approximately three along-tube Gaussian standard deviations.
    template_half_size = 60
    # Width of the negative side regions relative to the bright central stripe.
    side_width_multiplier = 2.0
    # Conversion factor from a Gaussian's standard deviation to its full width at half height.
    gaussian_fwhm_factor = 2.35482
    # Image-edge handling used to prevent artificial responses at the frame boundary.
    boundary_mode = "reflect"
    # Lower original-image percentile displayed as black.
    display_low_percentile = 1.0
    # Upper original-image percentile displayed as white.
    display_high_percentile = 99.8
    # Power applied only for displaying weak template scores.
    score_display_power = 0.7
    # Color map shared by all three continuous score images.
    score_color_map = "magma"
    # Size of the four-panel diagnostic figure in inches.
    figure_size = (18, 12)
    # Resolution of the saved diagnostic figure.
    figure_dpi = 150

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"

    print(f"Reading frame {frame_index} from {input_path}...")
    frame = tifffile.imread(input_path, key=frame_index).astype(np.float32)
    assert frame.ndim == 2, f"Expected a grayscale frame, got {frame.shape}"

    print(
        f"Applying {median_filter_passes} median-filter pass with a "
        f"{median_filter_size}x{median_filter_size} neighborhood..."
    )
    filtered = frame
    for _ in range(median_filter_passes):
        filtered = median_filter(filtered, size=median_filter_size)

    padded_filtered = np.pad(
        filtered,
        pad_width=template_half_size,
        mode=boundary_mode,
    )
    scores_by_step = {}
    angles_by_step = {}

    for angle_step in angle_steps:
        template_angles = tuple(range(0, 180, angle_step))
        print(
            f"Testing {angle_step}-degree steps "
            f"({len(template_angles)} templates)..."
        )
        best_score = np.zeros_like(filtered, dtype=np.float32)
        best_angle = np.zeros_like(filtered, dtype=np.float32)

        for angle_degrees in template_angles:
            template = make_gaussian_line_template(
                tube_width,
                angle_degrees,
                template_half_size,
                along_half_height_length,
                side_width_multiplier,
                gaussian_fwhm_factor,
            )
            padded_response = fftconvolve(padded_filtered, template, mode="same")
            response = padded_response[
                template_half_size:-template_half_size,
                template_half_size:-template_half_size,
            ]
            response = np.maximum(response, 0.0)
            stronger_match = response > best_score
            best_score[stronger_match] = response[stronger_match]
            best_angle[stronger_match] = angle_degrees

        scores_by_step[angle_step] = best_score
        angles_by_step[angle_step] = best_angle

    shared_maximum_score = max(score.max() for score in scores_by_step.values())
    assert shared_maximum_score > 0.0
    for angle_step in angle_steps:
        scores_by_step[angle_step] /= shared_maximum_score

    original_display = scale_for_display(
        frame, display_low_percentile, display_high_percentile
    )
    figure, axes = plt.subplots(2, 2, figsize=figure_size)
    axes[0, 0].imshow(original_display, cmap="gray")
    axes[0, 0].set_title("Original frame")

    score_artist = None
    for axis, angle_step in zip(axes.flat[1:], angle_steps):
        score_display = np.power(scores_by_step[angle_step], score_display_power)
        score_artist = axis.imshow(
            score_display,
            cmap=score_color_map,
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_title(
            f"Angle step {angle_step} degrees, length "
            f"{along_half_height_length:.0f} px"
        )
    assert score_artist is not None
    figure.colorbar(score_artist, ax=list(axes.flat), fraction=0.025, pad=0.02)
    for axis in axes.flat:
        axis.set_axis_off()

    output_directory.mkdir(parents=True, exist_ok=True)
    output_stem = f"frame_{frame_index:03d}"
    diagnostic_path = output_directory / f"{output_stem}_diagnostic.png"
    figure.savefig(diagnostic_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    for angle_step in angle_steps:
        score_path = output_directory / (
            f"{output_stem}_angle_step_{angle_step}_score.tif"
        )
        angle_path = output_directory / (
            f"{output_stem}_angle_step_{angle_step}_angle_degrees.tif"
        )
        tifffile.imwrite(
            score_path,
            scores_by_step[angle_step],
            photometric="minisblack",
        )
        tifffile.imwrite(
            angle_path,
            angles_by_step[angle_step],
            photometric="minisblack",
        )
        print(f"Saved {angle_step}-degree score: {score_path}")
        print(f"Saved {angle_step}-degree angle: {angle_path}")

    print(f"Saved comparison image: {diagnostic_path}")
