"""Plot the best-matching tube angle at every pixel."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import median_filter
from scipy.signal import fftconvolve
import tifffile


def make_super_gaussian_line_template(
    tube_width,
    angle_degrees,
    template_half_size,
    along_half_height_length,
    super_gaussian_power,
    side_width_multiplier,
    gaussian_fwhm_factor,
):
    """Build a rotated super-Gaussian bright-center, dark-sides template."""
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
    along_scale = along_half_height_length / (
        2.0 * (2.0 * np.log(2.0)) ** (1.0 / super_gaussian_power)
    )
    along_weight = np.exp(
        -0.5 * (np.abs(along) / along_scale) ** super_gaussian_power
    )
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
    # Read-only microscopy TIFF stack used for this orientation-field test.
    input_path = Path("test_data/test_1.tif")
    # Single output image, overwritten on every run.
    output_path = Path("test_outputs/max_angle.png")
    # Full response stack with one image for every tested angle.
    angle_stack_path = Path("test_outputs/angle_responses.tif")
    # Zero-based TIFF frame tested by this script.
    frame_index = 80
    # Width and height of the square median-filter neighborhood in pixels.
    median_filter_size = 3
    # Number of median-filter passes selected from the manual measurements.
    median_filter_passes = 1
    # Fixed cross-tube width selected from the measured typical width.
    tube_width = 6.0
    # Full along-tube length between half-height points.
    along_half_height_length = 30.0
    # Power producing a flat-centered along-tube function with smooth ends.
    super_gaussian_power = 4.0
    # Angular increment used to generate the 90 tested orientations.
    angle_step = 2
    # Half-size of the square template, providing room for smooth tails.
    template_half_size = 36
    # Width of the negative side regions relative to the bright central stripe.
    side_width_multiplier = 2.0
    # Conversion factor from a Gaussian's standard deviation to its full width at half height.
    gaussian_fwhm_factor = 2.35482
    # Image-edge handling used to prevent artificial responses at the frame boundary.
    boundary_mode = "reflect"
    # Angular interval between console progress messages.
    progress_angle_interval = 30
    # Cyclic color map because zero and 180 degrees are the same line direction.
    angle_color_map = "twilight"
    # Size of the maximum-angle figure in inches.
    figure_size = (10, 9)
    # Resolution of the saved maximum-angle figure.
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
    best_score = np.zeros_like(filtered, dtype=np.float32)
    best_angle = np.zeros_like(filtered, dtype=np.float32)
    template_angles = tuple(range(0, 180, angle_step))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    angle_response_stack = tifffile.memmap(
        angle_stack_path,
        shape=(len(template_angles), *filtered.shape),
        dtype=np.float32,
        photometric="minisblack",
        bigtiff=True,
        metadata={
            "axes": "QYX",
            "angle_degrees": template_angles,
        },
    )

    print(f"Testing {len(template_angles)} orientations at {angle_step}-degree steps...")
    for angle_index, angle_degrees in enumerate(template_angles):
        if angle_degrees % progress_angle_interval == 0:
            print(f"Processing angle {angle_degrees} degrees...")
        template = make_super_gaussian_line_template(
            tube_width,
            angle_degrees,
            template_half_size,
            along_half_height_length,
            super_gaussian_power,
            side_width_multiplier,
            gaussian_fwhm_factor,
        )
        padded_response = fftconvolve(padded_filtered, template, mode="same")
        response = padded_response[
            template_half_size:-template_half_size,
            template_half_size:-template_half_size,
        ]
        response = np.maximum(response, 0.0)
        angle_response_stack[angle_index] = response

        stronger_match = response > best_score
        best_score[stronger_match] = response[stronger_match]
        best_angle[stronger_match] = angle_degrees

    angle_response_stack.flush()
    del angle_response_stack

    maximum_score = float(best_score.max())
    assert maximum_score > 0.0
    figure, axis = plt.subplots(figsize=figure_size)
    angle_artist = axis.imshow(
        best_angle,
        cmap=angle_color_map,
        vmin=0.0,
        vmax=180.0,
        interpolation="nearest",
    )
    axis.set_title("Best-matching angle at every pixel")
    axis.set_axis_off()
    color_bar = figure.colorbar(angle_artist, ax=axis, fraction=0.046, pad=0.04)
    color_bar.set_label("Angle (degrees)")
    color_bar.set_ticks(np.arange(0, 181, 30))
    figure.tight_layout()

    figure.savefig(output_path, dpi=figure_dpi, bbox_inches="tight")
    plt.close(figure)

    print(f"Saved maximum-angle image: {output_path}")
    print(f"Saved all angle responses: {angle_stack_path}")
