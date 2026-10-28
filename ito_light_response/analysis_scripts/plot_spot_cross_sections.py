"""Plot horizontal and vertical illumination profiles through representative ITO spots."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import tifffile


plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 19,
    "axes.labelsize": 18,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
    "axes.linewidth": 1.6,
    "xtick.major.width": 1.6,
    "ytick.major.width": 1.6,
    "figure.facecolor": "white",
})


def find_spot_center(corrected_image, threshold_fraction):
    """Find the intensity-weighted center using positive pixels above a fraction of the image maximum."""
    positive_image = np.clip(corrected_image, 0.0, None)
    center_mask = positive_image > threshold_fraction * positive_image.max()
    y_coordinates, x_coordinates = np.indices(positive_image.shape)
    center_x = np.sum(x_coordinates[center_mask] * positive_image[center_mask]) / np.sum(positive_image[center_mask])
    center_y = np.sum(y_coordinates[center_mask] * positive_image[center_mask]) / np.sum(positive_image[center_mask])
    return int(round(center_x)), int(round(center_y))


def extract_cross_sections(corrected_image, center_x, center_y, profile_band_width):
    """Average horizontal and vertical strips of the requested odd width through (center_x, center_y)."""
    half_band_width = profile_band_width // 2
    horizontal_profile = corrected_image[center_y - half_band_width:center_y + half_band_width + 1, :].mean(axis=0)
    vertical_profile = corrected_image[:, center_x - half_band_width:center_x + half_band_width + 1].mean(axis=1)
    horizontal_positions = np.arange(corrected_image.shape[1]) - center_x
    vertical_positions = np.arange(corrected_image.shape[0]) - center_y
    return horizontal_positions, horizontal_profile, vertical_positions, vertical_profile


if __name__ == "__main__":
    data_dir = Path("ito_light_response/data/ito_characterization_1")  # Read-only characterization inputs.
    spot_image_dir = data_dir / "spot_sizes/images"  # Camera images from the spot-size sweep.
    background_path = data_dir / "intensity_sweep/images/back/Pos0/img_000000000_Default0_000.tif"  # Matched camera background.
    output_dir = Path("ito_light_response/output")  # Destination for the cross-section figures.
    dataset_names = ("1", "5")  # Smallest and largest spots retained in the spot-size response fit.
    dataset_descriptions = ("small spot", "large spot")  # Human-readable descriptions used in plot titles.
    threshold_fraction = 0.10  # Fraction of the corrected peak used to select pixels for center estimation.
    profile_band_width = 21  # Odd number of pixels averaged perpendicular to each cross-section.
    horizontal_color = "#2166ac"  # Blue used for the horizontal cross-section.
    vertical_color = "#b2182b"  # Red used to distinguish the vertical cross-section.
    figure_size = (8, 6.5)  # Width and height of each saved figure in inches.
    output_dpi = 300  # Resolution of each saved figure in dots per inch.

    background = tifffile.imread(background_path).astype(np.float64)
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name, dataset_description in zip(dataset_names, dataset_descriptions):
        image_path, = (spot_image_dir / dataset_name).rglob("*.tif")
        corrected_image = tifffile.imread(image_path).astype(np.float64) - background
        center_x, center_y = find_spot_center(corrected_image, threshold_fraction)
        horizontal_positions, horizontal_profile, vertical_positions, vertical_profile = extract_cross_sections(
            corrected_image, center_x, center_y, profile_band_width
        )

        output_path = output_dir / f"spot_cross_section_{dataset_description.replace(' ', '_')}.png"
        figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
        axis.plot(horizontal_positions, horizontal_profile, color=horizontal_color, linewidth=2.5, label="horizontal")
        axis.plot(vertical_positions, vertical_profile, color=vertical_color, linewidth=2.5, label="vertical")
        axis.axhline(0.0, color="0.65", linewidth=1.2, linestyle="--")
        axis.set_title(f"ITO {dataset_description} (dataset {dataset_name})")
        axis.set_xlabel("Position relative to spot center (pixels)")
        axis.set_ylabel("Corrected illumination (camera counts)")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.legend(frameon=False, loc="best", title=f"{profile_band_width}-pixel mean")
        figure.savefig(output_path, dpi=output_dpi)
        plt.close(figure)
        print(f"Dataset {dataset_name}: center=({center_x}, {center_y}); saved {output_path}.")
