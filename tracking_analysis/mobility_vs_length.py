"""Fit a per-tube mobility from each track and plot it against the tube length.
Each track is one tube followed over time. Within a track the along-axis drift should rise with the projected light
gradient through the origin, so the through-origin slope of v_parallel versus the projected gradient is that tube's
mobility. Fitting this per track turns the single pooled mobility into one number per tube, which can then be plotted
against the tube length to see whether shorter or longer tubes glide up the gradient more readily. Tracks with too few
measured frames sample only a narrow slice of the gradient and would give a meaningless slope, so they are dropped.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from tracking_analysis.tube_tracks import (
    add_light_gradient,
    add_projections,
    add_velocity,
    build_tube_tracks,
    load_datasets,
    simplify_dataframe,
)

# --- Physical calibration -------------------------------------------------------------------------------------------
FRAME_TIME_S = 0.5  # Seconds between successive frames.
MICRONS_PER_PIXEL = 221.66666 / 2048.0  # 2048 px span 221.66666 um in the imaging plane.
PIXELS_PER_FRAME_TO_UM_PER_S = MICRONS_PER_PIXEL / FRAME_TIME_S  # Convert a pixels/frame velocity to um/s.
INTENSITY_PER_PIXEL_TO_PER_UM = 1.0 / MICRONS_PER_PIXEL  # Convert an intensity/pixel gradient to a.u./um.
# A per-tube mobility is (um/s velocity) / (a.u./um gradient), so it scales by the ratio of the two calibrations.
SLOPE_TO_MOBILITY = PIXELS_PER_FRAME_TO_UM_PER_S / INTENSITY_PER_PIXEL_TO_PER_UM

# --- Publication styling --------------------------------------------------------------------------------------------
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
DATASET_LABELS = {"test_1": "Dataset 1", "test_2": "Dataset 2", "test_3": "Dataset 3"}  # Publication legend labels.
UNIT_MU_TRANSLATION = r"$\mu\mathrm{m}^2\,\mathrm{s}^{-1}\,\mathrm{a.u.}^{-1}$"  # Units of a translational mobility.


def through_origin_slope(gradient, velocity):
    """Return the through-origin slope for velocity = slope * gradient."""
    return np.dot(gradient, velocity) / np.dot(gradient, gradient)


if __name__ == "__main__":
    dataset_names = ["test_1", "test_2", "test_3"]  # Canonical track datasets pooled from tracking_data/.
    background_path = Path("test_data/background.tif")  # Read-only illumination profile shared by every movie.
    output_path = Path("analysis_outputs/mobility_vs_length.png")  # Figure written by this analysis.
    dataset_colors = {"test_1": "#2166ac", "test_2": "#b2182b", "test_3": "#1b7837"}  # One color per pooled movie.
    minimum_measured_frames = 5  # Fewest central-difference frames a track needs for a meaningful per-tube slope.
    figure_size = (8, 6.5)  # Width and height of the saved figure in inches.
    output_dpi = 300  # Resolution of the saved figure in dots per inch.

    ridges = simplify_dataframe(load_datasets(dataset_names))
    tube_tracks = build_tube_tracks(ridges)
    tube_tracks = add_velocity(tube_tracks)
    tube_tracks = add_light_gradient(tube_tracks, background_path)
    tube_tracks = add_projections(tube_tracks)
    measured = tube_tracks.dropna(subset=["v_parallel", "projected_gradient"])

    track_records = []
    for (dataset, track), track_rows in measured.groupby(["dataset", "track"], sort=True):
        if len(track_rows) < minimum_measured_frames:
            continue
        # The raw slope is in pixels/frame per intensity/pixel; rescale it to a physical mobility in um^2/s per a.u./um.
        raw_slope = through_origin_slope(track_rows["projected_gradient"].to_numpy(), track_rows["v_parallel"].to_numpy())
        mobility = raw_slope * SLOPE_TO_MOBILITY
        median_length = tube_tracks[(tube_tracks["dataset"] == dataset) & (tube_tracks["track"] == track)]["length"].median()
        track_records.append({"dataset": dataset, "mobility": mobility, "length": median_length * MICRONS_PER_PIXEL, "n_frames": len(track_rows)})

    n_tracks_total = measured.groupby(["dataset", "track"]).ngroups
    n_tracks_kept = len(track_records)
    lengths = np.array([record["length"] for record in track_records])  # Tube lengths already in um.
    mobilities = np.array([record["mobility"] for record in track_records])  # Per-tube parallel mobilities in um^2/s per a.u./um.
    pearson_r = np.corrcoef(lengths, mobilities)[0, 1]
    fit_slope, fit_intercept = np.polyfit(lengths, mobilities, 1)  # Straight-line trend of mobility with length.
    print(f"Kept {n_tracks_kept} of {n_tracks_total} tracks with at least {minimum_measured_frames} measured frames.")
    print(f"Per-tube mobility: median={np.median(mobilities):.3f}, range {mobilities.min():.3f} to {mobilities.max():.3f} um^2/s per a.u./um.")
    print(f"Tube length: median={np.median(lengths):.1f}, range {lengths.min():.1f} to {lengths.max():.1f} um.")
    print(f"Pearson correlation of mobility with length={pearson_r:.3f}; linear fit slope={fit_slope:.4f}, intercept={fit_intercept:.3f}.")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=figure_size, constrained_layout=True)
    for dataset_name, color in dataset_colors.items():
        dataset_rows = np.array([record["dataset"] == dataset_name for record in track_records])
        axis.scatter(lengths[dataset_rows], mobilities[dataset_rows], color=color, s=60, alpha=0.75, edgecolor="none", label=DATASET_LABELS[dataset_name])
    fit_lengths = np.array([lengths.min(), lengths.max()])
    axis.plot(fit_lengths, fit_slope * fit_lengths + fit_intercept, color="black", linewidth=3.0,
              label=rf"linear fit  ($r={pearson_r:.2f}$)")
    axis.axhline(0.0, color="0.6", linewidth=1.2, zorder=0)
    axis.set_title(rf"Per-tube mobility vs. length  ($N={n_tracks_kept}$ tubes)")
    axis.set_xlabel(r"Tube length  ($\mu$m)")
    axis.set_ylabel(rf"Mobility $\mu_\parallel$  ({UNIT_MU_TRANSLATION})")
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.legend(frameon=False, loc="best")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved mobility-versus-length figure to {output_path}.")
