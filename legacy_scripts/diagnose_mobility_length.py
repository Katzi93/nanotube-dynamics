"""Test whether the tube mobility-versus-length correlation is real or an artifact of the gradient definition.

The main analysis fits a per-tube through-origin mobility as the slope of parallel velocity against the whole-tube
gradient, which is the endpoint intensity difference divided by tube length, delta_I / L. If the true driving scales
with the total endpoint intensity difference delta_I rather than the per-length gradient delta_I / L, then dividing by
length forces mobility = v / (delta_I / L) = (v / delta_I) * L to rise with length by construction, with no underlying
length dependence. A second trap is the sublinear drift response: if longer tubes systematically sample a different
gradient range, the through-origin slope shifts with that range and fakes a length trend. This script recomputes the
per-tube mobility under the center-point, per-length endpoint, whole-tube, and length-unnormalized endpoint gradient
definitions, correlates each with length by Pearson and rank Spearman, jackknifes the whole-tube correlation to expose
outlier leverage, and reports whether longer tubes experience a systematically different gradient.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr
import tifffile

from legacy_scripts.analyze_tube_motion import (
    newest_track_file,
    load_and_merge_tube_geometry,
    fit_line_through_origin,
)
from legacy_scripts.diagnose_gradient_definition import gradient_definition_samples


def per_track_mobilities(gradient_magnitudes, parallel_velocities, interval_track_ids):
    """Return the track ids and each track's through-origin slope of parallel velocity against gradient magnitude."""
    track_ids = np.unique(interval_track_ids)
    mobilities = np.asarray(
        [fit_line_through_origin(gradient_magnitudes[interval_track_ids == track_id], parallel_velocities[interval_track_ids == track_id])[0] for track_id in track_ids]
    )
    return track_ids, mobilities


def jackknife_correlation_range(x_values, y_values):
    """Return the minimum and maximum Pearson correlation obtained by deleting one sample at a time."""
    sample_count = len(x_values)
    leave_one_out_correlations = []
    for excluded_index in range(sample_count):
        keep_mask = np.arange(sample_count) != excluded_index
        leave_one_out_correlations.append(np.corrcoef(x_values[keep_mask], y_values[keep_mask])[0, 1])
    leave_one_out_correlations = np.asarray(leave_one_out_correlations)
    return leave_one_out_correlations.min(), leave_one_out_correlations.max()


if __name__ == "__main__":
    track_globs = ["test_outputs/test_1_tube_tracks*.npz", "test_outputs/test_2_tube_tracks_*.npz", "test_outputs/test_3_tube_tracks_*.npz"]  # One glob per movie; the newest match of each is pooled (all share background.tif).
    track_paths = [newest_track_file(track_glob) for track_glob in track_globs]  # Newest track file resolved for every movie in the pool.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile.
    output_path = Path("test_outputs/test_combined_mobility_length_diagnosis.png")  # Diagnostic figure written by this script.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    frame_lag = 5  # Frame separation used for the direct drift measurement, matching the gradient-definition comparison.

    print(f"Reading tracks from {', '.join(str(track_path) for track_path in track_paths)}...")
    tube_positions_by_track, tube_angles_by_track = load_and_merge_tube_geometry(track_paths)
    print(f"Loaded {len(tube_positions_by_track)} tracks.")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    gradient_magnitudes, parallel_velocities, interval_track_ids, interval_tube_lengths, _ = gradient_definition_samples(
        tube_positions_by_track, tube_angles_by_track, illumination_gradient_x, illumination_gradient_y, smoothed_background, frame_lag
    )

    # The length-unnormalized endpoint driving is the endpoint intensity difference itself, delta_I = (delta_I / L) * L.
    gradient_magnitudes["endpoint_raw"] = gradient_magnitudes["endpoint"] * interval_tube_lengths
    parallel_velocities["endpoint_raw"] = parallel_velocities["endpoint"]

    definition_titles = {
        "center": "Center-point gradient (no length normalization)",
        "endpoint": "Endpoint difference per length (delta_I / L)",
        "whole_tube": "Whole-tube along-axis gradient (delta_I / L)",
        "endpoint_raw": "Endpoint difference, not per length (delta_I)",
    }
    definition_order = ["center", "endpoint", "whole_tube", "endpoint_raw"]

    mean_track_lengths = {}
    for track_id in np.unique(interval_track_ids):
        mean_track_lengths[track_id] = np.mean(interval_tube_lengths[interval_track_ids == track_id])

    definition_mobilities = {}
    definition_length_arrays = {}
    for name in definition_order:
        track_ids, mobilities = per_track_mobilities(gradient_magnitudes[name], parallel_velocities[name], interval_track_ids)
        track_lengths = np.asarray([mean_track_lengths[track_id] for track_id in track_ids])
        pearson_correlation = np.corrcoef(track_lengths, mobilities)[0, 1]
        spearman_correlation = spearmanr(track_lengths, mobilities).statistic
        definition_mobilities[name] = mobilities
        definition_length_arrays[name] = track_lengths
        print(
            f"{definition_titles[name]:48s}: Pearson r={pearson_correlation:+.3f}, Spearman rho={spearman_correlation:+.3f} across {len(mobilities)} tubes."
        )

    whole_tube_track_ids, whole_tube_mobilities = per_track_mobilities(gradient_magnitudes["whole_tube"], parallel_velocities["whole_tube"], interval_track_ids)
    whole_tube_lengths = np.asarray([mean_track_lengths[track_id] for track_id in whole_tube_track_ids])
    jackknife_minimum, jackknife_maximum = jackknife_correlation_range(whole_tube_lengths, whole_tube_mobilities)
    print(
        f"\nWhole-tube length-mobility Pearson correlation jackknife range (delete one tube): "
        f"[{jackknife_minimum:+.3f}, {jackknife_maximum:+.3f}]; a range crossing zero means a single tube drives the trend."
    )

    mean_gradient_per_track = np.asarray([np.mean(gradient_magnitudes["whole_tube"][interval_track_ids == track_id]) for track_id in whole_tube_track_ids])
    length_gradient_correlation = np.corrcoef(whole_tube_lengths, mean_gradient_per_track)[0, 1]
    print(
        f"Length versus mean whole-tube gradient experienced: Pearson r={length_gradient_correlation:+.3f} "
        f"(a strong value means the length-mobility trend is confounded by longer tubes sampling a different gradient range)."
    )

    figure, axes = plt.subplots(2, 2, figsize=(13, 11), constrained_layout=True)
    axes = axes.ravel()
    for panel_index, name in enumerate(definition_order):
        track_lengths = definition_length_arrays[name]
        mobilities = definition_mobilities[name]
        pearson_correlation = np.corrcoef(track_lengths, mobilities)[0, 1]
        slope, intercept = np.polyfit(track_lengths, mobilities, 1)
        fit_x = np.array([track_lengths.min(), track_lengths.max()])
        axes[panel_index].scatter(track_lengths, mobilities, s=30, color="#2166ac")
        axes[panel_index].plot(fit_x, intercept + slope * fit_x, color="black", linewidth=1.5)
        axes[panel_index].set_title(rf"{definition_titles[name]}: $r={pearson_correlation:+.3f}$")
        axes[panel_index].set_xlabel("Mean tube length (pixels)")
        axes[panel_index].set_ylabel(r"Through-origin mobility $\mu$")

    figure.suptitle(f"Is the mobility-length correlation real or a gradient-normalization artifact? ({frame_lag}-frame lag)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"\nSaved mobility-length diagnosis to {output_path}.")
