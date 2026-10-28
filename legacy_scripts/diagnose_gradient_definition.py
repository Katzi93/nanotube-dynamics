"""Compare gradient definitions to explain the drift-versus-gradient leveling.

The projected illumination gradient can be defined three ways for a finite tube: the local vector gradient
sampled only at the center of mass and projected onto the tube axis; the tip-minus-bottom endpoint
illumination difference per tube length; and the length-weighted mean of the along-axis gradient over every
ridge, which equals the endpoint difference divided by length for a straight tube. The center-point
definition can overstate the driving on a long tube spanning a curved illumination profile, sending such
tubes to high gradient while their true end-to-end driving and their velocity stay low, which produces an
apparent leveling. This script recomputes the direct lagged drift against all three definitions and reports
which one removes the leveling, then breaks the center-point leveling band down by track and tube length.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from tube_tracker.tube_tracking import polyline_center, polyline_length
from legacy_scripts.analyze_tube_motion import (
    newest_track_file,
    load_tube_geometry,
    principal_tube_axis,
    sample_vector_field,
    average_projected_gradient_over_tube,
    endpoint_activity_gradient,
    fit_line_through_origin,
    equal_count_bin_statistics,
)


def gradient_definition_samples(tube_positions_by_track, tube_angles_by_track, gradient_x, gradient_y, illumination_image, frame_lag):
    """Return, for every uninterrupted interval, the drift velocity projected onto three gradient definitions."""
    definition_names = ["center", "endpoint", "whole_tube"]
    gradient_magnitudes = {name: [] for name in definition_names}
    parallel_velocities = {name: [] for name in definition_names}
    interval_track_ids = []
    interval_tube_lengths = []
    interval_centers = []
    for track_id, frame_positions in tube_positions_by_track.items():
        available_frames = set(frame_positions)
        frame_angles = tube_angles_by_track[track_id]
        for current_frame in sorted(frame_positions):
            next_frame = current_frame + frame_lag
            interval_frames = range(current_frame, next_frame + 1)
            if not all(frame_index in available_frames for frame_index in interval_frames):
                continue
            center_vectors = []
            endpoint_vectors = []
            whole_tube_vectors = []
            tube_lengths = []
            for frame_index in interval_frames:
                positions = frame_positions[frame_index]
                tube_axis = principal_tube_axis(positions)
                center_gradient = sample_vector_field(gradient_x, gradient_y, polyline_center(positions))
                center_vectors.append(tube_axis * np.dot(tube_axis, center_gradient))
                endpoint_vectors.append(endpoint_activity_gradient(positions, illumination_image))
                whole_tube_vectors.append(
                    average_projected_gradient_over_tube(positions, frame_angles[frame_index], gradient_x, gradient_y)
                )
                tube_lengths.append(polyline_length(positions))
            interval_vectors = {"center": np.asarray(center_vectors), "endpoint": np.asarray(endpoint_vectors), "whole_tube": np.asarray(whole_tube_vectors)}
            displacement = polyline_center(frame_positions[next_frame]) - polyline_center(frame_positions[current_frame])
            velocity = displacement / frame_lag
            skip_interval = False
            interval_magnitudes = {}
            interval_projections = {}
            for name in definition_names:
                mean_vector = interval_vectors[name].mean(axis=0)
                magnitude = np.linalg.norm(mean_vector)
                if magnitude == 0.0:
                    skip_interval = True
                    break
                interval_magnitudes[name] = magnitude
                interval_projections[name] = np.dot(velocity, mean_vector / magnitude)
            if skip_interval:
                continue
            for name in definition_names:
                gradient_magnitudes[name].append(interval_magnitudes[name])
                parallel_velocities[name].append(interval_projections[name])
            interval_track_ids.append(track_id)
            interval_tube_lengths.append(np.mean(tube_lengths))
            interval_centers.append(polyline_center(frame_positions[current_frame]))
    gradient_magnitudes = {name: np.asarray(values) for name, values in gradient_magnitudes.items()}
    parallel_velocities = {name: np.asarray(values) for name, values in parallel_velocities.items()}
    return gradient_magnitudes, parallel_velocities, np.asarray(interval_track_ids), np.asarray(interval_tube_lengths), np.asarray(interval_centers)


if __name__ == "__main__":
    track_glob = "test_outputs/test_2_tube_tracks_*.npz"  # Newest timestamped tube-track file matching this pattern is tested.
    track_path = newest_track_file(track_glob)  # Most recent saved tracks, resolved from the timestamp in the file name.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile.
    output_path = Path("test_outputs/test_2_gradient_definition_diagnosis.png")  # Diagnostic figure written by this script.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    frame_lag = 5  # Frame separation used for the direct drift measurement in this comparison.
    bin_count = 9  # Number of equal-count gradient groups used to display each definition's mean drift curve.
    leveling_band_low = 4.0  # Lower projected-gradient bound of the center-point leveling band investigated by track.
    leveling_band_high = 6.5  # Upper projected-gradient bound of the center-point leveling band investigated by track.

    print(f"Reading tracks from {track_path}...")
    track_data = np.load(track_path, allow_pickle=False)
    tube_positions_by_track, tube_angles_by_track = load_tube_geometry(track_data)
    print(f"Loaded {len(tube_positions_by_track)} tracks.")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    gradient_magnitudes, parallel_velocities, interval_track_ids, interval_tube_lengths, interval_centers = gradient_definition_samples(
        tube_positions_by_track, tube_angles_by_track, illumination_gradient_x, illumination_gradient_y, smoothed_background, frame_lag
    )
    definition_titles = {"center": "Center-point gradient", "endpoint": "Endpoint tip-minus-bottom per length", "whole_tube": "Whole-tube mean along-axis gradient"}
    definition_order = ["center", "endpoint", "whole_tube"]
    definition_mobilities = {}
    definition_r_squared = {}
    for name in definition_order:
        mobility, r_squared = fit_line_through_origin(gradient_magnitudes[name], parallel_velocities[name])
        definition_mobilities[name] = mobility
        definition_r_squared[name] = r_squared
        print(f"{definition_titles[name]:42s}: mobility={mobility:.4f}, R^2={r_squared:.3f}, intervals={len(gradient_magnitudes[name])}.")

    center_magnitudes = gradient_magnitudes["center"]
    center_velocities = parallel_velocities["center"]
    band_mask = (center_magnitudes >= leveling_band_low) & (center_magnitudes <= leveling_band_high)
    band_track_ids = np.unique(interval_track_ids[band_mask])
    center_mobility = definition_mobilities["center"]
    print(f"\nCenter-point leveling band {leveling_band_low}-{leveling_band_high}: {band_mask.sum()} intervals across {len(band_track_ids)} tracks.")
    print("Per-track mean velocity shortfall below the linear expectation inside the band (negative means it drags the band down):")
    band_track_shortfalls = []
    for track_id in band_track_ids:
        track_band_mask = band_mask & (interval_track_ids == track_id)
        mean_gradient = np.mean(center_magnitudes[track_band_mask])
        mean_velocity = np.mean(center_velocities[track_band_mask])
        mean_length = np.mean(interval_tube_lengths[track_band_mask])
        shortfall = mean_velocity - center_mobility * mean_gradient
        band_track_shortfalls.append((track_id, track_band_mask.sum(), mean_length, shortfall))
    for track_id, count, mean_length, shortfall in sorted(band_track_shortfalls, key=lambda row: row[3]):
        print(f"  T{int(track_id):2d}: {count:3d} intervals, mean length={mean_length:6.1f} px, velocity shortfall={shortfall:+.3f} px/frame.")

    center_endpoint_discrepancy = center_magnitudes - gradient_magnitudes["endpoint"]
    length_discrepancy_correlation = np.corrcoef(interval_tube_lengths, center_endpoint_discrepancy)[0, 1]
    print(
        f"\nCorrelation between tube length and (center-point minus endpoint) gradient discrepancy: {length_discrepancy_correlation:.3f} "
        f"(positive means longer tubes read a larger center-point gradient than their true end-to-end driving)."
    )

    figure, axes = plt.subplots(1, 3, figsize=(19, 5.5), constrained_layout=True)
    for panel_index, name in enumerate(definition_order):
        magnitudes = gradient_magnitudes[name]
        velocities = parallel_velocities[name]
        binned_gradients, binned_velocities, binned_errors = equal_count_bin_statistics(magnitudes, velocities, bin_count)
        axes[panel_index].scatter(magnitudes, velocities, s=6, color="lightgray", alpha=0.3)
        axes[panel_index].errorbar(binned_gradients, binned_velocities, yerr=binned_errors, color="#2166ac", marker="o", linestyle="none", capsize=3)
        fit_x = np.array([0.0, magnitudes.max()])
        axes[panel_index].plot(fit_x, definition_mobilities[name] * fit_x, color="black", linewidth=1.5)
        axes[panel_index].axhline(0.0, color="gray", linewidth=0.7)
        axes[panel_index].set_title(rf"{definition_titles[name]}: $\mu$={definition_mobilities[name]:.3f}, $R_0^2$={definition_r_squared[name]:.3f}")
        axes[panel_index].set_xlabel("Projected gradient magnitude (intensity/pixel)")
        axes[panel_index].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")

    figure.suptitle(f"Gradient-definition comparison of the drift leveling ({frame_lag}-frame lag)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"\nSaved gradient-definition diagnosis to {output_path}.")
