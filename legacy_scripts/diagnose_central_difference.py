"""Measure the drift-versus-gradient mobility with a bare three-frame central difference as the least-assumption test.

The main analysis estimates velocity with a Savitzky-Golay polynomial fit over several frames, which smooths the
trajectory and therefore embeds a choice of window and polynomial order. The simplest possible velocity estimator makes
none of those choices: the central difference v(t) = (center(t + 1) - center(t - 1)) / 2, evaluated only at frames whose
immediate neighbors were both actually measured, with no interpolation and no smoothing. This script computes that bare
velocity, projects it onto the primary whole-tube along-axis gradient, fits the through-origin mobility, and plots the
parallel drift and the perpendicular control against the gradient. Agreement with the smoothed result shows the mobility
is not an artifact of the smoothing; the wider scatter here is the honest per-sample noise the smoothing averages away.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from tube_tracker.tube_tracking import polyline_center
from legacy_scripts.analyze_tube_motion import (
    newest_track_file,
    load_and_merge_tube_geometry,
    average_projected_gradient_over_tube,
    fit_line_through_origin,
    equal_count_bin_statistics,
)


def central_difference_drift_samples(tube_positions_by_track, tube_angles_by_track, gradient_x, gradient_y):
    """Return whole-tube gradient magnitudes with along-gradient and across-gradient central-difference velocities."""
    gradient_magnitudes = []
    parallel_velocities = []
    perpendicular_velocities = []
    for track_id, frame_positions in tube_positions_by_track.items():
        frame_angles = tube_angles_by_track[track_id]
        available_frames = set(frame_positions)
        for current_frame in sorted(frame_positions):
            if current_frame - 1 not in available_frames or current_frame + 1 not in available_frames:
                continue
            velocity = (polyline_center(frame_positions[current_frame + 1]) - polyline_center(frame_positions[current_frame - 1])) / 2.0
            whole_tube_gradient = average_projected_gradient_over_tube(
                frame_positions[current_frame], frame_angles[current_frame], gradient_x, gradient_y
            )
            gradient_magnitude = np.linalg.norm(whole_tube_gradient)
            if gradient_magnitude == 0.0:
                continue
            gradient_direction = whole_tube_gradient / gradient_magnitude
            perpendicular_direction = np.array([-gradient_direction[1], gradient_direction[0]])
            gradient_magnitudes.append(gradient_magnitude)
            parallel_velocities.append(np.dot(velocity, gradient_direction))
            perpendicular_velocities.append(np.dot(velocity, perpendicular_direction))
    return np.asarray(gradient_magnitudes), np.asarray(parallel_velocities), np.asarray(perpendicular_velocities)


if __name__ == "__main__":
    track_globs = ["test_outputs/test_1_tube_tracks*.npz", "test_outputs/test_2_tube_tracks_*.npz"]  # One glob per movie; the newest match of each is pooled (both share background.tif).
    track_paths = [newest_track_file(track_glob) for track_glob in track_globs]  # Newest track file resolved for every movie in the pool.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile shared by both movies.
    output_path = Path("test_outputs/test_combined_central_difference_diagnosis.png")  # Diagnostic figure written by this script.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    bin_count = 10  # Number of equal-count gradient groups used to display the mean drift with its uncertainty.

    print(f"Reading tracks from {', '.join(str(track_path) for track_path in track_paths)}...")
    tube_positions_by_track, tube_angles_by_track = load_and_merge_tube_geometry(track_paths)
    print(f"Loaded {len(tube_positions_by_track)} tracks.")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    gradient_magnitudes, parallel_velocities, perpendicular_velocities = central_difference_drift_samples(
        tube_positions_by_track, tube_angles_by_track, illumination_gradient_x, illumination_gradient_y
    )
    mobility, r_squared = fit_line_through_origin(gradient_magnitudes, parallel_velocities)
    mean_perpendicular_velocity = np.mean(perpendicular_velocities)
    binned_gradients, binned_parallel, binned_parallel_errors = equal_count_bin_statistics(gradient_magnitudes, parallel_velocities, bin_count)
    _, binned_perpendicular, binned_perpendicular_errors = equal_count_bin_statistics(gradient_magnitudes, perpendicular_velocities, bin_count)
    print(f"Bare central-difference samples: {len(gradient_magnitudes)} across {len(tube_positions_by_track)} tracks.")
    print(f"Through-origin mobility={mobility:.4f}, R-squared={r_squared:.3f}, mean v_perp={mean_perpendicular_velocity:+.4f} px/frame (near zero confirms drift is along-gradient).")

    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), constrained_layout=True)
    fit_x = np.array([0.0, gradient_magnitudes.max()])
    axes[0].scatter(gradient_magnitudes, parallel_velocities, s=6, color="lightgray", alpha=0.3)
    axes[0].errorbar(binned_gradients, binned_parallel, yerr=binned_parallel_errors, color="#2166ac", marker="o", linestyle="none", capsize=3)
    axes[0].plot(fit_x, mobility * fit_x, color="black", linewidth=1.5)
    axes[0].axhline(0.0, color="gray", linewidth=0.7)
    axes[0].set_title(rf"Bare central-difference drift: $\mu={mobility:.3f}$, $R_0^2={r_squared:.3f}$")
    axes[0].set_xlabel(r"Whole-tube along-axis gradient $|\langle u(u \cdot \nabla I)\rangle_L|$ (intensity/pixel)")
    axes[0].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")

    axes[1].scatter(gradient_magnitudes, perpendicular_velocities, s=6, color="lightgray", alpha=0.3)
    axes[1].errorbar(binned_gradients, binned_perpendicular, yerr=binned_perpendicular_errors, color="#b2182b", marker="o", linestyle="none", capsize=3)
    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].set_title(r"Perpendicular control $v_{\perp}$ (central difference)")
    axes[1].set_xlabel(r"Whole-tube along-axis gradient $|\langle u(u \cdot \nabla I)\rangle_L|$ (intensity/pixel)")
    axes[1].set_ylabel(r"$v_{\perp}$ (pixels/frame)")

    figure.suptitle("Least-assumption central-difference drift versus whole-tube gradient (combined movies)")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved central-difference diagnosis to {output_path}.")
