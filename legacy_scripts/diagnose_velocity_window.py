"""Test whether the Savitzky-Golay velocity window distorts the drift-versus-gradient relationship.

The main motion analysis estimates each velocity with a Savitzky-Golay derivative over a fixed number of
frames (velocity_fitting_window). A wide window low-pass filters the trajectory and attenuates the fastest
transient excursions, which occur precisely where the illumination gradient is steepest, so a window that is
too wide can flatten the high-gradient end of the velocity-versus-gradient curve. This script recomputes the
same projected-gradient velocity samples for several velocity windows and reports how the fitted mobility and
the binned curve shape change. If the leveling is a window artifact, it weakens at short windows and the
fitted mobility rises; if the shape is stable across windows, the window is not responsible.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
from scipy.signal import savgol_filter
import tifffile

from legacy_scripts.analyze_tube_motion import (
    newest_track_file,
    load_and_merge_tube_geometry,
    interpolate_track_geometry,
    sample_vector_field,
    fit_line_through_origin,
    equal_count_bin_statistics,
)


def velocity_gradient_samples(tube_positions_by_track, gradient_x, gradient_y, velocity_window, orientation_window, polynomial_order):
    """Return projected-gradient magnitudes with along-gradient and across-gradient velocities from Savitzky-Golay velocity fits."""
    projected_gradient_magnitudes = []
    parallel_velocities = []
    perpendicular_velocities = []
    for frame_positions in tube_positions_by_track.values():
        frame_indices, interpolated_centers, interpolated_axes, _ = interpolate_track_geometry(frame_positions)
        if len(frame_indices) < max(velocity_window, orientation_window):
            continue
        smoothed_centers = savgol_filter(interpolated_centers, velocity_window, polynomial_order, axis=0, mode="interp")
        fitted_velocities = savgol_filter(interpolated_centers, velocity_window, polynomial_order, deriv=1, delta=1.0, axis=0, mode="interp")
        doubled_orientation_angles = np.unwrap(2.0 * np.arctan2(interpolated_axes[:, 1], interpolated_axes[:, 0]))
        smoothed_orientation_angles = 0.5 * savgol_filter(doubled_orientation_angles, orientation_window, polynomial_order, axis=0, mode="interp")
        smoothed_axes = np.column_stack((np.cos(smoothed_orientation_angles), np.sin(smoothed_orientation_angles)))
        half_window = max(velocity_window // 2, orientation_window // 2)
        for current_index in range(half_window, len(frame_indices) - half_window):
            velocity = fitted_velocities[current_index]
            speed = np.linalg.norm(velocity)
            tube_axis = smoothed_axes[current_index]
            illumination_gradient = sample_vector_field(gradient_x, gradient_y, smoothed_centers[current_index])
            gradient_magnitude = np.linalg.norm(illumination_gradient)
            if speed == 0.0 or gradient_magnitude == 0.0:
                continue
            projected_gradient = tube_axis * np.dot(tube_axis, illumination_gradient)
            projected_gradient_magnitude = np.linalg.norm(projected_gradient)
            if projected_gradient_magnitude == 0.0:
                continue
            projected_gradient_direction = projected_gradient / projected_gradient_magnitude
            perpendicular_direction = np.array([-projected_gradient_direction[1], projected_gradient_direction[0]])
            projected_gradient_magnitudes.append(projected_gradient_magnitude)
            parallel_velocities.append(np.dot(velocity, projected_gradient_direction))
            perpendicular_velocities.append(np.dot(velocity, perpendicular_direction))
    return np.asarray(projected_gradient_magnitudes), np.asarray(parallel_velocities), np.asarray(perpendicular_velocities)


if __name__ == "__main__":
    track_globs = ["test_outputs/test_1_tube_tracks*.npz", "test_outputs/test_2_tube_tracks_*.npz", "test_outputs/test_3_tube_tracks_*.npz"]  # One glob per movie; the newest match of each is pooled (all share background.tif).
    track_paths = [newest_track_file(track_glob) for track_glob in track_globs]  # Newest track file resolved for every movie in the pool.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile shared by both movies.
    output_path = Path("test_outputs/test_combined_velocity_window_diagnosis.png")  # Diagnostic figure written by this script.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    velocity_windows = [3, 5, 7, 9, 11, 15, 21]  # Odd Savitzky-Golay velocity windows compared to reveal any window dependence; 3 is a plain central difference.
    orientation_window = 7  # Fixed odd Savitzky-Golay orientation window held constant across the sweep.
    polynomial_order = 2  # Polynomial order fitted to each local trajectory and orientation.
    bin_count = 8  # Number of equal-count gradient groups used to display each window's mean drift curve.

    print(f"Reading tracks from {', '.join(str(track_path) for track_path in track_paths)}...")
    tube_positions_by_track, _ = load_and_merge_tube_geometry(track_paths)
    print(f"Loaded {len(tube_positions_by_track)} tracks.")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    window_mobilities = []
    window_r_squared_values = []
    window_sample_counts = []
    binned_parallel_curves = {}
    binned_perpendicular_curves = {}
    for velocity_window in velocity_windows:
        gradient_magnitudes, parallel_velocities, perpendicular_velocities = velocity_gradient_samples(
            tube_positions_by_track, illumination_gradient_x, illumination_gradient_y, velocity_window, orientation_window, polynomial_order
        )
        mobility, r_squared = fit_line_through_origin(gradient_magnitudes, parallel_velocities)
        binned_gradients, binned_parallel, binned_parallel_errors = equal_count_bin_statistics(gradient_magnitudes, parallel_velocities, bin_count)
        _, binned_perpendicular, binned_perpendicular_errors = equal_count_bin_statistics(gradient_magnitudes, perpendicular_velocities, bin_count)
        window_mobilities.append(mobility)
        window_r_squared_values.append(r_squared)
        window_sample_counts.append(len(gradient_magnitudes))
        binned_parallel_curves[velocity_window] = (binned_gradients, binned_parallel, binned_parallel_errors)
        binned_perpendicular_curves[velocity_window] = (binned_gradients, binned_perpendicular, binned_perpendicular_errors)
        mean_perpendicular_velocity = np.mean(perpendicular_velocities)
        print(
            f"Window {velocity_window:2d}: samples={len(gradient_magnitudes):5d}, mobility={mobility:.4f}, R^2={r_squared:.3f}, "
            f"mean v_perp={mean_perpendicular_velocity:+.4f} px/frame (near zero confirms drift is along-gradient)."
        )

    window_mobilities = np.asarray(window_mobilities)
    print(
        f"Mobility change from widest to narrowest window: {window_mobilities[0] / window_mobilities[-1]:.2f}x "
        f"(a value well above 1 supports a velocity-window artifact)."
    )

    figure, axes = plt.subplots(1, 3, figsize=(19, 5), constrained_layout=True)
    window_color_map = plt.cm.plasma(np.linspace(0.0, 0.85, len(velocity_windows)))
    for window_color, velocity_window in zip(window_color_map, velocity_windows):
        binned_gradients, binned_parallel, binned_parallel_errors = binned_parallel_curves[velocity_window]
        axes[0].errorbar(binned_gradients, binned_parallel, yerr=binned_parallel_errors, color=window_color, marker="o", capsize=2, label=f"window {velocity_window}")
        binned_gradients, binned_perpendicular, binned_perpendicular_errors = binned_perpendicular_curves[velocity_window]
        axes[1].errorbar(binned_gradients, binned_perpendicular, yerr=binned_perpendicular_errors, color=window_color, marker="o", capsize=2, label=f"window {velocity_window}")
    axes[0].axhline(0.0, color="gray", linewidth=0.7)
    axes[0].set_title("Binned parallel velocity versus projected gradient per velocity window")
    axes[0].set_xlabel(r"Projected illumination gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    axes[0].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")
    axes[0].legend()

    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].set_title("Binned perpendicular velocity versus projected gradient per velocity window")
    axes[1].set_xlabel(r"Projected illumination gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    axes[1].set_ylabel(r"$v_{\perp}$ (pixels/frame)")
    axes[1].legend()

    axes[2].plot(velocity_windows, window_mobilities, color="#762a83", marker="o")
    axes[2].set_title("Fitted through-origin mobility versus velocity window")
    axes[2].set_xlabel("Savitzky-Golay velocity window (frames)")
    axes[2].set_ylabel(r"Mobility $\mu$ (pixels$^2$/intensity/frame)")

    figure.suptitle("Velocity-window diagnosis of the drift-versus-gradient leveling")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved velocity-window diagnosis to {output_path}.")
