"""Test whether the drift-versus-gradient leveling at high projected gradient is a finite-lag artifact.

The direct drift measurement averages the projected illumination gradient over a frame lag and divides
the net center displacement by that lag. A tube on a steep slope drifts up-gradient toward the illumination
maximum, where the gradient of a smooth profile vanishes, so over a long lag it decelerates and its net
velocity falls below the linear prediction v = mobility * gradient. That produces an apparent leveling at
high projected gradient. If the leveling is this finite-lag artifact, its knee weakens and the fitted
mobility rises as the lag shrinks; if the response is genuinely sub-linear, the shape is stable across lags.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage import gaussian_filter
import tifffile

from tube_tracker.tube_tracking import polyline_center
from legacy_scripts.analyze_tube_motion import (
    newest_track_file,
    load_tube_geometry,
    principal_tube_axis,
    sample_vector_field,
    fit_line_through_origin,
    equal_count_bin_statistics,
)


def drift_measurements_with_window_gradients(tube_positions_by_track, gradient_x, gradient_y, frame_lag):
    """Return per-interval mean projected gradient, parallel velocity, and start and end projected-gradient magnitudes."""
    mean_gradient_magnitudes = []
    parallel_velocities = []
    start_gradient_magnitudes = []
    end_gradient_magnitudes = []
    for frame_positions in tube_positions_by_track.values():
        available_frames = set(frame_positions)
        for current_frame in sorted(frame_positions):
            next_frame = current_frame + frame_lag
            interval_frames = range(current_frame, next_frame + 1)
            if not all(frame_index in available_frames for frame_index in interval_frames):
                continue
            projected_gradients = []
            for frame_index in interval_frames:
                positions = frame_positions[frame_index]
                center = polyline_center(positions)
                tube_axis = principal_tube_axis(positions)
                illumination_gradient = sample_vector_field(gradient_x, gradient_y, center)
                projected_gradients.append(tube_axis * np.dot(tube_axis, illumination_gradient))
            projected_gradients = np.asarray(projected_gradients)
            mean_projected_gradient = projected_gradients.mean(axis=0)
            mean_projected_magnitude = np.linalg.norm(mean_projected_gradient)
            if mean_projected_magnitude == 0.0:
                continue
            projected_direction = mean_projected_gradient / mean_projected_magnitude
            displacement = polyline_center(frame_positions[next_frame]) - polyline_center(frame_positions[current_frame])
            velocity = displacement / frame_lag
            mean_gradient_magnitudes.append(mean_projected_magnitude)
            parallel_velocities.append(np.dot(velocity, projected_direction))
            start_gradient_magnitudes.append(np.linalg.norm(projected_gradients[0]))
            end_gradient_magnitudes.append(np.linalg.norm(projected_gradients[-1]))
    return (
        np.asarray(mean_gradient_magnitudes),
        np.asarray(parallel_velocities),
        np.asarray(start_gradient_magnitudes),
        np.asarray(end_gradient_magnitudes),
    )


if __name__ == "__main__":
    track_glob = "test_outputs/test_2_tube_tracks_*.npz"  # Newest timestamped tube-track file matching this pattern is diagnosed.
    track_path = newest_track_file(track_glob)  # Most recent saved tracks, resolved from the timestamp in the file name.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile.
    output_path = Path("test_outputs/test_2_drift_lag_diagnosis.png")  # Diagnostic figure written by this script.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    lag_values = [3, 5, 7, 9, 11, 15]  # Frame lags compared to reveal any lag dependence of the leveling.
    bin_count = 8  # Number of equal-count gradient groups used to display each lag's mean drift curve.
    high_gradient_quantile = 0.75  # Lower quantile of the mean projected gradient defining the high-gradient intervals.

    print(f"Reading tracks from {track_path}...")
    track_data = np.load(track_path, allow_pickle=False)
    tube_positions_by_track, _ = load_tube_geometry(track_data)
    print(f"Loaded {len(tube_positions_by_track)} tracks.")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    lag_mobilities = []
    lag_r_squared_values = []
    lag_high_gradient_end_minus_start = []
    lag_interval_counts = []
    binned_curves = {}
    for frame_lag in lag_values:
        mean_gradients, parallel_velocities, start_gradients, end_gradients = drift_measurements_with_window_gradients(
            tube_positions_by_track, illumination_gradient_x, illumination_gradient_y, frame_lag
        )
        mobility, r_squared = fit_line_through_origin(mean_gradients, parallel_velocities)
        binned_gradients, binned_velocities, binned_errors = equal_count_bin_statistics(mean_gradients, parallel_velocities, bin_count)
        high_gradient_threshold = np.quantile(mean_gradients, high_gradient_quantile)
        high_gradient_mask = mean_gradients >= high_gradient_threshold
        end_minus_start = np.mean(end_gradients[high_gradient_mask] - start_gradients[high_gradient_mask])
        lag_mobilities.append(mobility)
        lag_r_squared_values.append(r_squared)
        lag_high_gradient_end_minus_start.append(end_minus_start)
        lag_interval_counts.append(len(mean_gradients))
        binned_curves[frame_lag] = (binned_gradients, binned_velocities, binned_errors)
        print(
            f"Lag {frame_lag:2d}: intervals={len(mean_gradients):4d}, mobility={mobility:.4f}, R^2={r_squared:.3f}, "
            f"high-gradient end-minus-start projected gradient={end_minus_start:+.4f} (negative means climbing toward the flat peak)."
        )

    lag_mobilities = np.asarray(lag_mobilities)
    print(
        f"Mobility change from longest to shortest lag: {lag_mobilities[0] / lag_mobilities[-1]:.2f}x "
        f"(a value well above 1 supports a finite-lag artifact)."
    )

    figure, axes = plt.subplots(1, 3, figsize=(18, 5), constrained_layout=True)
    lag_color_map = plt.cm.viridis(np.linspace(0.0, 0.9, len(lag_values)))
    for lag_color, frame_lag in zip(lag_color_map, lag_values):
        binned_gradients, binned_velocities, binned_errors = binned_curves[frame_lag]
        axes[0].errorbar(binned_gradients, binned_velocities, yerr=binned_errors, color=lag_color, marker="o", capsize=2, label=f"lag {frame_lag}")
    axes[0].axhline(0.0, color="gray", linewidth=0.7)
    axes[0].set_title("Binned parallel drift versus projected gradient per lag")
    axes[0].set_xlabel(r"Mean projected gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    axes[0].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")
    axes[0].legend()

    axes[1].plot(lag_values, lag_mobilities, color="#2166ac", marker="o")
    axes[1].set_title("Fitted through-origin mobility versus lag")
    axes[1].set_xlabel("Frame lag")
    axes[1].set_ylabel(r"Mobility $\mu$ (pixels$^2$/intensity/frame)")

    axes[2].axhline(0.0, color="gray", linewidth=0.7)
    axes[2].plot(lag_values, lag_high_gradient_end_minus_start, color="#b2182b", marker="o")
    axes[2].set_title("High-gradient intervals: end-minus-start projected gradient")
    axes[2].set_xlabel("Frame lag")
    axes[2].set_ylabel(r"$\langle g_{\mathrm{end}} - g_{\mathrm{start}} \rangle$ (intensity/pixel)")

    figure.suptitle("Finite-lag diagnosis of the drift-versus-gradient leveling")
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    print(f"Saved finite-lag drift diagnosis to {output_path}.")
