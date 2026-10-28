"""Compare tracked nanotube motion with tube orientation and the illumination gradient."""

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, Normalize
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates
import tifffile

from tube_tracker.tube_tracking import polyline_center, polyline_length


def newest_track_file(track_glob):
    """Return the most recent tube-track file matching a glob, ordered by the timestamp embedded in the file name."""
    matching_paths = sorted(Path().glob(track_glob))
    if not matching_paths:
        raise FileNotFoundError(f"No tube-track files match {track_glob}.")
    return matching_paths[-1]


def load_tube_geometry(track_data):
    """Return the ordered ridge-center positions and angles for every stored track and frame."""
    tube_positions = {}
    tube_angles = {}
    for track_id in track_data["track_ids"]:
        tube_positions[int(track_id)] = {}
        tube_angles[int(track_id)] = {}
        track_rows = track_data["ridge_track_ids"] == track_id
        for frame_index in np.unique(track_data["ridge_frame_indices"][track_rows]):
            frame_rows = track_rows & (track_data["ridge_frame_indices"] == frame_index)
            ridge_order = np.argsort(track_data["ridge_indices"][frame_rows])
            x_positions = track_data["ridge_centers_x"][frame_rows][ridge_order]
            y_positions = track_data["ridge_centers_y"][frame_rows][ridge_order]
            tube_positions[int(track_id)][int(frame_index)] = np.column_stack((x_positions, y_positions))
            tube_angles[int(track_id)][int(frame_index)] = track_data["ridge_angles"][frame_rows][ridge_order]
    return tube_positions, tube_angles


def load_and_merge_tube_geometry(track_paths):
    """Return combined ridge-center positions and angles from several track files, offsetting track ids so they never collide."""
    merged_positions = {}
    merged_angles = {}
    track_id_offset = 0
    for track_path in track_paths:
        track_data = np.load(track_path, allow_pickle=False)
        tube_positions, tube_angles = load_tube_geometry(track_data)
        for track_id in sorted(tube_positions):
            merged_positions[track_id + track_id_offset] = tube_positions[track_id]
            merged_angles[track_id + track_id_offset] = tube_angles[track_id]
        track_id_offset += max(tube_positions) + 1
    return merged_positions, merged_angles


def principal_tube_axis(positions):
    """Return the undirected unit vector along the largest spatial variation of the ridge centers."""
    centered_positions = positions - positions.mean(axis=0)
    covariance = centered_positions.T @ centered_positions
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    return eigenvectors[:, np.argmax(eigenvalues)]


def sample_vector_field(vector_field_x, vector_field_y, position):
    """Interpolate a two-dimensional vector field at one floating-point image position."""
    sampling_coordinates = np.array([[position[1]], [position[0]]])
    sampled_x = map_coordinates(vector_field_x, sampling_coordinates, order=1)[0]
    sampled_y = map_coordinates(vector_field_y, sampling_coordinates, order=1)[0]
    return np.array([sampled_x, sampled_y])


def average_projected_gradient_over_tube(positions, angles, gradient_x, gradient_y):
    """Return the tube-length-weighted mean of u * (u dot gradient) across every ridge."""
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    sampling_coordinates = np.vstack((positions[:, 1], positions[:, 0]))
    gradients = np.column_stack(
        (map_coordinates(gradient_x, sampling_coordinates, order=1), map_coordinates(gradient_y, sampling_coordinates, order=1))
    )
    projected_gradients = directions * np.sum(directions * gradients, axis=1)[:, np.newaxis]
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    ridge_weights = np.concatenate(([segment_lengths[0] / 2.0], (segment_lengths[:-1] + segment_lengths[1:]) / 2.0, [segment_lengths[-1] / 2.0]))
    return np.average(projected_gradients, axis=0, weights=ridge_weights)


def endpoint_activity_gradient(positions, illumination_image):
    """Return the endpoint illumination difference per tube length as a directed image-plane vector."""
    endpoint_coordinates = np.array([[positions[0, 1], positions[-1, 1]], [positions[0, 0], positions[-1, 0]]])
    endpoint_illuminations = map_coordinates(illumination_image, endpoint_coordinates, order=1)
    endpoint_direction = positions[-1] - positions[0]
    endpoint_direction /= np.linalg.norm(endpoint_direction)
    illumination_difference_per_length = (endpoint_illuminations[-1] - endpoint_illuminations[0]) / polyline_length(positions)
    return endpoint_direction * illumination_difference_per_length


def signed_angle(reference_vector, measured_vector):
    """Return the signed image-plane angle from one vector to another in radians."""
    cross_product = reference_vector[0] * measured_vector[1] - reference_vector[1] * measured_vector[0]
    return np.arctan2(cross_product, np.dot(reference_vector, measured_vector))


def fit_line_through_origin(x_values, y_values):
    """Return the slope and through-origin R-squared for y = slope * x."""
    slope = np.dot(x_values, y_values) / np.dot(x_values, x_values)
    residuals = y_values - slope * x_values
    r_squared = 1.0 - np.dot(residuals, residuals) / np.dot(y_values, y_values)
    return slope, r_squared


def calculate_lagged_mean_squared_displacements(tube_positions_by_track, maximum_lag):
    """Return raw-center mean-squared displacements along and across each tube for every frame lag."""
    lag_values = np.arange(1, maximum_lag + 1)
    parallel_mean_squared_displacements = []
    perpendicular_mean_squared_displacements = []
    parallel_standard_errors = []
    perpendicular_standard_errors = []
    displacement_counts = []
    for lag in lag_values:
        squared_parallel_displacements = []
        squared_perpendicular_displacements = []
        for frame_positions in tube_positions_by_track.values():
            available_frames = set(frame_positions)
            for current_frame in sorted(frame_positions):
                next_frame = current_frame + lag
                if not all(frame_index in available_frames for frame_index in range(current_frame, next_frame + 1)):
                    continue
                current_center = polyline_center(frame_positions[current_frame])
                next_center = polyline_center(frame_positions[next_frame])
                current_axis = principal_tube_axis(frame_positions[current_frame])
                next_axis = principal_tube_axis(frame_positions[next_frame])
                if np.dot(current_axis, next_axis) < 0.0:
                    next_axis = -next_axis
                mean_axis = current_axis + next_axis
                mean_axis /= np.linalg.norm(mean_axis)
                perpendicular_axis = np.array([-mean_axis[1], mean_axis[0]])
                displacement = next_center - current_center
                squared_parallel_displacements.append(np.dot(displacement, mean_axis) ** 2)
                squared_perpendicular_displacements.append(np.dot(displacement, perpendicular_axis) ** 2)
        squared_parallel_displacements = np.asarray(squared_parallel_displacements)
        squared_perpendicular_displacements = np.asarray(squared_perpendicular_displacements)
        parallel_mean_squared_displacements.append(np.mean(squared_parallel_displacements))
        perpendicular_mean_squared_displacements.append(np.mean(squared_perpendicular_displacements))
        parallel_standard_errors.append(np.std(squared_parallel_displacements, ddof=1) / np.sqrt(len(squared_parallel_displacements)))
        perpendicular_standard_errors.append(np.std(squared_perpendicular_displacements, ddof=1) / np.sqrt(len(squared_perpendicular_displacements)))
        displacement_counts.append(len(squared_parallel_displacements))
    return (
        lag_values,
        np.asarray(parallel_mean_squared_displacements),
        np.asarray(perpendicular_mean_squared_displacements),
        np.asarray(parallel_standard_errors),
        np.asarray(perpendicular_standard_errors),
        np.asarray(displacement_counts),
    )


def coefficient_of_determination(measured_values, fitted_values):
    """Return the ordinary coefficient of determination around the measured mean."""
    residual_sum = np.sum((measured_values - fitted_values) ** 2)
    total_sum = np.sum((measured_values - np.mean(measured_values)) ** 2)
    return 1.0 - residual_sum / total_sum


def calculate_direct_drift_measurements(tube_positions_by_track, gradient_x, gradient_y, frame_lag):
    """Return raw lagged velocities and the mean projected gradient across every uninterrupted interval."""
    projected_gradient_magnitudes = []
    parallel_velocities = []
    perpendicular_velocities = []
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
            mean_projected_gradient = np.mean(projected_gradients, axis=0)
            projected_gradient_magnitude = np.linalg.norm(mean_projected_gradient)
            projected_gradient_direction = mean_projected_gradient / projected_gradient_magnitude
            perpendicular_direction = np.array([-projected_gradient_direction[1], projected_gradient_direction[0]])
            displacement = polyline_center(frame_positions[next_frame]) - polyline_center(frame_positions[current_frame])
            velocity = displacement / frame_lag
            projected_gradient_magnitudes.append(projected_gradient_magnitude)
            parallel_velocities.append(np.dot(velocity, projected_gradient_direction))
            perpendicular_velocities.append(np.dot(velocity, perpendicular_direction))
    return np.asarray(projected_gradient_magnitudes), np.asarray(parallel_velocities), np.asarray(perpendicular_velocities)


def equal_count_bin_statistics(x_values, y_values, bin_count):
    """Return means and standard errors after sorting measurements into equally populated x bins."""
    sorted_indices = np.argsort(x_values)
    index_bins = np.array_split(sorted_indices, bin_count)
    mean_x = np.asarray([np.mean(x_values[index_bin]) for index_bin in index_bins])
    mean_y = np.asarray([np.mean(y_values[index_bin]) for index_bin in index_bins])
    standard_error_y = np.asarray([np.std(y_values[index_bin], ddof=1) / np.sqrt(len(index_bin)) for index_bin in index_bins])
    return mean_x, mean_y, standard_error_y


def interpolate_track_geometry(frame_positions):
    """Return centers and undirected tube axes on every frame between the first and last measurement."""
    measured_frames = np.asarray(sorted(frame_positions))
    measured_centers = np.asarray([polyline_center(frame_positions[frame_index]) for frame_index in measured_frames])
    measured_axes = np.asarray([principal_tube_axis(frame_positions[frame_index]) for frame_index in measured_frames])
    measured_angles = np.arctan2(measured_axes[:, 1], measured_axes[:, 0])
    unwrapped_doubled_angles = np.unwrap(2.0 * measured_angles)

    full_frames = np.arange(measured_frames[0], measured_frames[-1] + 1)
    interpolated_centers_x = np.interp(full_frames, measured_frames, measured_centers[:, 0])
    interpolated_centers_y = np.interp(full_frames, measured_frames, measured_centers[:, 1])
    interpolated_centers = np.column_stack((interpolated_centers_x, interpolated_centers_y))
    interpolated_angles = 0.5 * np.interp(full_frames, measured_frames, unwrapped_doubled_angles)
    interpolated_axes = np.column_stack((np.cos(interpolated_angles), np.sin(interpolated_angles)))
    measured_frame_mask = np.isin(full_frames, measured_frames)
    return full_frames, interpolated_centers, interpolated_axes, measured_frame_mask


if __name__ == "__main__":
    track_globs = ["test_outputs/test_1_tube_tracks*.npz", "test_outputs/test_2_tube_tracks_*.npz", "test_outputs/test_3_tube_tracks_*.npz"]  # One glob per movie; the newest match of each is pooled (all share background.tif).
    track_paths = [newest_track_file(track_glob) for track_glob in track_globs]  # Newest track file resolved for every movie in the pool.
    background_path = Path("test_data/background.tif")  # Read-only image containing the illumination profile shared by both movies.
    output_path = Path("test_outputs/test_combined_motion_analysis.png")  # Single figure written by this analysis.
    noise_output_path = Path("test_outputs/test_combined_noise_analysis.png")  # Lag-time noise analysis written separately.
    drift_output_path = Path("test_outputs/test_combined_direct_drift_analysis.png")  # Direct raw-displacement drift test written separately.
    excluded_track_ids = set()  # Tracks omitted from this analysis.
    background_smoothing_sigma = 30.0  # Gaussian standard deviation in pixels used before calculating the illumination gradient.
    background_low_percentile = 1.0  # Background-image percentile displayed as black in the trajectory panel.
    background_high_percentile = 99.5  # Background-image percentile displayed as white in the trajectory panel.
    maximum_noise_lag = 15  # Largest frame separation used for the raw-center mean-squared-displacement analysis.
    noise_figure_size = (12, 5)  # Width and height of the noise-analysis figure in inches.
    direct_drift_frame_lag = 11  # Raw frame separation used for the direct drift-versus-gradient fit.
    direct_drift_bin_count = 10  # Number of equal-count gradient groups used only to display mean velocities and uncertainty.
    drift_figure_size = (12, 5)  # Width and height of the direct-drift figure in inches.
    shortest_tube_color = "#2166ac"  # Blue assigned to the shortest measured tube.
    longest_tube_color = "#b2182b"  # Red assigned to the longest measured tube.
    figure_size = (22, 12)  # Width and height of the saved figure in inches.
    output_dpi = 180  # Resolution of the saved figure in dots per inch.

    print(f"Reading tracks from {', '.join(str(track_path) for track_path in track_paths)}...")
    tube_positions_by_track, tube_angles_by_track = load_and_merge_tube_geometry(track_paths)
    for excluded_track_id in excluded_track_ids:
        del tube_positions_by_track[excluded_track_id]
        del tube_angles_by_track[excluded_track_id]
    mean_tube_lengths = {
        track_id: np.mean([polyline_length(positions) for positions in frame_positions.values()])
        for track_id, frame_positions in tube_positions_by_track.items()
    }
    print(f"Excluded tracks from the analysis: {sorted(excluded_track_ids)}")
    print(f"Reading illumination profile from {background_path}...")
    background = tifffile.imread(background_path).astype(np.float32)
    smoothed_background = gaussian_filter(background, background_smoothing_sigma)
    illumination_gradient_y, illumination_gradient_x = np.gradient(smoothed_background)

    transition_track_ids = []
    axis_velocity_angles = []
    velocity_gradient_angles = []
    projected_gradient_magnitudes = []
    velocities_along_projected_gradient = []
    velocities_perpendicular_to_projected_gradient = []
    whole_tube_gradient_magnitudes = []
    velocities_along_whole_tube_gradient = []
    velocities_perpendicular_to_whole_tube_gradient = []
    endpoint_gradient_magnitudes = []
    velocities_along_endpoint_gradient = []
    trajectory_centers = {}
    excluded_undefined_velocities = 0
    excluded_short_tracks = 0

    for track_id, frame_positions in tube_positions_by_track.items():
        frame_angles = tube_angles_by_track[track_id]
        available_frames = set(frame_positions)
        measured_frames = np.asarray(sorted(frame_positions))
        measured_centers = np.asarray([polyline_center(frame_positions[frame_index]) for frame_index in measured_frames])
        trajectory_centers[track_id] = (measured_frames, measured_centers)
        track_sample_count = 0
        for current_frame in measured_frames:
            # The bare central difference needs both immediate neighbors actually measured; interpolated gap frames are never used.
            if current_frame - 1 not in available_frames or current_frame + 1 not in available_frames:
                continue
            velocity = (polyline_center(frame_positions[current_frame + 1]) - polyline_center(frame_positions[current_frame - 1])) / 2.0
            speed = np.linalg.norm(velocity)
            positions = frame_positions[current_frame]
            tube_axis = principal_tube_axis(positions)
            illumination_gradient = sample_vector_field(illumination_gradient_x, illumination_gradient_y, polyline_center(positions))
            gradient_magnitude = np.linalg.norm(illumination_gradient)
            if speed == 0.0 or gradient_magnitude == 0.0:
                excluded_undefined_velocities += 1
                continue

            axis_velocity_cosine = np.abs(np.dot(tube_axis, velocity)) / speed
            gradient_along_tube = np.dot(tube_axis, illumination_gradient)
            projected_gradient = tube_axis * gradient_along_tube
            projected_gradient_magnitude = np.linalg.norm(projected_gradient)
            if projected_gradient_magnitude == 0.0:
                excluded_undefined_velocities += 1
                continue
            projected_gradient_direction = projected_gradient / projected_gradient_magnitude
            perpendicular_direction = np.array([-projected_gradient_direction[1], projected_gradient_direction[0]])
            whole_tube_gradient = average_projected_gradient_over_tube(positions, frame_angles[current_frame], illumination_gradient_x, illumination_gradient_y)
            whole_tube_gradient_magnitude = np.linalg.norm(whole_tube_gradient)
            if whole_tube_gradient_magnitude == 0.0:
                excluded_undefined_velocities += 1
                continue
            whole_tube_gradient_direction = whole_tube_gradient / whole_tube_gradient_magnitude
            whole_tube_perpendicular_direction = np.array([-whole_tube_gradient_direction[1], whole_tube_gradient_direction[0]])
            endpoint_gradient = endpoint_activity_gradient(positions, smoothed_background)
            endpoint_gradient_magnitude = np.linalg.norm(endpoint_gradient)
            if endpoint_gradient_magnitude == 0.0:
                excluded_undefined_velocities += 1
                continue
            endpoint_gradient_direction = endpoint_gradient / endpoint_gradient_magnitude
            transition_track_ids.append(track_id)
            axis_velocity_angles.append(np.arccos(np.clip(axis_velocity_cosine, 0.0, 1.0)))
            velocity_gradient_angles.append(signed_angle(illumination_gradient, velocity))
            projected_gradient_magnitudes.append(projected_gradient_magnitude)
            velocities_along_projected_gradient.append(np.dot(velocity, projected_gradient_direction))
            velocities_perpendicular_to_projected_gradient.append(np.dot(velocity, perpendicular_direction))
            whole_tube_gradient_magnitudes.append(whole_tube_gradient_magnitude)
            velocities_along_whole_tube_gradient.append(np.dot(velocity, whole_tube_gradient_direction))
            velocities_perpendicular_to_whole_tube_gradient.append(np.dot(velocity, whole_tube_perpendicular_direction))
            endpoint_gradient_magnitudes.append(endpoint_gradient_magnitude)
            velocities_along_endpoint_gradient.append(np.dot(velocity, endpoint_gradient_direction))
            track_sample_count += 1
        if track_sample_count == 0:
            excluded_short_tracks += 1

    transition_track_ids = np.asarray(transition_track_ids)
    axis_velocity_angles = np.rad2deg(axis_velocity_angles)
    velocity_gradient_angles = np.abs(np.rad2deg(velocity_gradient_angles))
    projected_gradient_magnitudes = np.asarray(projected_gradient_magnitudes)
    velocities_along_projected_gradient = np.asarray(velocities_along_projected_gradient)
    velocities_perpendicular_to_projected_gradient = np.asarray(velocities_perpendicular_to_projected_gradient)
    whole_tube_gradient_magnitudes = np.asarray(whole_tube_gradient_magnitudes)
    velocities_along_whole_tube_gradient = np.asarray(velocities_along_whole_tube_gradient)
    velocities_perpendicular_to_whole_tube_gradient = np.asarray(velocities_perpendicular_to_whole_tube_gradient)
    endpoint_gradient_magnitudes = np.asarray(endpoint_gradient_magnitudes)
    velocities_along_endpoint_gradient = np.asarray(velocities_along_endpoint_gradient)
    track_count = len(tube_positions_by_track)
    transition_count = len(axis_velocity_angles)
    median_axis_velocity_angle = np.median(axis_velocity_angles)
    fraction_within_30_degrees_of_axis = np.mean(axis_velocity_angles <= 30.0)
    median_velocity_gradient_angle = np.median(np.abs(velocity_gradient_angles))
    fraction_moving_up_gradient = np.mean(np.abs(velocity_gradient_angles) < 90.0)
    gradient_slope, gradient_r_squared = fit_line_through_origin(projected_gradient_magnitudes, velocities_along_projected_gradient)
    endpoint_slope, endpoint_r_squared = fit_line_through_origin(endpoint_gradient_magnitudes, velocities_along_endpoint_gradient)
    whole_tube_slope, whole_tube_r_squared = fit_line_through_origin(whole_tube_gradient_magnitudes, velocities_along_whole_tube_gradient)
    mobility_track_ids = np.unique(transition_track_ids)
    mobility_tube_lengths = np.asarray([mean_tube_lengths[track_id] for track_id in mobility_track_ids])
    tube_mobilities = np.asarray(
        [
            fit_line_through_origin(
                whole_tube_gradient_magnitudes[transition_track_ids == track_id], velocities_along_whole_tube_gradient[transition_track_ids == track_id]
            )[0]
            for track_id in mobility_track_ids
        ]
    )
    mobility_length_correlation = np.corrcoef(mobility_tube_lengths, tube_mobilities)[0, 1]
    mobility_length_slope, mobility_length_intercept = np.polyfit(mobility_tube_lengths, tube_mobilities, 1)
    print(f"Analyzing {transition_count} bare central-difference velocities across {track_count} tracks.")
    print(f"Excluded {excluded_short_tracks} tracks too short for any central difference and {excluded_undefined_velocities} undefined velocity measurements.")
    print(
        f"Median tube-axis/velocity angle={median_axis_velocity_angle:.1f} degrees; "
        f"fraction within 30 degrees={fraction_within_30_degrees_of_axis:.3f}."
    )
    print(
        f"Median velocity/gradient angle={median_velocity_gradient_angle:.1f} degrees; "
        f"fraction moving up-gradient={fraction_moving_up_gradient:.3f}."
    )
    print(f"Primary whole-tube gradient fit through zero: slope={whole_tube_slope:.4f}, R-squared={whole_tube_r_squared:.3f}.")
    print(f"Endpoint tip-minus-bottom fit through zero: slope={endpoint_slope:.4f}, R-squared={endpoint_r_squared:.3f}.")
    print(f"Legacy center-point gradient fit through zero: slope={gradient_slope:.4f}, R-squared={gradient_r_squared:.3f}.")
    print(f"Tube mobility versus length: Pearson correlation={mobility_length_correlation:.3f} across {len(tube_mobilities)} tubes.")

    figure, axes = plt.subplots(2, 3, figsize=figure_size, constrained_layout=True)
    axes = axes.ravel()
    background_minimum, background_maximum = np.percentile(background, [background_low_percentile, background_high_percentile])
    axes[0].imshow(background, cmap="gray", vmin=background_minimum, vmax=background_maximum, interpolation="nearest")
    tube_length_minimum = min(mean_tube_lengths.values())
    tube_length_maximum = max(mean_tube_lengths.values())
    tube_length_color_map = LinearSegmentedColormap.from_list("tube_length", [shortest_tube_color, longest_tube_color])
    tube_length_normalization = Normalize(tube_length_minimum, tube_length_maximum)
    track_colors = {
        track_id: tube_length_color_map(tube_length_normalization(mean_tube_lengths[track_id])) for track_id in tube_positions_by_track
    }
    for track_id, (measured_frames, centers) in trajectory_centers.items():
        color = track_colors[track_id]
        for position_index in range(len(centers) - 1):
            line_style = "-" if measured_frames[position_index + 1] - measured_frames[position_index] == 1 else ":"
            axes[0].plot(centers[position_index : position_index + 2, 0], centers[position_index : position_index + 2, 1], color=color, linestyle=line_style)
        axes[0].scatter(centers[:, 0], centers[:, 1], s=4, color=color)
        axes[0].text(centers[0, 0], centers[0, 1], f"T{track_id}", color=color, fontsize=8)
    axes[0].set_title("Tracked tube centers over illumination profile")
    axes[0].set_xlabel("x position (pixels)")
    axes[0].set_ylabel("y position (pixels)")

    for track_id in tube_positions_by_track:
        track_rows = transition_track_ids == track_id
        if not np.any(track_rows):
            continue
        axes[1].scatter(
            whole_tube_gradient_magnitudes[track_rows], velocities_along_whole_tube_gradient[track_rows], color=track_colors[track_id], s=12, alpha=0.6
        )
        axes[2].scatter(
            whole_tube_gradient_magnitudes[track_rows], velocities_perpendicular_to_whole_tube_gradient[track_rows], color=track_colors[track_id], s=12, alpha=0.6
        )
        axes[3].scatter(
            endpoint_gradient_magnitudes[track_rows], velocities_along_endpoint_gradient[track_rows], color=track_colors[track_id], s=12, alpha=0.6
        )
        axes[4].scatter(
            projected_gradient_magnitudes[track_rows], velocities_along_projected_gradient[track_rows], color=track_colors[track_id], s=12, alpha=0.6
        )
    whole_tube_fit_x = np.array([0.0, whole_tube_gradient_magnitudes.max()])
    axes[1].plot(whole_tube_fit_x, whole_tube_slope * whole_tube_fit_x, color="black", linewidth=1.5)
    axes[1].axhline(0.0, color="gray", linewidth=0.7)
    axes[1].set_title(rf"Primary whole-tube gradient: $\mu={whole_tube_slope:.3f}$, $R_0^2={whole_tube_r_squared:.3f}$")
    axes[1].set_xlabel(r"Whole-tube along-axis gradient $|\langle u(u \cdot \nabla I)\rangle_L|$ (intensity/pixel)")
    axes[1].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")

    axes[2].axhline(0.0, color="gray", linewidth=0.7)
    axes[2].set_title(r"Perpendicular velocity $v_{\perp}$ (whole-tube gradient)")
    axes[2].set_xlabel(r"Whole-tube along-axis gradient $|\langle u(u \cdot \nabla I)\rangle_L|$ (intensity/pixel)")
    axes[2].set_ylabel(r"$v_{\perp}$ (pixels/frame)")

    endpoint_fit_x = np.array([0.0, endpoint_gradient_magnitudes.max()])
    axes[3].plot(endpoint_fit_x, endpoint_slope * endpoint_fit_x, color="black", linewidth=1.5)
    axes[3].axhline(0.0, color="gray", linewidth=0.7)
    axes[3].set_title(rf"Endpoint tip-minus-bottom per length: $\mu={endpoint_slope:.3f}$, $R_0^2={endpoint_r_squared:.3f}$")
    axes[3].set_xlabel(r"$|I_{\mathrm{end},2}-I_{\mathrm{end},1}|/L$ (intensity/pixel)")
    axes[3].set_ylabel("Velocity toward the brighter endpoint (pixels/frame)")

    gradient_fit_x = np.array([0.0, projected_gradient_magnitudes.max()])
    axes[4].plot(gradient_fit_x, gradient_slope * gradient_fit_x, color="black", linewidth=1.5)
    axes[4].axhline(0.0, color="gray", linewidth=0.7)
    axes[4].set_title(rf"Legacy center-point gradient: $\mu={gradient_slope:.3f}$, $R_0^2={gradient_r_squared:.3f}$")
    axes[4].set_xlabel(r"Center-point projected gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    axes[4].set_ylabel(r"$v_{\parallel}$ (pixels/frame)")

    for track_id, tube_length, tube_mobility in zip(mobility_track_ids, mobility_tube_lengths, tube_mobilities):
        axes[5].scatter(tube_length, tube_mobility, color=track_colors[track_id], s=35)
        axes[5].text(tube_length, tube_mobility, f" T{track_id}", color=track_colors[track_id], fontsize=8)
    mobility_fit_x = np.array([mobility_tube_lengths.min(), mobility_tube_lengths.max()])
    axes[5].plot(mobility_fit_x, mobility_length_intercept + mobility_length_slope * mobility_fit_x, color="black", linewidth=1.5)
    axes[5].set_title(rf"Tube mobility versus length: $r={mobility_length_correlation:.3f}$")
    axes[5].set_xlabel("Mean tube length (pixels)")
    axes[5].set_ylabel(r"Tube mobility $\mu$ (pixels$^2$/intensity/frame)")

    figure.suptitle(f"Nanotube motion: {track_count} tracks, {transition_count} bare central-difference velocities")
    tube_length_color_bar = figure.colorbar(
        plt.cm.ScalarMappable(norm=tube_length_normalization, cmap=tube_length_color_map), ax=list(axes), orientation="horizontal", fraction=0.03, pad=0.08, aspect=50
    )
    tube_length_color_bar.set_label("Mean tube length (pixels)")
    figure.savefig(output_path, dpi=output_dpi)
    plt.close(figure)
    print(f"Saved motion analysis figure to {output_path}.")

    print(f"Calculating raw-center mean-squared displacements over frame lags 1 through {maximum_noise_lag}...")
    (
        noise_lags,
        parallel_mean_squared_displacements,
        perpendicular_mean_squared_displacements,
        parallel_standard_errors,
        perpendicular_standard_errors,
        displacement_counts,
    ) = calculate_lagged_mean_squared_displacements(tube_positions_by_track, maximum_noise_lag)
    perpendicular_slope, perpendicular_intercept = np.polyfit(noise_lags, perpendicular_mean_squared_displacements, 1)
    perpendicular_fit = perpendicular_intercept + perpendicular_slope * noise_lags
    perpendicular_fit_r_squared = coefficient_of_determination(perpendicular_mean_squared_displacements, perpendicular_fit)
    parallel_fit_matrix = np.column_stack((noise_lags**2, noise_lags, np.ones_like(noise_lags)))
    parallel_quadratic_coefficient, parallel_linear_coefficient, parallel_intercept = np.linalg.lstsq(
        parallel_fit_matrix, parallel_mean_squared_displacements, rcond=None
    )[0]
    parallel_fit = parallel_fit_matrix @ np.array([parallel_quadratic_coefficient, parallel_linear_coefficient, parallel_intercept])
    parallel_fit_r_squared = coefficient_of_determination(parallel_mean_squared_displacements, parallel_fit)
    perpendicular_diffusion = perpendicular_slope / 2.0
    parallel_diffusion = parallel_linear_coefficient / 2.0
    root_mean_squared_drift = np.sqrt(parallel_quadratic_coefficient)
    print(
        f"Perpendicular MSD fit: D={perpendicular_diffusion:.3f} pixels^2/frame, intercept={perpendicular_intercept:.3f} pixels^2, "
        f"R-squared={perpendicular_fit_r_squared:.3f}; the negative intercept does not support a localization-precision estimate."
    )
    print(
        f"Parallel MSD fit: RMS drift={root_mean_squared_drift:.3f} pixels/frame, D={parallel_diffusion:.3f} pixels^2/frame, "
        f"R-squared={parallel_fit_r_squared:.3f}."
    )

    noise_figure, noise_axes = plt.subplots(1, 2, figsize=noise_figure_size, constrained_layout=True)
    noise_axes[0].errorbar(
        noise_lags, parallel_mean_squared_displacements, yerr=parallel_standard_errors, color="#2166ac", marker="o", linestyle="none", capsize=2
    )
    noise_axes[0].plot(noise_lags, parallel_fit, color="black", linewidth=1.5)
    noise_axes[0].set_title(rf"Parallel MSD: $R^2={parallel_fit_r_squared:.3f}$")
    noise_axes[0].set_xlabel("Frame lag")
    noise_axes[0].set_ylabel(r"$\langle \Delta r_{\parallel}^2 \rangle$ (pixels$^2$)")
    noise_axes[1].errorbar(
        noise_lags, perpendicular_mean_squared_displacements, yerr=perpendicular_standard_errors, color="#b2182b", marker="o", linestyle="none", capsize=2
    )
    noise_axes[1].plot(noise_lags, perpendicular_fit, color="black", linewidth=1.5)
    noise_axes[1].set_title(rf"Perpendicular MSD: $R^2={perpendicular_fit_r_squared:.3f}$")
    noise_axes[1].set_xlabel("Frame lag")
    noise_axes[1].set_ylabel(r"$\langle \Delta r_{\perp}^2 \rangle$ (pixels$^2$)")
    noise_figure.suptitle(f"Raw-center displacement noise; pair counts {displacement_counts[0]} to {displacement_counts[-1]}")
    noise_figure.savefig(noise_output_path, dpi=output_dpi)
    plt.close(noise_figure)
    print(f"Saved lag-time noise analysis to {noise_output_path}.")

    print(f"Fitting raw {direct_drift_frame_lag}-frame displacements directly to the projected illumination gradient...")
    direct_gradients, direct_parallel_velocities, direct_perpendicular_velocities = calculate_direct_drift_measurements(
        tube_positions_by_track, illumination_gradient_x, illumination_gradient_y, direct_drift_frame_lag
    )
    direct_mobility, direct_r_squared = fit_line_through_origin(direct_gradients, direct_parallel_velocities)
    direct_parallel_residuals = direct_parallel_velocities - direct_mobility * direct_gradients
    direct_parallel_diffusion = direct_drift_frame_lag * np.var(direct_parallel_residuals, ddof=1) / 2.0
    direct_perpendicular_diffusion = direct_drift_frame_lag * np.var(direct_perpendicular_velocities, ddof=1) / 2.0
    binned_gradients, binned_parallel_velocities, binned_parallel_errors = equal_count_bin_statistics(
        direct_gradients, direct_parallel_velocities, direct_drift_bin_count
    )
    _, binned_perpendicular_velocities, binned_perpendicular_errors = equal_count_bin_statistics(
        direct_gradients, direct_perpendicular_velocities, direct_drift_bin_count
    )
    print(
        f"Direct drift fit: mobility={direct_mobility:.4f}, R-squared={direct_r_squared:.3f}, "
        f"parallel residual D={direct_parallel_diffusion:.3f} pixels^2/frame, perpendicular D={direct_perpendicular_diffusion:.3f} pixels^2/frame."
    )

    drift_figure, drift_axes = plt.subplots(1, 2, figsize=drift_figure_size, constrained_layout=True)
    direct_fit_x = np.array([0.0, direct_gradients.max()])
    drift_axes[0].scatter(direct_gradients, direct_parallel_velocities, color="lightgray", s=8, alpha=0.35)
    drift_axes[0].errorbar(
        binned_gradients, binned_parallel_velocities, yerr=binned_parallel_errors, color="#2166ac", marker="o", linestyle="none", capsize=3
    )
    drift_axes[0].plot(direct_fit_x, direct_mobility * direct_fit_x, color="black", linewidth=1.5)
    drift_axes[0].axhline(0.0, color="gray", linewidth=0.7)
    drift_axes[0].set_title(rf"Direct parallel drift: $R_0^2={direct_r_squared:.3f}$")
    drift_axes[0].set_xlabel(r"Mean projected gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    drift_axes[0].set_ylabel(rf"Raw $v_{{\parallel}}$ over {direct_drift_frame_lag} frames (pixels/frame)")
    drift_axes[1].scatter(direct_gradients, direct_perpendicular_velocities, color="lightgray", s=8, alpha=0.35)
    drift_axes[1].errorbar(
        binned_gradients,
        binned_perpendicular_velocities,
        yerr=binned_perpendicular_errors,
        color="#b2182b",
        marker="o",
        linestyle="none",
        capsize=3,
    )
    drift_axes[1].axhline(0.0, color="black", linewidth=1.5)
    drift_axes[1].set_title("Direct perpendicular drift")
    drift_axes[1].set_xlabel(r"Mean projected gradient $|u \cdot \nabla I|$ (intensity/pixel)")
    drift_axes[1].set_ylabel(rf"Raw $v_{{\perp}}$ over {direct_drift_frame_lag} frames (pixels/frame)")
    drift_figure.suptitle(f"Raw {direct_drift_frame_lag}-frame displacement model; colored points are {direct_drift_bin_count} equal-count means")
    drift_figure.savefig(drift_output_path, dpi=output_dpi)
    plt.close(drift_figure)
    print(f"Saved direct drift analysis to {drift_output_path}.")
