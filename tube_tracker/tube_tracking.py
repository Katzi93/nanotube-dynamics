"""Track a constructed tube by generating and comparing next-frame candidates."""

from time import perf_counter

import numpy as np

from tube_tracker import configuration
from tube_tracker.ridge_detection import find_likely_ridge
from tube_tracker.tube_construction import construct_tube


def tube_positions(tube):
    """Return the ordered center positions of a constructed tube."""
    return np.asarray([ridge["center"] for ridge in tube])


def create_tube_track(track_id, frame_index, tube):
    """Create one physical tube track containing its first fitted frame."""
    initial_length = polyline_length(tube_positions(tube))
    return {
        "track_id": track_id,
        "initial_frame": frame_index,
        "tracking_started": False,
        "average_length": initial_length,
        "tubes": {frame_index: tube},
    }


def add_tube_to_track(tube_track, frame_index, new_tube):
    """Add one newly tracked frame to an existing physical tube track."""
    tracked_frame_count = len(tube_track["tubes"])
    new_tube_length = polyline_length(tube_positions(new_tube))
    tube_track["average_length"] = (tracked_frame_count * tube_track["average_length"] + new_tube_length) / (tracked_frame_count + 1)
    tube_track["tubes"][frame_index] = new_tube


def polyline_length(positions):
    """Return the total length of an ordered sequence of positions."""
    return np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()


def polyline_center(positions):
    """Return the length-weighted center of an ordered polyline."""
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    segment_centers = (positions[:-1] + positions[1:]) / 2.0
    return np.sum(segment_lengths[:, np.newaxis] * segment_centers, axis=0) / segment_lengths.sum()


def validate_candidate(previous_tube, candidate_tube, average_tube_length):
    """Return the reasons a complete candidate tube differs too much from the previous tube."""
    candidate_positions = tube_positions(candidate_tube)
    candidate_length = polyline_length(candidate_positions)
    relative_length_difference = abs(candidate_length / average_tube_length - 1.0)
    rejection_reasons = []
    if relative_length_difference > configuration.maximum_relative_length_difference:
        rejection_reasons.append(f"relative length difference {relative_length_difference:.3f} is too large")
        return rejection_reasons

    previous_positions = tube_positions(previous_tube)
    previous_center = polyline_center(previous_positions)
    candidate_center = polyline_center(candidate_positions)
    center_position_difference = np.linalg.norm(candidate_center - previous_center)
    if center_position_difference > configuration.maximum_center_position_difference:
        rejection_reasons.append(f"center position difference {center_position_difference:.2f} pixels is too large")

    previous_direction = previous_positions[-1] - previous_positions[0]
    previous_direction /= np.linalg.norm(previous_direction)
    candidate_direction = candidate_positions[-1] - candidate_positions[0]
    candidate_direction /= np.linalg.norm(candidate_direction)
    # The absolute dot product treats a tube traced in reverse order as the same undirected tube.
    direction_dot_product = np.abs(np.dot(previous_direction, candidate_direction))
    overall_angle_difference = np.arccos(np.clip(direction_dot_product, 0.0, 1.0))
    if overall_angle_difference > configuration.maximum_overall_angle_difference:
        overall_angle_difference_degrees = np.rad2deg(overall_angle_difference)
        rejection_reasons.append(f"overall angle difference {overall_angle_difference_degrees:.2f} degrees is too large")
    return rejection_reasons


def track_tube(image, previous_tube, average_tube_length, convolution_kernels, random_generator):
    """Try previous ridge centers and return the first valid next-frame tube plus the number of trials used."""
    tracking_start_time = perf_counter()
    convolution_time = 0.0
    tube_construction_time = 0.0
    validation_time = 0.0

    ridge_indices = np.arange(len(previous_tube))
    central_start = len(previous_tube) // 4
    central_stop = len(previous_tube) - central_start
    central_ridge_indices = random_generator.permutation(ridge_indices[central_start:central_stop])
    outer_ridge_indices = random_generator.permutation(np.concatenate((ridge_indices[:central_start], ridge_indices[central_stop:])))
    trial_ridge_indices = np.concatenate((central_ridge_indices, outer_ridge_indices))

    for trial_number, ridge_index in enumerate(trial_ridge_indices, start=1):
        seed_position = previous_tube[ridge_index]["center"]
        convolution_start_time = perf_counter()
        predicted_center, predicted_angle, convolution_score = find_likely_ridge(
            image,
            seed_position,
            convolution_kernels,
            configuration.convolution_search_radius,
        )
        convolution_time += perf_counter() - convolution_start_time
        if configuration.tracking_verbose:
            print(
                f"Trial {trial_number}, ridge {ridge_index}: seed=({seed_position[0]:.2f}, {seed_position[1]:.2f}), "
                f"convolution score={convolution_score:.2f}"
            )
        tube_construction_start_time = perf_counter()
        candidate_tube = construct_tube(image, predicted_center, predicted_angle)
        tube_construction_time += perf_counter() - tube_construction_start_time
        if not candidate_tube:
            if configuration.tracking_verbose:
                print(f"Trial {trial_number} rejected: no tube was constructed.")
            continue

        validation_start_time = perf_counter()
        rejection_reasons = validate_candidate(previous_tube, candidate_tube, average_tube_length)
        validation_time += perf_counter() - validation_start_time
        if rejection_reasons:
            if configuration.tracking_verbose:
                print(f"Trial {trial_number} rejected: " + "; ".join(rejection_reasons))
            continue

        new_tube = candidate_tube
        new_tube_length = polyline_length(tube_positions(new_tube))
        relative_length_difference = abs(new_tube_length / average_tube_length - 1.0)
        total_tracking_time = perf_counter() - tracking_start_time
        if configuration.tracking_verbose:
            print(
                f"Trial {trial_number} accepted: new tube length={new_tube_length:.2f}, "
                f"relative length difference={relative_length_difference:.3f}."
            )
            print(
                f"Timing: convolution={convolution_time:.2f}s, tube construction={tube_construction_time:.2f}s, "
                f"validation={validation_time:.2f}s, total={total_tracking_time:.2f}s."
            )
        return new_tube, trial_number

    total_tracking_time = perf_counter() - tracking_start_time
    if configuration.tracking_verbose:
        print(
            f"Timing: convolution={convolution_time:.2f}s, tube construction={tube_construction_time:.2f}s, "
            f"validation={validation_time:.2f}s, total={total_tracking_time:.2f}s."
        )
        print("No candidate tube was accepted in this frame.")
    return None, len(trial_ridge_indices)
