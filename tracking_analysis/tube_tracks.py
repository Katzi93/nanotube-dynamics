"""Load saved tube-track datasets and reduce them to one geometric record per tube for motion analysis.

The tracker saves each movie as a flat columnar ``.npz`` in which every fitted ridge is one row and every ridge
property is a parallel array keyed by (track, frame, ridge index). ``load_datasets`` reads one or more of those files
by dataset name from a common directory and stacks them into a single pandas DataFrame with one row per ridge, labeled
by dataset name. Track ids are only unique within one movie, so the identity of a ridge across the pool is the tuple
(dataset, track, frame, ridge).

``build_tube_tracks`` collapses that per-ridge table into one row per tube: its center, its orientation, its two tip
locations, and its length. The orientation is the axial mean of the ridge angles. Ridge angles are undirected and
stored in [0, pi), so a naive arithmetic mean is wrong for a tube whose ridges straddle 0 and pi; the mean is therefore
taken on the doubled angles, mean orientation = 0.5 * atan2(mean sin 2*theta, mean cos 2*theta), which is the correct
mean direction of a set of undirected lines. The center is the mean of the ridge centers, the two tips are the centers
of the first and last ridge along the tube (ordered by ridge index), and the length is the contour length of the
ridge-center chain. The Euclidean tip-to-tip distance is not stored here; it is computed only for the light gradient.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import tifffile
from scipy.ndimage import map_coordinates


def load_datasets(dataset_names, data_directory="tracking_data"):
    """Return one ridge DataFrame pooled from several saved datasets, labeled by dataset name."""
    # Maps the tracker's saved ridge-column names to the shorter names used throughout the analysis.
    ridge_column_names = {
        "ridge_track_ids": "track",
        "ridge_frame_indices": "frame",
        "ridge_indices": "ridge",
        "ridge_centers_x": "x",
        "ridge_centers_y": "y",
        "ridge_angles": "angle",
        "ridge_perpendicular_shifts": "perp_shift",
        "ridge_perpendicular_fwhms": "perp_fwhm",
        "ridge_amplitudes": "amplitude",
        "ridge_backgrounds": "background",
        "ridge_errors": "error",
        "ridge_converged": "converged",
        "ridge_function_evaluations": "n_evals",
    }
    data_directory = Path(data_directory)
    dataset_frames = []
    for dataset_name in dataset_names:
        track_data = np.load(data_directory / f"{dataset_name}.npz", allow_pickle=False)
        ridges = pd.DataFrame({new_name: track_data[saved_name] for saved_name, new_name in ridge_column_names.items()})
        ridges.insert(0, "dataset", dataset_name)
        dataset_frames.append(ridges)
    return pd.concat(dataset_frames, ignore_index=True)


def simplify_dataframe(ridges):
    """Return the ridge DataFrame reduced to the geometry columns, dropping the per-fit diagnostics."""
    relevant_columns = ["dataset", "track", "frame", "ridge", "x", "y", "angle"]  # Fields relevant for motion analysis.
    return ridges[relevant_columns].copy()


def average_orientation_angle(angles):
    """Return the axial mean orientation in [0, pi) of a set of undirected ridge angles given in radians."""
    mean_double_angle = np.arctan2(np.mean(np.sin(2.0 * angles)), np.mean(np.cos(2.0 * angles)))
    return (0.5 * mean_double_angle) % np.pi


def contour_length(ordered_centers):
    """Return the tube length as the summed distance between consecutive ridge centers ordered along the tube."""
    return np.linalg.norm(np.diff(ordered_centers, axis=0), axis=1).sum()


def tube_center(ordered_centers):
    """Return the tube center as the mean of its ridge centers."""
    return ordered_centers.mean(axis=0)


def build_tube_tracks(ridges):
    """Return one row per tube (dataset, track, frame) with center, orientation, both tip locations, and length."""
    tube_records = []
    for (dataset, track, frame), tube in ridges.groupby(["dataset", "track", "frame"], sort=True):
        ordered_centers = tube.sort_values("ridge")[["x", "y"]].to_numpy()
        center = tube_center(ordered_centers)
        orientation = average_orientation_angle(tube["angle"].to_numpy())
        first_tip = ordered_centers[0]
        last_tip = ordered_centers[-1]
        length = contour_length(ordered_centers)
        tube_records.append({
            "dataset": dataset,
            "track": track,
            "frame": frame,
            "center_x": center[0],
            "center_y": center[1],
            "orientation": orientation,
            "tip1_x": first_tip[0],
            "tip1_y": first_tip[1],
            "tip2_x": last_tip[0],
            "tip2_y": last_tip[1],
            "length": length,
        })
    return pd.DataFrame(tube_records)


def axial_angle_difference(later_angle, earlier_angle):
    """Return the signed change between two undirected orientations, wrapped into (-pi/2, pi/2]."""
    return (later_angle - earlier_angle + np.pi / 2.0) % np.pi - np.pi / 2.0


def fill_missing_frames(tube_tracks, max_gap=1):
    """Return the tube table with short internal gaps in each track filled by interpolating the missing tube records.

    A track drops a frame whenever its tube was not detected, which leaves a hole that blocks the central-difference
    velocity on both sides. For every run of consecutive missing frames no longer than max_gap between two measured
    frames of the same track, one tube record is inserted per missing frame. Its center, tip locations, and length are
    linear interpolations of the two bracketing measured frames; its orientation is interpolated along the shortest
    axial path so that undirected angles straddling 0 and pi are handled correctly. Inserted rows carry interpolated
    = True; original rows carry interpolated = False. Gaps longer than max_gap are left open, since interpolating a
    tube across many unseen frames is not justified.
    """
    linear_columns = ["center_x", "center_y", "tip1_x", "tip1_y", "tip2_x", "tip2_y", "length"]
    tube_tracks = tube_tracks.copy()
    tube_tracks["interpolated"] = False
    filled_records = []
    for (dataset, track), track_rows in tube_tracks.groupby(["dataset", "track"], sort=False):
        track_rows = track_rows.sort_values("frame")
        frame_to_row = {row.frame: row for row in track_rows.itertuples()}
        measured_frames = sorted(frame_to_row)
        for earlier_frame, later_frame in zip(measured_frames[:-1], measured_frames[1:]):
            gap = later_frame - earlier_frame - 1
            if gap < 1 or gap > max_gap:
                continue
            earlier_row = frame_to_row[earlier_frame]
            later_row = frame_to_row[later_frame]
            orientation_step = axial_angle_difference(later_row.orientation, earlier_row.orientation)
            for missing_frame in range(earlier_frame + 1, later_frame):
                fraction = (missing_frame - earlier_frame) / (later_frame - earlier_frame)
                record = {"dataset": dataset, "track": track, "frame": missing_frame, "interpolated": True}
                for column in linear_columns:
                    record[column] = getattr(earlier_row, column) + fraction * (getattr(later_row, column) - getattr(earlier_row, column))
                record["orientation"] = (earlier_row.orientation + fraction * orientation_step) % np.pi
                filled_records.append(record)
    if filled_records:
        tube_tracks = pd.concat([tube_tracks, pd.DataFrame(filled_records)], ignore_index=True)
    return tube_tracks.sort_values(["dataset", "track", "frame"]).reset_index(drop=True)


def add_velocity(tube_tracks):
    """Return the tube table with the central-difference velocity vector and angular velocity added per track.

    For each frame whose immediate neighbors t - 1 and t + 1 were both measured in the same track, the velocity is the
    bare central difference (center(t + 1) - center(t - 1)) / 2 in pixels per frame, stored as the vector velocity_x,
    velocity_y. The angular velocity is the axial change of orientation over the same two frames divided by two, in
    radians per frame. Frames without both measured neighbors keep NaN rather than an invented value.
    """
    tube_tracks = tube_tracks.sort_values(["dataset", "track", "frame"]).reset_index(drop=True)
    velocity_x = np.full(len(tube_tracks), np.nan)
    velocity_y = np.full(len(tube_tracks), np.nan)
    angular_velocity = np.full(len(tube_tracks), np.nan)
    for _, track_rows in tube_tracks.groupby(["dataset", "track"], sort=False):
        frame_to_center = {row.frame: np.array([row.center_x, row.center_y]) for row in track_rows.itertuples()}
        frame_to_orientation = {row.frame: row.orientation for row in track_rows.itertuples()}
        for row in track_rows.itertuples():
            frame = row.frame
            if frame - 1 not in frame_to_center or frame + 1 not in frame_to_center:
                continue
            velocity = (frame_to_center[frame + 1] - frame_to_center[frame - 1]) / 2.0
            velocity_x[row.Index] = velocity[0]
            velocity_y[row.Index] = velocity[1]
            angular_velocity[row.Index] = axial_angle_difference(frame_to_orientation[frame + 1], frame_to_orientation[frame - 1]) / 2.0
    tube_tracks["velocity_x"] = velocity_x
    tube_tracks["velocity_y"] = velocity_y
    tube_tracks["angular_velocity"] = angular_velocity
    return tube_tracks


def add_light_gradient(tube_tracks, background_path):
    """Return the tube table with the tip-to-tip illumination gradient vector added as light_grad_x and light_grad_y.

    The illumination image is sampled by bilinear interpolation at both tip locations. The gradient is the finite
    difference of illumination between the tips divided by their Euclidean distance, delta_I / L, carried by the unit
    vector from tip1 toward tip2. Because delta_I = I(tip2) - I(tip1) sets the sign, the vector points from the dimmer
    tip toward the brighter tip, and its magnitude is |delta_I| / L.
    """
    background = tifffile.imread(background_path).astype(np.float64)
    tip1_illumination = map_coordinates(background, np.vstack((tube_tracks["tip1_y"], tube_tracks["tip1_x"])), order=1)
    tip2_illumination = map_coordinates(background, np.vstack((tube_tracks["tip2_y"], tube_tracks["tip2_x"])), order=1)
    tip_displacement_x = tube_tracks["tip2_x"] - tube_tracks["tip1_x"]
    tip_displacement_y = tube_tracks["tip2_y"] - tube_tracks["tip1_y"]
    tip_distance = np.hypot(tip_displacement_x, tip_displacement_y)
    illumination_difference = tip2_illumination - tip1_illumination
    tube_tracks = tube_tracks.copy()
    tube_tracks["light_grad_x"] = tip_displacement_x * illumination_difference / tip_distance**2
    tube_tracks["light_grad_y"] = tip_displacement_y * illumination_difference / tip_distance**2
    return tube_tracks


def add_projections(tube_tracks):
    """Return the tube table with the velocity split along and across the orientation axis and the projected gradient.

    The orientation axis t = (cos theta, sin theta) is undirected, so its sign is fixed by the light gradient: t is
    flipped when needed so that the projected gradient ∇I·t is nonnegative and t points toward the brighter tip. With
    that convention v_parallel = v·t is signed, positive when the tube drifts toward the brighter tip, and the
    perpendicular axis t_perp = (-t_y, t_x) gives v_perp = v·t_perp as a symmetric control that should average to zero.
    """
    tube_tracks = tube_tracks.copy()
    axis_x = np.cos(tube_tracks["orientation"])
    axis_y = np.sin(tube_tracks["orientation"])
    gradient_along_axis = tube_tracks["light_grad_x"] * axis_x + tube_tracks["light_grad_y"] * axis_y
    orientation_sign = np.where(gradient_along_axis < 0.0, -1.0, 1.0)
    axis_x *= orientation_sign
    axis_y *= orientation_sign
    tube_tracks["projected_gradient"] = tube_tracks["light_grad_x"] * axis_x + tube_tracks["light_grad_y"] * axis_y
    tube_tracks["v_parallel"] = tube_tracks["velocity_x"] * axis_x + tube_tracks["velocity_y"] * axis_y
    tube_tracks["v_perp"] = tube_tracks["velocity_x"] * (-axis_y) + tube_tracks["velocity_y"] * axis_x
    return tube_tracks


if __name__ == "__main__":
    dataset_names = ["test_1", "test_2", "test_3"]  # Canonical track datasets pooled from tracking_data/.

    background_path = Path("test_data/background.tif")  # Read-only illumination profile shared by every movie.

    ridges = simplify_dataframe(load_datasets(dataset_names))
    tube_tracks = build_tube_tracks(ridges)
    tube_tracks = add_velocity(tube_tracks)
    tube_tracks = add_light_gradient(tube_tracks, background_path)
    tube_tracks = add_projections(tube_tracks)
    speed = np.hypot(tube_tracks["velocity_x"], tube_tracks["velocity_y"])
    print(f"Built {len(tube_tracks)} tube records across {tube_tracks.groupby(['dataset', 'track']).ngroups} tracks.")
    print(f"Tube length: median={tube_tracks['length'].median():.1f}, range {tube_tracks['length'].min():.1f} to {tube_tracks['length'].max():.1f} pixels.")
    print(f"Central-difference velocities: {speed.notna().sum()} of {len(tube_tracks)} tubes; median speed={speed.median():.2f} pixels/frame.")
    print(f"Median projected gradient={tube_tracks['projected_gradient'].median():.4f} intensity/pixel.")
    print(f"Median v_parallel={tube_tracks['v_parallel'].median():.3f}, median v_perp={tube_tracks['v_perp'].median():.3f} pixels/frame.")
    print(tube_tracks[["dataset", "track", "frame", "v_parallel", "v_perp", "angular_velocity", "projected_gradient"]].head())
