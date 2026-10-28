"""Track one tube using complete traces launched from previous-frame seeds."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import tifffile

from test_raw_image_ridge_fit import (
    extend_tube,
    find_local_initial_guess,
    fit_local_ridge,
)


def polyline_length(positions):
    """Return the total length of one ordered point chain."""
    return np.linalg.norm(np.diff(positions, axis=0), axis=1).sum()


def positions_at_length_fractions(positions, length_fractions):
    """Interpolate positions at requested fractions of total curve length."""
    segment_lengths = np.linalg.norm(np.diff(positions, axis=0), axis=1)
    cumulative_lengths = np.concatenate(([0.0], np.cumsum(segment_lengths)))
    assert cumulative_lengths[-1] > 0.0
    target_lengths = np.asarray(length_fractions) * cumulative_lengths[-1]
    return np.column_stack(
        (
            np.interp(target_lengths, cumulative_lengths, positions[:, 0]),
            np.interp(target_lengths, cumulative_lengths, positions[:, 1]),
        )
    )


def resample_polyline(positions, point_count):
    """Return equally spaced positions along an ordered point chain."""
    return positions_at_length_fractions(
        positions,
        np.linspace(0.0, 1.0, point_count),
    )


def trace_tube_from_seed(
    image,
    seed_position,
    convolution_search_radius,
    trace_parameters,
    trace_label,
    print_accepted_points,
):
    """Generate one complete bidirectional tube trace from one seed."""
    print(
        f"{trace_label}: seed=({seed_position[0]:.2f}, "
        f"{seed_position[1]:.2f})"
    )
    initial_center, initial_angle_degrees, convolution_score = (
        find_local_initial_guess(
            image,
            seed_position,
            convolution_search_radius,
            trace_parameters["convolution_median_filter_size"],
            trace_parameters["convolution_tube_width"],
            trace_parameters["convolution_angle_step_degrees"],
            trace_parameters["convolution_template_half_size"],
            trace_parameters["convolution_along_half_height_length"],
            trace_parameters["convolution_super_gaussian_power"],
            trace_parameters["convolution_side_width_multiplier"],
            trace_parameters["gaussian_fwhm_factor"],
        )
    )
    print(
        f"{trace_label}: convolution center=({initial_center[0]:.2f}, "
        f"{initial_center[1]:.2f}), angle={initial_angle_degrees:.2f}, "
        f"score={convolution_score:.2f}"
    )
    try:
        initial_result = fit_local_ridge(
            image,
            initial_center,
            initial_angle_degrees,
            trace_parameters["fitting_radius"],
            trace_parameters["initial_center_search_range"],
            trace_parameters["initial_angle_search_range_degrees"],
            trace_parameters["fitting_minimum_sigma"],
            trace_parameters["fitting_maximum_sigma"],
            trace_parameters["initial_sigma"],
            trace_parameters["initial_fitting_iterations"],
            trace_parameters["maximum_function_evaluations"],
        )
    except RuntimeError as error:
        print(f"{trace_label} rejected: {error}")
        return {"positions": np.empty((0, 2), dtype=np.float64), "length": 0.0}
    extension_arguments = (
        trace_parameters["extension_step_length"],
        trace_parameters["fitting_radius"],
        trace_parameters["extension_center_search_range"],
        trace_parameters["extension_angle_search_range_degrees"],
        trace_parameters["fitting_minimum_sigma"],
        trace_parameters["fitting_maximum_sigma"],
        trace_parameters["extension_fitting_iterations"],
        trace_parameters["maximum_function_evaluations"],
        trace_parameters["maximum_extension_points"],
        trace_parameters["minimum_forward_progress"],
        trace_parameters["maximum_prediction_correction"],
        trace_parameters["maximum_angle_change_degrees"],
        trace_parameters["minimum_amplitude_to_error"],
        trace_parameters["minimum_flat_error_reduction"],
        trace_parameters["accepted_minimum_sigma"],
        trace_parameters["accepted_maximum_sigma"],
        print_accepted_points,
    )
    backward_results = extend_tube(
        image,
        initial_result,
        -1,
        *extension_arguments,
    )
    forward_results = extend_tube(
        image,
        initial_result,
        1,
        *extension_arguments,
    )
    ordered_results = tuple(
        list(reversed(backward_results)) + [initial_result] + forward_results
    )
    positions = np.asarray([result.center for result in ordered_results])
    trace_length = polyline_length(positions) if len(positions) >= 2 else 0.0
    print(
        f"{trace_label}: {len(positions)} points, length={trace_length:.2f}"
    )
    return {"positions": positions, "length": trace_length}


def compare_candidate(reference_trace, candidate_trace, comparison_point_count):
    """Compare complete tube center position and orientation."""
    reference_positions = resample_polyline(
        reference_trace["positions"],
        comparison_point_count,
    )
    candidate_positions = resample_polyline(
        candidate_trace["positions"],
        comparison_point_count,
    )
    reference_center = reference_trace["positions"].mean(axis=0)
    candidate_center = candidate_trace["positions"].mean(axis=0)
    center_position_difference = np.linalg.norm(
        candidate_center - reference_center
    )

    reference_tangents = np.gradient(reference_positions, axis=0)
    candidate_tangents = np.gradient(candidate_positions, axis=0)
    reference_tangents /= np.linalg.norm(
        reference_tangents,
        axis=1,
        keepdims=True,
    )
    candidate_tangents /= np.linalg.norm(
        candidate_tangents,
        axis=1,
        keepdims=True,
    )
    forward_tangent_dot_products = np.abs(
        np.sum(reference_tangents * candidate_tangents, axis=1)
    )
    reversed_tangent_dot_products = np.abs(
        np.sum(reference_tangents * candidate_tangents[::-1], axis=1)
    )
    forward_orientation_difference = np.rad2deg(
        np.arccos(np.clip(forward_tangent_dot_products, 0.0, 1.0))
    ).mean()
    reversed_orientation_difference = np.rad2deg(
        np.arccos(np.clip(reversed_tangent_dot_products, 0.0, 1.0))
    ).mean()
    return {
        "trace": candidate_trace,
        "relative_length_difference": abs(
            candidate_trace["length"] / reference_trace["length"] - 1.0
        ),
        "center_position_difference": center_position_difference,
        "mean_orientation_difference_degrees": min(
            forward_orientation_difference,
            reversed_orientation_difference,
        ),
    }


def candidate_agreement_score(
    comparison,
    maximum_relative_length_difference,
    maximum_center_position_difference,
    maximum_mean_orientation_difference_degrees,
):
    """Combine normalized length, position, and orientation disagreement."""
    return (
        (
            comparison["relative_length_difference"]
            / maximum_relative_length_difference
        )
        ** 2
        + (
            comparison["center_position_difference"]
            / maximum_center_position_difference
        )
        ** 2
        + (
            comparison["mean_orientation_difference_degrees"]
            / maximum_mean_orientation_difference_degrees
        )
        ** 2
    )


def draw_labeled_trace(
    frame,
    trace_positions,
    frame_index,
    display_low_percentile,
    display_high_percentile,
    trace_color,
    trace_outline_color,
    trace_outline_width,
    trace_line_width,
    label_color,
    label_position,
    label_font_size,
):
    """Return one RGB frame with its selected trace and frame number drawn."""
    display_minimum, display_maximum = np.percentile(
        frame,
        [display_low_percentile, display_high_percentile],
    )
    assert display_maximum > display_minimum
    scaled_frame = np.clip(
        (frame - display_minimum) / (display_maximum - display_minimum),
        0.0,
        1.0,
    )
    gray_frame = np.round(255.0 * scaled_frame).astype(np.uint8)
    rgb_frame = np.repeat(gray_frame[:, :, np.newaxis], 3, axis=2)
    labeled_image = Image.fromarray(rgb_frame)
    drawing = ImageDraw.Draw(labeled_image)
    drawing.line(
        [tuple(position) for position in trace_positions],
        fill=trace_outline_color,
        width=trace_outline_width,
        joint="curve",
    )
    drawing.line(
        [tuple(position) for position in trace_positions],
        fill=trace_color,
        width=trace_line_width,
        joint="curve",
    )
    label_font = ImageFont.load_default(size=label_font_size)
    drawing.text(
        label_position,
        f"Frame {frame_index}",
        fill=label_color,
        font=label_font,
        stroke_width=2,
        stroke_fill=(0, 0, 0),
    )
    return np.asarray(labeled_image)


def write_labeled_stack(output_path, labeled_frames, playback_frames_per_second):
    """Overwrite one ImageJ-compatible RGB TIFF with completed frames."""
    tifffile.imwrite(
        output_path,
        labeled_frames,
        imagej=True,
        photometric="rgb",
        metadata={
            "axes": "TYXS",
            "fps": playback_frames_per_second,
        },
    )


if __name__ == "__main__":
    # Read-only microscopy TIFF stack containing the tracked frames.
    input_path = Path("test_data/test_2.tif")
    # Distinct labeled RGB TIFF stack for the 30-frame legacy-tracker demonstration.
    output_path = Path("test_outputs/legacy_tracker_test2_frames_070_099.tif")
    # Thirty consecutive zero-based frames from frame 70 through frame 99.
    frame_indices = tuple(range(70, 100))
    # Fractions along each old trace used as five next-frame seeds.
    automatic_seed_length_fractions = (0.15, 0.325, 0.5, 0.675, 0.85)
    # Local convolution radius in pixels around the manual first-frame click.
    manual_seed_search_radius = 8
    # Larger local convolution radius around automatic next-frame seeds.
    automatic_seed_search_radius = 15
    # Number of equally spaced positions used to compare complete traces.
    comparison_point_count = 50
    # Largest accepted fractional length change between consecutive frames.
    maximum_relative_length_difference = 0.40
    # Largest accepted tube-center movement in pixels between frames.
    maximum_center_position_difference = 25.0
    # Largest accepted mean orientation change between frames in degrees.
    maximum_mean_orientation_difference_degrees = 30.0
    # Lower raw-intensity percentile displayed as black in the TIFF stack.
    display_low_percentile = 1.0
    # Upper raw-intensity percentile displayed as white in the TIFF stack.
    display_high_percentile = 99.8
    # RGB color used for the selected tube trace.
    trace_color = (0, 255, 255)
    # RGB outline color separating the trace from bright and dark pixels.
    trace_outline_color = (0, 0, 0)
    # Width of the black trace outline in output-image pixels.
    trace_outline_width = 7
    # Tube-trace line width in output-image pixels.
    trace_line_width = 3
    # RGB color used for the frame-number label.
    label_color = (255, 255, 0)
    # Upper-left pixel position of the frame-number label.
    label_position = (20, 20)
    # Frame-number font size in output-image pixels.
    label_font_size = 36
    # Playback speed stored in the ImageJ TIFF metadata.
    playback_frames_per_second = 2.0
    # Shared settings for every complete single-frame tube trace.
    trace_parameters = {
        "convolution_median_filter_size": 3,
        "convolution_tube_width": 6.0,
        "convolution_angle_step_degrees": 2,
        "convolution_template_half_size": 36,
        "convolution_along_half_height_length": 30.0,
        "convolution_super_gaussian_power": 4.0,
        "convolution_side_width_multiplier": 2.0,
        "gaussian_fwhm_factor": 2.35482,
        "fitting_radius": 15.0,
        "extension_step_length": 6.0,
        "initial_center_search_range": 8.0,
        "initial_angle_search_range_degrees": 30.0,
        "initial_fitting_iterations": 3,
        "extension_center_search_range": 3.0,
        "extension_angle_search_range_degrees": 15.0,
        "extension_fitting_iterations": 2,
        "fitting_minimum_sigma": 0.5,
        "fitting_maximum_sigma": 8.0,
        "initial_sigma": 6.20 / 2.35482,
        "maximum_function_evaluations": 500,
        "maximum_extension_points": 100,
        "minimum_forward_progress": 3.5,
        "maximum_prediction_correction": 6.0,
        "maximum_angle_change_degrees": 20.0,
        "minimum_amplitude_to_error": 1.0,
        "minimum_flat_error_reduction": 0.05,
        "accepted_minimum_sigma": 2.0 / 2.35482,
        "accepted_maximum_sigma": 12.0 / 2.35482,
    }

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"
    assert len(frame_indices) == 30
    assert len(automatic_seed_length_fractions) == 5

    print(f"Reading reference frame {frame_indices[0]} from {input_path}...")
    reference_frame = tifffile.imread(
        input_path,
        key=frame_indices[0],
    ).astype(np.float32)
    assert reference_frame.ndim == 2
    selection_minimum, selection_maximum = np.percentile(
        reference_frame,
        [display_low_percentile, display_high_percentile],
    )
    selection_figure, selection_axis = plt.subplots(figsize=(11, 10))
    selection_axis.imshow(
        reference_frame,
        cmap="gray",
        vmin=selection_minimum,
        vmax=selection_maximum,
        interpolation="nearest",
    )
    selection_axis.set_title(f"Click one tube in frame {frame_indices[0]}")
    selection_axis.set_axis_off()
    print(f"Click one tube in frame {frame_indices[0]}...")
    clicked_points = plt.ginput(1, timeout=-1)
    plt.close(selection_figure)
    assert len(clicked_points) == 1

    selected_trace = trace_tube_from_seed(
        reference_frame,
        np.asarray(clicked_points[0]),
        manual_seed_search_radius,
        trace_parameters,
        f"Frame {frame_indices[0]} reference",
        True,
    )
    assert len(selected_trace["positions"]) >= 3
    assert selected_trace["length"] > 0.0

    labeled_frames = np.empty(
        (len(frame_indices), *reference_frame.shape, 3),
        dtype=np.uint8,
    )
    labeled_frames[0] = draw_labeled_trace(
        reference_frame,
        selected_trace["positions"],
        frame_indices[0],
        display_low_percentile,
        display_high_percentile,
        trace_color,
        trace_outline_color,
        trace_outline_width,
        trace_line_width,
        label_color,
        label_position,
        label_font_size,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_labeled_stack(
        output_path,
        labeled_frames[:1],
        playback_frames_per_second,
    )
    print(f"Wrote partial tracking TIFF through frame {frame_indices[0]}.")

    for output_frame_index, frame_index in enumerate(frame_indices[1:], start=1):
        print(f"\nTracking frame {frame_index} from the selected previous trace...")
        frame = tifffile.imread(input_path, key=frame_index).astype(np.float32)
        automatic_seed_positions = positions_at_length_fractions(
            selected_trace["positions"],
            automatic_seed_length_fractions,
        )
        accepted_candidates = []
        for candidate_index, seed_position in enumerate(
            automatic_seed_positions,
            start=1,
        ):
            candidate_label = f"Frame {frame_index} candidate {candidate_index}"
            candidate_trace = trace_tube_from_seed(
                frame,
                seed_position,
                automatic_seed_search_radius,
                trace_parameters,
                candidate_label,
                False,
            )
            rejection_reasons = []
            if len(candidate_trace["positions"]) < 3:
                rejection_reasons.append(
                    f"only {len(candidate_trace['positions'])} fitted points"
                )
            if candidate_trace["length"] <= 0.0:
                rejection_reasons.append("zero trace length")
            if rejection_reasons:
                print(
                    f"{candidate_label} rejected: "
                    + "; ".join(rejection_reasons)
                )
                continue

            comparison = compare_candidate(
                selected_trace,
                candidate_trace,
                comparison_point_count,
            )
            if (
                comparison["relative_length_difference"]
                > maximum_relative_length_difference
            ):
                rejection_reasons.append(
                    f"length difference "
                    f"{100.0 * comparison['relative_length_difference']:.1f}% > "
                    f"{100.0 * maximum_relative_length_difference:.1f}%"
                )
            if (
                comparison["center_position_difference"]
                > maximum_center_position_difference
            ):
                rejection_reasons.append(
                    f"center position difference "
                    f"{comparison['center_position_difference']:.2f} > "
                    f"{maximum_center_position_difference:.2f} pixels"
                )
            if (
                comparison["mean_orientation_difference_degrees"]
                > maximum_mean_orientation_difference_degrees
            ):
                rejection_reasons.append(
                    f"mean orientation difference "
                    f"{comparison['mean_orientation_difference_degrees']:.2f} > "
                    f"{maximum_mean_orientation_difference_degrees:.2f} degrees"
                )

            print(
                f"{candidate_label}: length difference="
                f"{100.0 * comparison['relative_length_difference']:.1f}%, "
                f"center position difference="
                f"{comparison['center_position_difference']:.2f} pixels, "
                f"orientation difference="
                f"{comparison['mean_orientation_difference_degrees']:.2f} degrees"
            )
            if rejection_reasons:
                print(
                    f"{candidate_label} rejected: "
                    + "; ".join(rejection_reasons)
                )
                continue

            agreement_score = candidate_agreement_score(
                comparison,
                maximum_relative_length_difference,
                maximum_center_position_difference,
                maximum_mean_orientation_difference_degrees,
            )
            print(
                f"{candidate_label} accepted with geometric disagreement "
                f"score={agreement_score:.3f}"
            )
            accepted_candidates.append((agreement_score, comparison))

        assert accepted_candidates, (
            f"No frame-{frame_index} candidate passed every comparison"
        )
        selected_score, selected_comparison = min(
            accepted_candidates,
            key=lambda scored_comparison: scored_comparison[0],
        )
        selected_trace = selected_comparison["trace"]
        print(
            f"Selected frame {frame_index}: score={selected_score:.3f}, "
            f"length={selected_trace['length']:.2f}, length difference="
            f"{100.0 * selected_comparison['relative_length_difference']:.1f}%, "
            f"center position difference="
            f"{selected_comparison['center_position_difference']:.2f} pixels, "
            f"orientation difference="
            f"{selected_comparison['mean_orientation_difference_degrees']:.2f} "
            f"degrees"
        )
        labeled_frames[output_frame_index] = draw_labeled_trace(
            frame,
            selected_trace["positions"],
            frame_index,
            display_low_percentile,
            display_high_percentile,
            trace_color,
            trace_outline_color,
            trace_outline_width,
            trace_line_width,
            label_color,
            label_position,
            label_font_size,
        )
        write_labeled_stack(
            output_path,
            labeled_frames[: output_frame_index + 1],
            playback_frames_per_second,
        )
        print(f"Updated tracking TIFF through frame {frame_index}.")

    print(f"Saved labeled tracking TIFF stack: {output_path}")
