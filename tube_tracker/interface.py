"""Tkinter interface components for viewing and interacting with microscopy TIFF stacks."""

import json
import tkinter as tk
from datetime import datetime
from pathlib import Path
from time import perf_counter

import numpy as np
from PIL import Image, ImageTk

from tube_tracker import configuration
from tube_tracker.ridge_detection import find_likely_ridge
from tube_tracker.tube_construction import construct_tube
from tube_tracker.tube_tracking import add_tube_to_track, create_tube_track, track_tube, tube_positions


def save_tube_tracks(output_path, source_path, tube_tracks):
    """Save track metadata, fitted ridge parameters, and analysis parameters in one compressed NumPy file."""
    track_ids = np.asarray(list(tube_tracks), dtype=np.int64)
    initial_frames = np.asarray([tube_tracks[track_id]["initial_frame"] for track_id in track_ids], dtype=np.int64)
    tracking_started = np.asarray([tube_tracks[track_id]["tracking_started"] for track_id in track_ids], dtype=bool)
    average_lengths = np.asarray([tube_tracks[track_id]["average_length"] for track_id in track_ids], dtype=np.float64)

    ridge_track_ids = []
    ridge_frame_indices = []
    ridge_indices = []
    ridge_centers_x = []
    ridge_centers_y = []
    ridge_angles = []
    ridge_perpendicular_shifts = []
    ridge_perpendicular_fwhms = []
    ridge_amplitudes = []
    ridge_backgrounds = []
    ridge_errors = []
    ridge_converged = []
    ridge_function_evaluations = []
    for track_id, tube_track in tube_tracks.items():
        for frame_index, tube in sorted(tube_track["tubes"].items()):
            for ridge_index, ridge in enumerate(tube):
                ridge_track_ids.append(track_id)
                ridge_frame_indices.append(frame_index)
                ridge_indices.append(ridge_index)
                ridge_centers_x.append(ridge["center"][0])
                ridge_centers_y.append(ridge["center"][1])
                ridge_angles.append(ridge["angle"])
                ridge_perpendicular_shifts.append(ridge["perpendicular_shift"])
                ridge_perpendicular_fwhms.append(ridge["perpendicular_fwhm"])
                ridge_amplitudes.append(ridge["amplitude"])
                ridge_backgrounds.append(ridge["background"])
                ridge_errors.append(ridge["root_mean_squared_error"])
                ridge_converged.append(ridge["converged"])
                ridge_function_evaluations.append(ridge["function_evaluations"])

    analysis_parameters = {name: getattr(configuration, name) for name in configuration.__annotations__}
    np.savez_compressed(
        output_path,
        format_version=np.asarray(1, dtype=np.int64),
        source_tiff_path=np.asarray(str(source_path)),
        analysis_parameters=np.asarray(json.dumps(analysis_parameters)),
        track_ids=track_ids,
        initial_frames=initial_frames,
        tracking_started=tracking_started,
        average_lengths=average_lengths,
        ridge_track_ids=np.asarray(ridge_track_ids, dtype=np.int64),
        ridge_frame_indices=np.asarray(ridge_frame_indices, dtype=np.int64),
        ridge_indices=np.asarray(ridge_indices, dtype=np.int64),
        ridge_centers_x=np.asarray(ridge_centers_x, dtype=np.float64),
        ridge_centers_y=np.asarray(ridge_centers_y, dtype=np.float64),
        ridge_angles=np.asarray(ridge_angles, dtype=np.float64),
        ridge_perpendicular_shifts=np.asarray(ridge_perpendicular_shifts, dtype=np.float64),
        ridge_perpendicular_fwhms=np.asarray(ridge_perpendicular_fwhms, dtype=np.float64),
        ridge_amplitudes=np.asarray(ridge_amplitudes, dtype=np.float64),
        ridge_backgrounds=np.asarray(ridge_backgrounds, dtype=np.float64),
        ridge_errors=np.asarray(ridge_errors, dtype=np.float64),
        ridge_converged=np.asarray(ridge_converged, dtype=bool),
        ridge_function_evaluations=np.asarray(ridge_function_evaluations, dtype=np.int64),
    )
    print(f"Saved {len(tube_tracks)} tube tracks to {output_path}.")


def timestamped_output_path(output_path):
    """Insert the current date and time into a path so no earlier saved file is ever overwritten."""
    output_path = Path(output_path)
    timestamp = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    return output_path.with_name(f"{output_path.stem}_{timestamp}{output_path.suffix}")


def show_tiff_stack(
    source_path,
    output_path,
    tiff_stack,
    convolution_kernels,
    random_generator,
    display_low_percentile,
    display_high_percentile,
    window_title,
    display_width,
    display_height,
    initialized_tube_color,
    tracked_tube_colors,
    tube_line_width,
    tube_label_offset,
    tube_removal_distance,
):
    """Display a TIFF stack and construct one stored tube track from each image click."""
    frame_count, image_height, image_width = tiff_stack.shape
    displayed_photo = None
    tube_tracks = {}
    next_track_id = 1
    active_track_ids = []
    failed_frame_counts = {}
    tracking_frame_index = None
    tracking_running = False
    initial_display_minimum, initial_display_maximum = np.percentile(tiff_stack[0], [display_low_percentile, display_high_percentile])
    intensity_minimum = float(tiff_stack.min())
    intensity_maximum = float(tiff_stack.max())
    image_scale = min(display_width / image_width, display_height / image_height)
    shown_width = max(1, round(image_width * image_scale))
    shown_height = max(1, round(image_height * image_scale))
    shown_size = (shown_width, shown_height)
    horizontal_image_scale = shown_width / image_width
    vertical_image_scale = shown_height / image_height
    image_left = (display_width - shown_width) / 2.0
    image_top = (display_height - shown_height) / 2.0

    window = tk.Tk()
    window.title(window_title)

    image_canvas = tk.Canvas(window, width=display_width, height=display_height, background="black", highlightthickness=0)
    image_canvas.grid(row=0, column=0, columnspan=3)
    canvas_image = image_canvas.create_image(display_width / 2, display_height / 2, anchor="center")
    frame_label = tk.Label(window)
    frame_label.grid(row=2, column=0, columnspan=3, pady=(0, 8))

    track_panel = tk.Frame(window)
    track_panel.grid(row=0, column=3, rowspan=6, padx=8, pady=8, sticky="ns")
    track_button = tk.Button(track_panel, text="Track tubes")
    track_button.pack(fill="x", pady=(0, 8))
    save_button = tk.Button(track_panel, text="Save tracks")
    save_button.pack(fill="x", pady=(0, 12))
    tk.Label(track_panel, text="Initialized tracks").pack(anchor="w")
    initialized_track_list = tk.Listbox(track_panel, width=24, background="black", exportselection=False)
    initialized_track_list.pack(fill="both", expand=True, pady=(0, 12))
    tk.Label(track_panel, text="Tracked tracks").pack(anchor="w")
    tracked_track_list = tk.Listbox(track_panel, width=24, background="black", exportselection=False)
    tracked_track_list.pack(fill="both", expand=True)

    def tube_canvas_positions(tube):
        """Convert the fitted source-image coordinates of one tube to canvas coordinates."""
        positions = tube_positions(tube)
        canvas_positions = np.empty_like(positions)
        canvas_positions[:, 0] = image_left + horizontal_image_scale * positions[:, 0]
        canvas_positions[:, 1] = image_top + vertical_image_scale * positions[:, 1]
        return canvas_positions

    def track_color(tube_track):
        """Return white for an initialized track or its stable palette color after tracking starts."""
        if not tube_track["tracking_started"]:
            return initialized_tube_color
        return tracked_tube_colors[(tube_track["track_id"] - 1) % len(tracked_tube_colors)]

    def draw_tubes(frame_index):
        """Draw every stored tube belonging to one frame over the displayed image."""
        image_canvas.delete("tube")
        for tube_track in tube_tracks.values():
            if frame_index not in tube_track["tubes"]:
                continue
            tube_color = track_color(tube_track)
            canvas_positions = tube_canvas_positions(tube_track["tubes"][frame_index])
            if len(canvas_positions) == 1:
                canvas_positions = np.repeat(canvas_positions, 2, axis=0)
            image_canvas.create_line(
                *canvas_positions.ravel(), fill=tube_color, width=tube_line_width, capstyle=tk.ROUND, joinstyle=tk.ROUND, tags="tube"
            )
            label_position = canvas_positions[-1] + np.array([tube_label_offset, -tube_label_offset])
            image_canvas.create_text(
                *label_position, text=f"T{tube_track['track_id']}", fill=tube_color, anchor="sw", font=("TkDefaultFont", 10, "bold"), tags="tube"
            )

    def update_track_lists():
        """Show initialized and previously tracked tube IDs in separate lists."""
        initialized_track_list.delete(0, tk.END)
        tracked_track_list.delete(0, tk.END)
        for tube_track in tube_tracks.values():
            track_description = f"T{tube_track['track_id']} - frame {tube_track['initial_frame']}"
            tube_color = track_color(tube_track)
            if tube_track["tracking_started"]:
                track_list = tracked_track_list
            else:
                track_list = initialized_track_list
            track_list.insert(tk.END, track_description)
            track_list.itemconfigure(track_list.size() - 1, foreground=tube_color)

    def display_frame(frame_index):
        """Convert and draw one raw frame without copying the complete TIFF stack."""
        nonlocal displayed_photo
        current_frame_index = int(float(frame_index))
        display_minimum = float(black_level_slider.get())
        display_maximum = float(white_level_slider.get())
        if display_minimum >= display_maximum:
            return
        frame = tiff_stack[current_frame_index].astype(np.float32)
        scaled_frame = np.clip((frame - display_minimum) / (display_maximum - display_minimum), 0.0, 1.0)
        green_intensity = np.round(255.0 * scaled_frame).astype(np.uint8)
        colored_frame = np.zeros((image_height, image_width, 3), dtype=np.uint8)
        colored_frame[:, :, 1] = green_intensity
        shown_frame = Image.fromarray(colored_frame).resize(shown_size, Image.Resampling.NEAREST)
        displayed_photo = ImageTk.PhotoImage(shown_frame)
        image_canvas.itemconfigure(canvas_image, image=displayed_photo)
        draw_tubes(current_frame_index)
        frame_label.configure(text=f"Frame {current_frame_index + 1} of {frame_count}  (index {current_frame_index})")

    def construct_clicked_tube(event):
        """Convert one canvas click to source coordinates, construct its tube, and store a new track."""
        nonlocal next_track_id
        if tracking_running:
            print("Wait for the current tracking run to finish before initializing another tube.")
            return
        if not image_left <= event.x < image_left + shown_width or not image_top <= event.y < image_top + shown_height:
            print("The click was outside the displayed microscopy image.")
            return

        frame_index = frame_slider.get()
        clicked_center = np.array(
            [(event.x - image_left) / horizontal_image_scale, (event.y - image_top) / vertical_image_scale]
        )
        print(f"Constructing tube {next_track_id} in frame {frame_index} from click ({clicked_center[0]:.2f}, {clicked_center[1]:.2f})...")
        image = tiff_stack[frame_index].astype(np.float32)
        predicted_center, predicted_angle, convolution_score = find_likely_ridge(
            image,
            clicked_center,
            convolution_kernels,
            configuration.convolution_search_radius,
        )
        tube = construct_tube(image, predicted_center, predicted_angle)
        if not tube:
            print(f"No valid tube was constructed from click ({clicked_center[0]:.2f}, {clicked_center[1]:.2f}).")
            return

        tube_tracks[next_track_id] = create_tube_track(next_track_id, frame_index, tube)
        print(f"Stored tube {next_track_id} with {len(tube)} ridges; convolution score={convolution_score:.2f}.")
        next_track_id += 1
        draw_tubes(frame_index)
        update_track_lists()

    def remove_clicked_tube(event):
        """Remove the tube whose displayed center line is closest to a control-click."""
        if tracking_running:
            print("Wait for the current tracking run to finish before removing a tube.")
            return "break"
        frame_index = frame_slider.get()
        clicked_position = np.array([event.x, event.y])
        closest_track_id = None
        closest_distance = np.inf
        for track_id, tube_track in tube_tracks.items():
            if frame_index not in tube_track["tubes"]:
                continue
            canvas_positions = tube_canvas_positions(tube_track["tubes"][frame_index])
            if len(canvas_positions) == 1:
                distance = np.linalg.norm(clicked_position - canvas_positions[0])
            else:
                segment_starts = canvas_positions[:-1]
                segment_vectors = canvas_positions[1:] - segment_starts
                projection_numerators = np.sum((clicked_position - segment_starts) * segment_vectors, axis=1)
                projection_denominators = np.sum(segment_vectors**2, axis=1)
                projection_fractions = projection_numerators / projection_denominators
                projection_fractions = np.clip(projection_fractions, 0.0, 1.0)
                closest_segment_positions = segment_starts + projection_fractions[:, np.newaxis] * segment_vectors
                distance = np.min(np.linalg.norm(clicked_position - closest_segment_positions, axis=1))
            if distance < closest_distance:
                closest_track_id = track_id
                closest_distance = distance

        if closest_distance > tube_removal_distance:
            print(f"No tube was within {tube_removal_distance:.1f} screen pixels of the control-click.")
            return "break"
        del tube_tracks[closest_track_id]
        print(f"Removed tube {closest_track_id} from the initialized tube tracks.")
        draw_tubes(frame_index)
        update_track_lists()
        return "break"

    def finish_tracking():
        """Restore the interface controls after every active tube has stopped or the stack has ended."""
        nonlocal tracking_running
        tracking_running = False
        track_button.configure(state=tk.NORMAL)
        update_track_lists()
        draw_tubes(frame_slider.get())
        print("Finished tracking the initialized tubes.")

    def track_next_frame():
        """Track every active tube requiring one frame and then yield control back to Tkinter."""
        nonlocal tracking_frame_index, active_track_ids, failed_frame_counts
        if tracking_frame_index >= frame_count or not active_track_ids:
            finish_tracking()
            return

        track_ids_in_frame = [
            track_id for track_id in active_track_ids if tube_tracks[track_id]["initial_frame"] < tracking_frame_index
        ]
        if track_ids_in_frame:
            print(f"=== Frame {tracking_frame_index}/{frame_count - 1}: tracking {len(track_ids_in_frame)} tubes ===")
            image = tiff_stack[tracking_frame_index].astype(np.float32)
            for track_id in track_ids_in_frame:
                tube_track = tube_tracks[track_id]
                last_valid_frame = max(tube_track["tubes"])
                previous_tube = tube_track["tubes"][last_valid_frame]
                tube_tracking_start_time = perf_counter()
                new_tube, trial_count = track_tube(
                    image,
                    previous_tube,
                    tube_track["average_length"],
                    convolution_kernels,
                    random_generator,
                )
                tube_tracking_time = perf_counter() - tube_tracking_start_time
                if new_tube is None:
                    failed_frame_counts[track_id] += 1
                    if failed_frame_counts[track_id] > configuration.maximum_rescue_frames:
                        active_track_ids.remove(track_id)
                        print(
                            f"Frame {tracking_frame_index}/{frame_count - 1} | T{track_id} | not found after {trial_count} trials | "
                            f"terminated after {failed_frame_counts[track_id]} missing frames | {tube_tracking_time:.2f}s"
                        )
                    else:
                        remaining_rescue_frames = configuration.maximum_rescue_frames - failed_frame_counts[track_id] + 1
                        print(
                            f"Frame {tracking_frame_index}/{frame_count - 1} | T{track_id} | not found after {trial_count} trials | "
                            f"{remaining_rescue_frames} rescue frames remain | {tube_tracking_time:.2f}s"
                        )
                else:
                    recovery_text = ""
                    if failed_frame_counts[track_id] > 0:
                        recovery_text = f" | recovered after {failed_frame_counts[track_id]} missing frames"
                    add_tube_to_track(tube_track, tracking_frame_index, new_tube)
                    failed_frame_counts[track_id] = 0
                    print(
                        f"Frame {tracking_frame_index}/{frame_count - 1} | T{track_id} | accepted on trial {trial_count}"
                        f"{recovery_text} | {tube_tracking_time:.2f}s"
                    )

        frame_slider.set(tracking_frame_index)
        tracking_frame_index += 1
        window.after(1, track_next_frame)

    def start_tracking():
        """Start forward tracking for every tube that has only been initialized."""
        nonlocal active_track_ids, failed_frame_counts, tracking_frame_index, tracking_running
        active_track_ids = [track_id for track_id, tube_track in tube_tracks.items() if not tube_track["tracking_started"]]
        if not active_track_ids:
            print("There are no initialized tubes waiting to be tracked.")
            return

        for track_id in active_track_ids:
            tube_tracks[track_id]["tracking_started"] = True
        failed_frame_counts = {track_id: 0 for track_id in active_track_ids}
        tracking_frame_index = min(tube_tracks[track_id]["initial_frame"] + 1 for track_id in active_track_ids)
        tracking_running = True
        track_button.configure(state=tk.DISABLED)
        update_track_lists()
        draw_tubes(frame_slider.get())
        print(f"Starting forward tracking for {len(active_track_ids)} initialized tubes...")
        window.after(1, track_next_frame)

    def change_frame(frame_change):
        """Move by one frame while keeping the frame index inside the TIFF stack."""
        new_frame_index = min(max(frame_slider.get() + frame_change, 0), frame_count - 1)
        frame_slider.set(new_frame_index)

    def change_contrast(unused_slider_value=None):
        """Apply the raw intensity levels chosen with the black and white sliders."""
        display_frame(frame_slider.get())

    def auto_contrast():
        """Set the display levels from the configured percentiles of the current frame."""
        frame = tiff_stack[frame_slider.get()]
        automatic_minimum, automatic_maximum = np.percentile(frame, [display_low_percentile, display_high_percentile])
        black_level_slider.set(automatic_minimum)
        white_level_slider.set(automatic_maximum)
        change_contrast()

    def reset_contrast():
        """Restore the display levels used when the interface was opened."""
        black_level_slider.set(initial_display_minimum)
        white_level_slider.set(initial_display_maximum)
        change_contrast()

    def save_current_tracks():
        """Save every accumulated tube track to a fresh timestamped file without overwriting earlier saves."""
        if not tube_tracks:
            print("There are no tube tracks to save yet.")
            return
        save_path = timestamped_output_path(output_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_tube_tracks(save_path, source_path, tube_tracks)

    tk.Button(window, text="Previous", command=lambda: change_frame(-1)).grid(row=1, column=0, padx=8, pady=8)

    frame_slider = tk.Scale(window, from_=0, to=frame_count - 1, orient="horizontal", command=display_frame, showvalue=False)
    frame_slider.grid(row=1, column=1, padx=8, pady=8, sticky="ew")

    tk.Button(window, text="Next", command=lambda: change_frame(1)).grid(row=1, column=2, padx=8, pady=8)

    tk.Label(window, text="Black level").grid(row=3, column=0, padx=8)
    black_level_slider = tk.Scale(window, from_=intensity_minimum, to=intensity_maximum, orient="horizontal", resolution=1)
    black_level_slider.set(initial_display_minimum)
    black_level_slider.grid(row=3, column=1, columnspan=2, padx=8, sticky="ew")

    tk.Label(window, text="White level").grid(row=4, column=0, padx=8)
    white_level_slider = tk.Scale(window, from_=intensity_minimum, to=intensity_maximum, orient="horizontal", resolution=1)
    white_level_slider.set(initial_display_maximum)
    white_level_slider.grid(row=4, column=1, columnspan=2, padx=8, sticky="ew")

    tk.Button(window, text="Auto", command=auto_contrast).grid(row=5, column=1, padx=8, pady=8, sticky="e")
    tk.Button(window, text="Reset", command=reset_contrast).grid(row=5, column=2, padx=8, pady=8, sticky="w")

    black_level_slider.configure(command=change_contrast)
    white_level_slider.configure(command=change_contrast)
    track_button.configure(command=start_tracking)
    save_button.configure(command=save_current_tracks)

    window.columnconfigure(1, weight=1)
    window.bind("<Left>", lambda event: change_frame(-1))
    window.bind("<Right>", lambda event: change_frame(1))
    image_canvas.bind("<Button-1>", construct_clicked_tube)
    image_canvas.bind("<Control-Button-1>", remove_clicked_tube)

    display_frame(0)
    window.mainloop()
    return tube_tracks
