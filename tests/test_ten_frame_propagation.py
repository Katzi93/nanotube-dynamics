"""Test whether one frame-0 box can track a nanotube through ten frames."""

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import sam3.model.decoder as sam3_decoder
from sam3.model_builder import build_sam3_predictor
import tifffile
import torch
from torch.nn.attention import SDPBackend, sdpa_kernel


def scale_frame_to_rgb(frame, low_value, high_value):
    """Apply fixed grayscale contrast limits and return an RGB uint8 image."""
    scaled_frame = np.clip(
        (frame.astype(np.float32) - low_value) / (high_value - low_value),
        0.0,
        1.0,
    )
    grayscale_uint8 = np.rint(scaled_frame * 255).astype(np.uint8)
    return np.repeat(grayscale_uint8[:, :, np.newaxis], 3, axis=2)


def save_overlay(rgb_array, mask, output_path, mask_color, mask_alpha):
    """Save one RGB frame with a colored binary-mask overlay."""
    overlay = rgb_array.astype(np.float32)
    overlay[mask] = (
        (1.0 - mask_alpha) * overlay[mask]
        + mask_alpha * np.asarray(mask_color, dtype=np.float32)
    )
    Image.fromarray(np.rint(overlay).astype(np.uint8)).save(output_path)


def select_compatible_attention_backends(requested_backend):
    """Allow PyTorch's math fallback when SAM requests unavailable Windows flash."""
    assert requested_backend == SDPBackend.FLASH_ATTENTION
    return sdpa_kernel(
        [
            SDPBackend.FLASH_ATTENTION,
            SDPBackend.EFFICIENT_ATTENTION,
            SDPBackend.MATH,
        ]
    )


if __name__ == "__main__":
    # Read-only microscopy TIFF stack used for the tracking test.
    input_path = Path("test_data/test_1.tif")
    # Local SAM 3.1 checkpoint downloaded from Hugging Face.
    checkpoint_path = Path("checkpoints/sam3.1_multiplex.pt")
    # Directory for generated inputs, masks, and overlays.
    output_directory = Path("test_outputs/sam3_ten_frame_tracking")
    # Directory containing JPEG frames required by SAM's video API.
    frame_directory = output_directory / "input_frames"
    # Number of consecutive frames included in this small test.
    frame_count = 10
    # First TIFF frame included in the test.
    first_frame_index = 0
    # Lower percentile of frame 0 mapped to black for every test frame.
    contrast_low_percentile = 1.0
    # Upper percentile of frame 0 mapped to white for every test frame.
    contrast_high_percentile = 99.8
    # Gaussian blur radius applied to every contrast-scaled frame.
    gaussian_blur_radius = 1.5
    # Frame-0 box in pixel coordinates as x_min, y_min, x_max, y_max.
    initial_box = (1260, 1000, 1450, 1150)
    # Minimum object probability retained in SAM's returned tracking output.
    output_probability_threshold = 0.0
    # Maximum number of objects exposed by this single-object test.
    max_num_objects = 1
    # Native object bucket size encoded by the SAM 3.1 checkpoint.
    multiplex_count = 16
    # Avoid the optional Flash Attention 3 dependency on the laptop GPU.
    use_flash_attention = False
    # Use the checkpoint-compatible rotary position encoding setting.
    use_real_rope = False
    # Allow faster Ampere matrix operations during inference.
    allow_tf32 = True
    # Permit math attention because PyTorch's Windows build lacks flash attention.
    use_windows_attention_fallback = True
    # RGB color used for tracked-mask overlays.
    mask_color = (0, 204, 255)
    # Opacity used for tracked-mask overlays.
    mask_alpha = 0.45
    # RGB color used to show the initialization box on frame 0.
    box_color = (0, 255, 0)
    # Pixel width used to draw the initialization box.
    box_line_width = 3

    assert input_path.is_file(), f"Missing input TIFF: {input_path}"
    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    print(f"Reading {frame_count} frames from {input_path}...")
    with tifffile.TiffFile(input_path) as tif:
        assert len(tif.pages) >= first_frame_index + frame_count
        frames = np.stack(
            [
                tif.pages[frame_index].asarray()
                for frame_index in range(
                    first_frame_index, first_frame_index + frame_count
                )
            ]
        )
    assert frames.ndim == 3, f"Expected grayscale frames, got shape {frames.shape}"

    contrast_low_value, contrast_high_value = np.percentile(
        frames[0], [contrast_low_percentile, contrast_high_percentile]
    )
    assert contrast_high_value > contrast_low_value
    print(
        f"Fixed contrast for every frame: {contrast_low_value:.1f} to "
        f"{contrast_high_value:.1f} intensity units"
    )

    frame_directory.mkdir(parents=True, exist_ok=True)
    rgb_frames = []
    for local_frame_index, frame in enumerate(frames):
        rgb_array = scale_frame_to_rgb(
            frame, contrast_low_value, contrast_high_value
        )
        rgb_image = Image.fromarray(rgb_array).filter(
            ImageFilter.GaussianBlur(radius=gaussian_blur_radius)
        )
        rgb_frames.append(np.asarray(rgb_image))
        rgb_image.save(
            frame_directory / f"{local_frame_index:05d}.jpg",
            quality=100,
            subsampling=0,
        )
    print(f"Generated {frame_count} SAM input frames in {frame_directory}")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    if use_windows_attention_fallback:
        sam3_decoder.sdpa_kernel = select_compatible_attention_backends
        print("Enabled the project-local Windows attention fallback")

    print("Loading the SAM 3.1 video predictor...")
    predictor = build_sam3_predictor(
        checkpoint_path=str(checkpoint_path),
        version="sam3.1",
        compile=False,
        warm_up=False,
        max_num_objects=max_num_objects,
        multiplex_count=multiplex_count,
        use_fa3=use_flash_attention,
        use_rope_real=use_real_rope,
        async_loading_frames=False,
    )
    tracking_model = predictor.model
    inference_state = tracking_model.init_state(
        resource_path=str(frame_directory),
        offload_video_to_cpu=False,
        async_loading_frames=False,
    )

    image_height = inference_state["orig_height"]
    image_width = inference_state["orig_width"]
    x_min, y_min, x_max, y_max = initial_box
    assert 0 <= x_min < x_max <= image_width
    assert 0 <= y_min < y_max <= image_height
    relative_box_xywh = [
        x_min / image_width,
        y_min / image_height,
        (x_max - x_min) / image_width,
        (y_max - y_min) / image_height,
    ]

    print(f"Finding the frame-0 object from box {initial_box}...")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, box_output = tracking_model.add_prompt(
            inference_state=inference_state,
            frame_idx=0,
            points=None,
            point_labels=None,
            boxes_xywh=torch.tensor([relative_box_xywh], dtype=torch.float32),
            box_labels=torch.tensor([1], dtype=torch.int32),
            clear_old_boxes=True,
            output_prob_thresh=output_probability_threshold,
        )
    assert returned_frame_index == 0
    box_object_ids = np.asarray(box_output["out_obj_ids"])
    box_scores = np.asarray(box_output["out_probs"])
    box_masks = np.asarray(box_output["out_binary_masks"])
    assert len(box_object_ids) > 0, "SAM returned no object for the initial box"
    selected_index = int(np.argmax(box_scores))
    tracked_object_id = int(box_object_ids[selected_index])
    selected_mask = box_masks[selected_index].astype(bool)
    selected_y, selected_x = np.nonzero(selected_mask)
    centroid_point = torch.tensor(
        [[selected_x.mean() / image_width, selected_y.mean() / image_height]],
        dtype=torch.float32,
    )
    positive_label = torch.tensor([1], dtype=torch.int32)

    print(
        f"Switching object {tracked_object_id} from visual search to exact tracking "
        f"at mask centroid ({selected_x.mean():.1f}, {selected_y.mean():.1f})..."
    )
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, initial_output = tracking_model.add_sam2_new_points(
            inference_state=inference_state,
            frame_idx=0,
            obj_id=tracked_object_id,
            points=centroid_point,
            labels=positive_label,
            clear_old_points=True,
            rel_coordinates=True,
            use_prev_mem_frame=False,
        )
    assert returned_frame_index == 0
    initial_object_ids = np.asarray(initial_output["out_obj_ids"])
    initial_scores = np.asarray(initial_output["out_probs"])
    matching_initial_indices = np.flatnonzero(initial_object_ids == tracked_object_id)
    assert len(matching_initial_indices) == 1
    selected_index = int(matching_initial_indices[0])
    print(
        f"Tracking object {tracked_object_id} with initial score "
        f"{initial_scores[selected_index]:.3f}..."
    )

    tracked_frame_count = 0
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        propagation_results = tracking_model.propagate_in_video(
            inference_state=inference_state,
            start_frame_idx=0,
            max_frame_num_to_track=frame_count,
            reverse=False,
            output_prob_thresh=output_probability_threshold,
        )
        for local_frame_index, output in propagation_results:
            object_ids = np.asarray(output["out_obj_ids"])
            matching_indices = np.flatnonzero(object_ids == tracked_object_id)
            assert len(matching_indices) <= 1

            if len(matching_indices) == 0:
                print(f"Frame {local_frame_index}: object absent")
                continue

            object_index = int(matching_indices[0])
            mask = np.asarray(output["out_binary_masks"])[object_index].astype(bool)
            score = float(np.asarray(output["out_probs"])[object_index])
            tracked_frame_count += 1
            print(
                f"Frame {local_frame_index}: score {score:.3f}, "
                f"mask pixels {mask.sum()}"
            )

            mask_path = output_directory / f"frame_{local_frame_index:03d}_mask.tif"
            overlay_path = output_directory / f"frame_{local_frame_index:03d}_overlay.png"
            tifffile.imwrite(mask_path, mask.astype(np.uint8), photometric="minisblack")
            save_overlay(
                rgb_frames[local_frame_index],
                mask,
                overlay_path,
                mask_color,
                mask_alpha,
            )
            if local_frame_index == 0:
                frame_zero_overlay = Image.open(overlay_path)
                box_drawing = ImageDraw.Draw(frame_zero_overlay)
                box_drawing.rectangle(
                    initial_box, outline=box_color, width=box_line_width
                )
                frame_zero_overlay.save(overlay_path)

    print(f"Tracked object present in {tracked_frame_count}/{frame_count} frames")
    print(f"Saved tracking results under {output_directory}")
