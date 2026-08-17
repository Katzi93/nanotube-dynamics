"""Test SAM 3.1 point segmentation against a known synthetic circle mask."""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from sam3.model_builder import build_sam3_predictor
import tifffile
import torch


def create_circle_image(image_size, circle_center, circle_radius, background, foreground):
    """Create an RGB image and ground-truth mask containing one filled circle."""
    y_coordinates, x_coordinates = np.ogrid[:image_size, :image_size]
    center_x, center_y = circle_center
    squared_distance = (x_coordinates - center_x) ** 2 + (
        y_coordinates - center_y
    ) ** 2
    circle_mask = squared_distance <= circle_radius**2

    grayscale_image = np.full(
        (image_size, image_size), background, dtype=np.uint8
    )
    grayscale_image[circle_mask] = foreground
    rgb_image = np.repeat(grayscale_image[:, :, np.newaxis], 3, axis=2)
    return rgb_image, circle_mask


def segment_from_point(
    tracking_model,
    inference_state,
    point_coordinate,
    point_label,
    frame_index,
    object_id,
    output_probability_threshold,
):
    """Prompt the SAM 3.1 tracker with one point and return its binary mask."""
    image_size = np.array(
        [inference_state["orig_width"], inference_state["orig_height"]],
        dtype=np.float32,
    )
    relative_point = np.asarray([point_coordinate], dtype=np.float32) / image_size
    point_labels = np.asarray([point_label], dtype=np.int32)

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, outputs = tracking_model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_index,
            points=torch.from_numpy(relative_point),
            point_labels=torch.from_numpy(point_labels),
            clear_old_points=True,
            output_prob_thresh=output_probability_threshold,
            obj_id=object_id,
            rel_coordinates=True,
        )

    assert returned_frame_index == frame_index
    object_ids = np.asarray(outputs["out_obj_ids"])
    masks = np.asarray(outputs["out_binary_masks"])
    output_scores = np.asarray(outputs["out_probs"])

    selected_indices = np.flatnonzero(object_ids == object_id)
    assert selected_indices.size == 1, (
        f"Expected one mask for object {object_id}, got object IDs {object_ids.tolist()}"
    )
    selected_index = int(selected_indices[0])
    return masks[selected_index].astype(bool), float(output_scores[selected_index])


def segment_from_box(
    tracking_model,
    inference_state,
    bounding_box_xywh,
    frame_index,
    output_probability_threshold,
):
    """Prompt the SAM 3.1 detector with one normalized box."""
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        returned_frame_index, outputs = tracking_model.add_prompt(
            inference_state=inference_state,
            frame_idx=frame_index,
            points=None,
            point_labels=None,
            boxes_xywh=torch.tensor([bounding_box_xywh], dtype=torch.float32),
            box_labels=torch.tensor([1], dtype=torch.int32),
            clear_old_boxes=True,
            output_prob_thresh=output_probability_threshold,
        )

    assert returned_frame_index == frame_index
    masks = np.asarray(outputs["out_binary_masks"])
    output_scores = np.asarray(outputs["out_probs"])
    assert masks.shape[0] > 0, "SAM returned no masks for the circle box prompt"

    selected_index = int(np.argmax(output_scores))
    return masks[selected_index].astype(bool), float(output_scores[selected_index])


def calculate_iou(predicted_mask, reference_mask):
    """Calculate intersection over union for two binary masks."""
    intersection = np.logical_and(predicted_mask, reference_mask).sum()
    union = np.logical_or(predicted_mask, reference_mask).sum()
    assert union > 0, "Cannot calculate IoU for two empty masks"
    return intersection / union


def save_results(
    rgb_image,
    reference_mask,
    predicted_mask,
    point_coordinate,
    output_directory,
    intersection_over_union,
):
    """Save the synthetic input, predicted mask, and comparison figure."""
    output_directory.mkdir(parents=True, exist_ok=True)
    mask_path = output_directory / "predicted_mask.tif"
    comparison_path = output_directory / "comparison.png"

    tifffile.imwrite(
        mask_path, predicted_mask.astype(np.uint8), photometric="minisblack"
    )

    figure, axes = plt.subplots(1, 3, figsize=(15, 5))
    axes[0].imshow(rgb_image)
    axes[0].scatter(
        point_coordinate[0], point_coordinate[1], c="lime", marker="+", s=100
    )
    axes[0].set_title("Synthetic input and prompt")
    axes[1].imshow(reference_mask, cmap="gray")
    axes[1].set_title("Known circle mask")
    axes[2].imshow(predicted_mask, cmap="gray")
    axes[2].set_title(f"SAM prediction, IoU={intersection_over_union:.3f}")
    for axis in axes:
        axis.axis("off")
    figure.tight_layout()
    figure.savefig(comparison_path, dpi=150)
    plt.close(figure)

    print(f"Saved predicted mask: {mask_path}")
    print(f"Saved comparison: {comparison_path}")


if __name__ == "__main__":
    # Local SAM 3.1 checkpoint downloaded from Hugging Face.
    checkpoint_path = Path("checkpoints/sam3.1_multiplex.pt")
    # Directory for the generated frame and segmentation results.
    output_directory = Path("test_outputs/sam3_synthetic_circle")
    # Directory exposed to SAM as a one-frame video.
    frame_directory = output_directory / "input_frames"
    # Width and height of the square synthetic image in pixels.
    image_size = 1024
    # Center of the synthetic circle in original image coordinates.
    circle_center = (512, 512)
    # Radius of the synthetic circle in pixels.
    circle_radius = 200
    # Grayscale intensity assigned to the background.
    background_intensity = 30
    # Grayscale intensity assigned to the circle.
    circle_intensity = 180
    # Visual prompt type used for the control test: "point" or "box".
    prompt_type = "box"
    # Positive point placed at the center of the circle.
    prompt_point = circle_center
    # SAM label identifying the prompt as foreground.
    prompt_label = 1
    # Normalized top-left x, top-left y, width, and height of the circle prompt box.
    prompt_box_xywh = (0.3, 0.3, 0.4, 0.4)
    # Zero-based index of the generated one-frame input.
    frame_index = 0
    # Numeric object identifier used by the tracking model.
    object_id = 1
    # Minimum object probability retained in the model output.
    output_probability_threshold = 0.0
    # Maximum number of objects exposed by this single-object test.
    max_num_objects = 1
    # Native object bucket size encoded by the SAM 3.1 checkpoint.
    multiplex_count = 16
    # Avoid the optional Flash Attention 3 dependency on the laptop GPU.
    use_flash_attention = False
    # Use the checkpoint's native complex-valued rotary position encoding.
    use_real_rope = False
    # Allow faster Ampere matrix operations during inference.
    allow_tf32 = True

    assert checkpoint_path.is_file(), f"Missing checkpoint: {checkpoint_path}"
    assert torch.cuda.is_available(), "PyTorch cannot access a CUDA GPU"

    rgb_image, reference_mask = create_circle_image(
        image_size,
        circle_center,
        circle_radius,
        background_intensity,
        circle_intensity,
    )
    frame_directory.mkdir(parents=True, exist_ok=True)
    generated_frame_path = frame_directory / "00000.jpg"
    Image.fromarray(rgb_image).save(generated_frame_path, quality=100, subsampling=0)
    print(f"Generated circle image: {generated_frame_path}")

    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32

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

    print("Starting the one-frame SAM session...")
    tracking_model = predictor.model
    inference_state = tracking_model.init_state(
        resource_path=str(frame_directory),
        offload_video_to_cpu=False,
        async_loading_frames=False,
    )

    if prompt_type == "point":
        print(f"Segmenting from positive point {prompt_point}...")
        predicted_mask, output_score = segment_from_point(
            tracking_model,
            inference_state,
            prompt_point,
            prompt_label,
            frame_index,
            object_id,
            output_probability_threshold,
        )
    elif prompt_type == "box":
        print(f"Segmenting from normalized box {prompt_box_xywh}...")
        predicted_mask, output_score = segment_from_box(
            tracking_model,
            inference_state,
            prompt_box_xywh,
            frame_index,
            output_probability_threshold,
        )
    else:
        raise ValueError(f"Unknown prompt type: {prompt_type}")
    intersection_over_union = calculate_iou(predicted_mask, reference_mask)

    print(f"SAM output score: {output_score:.3f}")
    print(f"Predicted pixels: {predicted_mask.sum()}")
    print(f"Reference pixels: {reference_mask.sum()}")
    print(f"Intersection over union: {intersection_over_union:.3f}")
    save_results(
        rgb_image,
        reference_mask,
        predicted_mask,
        prompt_point,
        output_directory,
        intersection_over_union,
    )
