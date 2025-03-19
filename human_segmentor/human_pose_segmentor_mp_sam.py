import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image

# Import SAM2 from your specified relative path
import sys
sys.path.append("submodules/segment-anything-2-real-time/sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize MediaPipe Pose with segmentation enabled
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, 
                     min_detection_confidence=0.1, min_tracking_confidence=0.1)

# Load SAM2 Model
checkpoint = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def extract_segmentation_points(image_path, num_points=10):
    """
    Extracts human pose keypoints and segmentation mask from an image using MediaPipe.
    Evenly samples points from the segmentation mask for SAM2 input.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    height, width, _ = image.shape
    segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # Get all foreground pixel locations
    mask_indices = np.column_stack(np.where(segmentation_mask == 1))  # (row, col)

    if len(mask_indices) == 0:
        print("No valid segmentation points found.")
        return None, None, None, None

    # Sort points to ensure even spacing
    mask_indices = mask_indices[np.lexsort((mask_indices[:, 1], mask_indices[:, 0]))]  # Sort by (y, x)

    # Evenly sample points from the sorted indices
    step = max(1, len(mask_indices) // num_points)  # Step size to evenly pick points
    sampled_indices = mask_indices[::step][:num_points]  # Take every `step`th point

    sampled_points = np.flip(sampled_indices, axis=1).copy()  # Convert (row, col) to (x, y) and fix negative strides

    # Draw the skeleton on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return image, sampled_points, annotated_image, segmentation_mask


def segment_human(image, sampled_points):
    """
    Uses SAM2 to refine segmentation using sampled points from MediaPipe's segmentation mask.
    """
    # Convert OpenCV BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image for segmentation
    sam_predictor.set_image(image_rgb)

    # Predict segmentation mask using sampled points
    masks, _, _ = sam_predictor.predict(
        point_coords=sampled_points, 
        point_labels=np.ones(len(sampled_points)),  # 1 for foreground
        multimask_output=False
    )

    return masks[0]  # Return the best mask

def replace_background(image, mask, reference_image):
    """
    Replaces the background of the segmented human image with pixels from the reference image.
    
    Arguments:
    - image: Original input image.
    - mask: Binary segmentation mask (1 for foreground, 0 for background).
    - reference_image: Image to take the background from.
    
    Returns:
    - Image with replaced background.
    """
    # Ensure the reference image is the same size as the original image
    reference_resized = cv2.resize(reference_image, (image.shape[1], image.shape[0]))

    # Create the final composited image
    result_image = image.copy()
    result_image[mask == 0] = reference_resized[mask == 0]  # Replace background pixels

    return result_image

def process_image(image_path, output_mask_path):
    """
    Full pipeline: Extracts segmentation points, displays skeleton, refines segmentation with SAM2, and saves mask.
    """
    image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(image_path)
    if sampled_points is None:
        return

    # Save and display skeleton image
    skeleton_image_path = "annotated_skeleton.png"
    cv2.imwrite(skeleton_image_path, annotated_image)

    # Segment using SAM2
    refined_mask = segment_human(image, sampled_points)

    # Save segmentation mask
    mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
    mask_image.save(output_mask_path)

    # Overlay segmentation on image for visualization
    segmented_image = image.copy()
    segmented_image[refined_mask > 0] = [255, 0, 0]  # Highlight mask in red

    # Display results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Skeleton")

    plt.subplot(1, 3, 2)
    plt.imshow(mediapipe_mask, cmap="gray")
    plt.title("MediaPipe Segmentation Mask")

    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image)
    plt.title("SAM2 Refined Mask")

    plt.show()

# Example Usage
process_image("hamer_detector/example_data/realsense-test/frame_000000.png", "output_mask.png")
