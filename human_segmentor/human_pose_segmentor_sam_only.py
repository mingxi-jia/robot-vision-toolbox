import os
import cv2
import numpy as np
import torch
import hydra
import mediapipe as mp
from PIL import Image
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

# Import SAM2 from your specified relative path
import sys
sys.path.append("submodules/segment-anything-2-real-time/sam2")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True, min_detection_confidence = 0.1, min_tracking_confidence = 0.1)

# Load SAM2 Model
import torch
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

checkpoint = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))



def extract_keypoints(image_path):
    """
    Extracts human pose keypoints from an image using MediaPipe.
    Returns keypoints in (x, y) format and the annotated skeleton image.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks:
        print("No person detected.")
        return None, None, None

    height, width, _ = image.shape
    keypoints = [(int(lm.x * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]

    # Draw the skeleton on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    return image, np.array(keypoints), annotated_image

def segment_human(image, keypoints):
    """
    Uses SAM2ImagePredictor to segment the human in an image using skeleton keypoints as input prompts.
    """
    # Convert OpenCV BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image for segmentation
    sam_predictor.set_image(image_rgb)

    # Predict segmentation mask using keypoints
    masks, _, _ = sam_predictor.predict(
        point_coords=keypoints, 
        point_labels=np.ones(len(keypoints)),  # 1 for foreground
        multimask_output=False
    )

    return masks[0]  # Return the best mask

def process_image(image_path, output_mask_path):
    """
    Full pipeline: Extracts keypoints, displays the skeleton, segments human, and saves the mask.
    """
    image, keypoints, annotated_image = extract_keypoints(image_path)
    if keypoints is None:
        return

    # Save and display skeleton image
    skeleton_image_path = "annotated_skeleton.png"
    cv2.imwrite(skeleton_image_path, annotated_image)
    # cv2.imshow("Skeleton Image", annotated_image)
    # cv2.waitKey(0)

    # Segment using SAM2
    human_mask = segment_human(image, keypoints)

    # Save segmentation mask
    mask_image = Image.fromarray((human_mask * 255).astype(np.uint8))
    mask_image.save(output_mask_path)

    # Overlay segmentation on image for visualization
    segmented_image = image.copy()
    segmented_image[human_mask > 0] = [255, 0, 0]  # Highlight mask in red

    # Display results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Skeleton")

    plt.subplot(1, 2, 2)
    plt.imshow(segmented_image)
    plt.title("Segmented Human Mask")

    plt.show()

    cv2.destroyAllWindows()

# Example Usage
process_image("hamer_detector/example_data/realsense-test/frame_000000.png", "output_mask.png")
