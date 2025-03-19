import os
import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image

# Initialize MediaPipe Pose with segmentation enabled
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True)

def extract_keypoints_and_mask(image_path):
    """
    Extracts human pose keypoints and segmentation mask from an image using MediaPipe.
    Returns keypoints in (x, y) format, the annotated skeleton image, and the segmentation mask.
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    # if not results.pose_landmarks or not results.segmentation_mask:
    #     print("No person detected.")
    #     return None, None, None, None

    height, width, _ = image.shape
    keypoints = [(int(lm.x * width), int(lm.y * height)) for lm in results.pose_landmarks.landmark]

    # Draw the skeleton on the image
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

    # Convert segmentation mask to binary mask
    human_mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    return image, np.array(keypoints), annotated_image, human_mask

def process_image(image_path, output_mask_path):
    """
    Full pipeline: Extracts keypoints, displays the skeleton, segments human, and saves the mask.
    """
    image, keypoints, annotated_image, human_mask = extract_keypoints_and_mask(image_path)
    if keypoints is None or human_mask is None:
        return

    # Save and display skeleton image
    skeleton_image_path = "annotated_skeleton.png"
    cv2.imwrite(skeleton_image_path, annotated_image)

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

# Example Usage
process_image("hamer_detector/example_data/realsense-test/frame_000000.png", "output_mask.png")
