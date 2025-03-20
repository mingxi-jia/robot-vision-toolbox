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
pose = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True, 
                     min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Load SAM2 Model
checkpoint = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))

def expand_mask(mask, kernel_size=5, iterations=1):
    """
    Expands the segmentation mask slightly using morphological dilation.
    
    Arguments:
    - mask: Binary segmentation mask (np.uint8, values 0 or 1).
    - kernel_size: Size of the dilation kernel (default=3).
    - iterations: Number of times to apply dilation (default=1).

    Returns:
    - Expanded segmentation mask.
    """
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    expanded_mask = cv2.dilate(mask, kernel, iterations=iterations)
    
    return expanded_mask


def extract_segmentation_points(image_path, num_points=10):
    """
    Extracts human pose keypoints and segmentation mask from an image using MediaPipe.
    Applies Gaussian blur to smooth the edges of the segmentation mask.
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if not results.pose_landmarks or results.segmentation_mask is None:
        # print(f"No valid segmentation found for {image_path}.")
        return None, None, None, None
    # drop frame if left or right wrist visibility is lower than a threshold:
    thre = 0.3
    left_wrist_visibility =  results.pose_landmarks.landmark[18].visibility
    right_wrist_visibility =  results.pose_landmarks.landmark[15].visibility
    
    if right_wrist_visibility< thre and left_wrist_visibility < thre:
        print("Did not detect wrist:", left_wrist_visibility, right_wrist_visibility)
        return None, None, None, None
    # print(results.pose_landmarks.landmark[15].visibility)
    height, width, _ = image.shape
    segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # Get all foreground pixel locations
    mask_indices = np.column_stack(np.where(segmentation_mask == 1))  # (row, col)

    if len(mask_indices) == 0:
        print(f"No valid segmentation points found for {image_path}.")
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
    Uses SAM2 to refine segmentation using evenly sampled points from MediaPipe's segmentation mask.
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

    segmentation_mask = masks[0]
        # 🔥 Apply Morphological OPENING to Remove Small Noise
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

    # 🔥 Apply Morphological CLOSING to Fill Small Holes
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))

    # 🔥 Expand the Mask Slightly (New Fix)
    segmentation_mask = expand_mask(segmentation_mask, kernel_size=7, iterations=1)


    return segmentation_mask  # Return the best mask

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
    result_image[mask == 1] = reference_resized[mask == 1]  # Replace background pixels

    # Replace the top 1/3 of the image with the reference background
    height = image.shape[0]
    top_section = height // 3  # Calculate the height for top 1/3
    result_image[:top_section, :] = reference_resized[:top_section, :]  # Replace top section

    return result_image

def process_image(image_path, output_mask_path, reference_path=None, output_final_path=None):
    """
    Full pipeline: Extracts segmentation points, displays skeleton, refines segmentation with SAM2,
    and optionally replaces the background using a reference image.
    """
    image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(image_path)
    if sampled_points is None:
        return

    # Save and display skeleton image
    skeleton_image_path = os.path.join(os.path.dirname(output_mask_path), "annotated_skeleton.png")
    # cv2.imwrite(skeleton_image_path, annotated_image)

    # Segment using SAM2
    refined_mask = segment_human(image, sampled_points)

    # Save segmentation mask
    mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
    mask_image.save(output_mask_path)

    final_result = None
    if reference_path and output_final_path:
        # Load reference background image
        reference_image = cv2.imread(reference_path)

        # Replace background
        final_result = replace_background(image, refined_mask, reference_image)

        # Save final composited image
        cv2.imwrite(output_final_path, final_result)

    # Display results
    plt.figure(figsize=(20, 5))

    plt.subplot(1, 4, 1)
    plt.imshow(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB))
    plt.title("Extracted Skeleton")

    plt.subplot(1, 4, 2)
    plt.imshow(mediapipe_mask, cmap="gray")
    plt.title("MediaPipe Segmentation Mask")

    plt.subplot(1, 4, 3)
    plt.imshow(refined_mask, cmap="gray")
    plt.title("SAM2 Refined Mask")

    if final_result is not None:
        plt.subplot(1, 4, 4)
        plt.imshow(cv2.cvtColor(final_result, cv2.COLOR_BGR2RGB))
        plt.title("Final Image with New Background")

    # plt.show()

def process_folder(input_folder, output_folder, reference_path=None):
    """
    Processes all images in a folder.
    
    Arguments:
    - input_folder: Folder containing images to process.
    - output_folder: Folder to save results.
    - reference_path: Optional reference background image for replacement.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_files = [f for f in os.listdir(input_folder) if f.endswith((".png", ".jpg", ".jpeg"))]
    
    for image_file in image_files:
        image_path = os.path.join(input_folder, image_file)
        mask_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_mask.png")
        final_output_path = os.path.join(output_folder, f"{os.path.splitext(image_file)[0]}_final.png")

        print(f"Processing {image_path}...")
        process_image(image_path, mask_output_path, reference_path, final_output_path if reference_path else None)


def process_video(video_path, output_folder):
    """
    Processes a video frame by frame, applying segmentation and replacing background with first frame.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))  # Get video FPS

    reference_image = None  # Placeholder for reference background

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Save first frame as reference background
        if frame_count == 0:
            reference_image = frame.copy()

        frame_filename = f"frame_{frame_count:06d}.png"
        # mask_output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_mask.png")
        final_output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_final.png")

        image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(frame)

        if sampled_points is not None:
            refined_mask = segment_human(image, sampled_points)

            # Save segmentation mask
            mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
            # mask_image.save(mask_output_path)

            final_result = None
            if reference_image is not None:
                final_result = replace_background(image, refined_mask, reference_image)
                cv2.imwrite(final_output_path, final_result)

        frame_count += 1

    cap.release()
    print(f"✅ Processed {frame_count} frames. Now compiling into video...")

    # Compile frames into a video using ffmpeg
    os.system(f"ffmpeg -framerate {fps} -i {output_folder}frame_%06d_final.png -c:v libx264 -pix_fmt yuv420p {output_folder}output_video.mp4")

    print("✅ Video processing complete! Output saved to:", output_folder)


def main():
    """
    Main function to process images from a folder.
    """
    # input_folder = "hamer_detector/example_data/realsense-test/"
    # output_folder = "hamer_detector/example_data/realsense-test-pose-seg/"
    # reference_image = "hamer_detector/example_data/realsense-test/frame_000180.png" # Set to None if no background replacement is needed
    ####################3 
    # input_folder = "hamer_detector/example_data/test-env/"
    # output_folder = "hamer_detector/example_data/test-env-pose-seg/"
    # reference_image = "hamer_detector/example_data/test-env/frame_000000.png" # Set to None if no background replacement is needed

    # process_folder(input_folder, output_folder, reference_image)



    video_path = "/home/xhe71/Downloads/human (1).mp4"  # Change this to your input video
    output_folder = "hamer_detector/example_data/test-env-pose-seg-2/"

    process_video(video_path, output_folder)

if __name__ == "__main__":
    main()
