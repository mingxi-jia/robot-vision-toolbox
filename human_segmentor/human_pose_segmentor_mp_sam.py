import os
import cv2
import numpy as np
import torch
import mediapipe as mp
import matplotlib.pyplot as plt
from PIL import Image
import torch

# Import SAM2 from your specified relative path
import sys
sys.path.append("submodules/segment-anything-2-real-time/sam2")

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor

# Initialize MediaPipe Pose with segmentation enabled
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True,
                           min_detection_confidence=0.1, min_tracking_confidence=0.3)
pose_static = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                            min_detection_confidence=0.1, min_tracking_confidence=0.1)

# Load SAM2 Model
checkpoint = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
sam_predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint).to("cuda"))

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

def compute_landmark_flow(prev_frame, curr_frame, prev_landmarks):
    """
    Computes optical flow from prev_frame to curr_frame at landmark positions.
    Returns the moved landmark positions and motion vectors.
    """
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)

    prev_pts = np.array(prev_landmarks, dtype=np.float32).reshape(-1, 1, 2)
    next_pts, status, _ = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, prev_pts, None)

    # Filter valid points
    good_prev = prev_pts[status == 1].reshape(-1, 2)
    good_next = next_pts[status == 1].reshape(-1, 2)

    return good_prev, good_next


#     return image, sampled_points, annotated_image, segmentation_mask
def extract_segmentation_points(image_path):
    """
    Extracts pose landmarks and segmentation mask from an image using MediaPipe.
    Keeps only the largest connected region in the mask.
    Returns pose keypoints, refined mask, and pose-annotated image.
    """
    if isinstance(image_path, str):
        image = cv2.imread(image_path)
    else:
        image = image_path
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose_video.process(image_rgb)
    if not results.pose_landmarks or results.segmentation_mask is None:
        results = pose_static.process(image_rgb)  # fallback to static mode for higher confidence
        if not results.pose_landmarks or results.segmentation_mask is None:
            return None, None, None, None

    height, width, _ = image.shape

    # ➕ Binarize segmentation mask
    raw_mask = (results.segmentation_mask > 0.5).astype(np.uint8)

    # 🔍 Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw_mask, connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        largest_mask = np.zeros_like(raw_mask)
        largest_mask[labels == largest_label] = 1
    else:
        largest_mask = raw_mask

    # ✋ Use pose landmarks as segmentation points from upper body landmarks
    upper_body_indices = [0, 11, 12, 13, 14, 15, 16, 23, 24]  # nose, shoulders, elbows, wrists, clavicle
    landmarks = results.pose_landmarks.landmark
    sampled_points = []
    for idx in upper_body_indices:
        lm = landmarks[idx]
        if lm.visibility > 0.5:
            x = int(lm.x * width)
            y = int(lm.y * height)
            sampled_points.append([x, y])
    # Add evenly spaced vertical points down the torso between shoulders and mid-torso
    if 11 in upper_body_indices and 12 in upper_body_indices and 23 in upper_body_indices and 24 in upper_body_indices:
        left_shoulder = landmarks[11]
        right_shoulder = landmarks[12]
        left_hip = landmarks[23]
        right_hip = landmarks[24]

        center_shoulder = ((left_shoulder.x + right_shoulder.x) / 2, (left_shoulder.y + right_shoulder.y) / 2)
        center_hip = ((left_hip.x + right_hip.x) / 2, (left_hip.y + right_hip.y) / 2)

        for t in np.linspace(0.2, 0.8, num=4):
            x = int((1 - t) * center_shoulder[0] * width + t * center_hip[0] * width)
            y = int((1 - t) * center_shoulder[1] * height + t * center_hip[1] * height)
            sampled_points.append([x, y])
    # Sample points along left and right arms (shoulder to wrist)
    def sample_line(lm_start, lm_end, steps=3):
        return [
            [int((1 - t) * lm_start.x * width + t * lm_end.x * width),
             int((1 - t) * lm_start.y * height + t * lm_end.y * height)]
            for t in np.linspace(0.2, 0.8, steps)
            if lm_start.visibility > 0.3 and lm_end.visibility > 0.2
        ]

    left_arm_points = sample_line(landmarks[11], landmarks[15])  # left shoulder to left wrist
    right_arm_points = sample_line(landmarks[12], landmarks[16])  # right shoulder to right wrist

    sampled_points.extend(left_arm_points)
    sampled_points.extend(right_arm_points)

    sampled_points = np.array(sampled_points)


    # Draw skeleton overlay
    annotated_image = image.copy()
    mp_drawing.draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style()
    )
    # cv2.imshow("landmarks", annotated_image)
    # cv2.waitKey(1)
    return image, sampled_points, annotated_image, largest_mask

def extend_arm_from_depth(hand_mask, depth_image, max_arm_length=300, depth_tolerance_ratio=0.05):
    """
    Grows a region from the hand mask in the depth image to estimate arm area.

    Args:
        hand_mask (np.ndarray): Binary mask indicating hand region.
        depth_image (np.ndarray): Depth image (in meters or millimeters).
        max_arm_length (int): Max number of pixels to grow.
        depth_tolerance_ratio (float): Percentage tolerance relative to starting depth (e.g. 0.05 = 5%).

    Returns:
        np.ndarray: Binary mask of the extended arm region.
    """
    h, w = hand_mask.shape
    arm_mask = np.zeros_like(hand_mask, dtype=np.uint8)

    # Find starting point from hand mask
    ys, xs = np.where(hand_mask > 0)
    if len(xs) == 0:
        return arm_mask  # No hand detected

    start_y = int(np.mean(ys))
    start_x = int(np.mean(xs))
    start_depth = depth_image[start_y, start_x]

    if start_depth == 0:
        return arm_mask  # Invalid depth

    # Compute dynamic tolerance window
    depth_tolerance = depth_tolerance_ratio * start_depth
    lower_bound = start_depth - depth_tolerance
    upper_bound = start_depth + depth_tolerance
    # Convert depth to grayscale if needed
    if len(depth_image.shape) == 3:
        depth_gray = cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
    else:
        depth_gray = depth_image.copy()

    # Threshold depth
    in_range_mask = (
        (depth_gray >= lower_bound) &
        (depth_gray <= upper_bound) &
        (depth_gray > 0)
    ).astype(np.uint8)

    # Connected components
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(in_range_mask, connectivity=8)

    # Select component that overlaps with hand
    selected_label = None
    for label_id in range(1, num_labels):
        component_mask = (labels == label_id).astype(np.uint8)
        overlap = cv2.bitwise_and(component_mask, hand_mask)
        if np.any(overlap):
            selected_label = label_id
            break

    if selected_label is not None:
        arm_mask[labels == selected_label] = 1

    return arm_mask


def segment_human(image, sampled_points):
    """
    Uses SAM2 to refine segmentation using evenly sampled points from MediaPipe's segmentation mask.
    Keeps only the largest connected region.
    """
    # Convert OpenCV BGR to RGB for SAM2
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Set image for segmentation
    sam_predictor.set_image(image_rgb)

    # Predict segmentation mask using sampled points
    # masks, _, _ = sam_predictor.predict(
    #     point_coords=sampled_points, 
    #     point_labels=np.ones(len(sampled_points)),  # 1 for foreground
    #     multimask_output=False
    # )
    # Convert inputs to tensors and move to CUDA
    point_coords = torch.tensor(sampled_points, dtype=torch.float32).to("cuda")
    point_labels = torch.ones(len(sampled_points), dtype=torch.int64).to("cuda")

    # Predict with GPU inputs
    masks, _, _ = sam_predictor.predict(
        point_coords=point_coords,
        point_labels=point_labels,
        multimask_output=False
    )
    segmentation_mask = masks[0]

    # 🔥 Morphological processing
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))
    segmentation_mask = cv2.morphologyEx(segmentation_mask, cv2.MORPH_CLOSE, np.ones((3,3), np.uint8))
    segmentation_mask = expand_mask(segmentation_mask, kernel_size=7, iterations=1)

    # 🔍 Keep only the largest connected component
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(segmentation_mask.astype(np.uint8), connectivity=8)
    if num_labels > 1:
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        filtered_mask = np.zeros_like(segmentation_mask, dtype=np.uint8)
        filtered_mask[labels == largest_label] = 1
        segmentation_mask = filtered_mask

    return segmentation_mask


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

    # Replace the top 12/5 of the image with the reference background
    height = image.shape[0]
    top_section = 2*height // 5  # Calculate the height for top 1/3
    result_image[:top_section, :] = reference_resized[:top_section, :]  # Replace top section

    return result_image

def sample_mask_points(hand_mask, num_points=10):
    """
    Samples points from a hand mask and visualizes them.

    Args:
        hand_mask (np.ndarray): Binary or grayscale hand mask (0–255).
        num_points (int): Number of points to sample.

    Returns:
        np.ndarray or None: Sampled (x, y) point coordinates.
    """
    if hand_mask is None:
        print("⚠️ No hand mask provided.")
        return None

    # Ensure binary mask and dilate

    # Get foreground pixel coordinates
    ys, xs = np.where(hand_mask > 0)
    if len(xs) == 0:
        print("⚠️ No foreground pixels in mask.")
        return None

    # Sample random indices
    indices = np.random.choice(len(xs), size=min(num_points, len(xs)), replace=False)
    sampled_points = np.stack([xs[indices], ys[indices]], axis=1)

    return sampled_points

def process_image_folder(image_folder, output_folder, background_path=None, hand_model_path = None, debug = True, depth_folder = None):
    """
    Processes a folder of images using MediaPipe + SAM2 segmentation,
    falls back to MediaPipe or optical flow when needed,
    and optionally replaces background using a reference image.
    """
    os.makedirs(output_folder, exist_ok=True)

    image_paths = sorted([
        os.path.join(image_folder, f)
        for f in os.listdir(image_folder)
        if f.endswith(('.jpg', '.png'))
    ])

    depth_paths = sorted([
        os.path.join(depth_folder, f)
        for f in os.listdir(depth_folder)
        if f.endswith('.npy')
    ])

    if len(depth_paths) < 1:
        depth_paths = sorted([
        os.path.join(depth_folder, f)
        for f in os.listdir(depth_folder)
        if f.endswith('.png')
    ])
    if background_path is not None:
        reference_image = cv2.imread(background_path)
        if reference_image is None:
            raise ValueError(f"❌ Could not read background image at {background_path}")
    else:
        reference_image = None

    prev_frame = None
    prev_landmarks = None

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)
        
        depth_image = np.load(depth_paths[idx])
        frame_count = int(image_paths[idx][-10:-4])

        final_output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_final.png")
        final_debug_output_path = os.path.join(output_folder, f"debug_frame_{frame_count:06d}_final[DEBUG].png")
        landmark_flow = None

        height, width = frame.shape[:2]
        # Load and sample hand mask points before SAM2
        suffix = f"{frame_count:06d}_handmask.png"
        matching_files = [f for f in os.listdir(hand_model_path) if f.endswith(suffix)]
        hand_mask = None
        if matching_files:
            mask_path = os.path.join(hand_model_path, matching_files[0])
            hand_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if hand_mask is not None:
                hand_mask = (hand_mask > 127).astype(np.uint8)
                ys, xs = np.where(hand_mask > 0)
        else:
            continue

        # Use MediaPipe to get pose landmarks (arm points)
        image, mp_sampled_points, _, mediapipe_mask = extract_segmentation_points(frame)

        # sample some points from loaded hand_mask
        hand_sampled_points = sample_mask_points(hand_mask, num_points = 5)


        # Decide whether to trust MediaPipe or fall back to depth
        use_depth_arm = False
        if hand_mask is not None and mediapipe_mask is not None:
            overlap = cv2.bitwise_and(hand_mask, mediapipe_mask)
            if not np.any(overlap):
                print("⚠️ No overlap between hand mask and MediaPipe mask — using depth to find arm.")
                use_depth_arm = True

        if use_depth_arm:
            arm_mask = extend_arm_from_depth(hand_mask, depth_image, depth_tolerance_ratio = 0.1)
            arm_sampled_points = sample_mask_points(arm_mask,num_points = 15)
        else:
            arm_sampled_points = mp_sampled_points
            arm_mask = mediapipe_mask

        # Get a union of all segmented points and use SAM to get the final mask
              # Combine sampled points
        full_sampled_points = []
        for pts in [hand_sampled_points, arm_sampled_points]:
            if pts is not None:
                full_sampled_points.append(pts)
        full_sampled_points = np.vstack(full_sampled_points) if full_sampled_points else None

        final_sam_mask = None

        # Case 1: run SAM normally
        if full_sampled_points is not None and len(full_sampled_points) >= 4:
            final_sam_mask = segment_human(frame, full_sampled_points)

        # Case 2: fallback to optical flow
        elif prev_frame is not None and prev_landmarks is not None:
            good_prev, good_next = compute_landmark_flow(prev_frame, frame, prev_landmarks)
            if len(good_next) >= 4:
                full_sampled_points = good_next.astype(int)
                print("🌀 Using optical flow fallback points.")
                final_sam_mask = segment_human(frame, full_sampled_points)
            else:
                print("❌ Optical flow fallback also failed.")
                continue  # skip frame

        else:
            print(f"❌ Skipping frame {frame_count} — no fallback available.")
            continue

        final_result = replace_background(frame, final_sam_mask, reference_image) if reference_image is not None else frame.copy()

        cv2.imwrite(final_output_path, final_result)

        # DEBUG:
        if debug:

            fig, axs = plt.subplots(2, 2, figsize=(14, 10))  # 2x2 grid

            # Fig 1: Original RGB image
            axs[0, 0].imshow(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            axs[0, 0].set_title("Original Image")
            axs[0, 0].axis('off')

            # Fig 2: Depth image
            depth_vis = depth_image if depth_image.ndim == 2 else cv2.cvtColor(depth_image, cv2.COLOR_BGR2GRAY)
            axs[0, 1].imshow(depth_vis, cmap='gray')
            axs[0, 1].set_title("Depth Image")
            axs[0, 1].axis('off')

            # Fig 3: Combined sample points and masks
            overlay_img = cv2.cvtColor(frame.copy(), cv2.COLOR_BGR2RGB)
            axs[1, 0].imshow(overlay_img)
            if mp_sampled_points is not None:
                axs[1, 0].scatter(mp_sampled_points[:, 0], mp_sampled_points[:, 1], color='green', label='Pose')
            if hand_sampled_points is not None:
                axs[1, 0].scatter(hand_sampled_points[:, 0], hand_sampled_points[:, 1], color='red', label='Hand')
            if arm_sampled_points is not None:
                axs[1, 0].scatter(arm_sampled_points[:, 0], arm_sampled_points[:, 1], color='blue', label='Arm')
            axs[1, 0].set_title("Sample Points on Mask Overlay")
            axs[1, 0].axis('off')
            axs[1, 0].legend(loc='lower right', fontsize='small')

            # Also draw semi-transparent masks on top
            combined_mask = np.zeros_like(frame[:, :, 0], dtype=np.uint8)
            if hand_mask is not None:
                combined_mask[hand_mask > 0] = 100
            if arm_mask is not None:
                combined_mask[arm_mask > 0] += 150  # additive to show overlap as brighter
            axs[1, 0].imshow(combined_mask, cmap='hot', alpha=0.4)

            # Fig 4: Final SAM mask
            axs[1, 1].imshow((final_sam_mask if final_sam_mask is not None else np.zeros_like(frame[:, :, 0])) * 255, cmap='gray')
            axs[1, 1].set_title("Final SAM Mask")
            axs[1, 1].axis('off')

            plt.tight_layout()
            plt.savefig(final_debug_output_path, dpi=300, bbox_inches='tight')



# image_folder = "/home/xhe71/Desktop/robotool_data/Color_frames"
# output_folder = "/home/xhe71/Desktop/robotool_data/Color_segmented"
# background_path = "/home/xhe71/Desktop/robotool_data/Color/color_000003.png"
# hand_model_path = "/home/xhe71/Desktop/robotool_data/Color_hamer"
# process_image_folder(image_folder, output_folder, background_path, hand_model_path)