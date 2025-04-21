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
pose_video = mp_pose.Pose(static_image_mode=False, model_complexity=2, enable_segmentation=True,
                           min_detection_confidence=0.3, min_tracking_confidence=0.3)
pose_static = mp_pose.Pose(static_image_mode=True, model_complexity=2, enable_segmentation=True,
                            min_detection_confidence=0.3, min_tracking_confidence=0.3)

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


def warp_mask_with_optical_flow(prev_frame, current_frame, prev_mask):
    prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
    curr_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(prev_gray, curr_gray, None,
                                        pyr_scale=0.5, levels=3, winsize=15,
                                        iterations=3, poly_n=5, poly_sigma=1.2, flags=0)

    h, w = prev_mask.shape
    flow_map = np.stack(np.meshgrid(np.arange(w), np.arange(h)), axis=-1).astype(np.float32)
    remap = (flow_map + flow).astype(np.float32)
    warped_mask = cv2.remap(prev_mask.astype(np.float32), remap[..., 0], remap[..., 1], interpolation=cv2.INTER_LINEAR)
    warped_mask = (warped_mask > 0.5).astype(np.uint8)

    return warped_mask, flow

def visualize_segmentation_debug(original, skeleton=None, mediapipe_mask=None, refined_mask=None, final=None,
                                  fallback_mask=None, flow=None, title_suffix="", save_path=None):
    """
    Displays segmentation and processing steps across 2 rows for debugging.
    Adds optical flow quiver plot using tracked pose landmarks.
    """
    visuals = []
    titles = []

    visuals.append(cv2.cvtColor(original, cv2.COLOR_BGR2RGB))
    titles.append(f"Original Frame {title_suffix}")

    if skeleton is not None:
        visuals.append(cv2.cvtColor(skeleton, cv2.COLOR_BGR2RGB))
        titles.append("Pose Skeleton")

    if mediapipe_mask is not None:
        visuals.append(mediapipe_mask)
        titles.append("MediaPipe Mask")

    if refined_mask is not None:
        visuals.append(refined_mask)
        titles.append("SAM2 Refined Mask")

    if fallback_mask is not None:
        visuals.append(fallback_mask)
        titles.append("Final Used Mask")

    if final is not None:
        visuals.append(cv2.cvtColor(final, cv2.COLOR_BGR2RGB))
        titles.append("Final Composite")

    if flow is not None:
        visuals.append("FLOW_PLOT")  # special marker
        titles.append("Landmark Optical Flow")

    total = len(visuals)
    cols = (total + 1) // 2
    rows = 2
    plt.figure(figsize=(4 * cols, 6))

    for idx, (img, title) in enumerate(zip(visuals, titles)):
        plt.subplot(rows, cols, idx + 1)
        if isinstance(img, str) and img == "FLOW_PLOT":
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            plt.imshow(gray, cmap="gray")
            if flow is not None and isinstance(flow, tuple) and len(flow) == 2:
                prev_pts, next_pts = flow
                fx = next_pts[:, 0] - prev_pts[:, 0]
                fy = next_pts[:, 1] - prev_pts[:, 1]
                plt.quiver(prev_pts[:, 0], prev_pts[:, 1], fx, fy, color='r', angles='xy', scale_units='xy', scale=1)

        plt.title(title)
        plt.axis("off")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path + title_suffix + ".png", dpi=150)



#     return image, sampled_points, annotated_image, segmentation_mask
def extract_segmentation_points(image_path, num_points=10):
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
            if lm_start.visibility > 0.5 and lm_end.visibility > 0.5
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
    masks, _, _ = sam_predictor.predict(
        point_coords=sampled_points, 
        point_labels=np.ones(len(sampled_points)),  # 1 for foreground
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

    # Replace the top 1/3 of the image with the reference background
    height = image.shape[0]
    top_section = 2*height // 5  # Calculate the height for top 1/3
    result_image[:top_section, :] = reference_resized[:top_section, :]  # Replace top section

    return result_image

def process_video(video_path, output_folder, background_path=None, hand_mask_folder = None, handedness = 'left'):
    """
    Processes a video frame by frame using MediaPipe + SAM2 segmentation,
    falls back to MediaPipe or optical flow when needed,
    and optionally replaces background using a reference image.
    """
    os.makedirs(output_folder, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    fps = int(cap.get(cv2.CAP_PROP_FPS))

    reference_image = None
    if background_path is not None:
        reference_image = cv2.imread(background_path)
        if reference_image is None:
            raise ValueError(f"❌ Could not read background image at {background_path}")

    prev_frame = None
    prev_mask = None
    prev_landmarks = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        final_output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_final.png")
        landmark_flow = None

        image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(frame)
        height, width = frame.shape[:2]

        # Define valid region: bottom 2/3 of the image
        if mediapipe_mask is not None:
            valid_region_mask = np.zeros_like(mediapipe_mask, dtype=np.uint8)
            valid_region_mask[int(height * (1 / 3)):, :] = 1
        else:
            valid_region_mask = np.ones((height, width), dtype=np.uint8)

        if sampled_points is not None:
            # Step 1: Run SAM2
            segmentation_mask = segment_human(image, sampled_points)
            refined_mask = segmentation_mask.copy()

            # Step 2: Mask out invalid regions
            masked_mp = np.logical_and(mediapipe_mask, valid_region_mask)
            masked_sam = np.logical_and(segmentation_mask, valid_region_mask)

            mp_area = np.sum(masked_mp)
            sam_area = np.sum(masked_sam)

            min_mask_area = 500  # can be tuned
            use_mp_mask = False

            if mp_area < min_mask_area:
                
                print(f"⚠️ Frame {frame_count}: MediaPipe mask too small (area={mp_area}) — ignoring it.")
                continue
            else:
                intersection = np.logical_and(masked_sam, masked_mp).sum()
                union = np.logical_or(masked_sam, masked_mp).sum()
                iou = intersection / union if union > 0 else 0
                print(f"ℹ️ Frame {frame_count}: IoU = {iou:.2f}")

                if iou < 0.1:
                    print(f"⚠️ Frame {frame_count}: Low IoU — using MediaPipe mask instead.")
                    refined_mask = mediapipe_mask.copy()
                    use_mp_mask = True

            # Step 3: Fallback to optical flow if both masks are poor
            if np.sum(np.logical_and(refined_mask, valid_region_mask)) < min_mask_area and not use_mp_mask:
                print(f"❌ Frame {frame_count}: Both SAM2 and MP failed — using optical flow fallback.")
                if prev_frame is not None and prev_mask is not None:
                    refined_mask, _ = warp_mask_with_optical_flow(prev_frame, frame, prev_mask)

            # Step 4: Replace background
            final_result = replace_background(image, refined_mask, reference_image) if reference_image is not None else frame.copy()

            # Step 5: Compute landmark flow
            if prev_frame is not None and prev_landmarks is not None:
                good_prev, good_next = compute_landmark_flow(prev_frame, frame, prev_landmarks)
                if len(good_prev) > 0:
                    landmark_flow = (good_prev, good_next)

            # Step 6: Save data for next frame
            prev_frame = frame.copy()
            prev_mask = refined_mask.copy()
            prev_landmarks = sampled_points.copy()

            # Step 7: Visualize & Save
            visualize_segmentation_debug(
                original=frame,
                skeleton=annotated_image,
                mediapipe_mask=masked_mp,
                refined_mask=masked_sam,
                fallback_mask=refined_mask,
                final=final_result,
                flow=landmark_flow,
                title_suffix=f"(Frame {frame_count})",
                save_path = "hamer_detector/example_data/test-env-pose-seg-2-tmp/"
            )

            cv2.imwrite(final_output_path, final_result)

        else:
            print(f"⚠️ Frame {frame_count}: No pose detected — using optical flow points as input to SAM2.")
            if prev_frame is not None and prev_landmarks is not None:
                good_prev, good_next = compute_landmark_flow(prev_frame, frame, prev_landmarks)
                if len(good_next) >= 4:
                    print(f"ℹ️ Using {len(good_next)} tracked landmarks for SAM2.")
                    segmentation_mask = segment_human(frame, good_next.astype(int))
                    refined_mask = segmentation_mask.copy()

                    final_result = replace_background(frame, refined_mask, reference_image) if reference_image is not None else frame.copy()

                    visualize_segmentation_debug(
                        original=frame,
                        fallback_mask=refined_mask,
                        final=final_result,
                        flow=(good_prev, good_next),
                        title_suffix=f"(Frame {frame_count} - SAM2 via Flow Landmarks)",
                        save_path = "hamer_detector/example_data/test-env-pose-seg-2-tmp/"
                    )

                    cv2.imwrite(final_output_path, final_result)

                    prev_frame = frame.copy()
                    prev_mask = refined_mask.copy()
                    prev_landmarks = good_next.copy()
                else:
                    print(f"❌ Frame {frame_count}: Not enough flow-tracked landmarks — skipping.")
                    frame_count += 1
                    continue
            else:
                print(f"❌ Skipping frame {frame_count} — no previous pose/landmarks.")
                frame_count += 1
                continue

        frame_count += 1

    cap.release()
    print(f"✅ Processed {frame_count} frames. Now compiling into video...")

    os.system(f"ffmpeg -framerate {fps} -i {output_folder}frame_%06d_final.png -c:v libx264 -pix_fmt yuv420p {output_folder}output_video.mp4")

    print("✅ Video processing complete! Output saved to:", output_folder)

def process_image_folder(image_folder, output_folder, background_path=None, hand_model_path = None):
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

    if background_path is not None:
        reference_image = cv2.imread(background_path)
        if reference_image is None:
            raise ValueError(f"❌ Could not read background image at {background_path}")
    else:
        reference_image = None

    prev_frame = None
    prev_mask = None
    prev_landmarks = None

    for idx, image_path in enumerate(image_paths):
        frame = cv2.imread(image_path)
        frame_count = int(image_paths[idx][-10:-4])
        # frame_count = idx  # Use index for naming

        final_output_path = os.path.join(output_folder, f"frame_{frame_count:06d}_final.png")
        landmark_flow = None

        image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(frame)
        height, width = frame.shape[:2]

        if mediapipe_mask is not None:
            valid_region_mask = np.zeros_like(mediapipe_mask, dtype=np.uint8)
            valid_region_mask[int(height * (1 / 3)):, :] = 1
        else:
            valid_region_mask = np.ones((height, width), dtype=np.uint8)

        if sampled_points is not None:
            segmentation_mask = segment_human(image, sampled_points)
            refined_mask = segmentation_mask.copy()

            masked_mp = np.logical_and(mediapipe_mask, valid_region_mask)
            masked_sam = np.logical_and(segmentation_mask, valid_region_mask)

            mp_area = np.sum(masked_mp)
            sam_area = np.sum(masked_sam)

            min_mask_area = 500
            use_mp_mask = False

            if mp_area < min_mask_area:
                print(f"⚠️ Frame {frame_count}: MediaPipe mask too small (area={mp_area}) — skipping.")
                continue
            else:
                intersection = np.logical_and(masked_sam, masked_mp).sum()
                union = np.logical_or(masked_sam, masked_mp).sum()
                iou = intersection / union if union > 0 else 0
                print(f"ℹ️ Frame {frame_count}: IoU = {iou:.2f}")

                if iou < 0.1:
                    print(f"⚠️ Frame {frame_count}: Low IoU — using MediaPipe mask instead.")
                    refined_mask = mediapipe_mask.copy()
                    use_mp_mask = True

            if np.sum(np.logical_and(refined_mask, valid_region_mask)) < min_mask_area and not use_mp_mask:
                print(f"❌ Frame {frame_count}: Both SAM2 and MP failed — using optical flow fallback.")
                if prev_frame is not None and prev_mask is not None:
                    refined_mask, _ = warp_mask_with_optical_flow(prev_frame, frame, prev_mask)

            # merge mask with Mano hand model
            mask_path = os.path.join(hand_model_path, f"frame_{frame_count:06d}_handmask.png")
            hand_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            if hand_mask is not None:
                hand_mask = (hand_mask > 127).astype(np.uint8)
                print(f"✔️ Loaded hand mask for frame {frame_count}")
                # Apply dilation
                kernel_size = 5  # Adjust for how much wider you want it
                iterations = 1   # Number of dilation passes

                kernel = np.ones((kernel_size, kernel_size), np.uint8)
                dilated_mask = cv2.dilate(hand_mask, kernel, iterations=iterations)

                # Combine with refined human mask
                refined_mask = np.logical_or(refined_mask, dilated_mask).astype(np.uint8)
            else:
                print(f"⚠️ No hand mask found for frame {frame_count}")
            # cv2.imshow("mask", refined_mask*255)
            # cv2.waitKey(0)
            # merge hand mask into refined mask
            final_result = replace_background(image, refined_mask, reference_image) if reference_image is not None else frame.copy()


            if prev_frame is not None and prev_landmarks is not None:
                good_prev, good_next = compute_landmark_flow(prev_frame, frame, prev_landmarks)
                if len(good_prev) > 0:
                    landmark_flow = (good_prev, good_next)

            prev_frame = frame.copy()
            prev_mask = refined_mask.copy()
            prev_landmarks = sampled_points.copy()

            visualize_segmentation_debug(
                original=frame,
                skeleton=annotated_image,
                mediapipe_mask=masked_mp,
                refined_mask=masked_sam,
                fallback_mask=refined_mask,
                final=final_result,
                flow=landmark_flow,
                title_suffix=f"(Frame {frame_count})",
                save_path=os.path.join(output_folder, "debug")
            )

            cv2.imwrite(final_output_path, final_result)

        else:
            print(f"⚠️ Frame {frame_count}: No pose detected — using optical flow points as input to SAM2.")
            if prev_frame is not None and prev_landmarks is not None:
                good_prev, good_next = compute_landmark_flow(prev_frame, frame, prev_landmarks)
                if len(good_next) >= 4:
                    print(f"ℹ️ Using {len(good_next)} tracked landmarks for SAM2.")
                    segmentation_mask = segment_human(frame, good_next.astype(int))
                    refined_mask = segmentation_mask.copy()

                    # merge mask with Mano hand model
                    mask_path = os.path.join(hand_model_path, f"frame_{frame_count:06d}_handmask.png")
                    hand_mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
                    if hand_mask is not None:
                        hand_mask = (hand_mask > 127).astype(np.uint8)
                        print(f"✔️ Loaded hand mask for frame {frame_count}")
                        # Apply dilation
                        kernel_size = 5  # Adjust for how much wider you want it
                        iterations = 1   # Number of dilation passes

                        kernel = np.ones((kernel_size, kernel_size), np.uint8)
                        dilated_mask = cv2.dilate(hand_mask, kernel, iterations=iterations)

                        # Combine with refined human mask
                        refined_mask = np.logical_or(refined_mask, dilated_mask).astype(np.uint8)
                        

                    final_result = replace_background(frame, refined_mask, reference_image) if reference_image is not None else frame.copy()

                    visualize_segmentation_debug(
                        original=frame,
                        fallback_mask=refined_mask,
                        final=final_result,
                        flow=(good_prev, good_next),
                        title_suffix=f"(Frame {frame_count} - SAM2 via Flow Landmarks)",
                        save_path=os.path.join(output_folder, "debug")
                    )

                    cv2.imwrite(final_output_path, final_result)

                    prev_frame = frame.copy()
                    prev_mask = refined_mask.copy()
                    prev_landmarks = good_next.copy()
                else:
                    print(f"❌ Frame {frame_count}: Not enough flow-tracked landmarks — skipping.")
                    continue
            else:
                print(f"❌ Skipping frame {frame_count} — no previous pose/landmarks.")
                continue

    print("✅ Image folder processing complete.")


# def main():
#     """
#     Main function to process images from a folder.
#     """

#     background_path = "human_segmentor/first_frame.png"
#     video_path = "/home/xhe71/Downloads/human (1).mp4"  # Change this to your input video
#     output_folder = "hamer_detector/example_data/test-env-pose-seg-2/"

#     process_video(video_path, output_folder, background_path)
    
# if __name__ == "__main__":
#     main()
#     import os
#     import argparse
#     cwd = os.getcwd()

#     parser = argparse.ArgumentParser(description="Human segmentation and background replacement using MediaPipe + SAM2")
#     parser.add_argument("--video_path", type=str, default= "/home/xhe71/Downloads/human (1).mp4",
#                         help="Path to input video file")
#     parser.add_argument("--output_folder", type=str, default= "hamer_detector/example_data/test-env-pose-seg-2/",
#                         help="Directory to save processed frames and output video")
#     parser.add_argument("--background_path", type=str, default= "human_segmentor/first_frame.png",
#                         help="Path to background image for replacement")


# if __name__ == "__main__":
#     import argparse
#     parser = argparse.ArgumentParser(description="Segment folder of images using MediaPipe + SAM2")
#     parser.add_argument("--image_folder", type=str,default= "/home/xhe71/Desktop/robotool_test/tmp_imgs", help="Path to input image folder")
#     parser.add_argument("--output_folder", type=str,default= "hamer_detector/example_data/test-env-pose-seg-2/", help="Path to save processed images and debug outputs")
#     parser.add_argument("--background_path", type=str, default= "human_segmentor/first_frame.png", help="Background image for replacement")
#     args = parser.parse_args()

#     process_image_folder(args.image_folder, args.output_folder, args.background_path)
