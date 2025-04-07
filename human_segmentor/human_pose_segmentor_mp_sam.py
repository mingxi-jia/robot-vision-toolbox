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
        elif img.ndim == 2:
            plt.imshow(img, cmap="gray")
        else:
            plt.imshow(img)
        plt.title(title)
        plt.axis("off")

    if save_path:
        plt.savefig(save_path)
    else:
        plt.tight_layout()
        plt.show()

    plt.close()



# def extract_segmentation_points(image_path, num_points=10):
#     """
#     Extracts human pose keypoints and segmentation mask from an image using MediaPipe.
#     Applies Gaussian blur to smooth the edges of the segmentation mask.
#     """
#     if isinstance(image_path, str):
#         image = cv2.imread(image_path)
#     else:
#         image = image_path
#     image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     results = pose.process(image_rgb)

#     if not results.pose_landmarks or results.segmentation_mask is None:
#         # print(f"No valid segmentation found for {image_path}.")
#         return None, None, None, None
#     # drop frame if left or right wrist visibility is lower than a threshold:
#     # thre = 0.3
#     # left_wrist_visibility =  results.pose_landmarks.landmark[18].visibility
#     # right_wrist_visibility =  results.pose_landmarks.landmark[15].visibility
    
#     # if right_wrist_visibility< thre and left_wrist_visibility < thre:
#     #     print("Did not detect wrist:", left_wrist_visibility, right_wrist_visibility)
#     #     return None, None, None, None
#     # print(results.pose_landmarks.landmark[15].visibility)
#     height, width, _ = image.shape
#     segmentation_mask = (results.segmentation_mask > 0.5).astype(np.uint8)

#     # Get all foreground pixel locations
#     mask_indices = np.column_stack(np.where(segmentation_mask == 1))  # (row, col)

#     if len(mask_indices) == 0:
#         print(f"No valid segmentation points found for {image_path}.")
#         return None, None, None, None

#     # Sort points to ensure even spacing
#     mask_indices = mask_indices[np.lexsort((mask_indices[:, 1], mask_indices[:, 0]))]  # Sort by (y, x)

#     # Evenly sample points from the sorted indices
#     step = max(1, len(mask_indices) // num_points)  # Step size to evenly pick points
#     sampled_indices = mask_indices[::step][:num_points]  # Take every `step`th point

#     sampled_points = np.flip(sampled_indices, axis=1).copy()  # Convert (row, col) to (x, y) and fix negative strides

#     # Draw the skeleton on the image
#     annotated_image = image.copy()
#     mp_drawing.draw_landmarks(
#         annotated_image,
#         results.pose_landmarks,
#         mp_pose.POSE_CONNECTIONS,
#         landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

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
    results = pose.process(image_rgb)

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

    # 🖼️ Show segmentation mask
    # plt.figure(figsize=(6, 6))
    # plt.imshow(largest_mask, cmap="gray")
    # plt.title("MediaPipe Largest Segment")
    # plt.axis("off")
    # plt.show()

    # ✋ Use pose landmarks as segmentation points
    landmarks = results.pose_landmarks.landmark
    sampled_points = []
    for lm in landmarks:
        if lm.visibility > 0.5:  # Only include visible keypoints
            x = int(lm.x * width)
            y = int(lm.y * height)
            sampled_points.append([x, y])
    sampled_points = np.array(sampled_points)


    # 🎯 Draw skeleton overlay
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
    top_section = height // 3  # Calculate the height for top 1/3
    result_image[:top_section, :] = reference_resized[:top_section, :]  # Replace top section

    return result_image

# def process_image(image_path, output_mask_path, reference_path=None, output_final_path=None):
#     """
#     Full pipeline: Extracts segmentation points, displays skeleton, refines segmentation with SAM2,
#     and optionally replaces the background using a reference image.
#     """
#     image, sampled_points, annotated_image, mediapipe_mask = extract_segmentation_points(image_path)
#     if sampled_points is None:
#         return

#     # Segment using SAM2
#     refined_mask = segment_human(image, sampled_points)

#     # Compare areas and fallback to MediaPipe mask if SAM2 is too small
#     sam_area = np.sum(refined_mask)
#     mp_area = np.sum(mediapipe_mask)
#     if sam_area < 0.75 * mp_area:
#         print("⚠️ Replacing SAM2 mask with MediaPipe mask (too small)")
#         refined_mask = mediapipe_mask

#     # Save segmentation mask
#     mask_image = Image.fromarray((refined_mask * 255).astype(np.uint8))
#     mask_image.save(output_mask_path)

#     final_result = None
#     if reference_path and output_final_path:
#         # Load reference background image
#         reference_image = cv2.imread(reference_path)

#         # Replace background
#         final_result = replace_background(image, refined_mask, reference_image)

#         # Save final composited image
#         cv2.imwrite(output_final_path, final_result)

#     # Debug visualization
#     visualize_segmentation_debug(
#         original=image,
#         skeleton=annotated_image,
#         mediapipe_mask=mediapipe_mask,
#         refined_mask=refined_mask,
#         final=final_result,
#         title_suffix=f"(Image)"
#     )
def process_video(video_path, output_folder, background_path=None):
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
            valid_region_mask = np.ones((height, frame.shape[1]), dtype=np.uint8)  # fallback: full image valid

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
                title_suffix=f"(Frame {frame_count})"
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
                        title_suffix=f"(Frame {frame_count} - SAM2 via Flow Landmarks)"
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

def main():
    """
    Main function to process images from a folder.
    """

    background_path = "human_segmentor/first_frame.png"
    video_path = "/home/xhe71/Downloads/human (1).mp4"  # Change this to your input video
    output_folder = "hamer_detector/example_data/test-env-pose-seg-2/"

    process_video(video_path, output_folder, background_path)
    
if __name__ == "__main__":
    main()

