from PIL import Image
import cv2
import torch
import base64
import os
import glob

# if using Apple MPS, fall back to CPU for unsupported ops
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import numpy as np
import torch
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import supervision as sv
from pathlib import Path
from supervision.assets import download_assets, VideoAssets
# Import SAM2 from your specified relative path
import sys
sys.path.append("submodules/segment-anything-2-real-time/sam2")
from sam2.build_sam import build_sam2
from sam2.build_sam import build_sam2_video_predictor

# torch.autocast(device_type="cuda", dtype=torch.bfloat16).__enter__()

# if torch.cuda.get_device_properties(0).major >= 8:
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True
    
# DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

from pathlib import Path

# Load SAM2 Model
CHECKPOINT = "submodules/segment-anything-2-real-time/checkpoints/sam2.1_hiera_large.pt"
CONFIG = "configs/sam2.1/sam2.1_hiera_l.yaml"
predictor = build_sam2_video_predictor(CONFIG, CHECKPOINT)


def show_mask(mask, ax, obj_id=None, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        cmap = plt.get_cmap("tab10")
        cmap_idx = 0 if obj_id is None else obj_id
        color = np.array([*cmap(cmap_idx)[:3], 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=200):
    pos_points = coords[labels==1]
    neg_points = coords[labels==0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white', linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))
    
def rename_images_to_jpeg_sequence(folder_path):
    folder = Path(folder_path)
    valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp']

    # Filter and sort image files
    image_paths = sorted([f for f in folder.iterdir() if f.suffix.lower() in valid_extensions])

    for idx, img_path in enumerate(image_paths):
        # Read image
        img = cv2.imread(str(img_path))
        if img is None:
            print(f"❌ Skipping unreadable file: {img_path}")
            continue

        # Create new file name
        new_name = f"{idx+0:06d}.jpg"
        new_path = folder / new_name

        # Save image as JPEG
        cv2.imwrite(str(new_path), img)

        # Remove original if not already .jpg with correct name
        if img_path.name != new_name:
            os.remove(img_path)

    print(f"✅ Renamed {len(image_paths)} images in '{folder_path}'.")

def annotate_points(image_path):
    """
    Open an image and allow user to click points and manually label them as 0 or 1.
    
    Returns:
        points_with_labels (list of dict): [{'x': int, 'y': int, 'label': 0 or 1}, ...]
    """
    image = cv2.imread(image_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    points_with_labels = []

    fig, ax = plt.subplots()
    ax.imshow(image_rgb)
    ax.set_title("Click to add points. Close window when done.")

    coords = []

    def onclick(event):
        if event.xdata is None or event.ydata is None:
            return
        x, y = int(event.xdata), int(event.ydata)
        if event.button == 1:
            label = 1
        elif event.button == 3:
            label = 0
        else:
            print("❌ Use left (1) for label 1 or right (3) for label 0 click.")
            return
        print(f"Clicked at ({x}, {y}) with label {label}")
        points_with_labels.append({'x': x, 'y': y, 'label': label})
        ax.plot(x, y, 'ro' if label == 1 else 'bo')
        fig.canvas.draw()

    cid = fig.canvas.mpl_connect('button_press_event', onclick)
    plt.show()

    return points_with_labels




def replace_background(image, mask, reference_image, ref_cam = 1):
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

    height, width = image.shape[:2]

    if ref_cam == 1:
        top_section = height // 6
        result_image[:top_section, :] = reference_resized[:top_section, :]
        result_image[:, 3 * width // 4:] = reference_resized[:, 3 * width // 4:]  # Right 1/4

    elif ref_cam == 2:
        top_section = height // 6
        result_image[:top_section, :] = reference_resized[:top_section, :]
        result_image[:, :width // 3] = reference_resized[:, :width // 3]  # Left 1/3

    else:
        top_section = 2 * height // 5
        result_image[:top_section, :] = reference_resized[:top_section, :]

    return result_image

# `video_dir` a directory of JPEG frames with filenames like `<frame_index>.jpg`
# Specify the folder path containing your image sequence

def run_sam2_segmentation(source_frames, hand_mask_dir, output_dir, reference_image_path = None, debug = False, ref_cam = 1):
    if not any(f.lower().endswith(".jpg") for f in os.listdir(source_frames)):
        rename_images_to_jpeg_sequence(source_frames)

    reference_image = cv2.imread(reference_image_path)
    hand_mask_paths = sorted(glob.glob(os.path.join(hand_mask_dir, "*_handmask.png")))
    video_dir = source_frames

    frame_names = sorted([
        p for p in os.listdir(video_dir)
        if os.path.splitext(p)[-1].lower() in [".jpg", ".jpeg"]
    ], key=lambda p: str(os.path.splitext(p)[0]))

    inference_state = predictor.init_state(video_path=video_dir)
    predictor.reset_state(inference_state)

    points_with_labels = []
    frame_idx_list = []

    selected_indices = np.linspace(1, len(hand_mask_paths) - 1, num=10, dtype=int)
    print(selected_indices)
    for i in selected_indices:
        mask_path = hand_mask_paths[i]
        if not os.path.exists(mask_path):
            print(f"❌ Mask file not found: {mask_path}, skipping.")
            continue
        print(mask_path)
        basename = os.path.basename(mask_path)
        frame_idx_str = basename.split('_')[0]
        frame_name = f"{int(frame_idx_str):06d}.jpg"
        if frame_name in frame_names:
            frame_idx = frame_names.index(frame_name)
        else:
            print(f"❌ Frame name {frame_name} not found in frame_names, skipping.")
            continue

        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            print(f"❌ Unable to read mask file: {mask_path}, skipping.")
            continue
        ys, xs = np.where(mask > 127)
        if len(xs) == 0:
            print(f"❌ No valid mask pixels in: {mask_path}, skipping.")
            continue

        centroid_x = int(np.median(xs))
        centroid_y = int(np.median(ys))

        points_with_labels.append({'x': centroid_x, 'y': centroid_y, 'label': 1})
        frame_idx_list.append(frame_idx)

    for point_entry, frame_idx in zip(points_with_labels, frame_idx_list):
        points = np.array([[point_entry['x'], point_entry['y']]], dtype=np.float32)
        labels = np.array([point_entry['label']], dtype=np.int32)

        _, out_obj_ids, out_mask_logits = predictor.add_new_points_or_box(
            inference_state=inference_state,
            frame_idx=frame_idx,
            obj_id=1,
            points=points,
            labels=labels,
        )

    video_segments = {}
    for out_frame_idx, out_obj_ids, out_mask_logits in predictor.propagate_in_video(inference_state):
        video_segments[out_frame_idx] = {
            out_obj_id: np.squeeze((out_mask_logits[i] > 0.0).cpu().numpy())
            for i, out_obj_id in enumerate(out_obj_ids)
        }

    os.makedirs(output_dir, exist_ok=True)
    for out_frame_idx in range(len(frame_names)):
        frame_path = os.path.join(video_dir, frame_names[out_frame_idx])
        image = cv2.imread(frame_path)
        if image is None:
            continue

        if out_frame_idx in video_segments:
            for out_obj_id, out_mask in video_segments[out_frame_idx].items():
                binary_mask = out_mask.astype(np.uint8)
                # ➕ Dilate the mask to add a 1-pixel contour
                kernel = np.ones((3, 3), np.uint8)
                binary_mask = cv2.dilate(binary_mask, kernel, iterations=1)

                # If debug mode, save overlay with green mask
                if debug:
                    colored_mask = np.zeros_like(image, dtype=np.uint8)
                    colored_mask[binary_mask == 1] = (0, 255, 0)
                    overlay_image = cv2.addWeighted(image, 0.5, colored_mask, 0.5, 0)
                    debug_path = os.path.join(output_dir, f"debug_{out_frame_idx:06d}.jpg")
                    cv2.imwrite(debug_path, overlay_image)

                # Save background replaced version
                if reference_image is not None:
                    replaced = replace_background(image, binary_mask, reference_image, ref_cam=ref_cam)
                    final_path = os.path.join(output_dir, f"frame_{out_frame_idx:06d}_final.jpg")
                    cv2.imwrite(final_path, replaced)
                    print(f"✅ Saved final image to: {final_path}")


# ref_cam = 3
# source_frames = f"/home/xhe71/Desktop/robotool_data/06232025/slow/cam{ref_cam}/rgb/"  # <-- Replace with your actual folder path
# hand_mask_dir = f"/home/xhe71/Desktop/robotool_data/06232025/slow/cam{ref_cam}/rgb_hamer"
# output_dir = f"/home/xhe71/Desktop/robotool_data/06232025/slow/cam{ref_cam}/test2"
# ref_image = f"/home/xhe71/Desktop/robotool_data/06232025/background/cam{ref_cam}/rgb/frame_000000.png"

# debug = True
# run_sam2_segmentation(source_frames, hand_mask_dir, output_dir, ref_image, debug, ref_cam = ref_cam)

