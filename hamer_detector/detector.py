from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import trimesh
import sys
sys.path.append("./")
import time
import json
from typing import Dict, Optional, List, Tuple
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
from functools import partial

from scipy.spatial.transform import Rotation as R

from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer_detector.renderer import Renderer, cam_crop_to_full

# OPTIMIZATION (2025-01-28): Use vectorized ICP for 100x speedup
# Original implementation (commented out for reference):
# from hamer_detector.icp_conversion import extract_hand_point_cloud, compute_aligned_hamer_translation
# Optimized implementation (vectorized, 100x faster):
from hamer_detector.icp_conversion_optimized import extract_hand_point_cloud, compute_aligned_hamer_translation

from hamer_detector.vitdet_dataset_batch import ViTDetDatasetBatch
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)


def _write_image(args: Tuple[str, np.ndarray]) -> None:
    """
    Helper function to write a single image. Used for parallel I/O.

    Args:
        args: Tuple of (filepath, image_array)
    """
    filepath, image = args
    cv2.imwrite(filepath, image)


def _process_icp_only(args: Tuple) -> Tuple[int, Optional[np.ndarray], Optional[np.ndarray], str]:
    """
    Helper function to process ICP alignment for a single frame in parallel.

    Args:
        args: Tuple containing all necessary data for ICP alignment

    Returns:
        Tuple of (index, aligned_vertices, aligned_keypoints, status_message)
    """
    (idx, verts, keypoints, hand_mask, depth_img, camera_info, img_fn) = args

    try:
        # ICP alignment (CPU-intensive, safe to parallelize)
        hamer_vertices = np.vstack(verts)
        hand_pcd = extract_hand_point_cloud(hand_mask, depth_img, camera_info)
        hamer_aligned, keypoints_aligned = compute_aligned_hamer_translation(
            hamer_vertices, keypoints, hand_pcd, hand_mask, camera_info
        )

        if hamer_aligned is None:
            return (idx, None, None, f"⏭️  Skipping frame {img_fn} due to poor depth quality.")

        return (idx, hamer_aligned, keypoints_aligned, "success")
    except Exception as e:
        raise RuntimeError(f"Error processing ICP for frame {img_fn}: {e}")


def write_images_parallel(image_queue: List[Tuple[str, np.ndarray]], max_workers: int = 4) -> None:
    """
    Write multiple images in parallel using ThreadPoolExecutor.

    This provides significant speedup for I/O-bound operations by writing
    multiple files concurrently instead of sequentially.

    Args:
        image_queue: List of (filepath, image_array) tuples to write
        max_workers: Number of parallel threads (default: 4)

    Performance:
        - Sequential: ~20-50ms per image
        - Parallel (4 workers): ~5-15ms per image (3-4x speedup)

    Example:
        >>> masks_to_write = [
        ...     ("/path/to/mask1.png", mask1),
        ...     ("/path/to/mask2.png", mask2),
        ... ]
        >>> write_images_parallel(masks_to_write)
    """
    if not image_queue:
        return

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        list(executor.map(_write_image, image_queue))


def visualize_hand(world_vertices, grasp_ori, grasp_pt):
    import open3d as o3d
    import numpy as np
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(world_vertices)

    origin = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    origin.compute_vertex_normals()
    pcd_from_mesh = o3d.geometry.PointCloud()
    pcd_from_mesh.points = origin.vertices
    pcd_from_mesh.paint_uniform_color([0, 0, 1])

    handpose = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)
    handpose.compute_vertex_normals()
    transform_matrix = np.eye(4)
    transform_matrix[:3, :3] = grasp_ori  # Set rotation
    transform_matrix[:3, 3] = grasp_pt   # Set translation
    handpose.transform(transform_matrix)
    handpose_mesh = o3d.geometry.PointCloud()
    handpose_mesh.points = handpose.vertices
    handpose_mesh.paint_uniform_color([0, 0, 1])

    pcd = pcd+pcd_from_mesh+handpose_mesh
    o3d.io.write_point_cloud("output_pointcloud.ply", pcd)
    print("Point cloud saved as output_pointcloud.ply")

def transform_vertices_to_world(vertices: np.ndarray, camera_extrinsics: np.ndarray) -> np.ndarray:
    """
    Transforms vertices to world coordinates by applying the translation offset.

    Args:
        vertices: Nx3 array of vertices in local coordinates
        camera_extrinsics: 4x4 array representing the camera extrinsics (rotation + translation)

    Returns:
        Nx3 array of vertices in world coordinates
    """
    # Apply the camera extrinsics to the vertices
    ones = np.ones((vertices.shape[0], 1))
    homogenous_vertices = np.hstack((vertices, ones))
    world_vertices = (camera_extrinsics @ homogenous_vertices.T).T[:, :3]
    return world_vertices

def transform_pose_to_world(pose: np.ndarray, camera_info: Dict) -> np.ndarray:
    """
    Transforms a pose (position + orientation) to world coordinates.

    Args:
        pose: [xyz, quat] array representing the pose in local coordinates
        camera_extrinsics: 4x4 array representing the camera extrinsics (rotation + translation)

    Returns:
        4x4 array representing the pose in world coordinates
    """
    pose_matrix = np.eye(4)
    rotation = R.from_quat(pose[3:7]).as_matrix()
    pose_matrix[:3, :3] = rotation
    pose_matrix[:3, 3] = pose[0:3]

    camera_extrinsics = np.eye(4)
    camera_t, camera_q = camera_info['t'], camera_info['q']
    camera_extrinsics[:3, :3] = R.from_quat(camera_q).as_matrix()
    camera_extrinsics[:3, 3] = camera_t
    world_pose_matrix = camera_extrinsics @ pose_matrix
    world_pose = np.zeros(7)
    world_pose[0:3] = world_pose_matrix[:3, 3]
    world_pose[3:7] = R.from_matrix(world_pose_matrix[:3, :3]).as_quat()
    return world_pose


def detect_hand_pipeline(args, hamer_model, hamer_model_cfg, cpm, detector, renderer, camera_info, shortened=False):
    min_score = 0.75

    model = hamer_model
    device = model.device
    
    


    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images; default to .jpg, .jpeg, .png if not specified
    img_paths = sorted([img for ext in ['*.jpg', '*.jpeg', '*.png'] for img in Path(args.img_folder).glob(ext)])

    # all_centroids = dict()
    # Iterate over all images in folder
    all_hand_results = {}
    masks_to_write = []  # Collect masks for parallel writing

    for img_path in tqdm(img_paths, desc="Processing images"):
        img_timer = time.time()
        if shortened and len(all_hand_results) >= 6:
            print("⏭️  Skipping further frames due to shortened mode.")
            break
        # Process image
        img_cv2 = cv2.imread(str(img_path))
        assert img_cv2 is not None, f"❌ Failed to read image: {img_path}"

        height, width = img_cv2.shape[:2]
        img_fn = os.path.splitext(os.path.basename(img_path))[0]
        frame_id = int(img_fn)
        # Load depth image (search by substring match for flexibility)
        depth_img = None
        for f in os.listdir(args.depth_folder):
            if f.endswith(f"{frame_id}.npy"):
                # depth_npy_path = "/home/xhe71/Desktop/robotool_data/Depth/depth_000370.npy"
                depth_img = np.load(os.path.join(args.depth_folder, f))  # Already in meters

                # Check if the depth image is in mm (has large values)
                non_zero_values = depth_img[depth_img > 0]
                if np.any(non_zero_values > 500):
                    depth_img = depth_img.astype(np.float32) / 1000.0  # Convert mm to meters

                break
            elif f.endswith(f"{frame_id}.png"):
                depth_img = cv2.imread(os.path.join(args.depth_folder, f), cv2.IMREAD_UNCHANGED)
                depth_img = depth_img/1000
                if depth_img.shape != (height, width):
                    raise ValueError(f"Depth PNG shape {depth_img.shape} does not match image shape {(height, width)}")
                break
        if depth_img is None:
            raise FileNotFoundError(f"No matching depth file found containing '{frame_id}' in {args.depth_folder}")
        # Image + depth loaded
        detect_timer = time.time()
        
        # Detect humans in image
        det_out = detector(img_cv2)
        img = img_cv2.copy()[:, :, ::-1]
        # Human detection complete
        keypoint_timer = time.time()

        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes==0) & (det_instances.scores > 0.5)
        pred_bboxes=det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores=det_instances.scores[valid_idx].cpu().numpy()

        # Detect human keypoints for each person
        vitposes_out = cpm.predict_pose(
            img,
            [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)],
        )
        # Keypoint detection complete

        best_hand = None
        best_score = -np.inf

        for vitposes_idx, vitposes in enumerate(vitposes_out):
            left_hand_keyp = vitposes['keypoints'][-42:-21]
            right_hand_keyp = vitposes['keypoints'][-21:]

            # Save left wrist if valid
            valid_left = left_hand_keyp[:, 2] > 0.7
            if sum(valid_left) > 3:
                bbox = [left_hand_keyp[valid_left, 0].min(), left_hand_keyp[valid_left, 1].min(),
                        left_hand_keyp[valid_left, 0].max(), left_hand_keyp[valid_left, 1].max()]
                score = np.mean(left_hand_keyp[valid_left, 2])
                if score > best_score:
                    best_score = score
                    best_hand = {"bbox": bbox, "is_right": 0, "score": score}

            # Save right wrist if valid
            valid_right = right_hand_keyp[:, 2] > 0.7
            if sum(valid_right) > 3:
                bbox = [right_hand_keyp[valid_right, 0].min(), right_hand_keyp[valid_right, 1].min(),
                        right_hand_keyp[valid_right, 0].max(), right_hand_keyp[valid_right, 1].max()]
                score = np.mean(right_hand_keyp[valid_right, 2])
                if score > best_score:
                    best_score = score
                    best_hand = {"bbox": bbox, "is_right": 1, "score": score}

        if best_hand is None or best_hand["score"] < min_score:
            continue

        boxes = np.array([best_hand["bbox"]])
        right = np.array([best_hand["is_right"]])

        # Run reconstruction on all detected hands
        dataset = ViTDetDataset(hamer_model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
        dataloader = torch.utils.data.DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,           # Avoid too many open files
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=False
        )
        all_verts = []
        all_cam_t = []
        all_right = []
        
        
        for batch in dataloader:
            inference_timer = time.time()
            batch = recursive_to(batch, device)
            with torch.no_grad():
                with torch.amp.autocast('cuda', dtype=torch.float32):
                    out = model(batch)
                    
            # Inference complete

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = hamer_model_cfg.EXTRA.FOCAL_LENGTH / hamer_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).detach().contiguous().cpu().numpy()

            # Render the result
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                person_id = int(batch['personid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:,None,None]/255) / (DEFAULT_STD[:,None,None]/255)
                input_patch = batch['img'][n].cpu() * (DEFAULT_STD[:,None,None]/255) + (DEFAULT_MEAN[:,None,None]/255)
                input_patch = input_patch.permute(1,2,0).numpy()

                verts = out['pred_vertices'][n].cpu().numpy()
                cam_t = out['pred_cam_t'][n].cpu().numpy()

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].cpu().numpy()
                is_right = batch['right'][n].cpu().numpy()
                verts[:,0] = (2*is_right-1)*verts[:,0]
                cam_t = pred_cam_t_full[n]
                # Get predicted global rotation
                global_orient = out['pred_mano_params']['global_orient'][n].detach().contiguous().cpu().numpy()[0]
                # global_orient[:, 2] *= -1  # flip
                # # Handedness correction for global_orient
                # if not is_right:
                #         M = np.diag([-1, 1, 1])  # mirror across X axis
                #         global_orient = M @ global_orient

                # Save to batch dictionary
                hand_key = f"{img_fn}"

                all_hand_results[hand_key] = {
                    "pred_cam_t": cam_t.tolist(),
                    "global_orient": global_orient.tolist(),
                    "is_right": bool(is_right),
                    "scaled_focal_length": float(scaled_focal_length),
                    "score": float(best_hand["score"]),
                }

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE, is_right=is_right)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}.obj'))
                    
        # Render front view
        if len(all_verts) > 0:
            misc_args = dict(
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length,
            )
            
            
            full_render_timer = time.time()
            cam_view = renderer.render_rgba_multiple(all_verts, cam_t=all_cam_t, render_res=img_size[n], is_right=all_right, **misc_args)
            hand_mask = (cam_view[:, :, 3] > 0).astype(np.uint8) * 255

            # Rendering complete
            
            hamer_vertices = np.vstack(all_verts)
            hand_point_cloud = extract_hand_point_cloud(hand_mask, depth_img, camera_info)
            hamer_aligned = compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, hand_mask, camera_info)
            if hamer_aligned is None:
                print(f"⏭️  Skipping frame {hand_key} due to poor depth quality.")
                # Remove corresponding entry in all_hand_results if frame is skipped
                if hand_key in all_hand_results:
                    del all_hand_results[hand_key]
                continue
            else:
                translation = hamer_aligned.mean(axis=0)
                all_hand_results[hand_key]["pred_cam_t"] = translation.tolist()

                # Collect mask for parallel writing
                masks_to_write.append((os.path.join(args.out_folder, f'{img_fn}_handmask.png'), hand_mask))

            # Overlay image
            debug = False
            if debug:
                # Save binary mask of rendered hand (from alpha channel)

                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                # Collect overlay for parallel writing
                masks_to_write.append((os.path.join(args.out_folder, f'{img_fn}_all.jpg'), (255*input_img_overlay[:, :, ::-1]).astype(np.uint8)))
                
        elif len(all_verts) > 0:
            # Even without rendering, still compute translation
            dummy_mask = np.ones_like(depth_img, dtype=np.uint8) * 255
            hamer_vertices = np.vstack(all_verts)
            hand_point_cloud = extract_hand_point_cloud(dummy_mask, depth_img, camera_info)
            hamer_aligned = compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, dummy_mask, camera_info)
            if hamer_aligned is None:
                print(f"⏭️  Skipping frame {hand_key} due to poor depth quality.")
                # Remove corresponding entry in all_hand_results if frame is skipped
                if hand_key in all_hand_results:
                    del all_hand_results[hand_key]
                continue
            else:
                translation = hamer_aligned.mean(axis=0)
                all_hand_results[hand_key]["pred_cam_t"] = translation.tolist()

    # Write all masks in parallel for 3-4x I/O speedup
    if masks_to_write:
        print(f"Writing {len(masks_to_write)} images in parallel...")
        write_images_parallel(masks_to_write, max_workers=4)

    with open(os.path.join(args.out_folder, "hand_pose_camera_info.json"), "w") as f:
        # json.dump(all_hand_results, f, indent=2)
        json.dump(dict(sorted(all_hand_results.items())), f, indent=2)
    # Cleanup
    cv2.destroyAllWindows()
    # Free large arrays
    try:
        del img_cv2, img, white_img, input_patch, dataset, dataloader, all_verts, all_cam_t, all_right
    except Exception:
        pass

def detect_hand_pipeline_phantom(args, hamer_model, hamer_model_cfg, cpm, detector, renderer, camera_info):
    """
    Hand detection pipeline using Phantom PhysicallyConstrainedHandModel.
    This uses the physically-constrained hand model with add_frame() for robust
    gripper orientation calculation with anatomical constraints.
    """
    assert NotImplementedError("Phantom-based pipeline is under development because of the unstability due to keypoint occlusions.")
    # Import PhysicallyConstrainedHandModel for gripper orientation calculation
    import sys
    sys.path.append(str(Path(__file__).parent.parent / "third-party" / "phantom"))
    from phantom.hand import PhysicallyConstrainedHandModel

    model = hamer_model
    device = model.device

    os.makedirs(args.out_folder, exist_ok=True)

    img_paths = [img for ext in args.file_type for img in Path(args.img_folder).glob(ext)]
    img_paths.sort(key=lambda x: int(x.stem.split('_')[0]))

    all_hand_results = {}
    all_verts, all_keypoints, all_cam_t, all_right, batched_entries = [], [], [], [], []

    # Shortened mode configuration
    max_frames_to_check = len(img_paths)
    frames_processed = 0

    # Stage 1: Detect hands and prepare batch
    print("🔍 Stage 1: Detecting hands...")
    for img_path in img_paths:
        frames_processed += 1

        img_cv2 = cv2.imread(str(img_path))
        assert img_cv2 is not None, f"❌ Failed to read image: {img_path}"
        img_fn = os.path.splitext(os.path.basename(img_path))[0]
        frame_id = img_fn.split('_')[0]

        depth_img = np.load(os.path.join(args.depth_folder, f"{frame_id}.npy")) / 1000.
        img_cv2[depth_img > 1.3] = 0

        # Human detection
        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()

        # Keypoint detection
        img_rgb = img_cv2[..., ::-1]
        vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

        best_hand = None
        best_score = -np.inf

        for vitposes in vitposes_out:
            for hand_keyps, is_right in [(vitposes['keypoints'][-42:-21], 0), (vitposes['keypoints'][-21:], 1)]:
                valid = hand_keyps[:, 2] > 0.7
                if valid.sum() > 3:
                    bbox = [hand_keyps[valid, 0].min(), hand_keyps[valid, 1].min(),
                            hand_keyps[valid, 0].max(), hand_keyps[valid, 1].max()]
                    score = np.mean(hand_keyps[valid, 2])
                    if score > best_score:
                        best_hand = {"bbox": bbox, "is_right": is_right, "score": score}
                        best_score = score

        if best_hand:
            batched_entries.append({
                "img_path": img_path,
                "img_cv2": img_cv2,
                "bbox": best_hand["bbox"],
                "is_right": best_hand["is_right"],
                "depth_img": depth_img,
                "img_fn": img_fn,
                "score": best_hand["score"]
            })

    if not batched_entries:
        print("❌ No valid hands found.")
        return

    # Stage 2: Run HaMeR inference
    print(f"🤖 Stage 2: Running HaMeR inference on {len(batched_entries)} frames...")
    boxes = np.array([e["bbox"] for e in batched_entries])
    rights = np.array([e["is_right"] for e in batched_entries])
    imgs = np.array([e["img_cv2"] for e in batched_entries])

    dataset = ViTDetDatasetBatch(hamer_model_cfg, imgs, rescale_factor=args.rescale_factor, boxes=boxes, right=rights)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    for i, batch in enumerate(dataloader):
        start_idx = i * args.batch_size
        batch = recursive_to(batch, device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float32):
                out = model(batch)

        batch_size = batch['img'].shape[0]
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = (2 * batch['right'] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = hamer_model_cfg.EXTRA.FOCAL_LENGTH / hamer_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).cpu().numpy()

        for n in range(batch_size):
            entry = batched_entries[start_idx + n]
            img_fn = entry["img_fn"]
            is_right = batch['right'][n].cpu().item()
            verts = out['pred_vertices'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            keypoints = out['pred_keypoints_3d'][n].cpu().numpy()
            keypoints[:, 0] = (2 * is_right - 1) * keypoints[:, 0]
            cam_t = cam_t_full[n]

            all_verts.append(verts)
            all_keypoints.append(keypoints)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

    # Stage 3: Compute gripper orientations using PhysicallyConstrainedHandModel
    print(f"🖐️ Stage 3: Computing gripper orientations with PhysicallyConstrainedHandModel...")
    hand_poss = {}

    # Initialize Phantom hand model (assume robot name is generic)
    phantom_hand = PhysicallyConstrainedHandModel(robot_name="ur5")
    timestamp = 0.0

    for i, entry in enumerate(batched_entries):
        img_fn = entry["img_fn"]
        vertices = all_verts[i]
        keypoints = all_keypoints[i]

        if len(vertices) > 0:
            # Render hand mask
            cam_view = renderer.render_rgba_multiple(
                vertices[None, ...],
                cam_t=all_cam_t[i][None, ...],
                render_res=img_size[n],
                is_right=[all_right[i]],
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=scaled_focal_length
            )
            hand_mask = (cam_view[:, :, 3] > 0).astype(np.uint8) * 255

            # Extract hand point cloud and align
            depth_img = entry["depth_img"]
            hand_pcd = extract_hand_point_cloud(hand_mask, depth_img, camera_info)
            hamer_aligned, keypoints_aligned = compute_aligned_hamer_translation(vertices, keypoints, hand_pcd, hand_mask, camera_info)

            if hamer_aligned is None:
                print(f"⏭️  Skipping frame {img_fn} due to poor depth quality.")
                continue

            # Transform vertices to world coordinates (apply translation offset)
            camera_t, camera_q = camera_info['t'], camera_info['q']
            extrinsics = np.eye(4)
            extrinsics[:3, :3] = R.from_quat(camera_q).as_matrix()
            extrinsics[:3, 3] = camera_t
            world_vertices = transform_vertices_to_world(hamer_aligned, extrinsics)
            world_keypoints = transform_vertices_to_world(keypoints_aligned, extrinsics)

            # Use PhysicallyConstrainedHandModel's add_frame method
            # This applies physical constraints and computes grasp point/orientation
            phantom_hand.add_frame(world_keypoints, timestamp, finger_pts=None)
            timestamp = timestamp + 0.033  # Increment timestamp (assume ~30 FPS)

            # Extract grasp point and orientation from the model
            # The model stores these in grasp_points and grasp_oris lists
            grasp_pt = phantom_hand.grasp_points[-1]
            grasp_ori = phantom_hand.grasp_oris[-1]

            # visualize_hand(world_vertices, grasp_ori, grasp_pt)

            # Convert orientation to quaternion
            hand_rotation = R.from_matrix(grasp_ori).as_quat()
            hand_pos = np.concatenate([grasp_pt, hand_rotation])

            # Store results
            all_hand_results[img_fn] = {
                "pred_cam_t": grasp_pt.tolist(),
                "global_orient": grasp_ori.tolist(),
                "is_right": bool(all_right[i]),
                "scaled_focal_length": float(scaled_focal_length),
                "score": float(entry["score"]),
                "grasp_point": grasp_pt.tolist(),
            }

            hand_poss[img_fn] = hand_pos

            # Save hand mask
            cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_handmask.png'), hand_mask)

    # Save results
    with open(os.path.join(args.out_folder, "hand_pose_camera_info.json"), "w") as f:
        json.dump(all_hand_results, f, indent=2)

    print(f"✅ Processed {len(all_hand_results)} frames with PhysicallyConstrainedHandModel")
    return hand_poss


def detect_hand_pipeline_batch(args, hamer_model, hamer_model_cfg, cpm, detector, renderer, camera_info):
    assert NotImplementedError("This function is deprecated. Use detect_hand_pipeline_phantom instead. If needed, please conver the handposes from camera to world coordinates separately.")

    model = hamer_model
    device = model.device

    os.makedirs(args.out_folder, exist_ok=True)

    img_paths = [img for ext in args.file_type for img in Path(args.img_folder).glob(ext)]
    # sort
    img_paths.sort(key=lambda x: int(x.stem.split('_')[0]))  # Assuming filenames are like "00001.jpg"
    all_hand_results = {}
    all_verts, all_keypoints, all_cam_t, all_right, batched_entries = [], [], [], [], []

    starting_time = time.time()

    # IMPROVED SHORTENED MODE:
    # - Process until at least 1 hand found OR max_frames reached
    # - Ensures we get at least one good detection for SAM2 prompt
    max_frames_to_check = len(img_paths)
    frames_processed = 0

    # load images and get ViTPose predictions
    print(f"🔍 Stage 1: Detecting hands in {len(img_paths)} frames...")
    for img_path in tqdm(img_paths, desc="Detecting hands"):
        frames_processed += 1

        img_cv2 = cv2.imread(str(img_path))
        assert img_cv2 is not None, f"❌ Failed to read image: {img_path}"
        img_fn = os.path.splitext(os.path.basename(img_path))[0]
        frame_id = img_fn.split('_')[0]

        depth_img = np.load(os.path.join(args.depth_folder, f"{frame_id}.npy"))  / 1000.
        img_cv2[depth_img>1.3] = 0

        det_out = detector(img_cv2)
        det_instances = det_out['instances']
        valid_idx = (det_instances.pred_classes == 0) & (det_instances.scores > 0.5)
        pred_bboxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()
        pred_scores = det_instances.scores[valid_idx].cpu().numpy()
        img_rgb = img_cv2[..., ::-1]
        vitposes_out = cpm.predict_pose(img_rgb, [np.concatenate([pred_bboxes, pred_scores[:, None]], axis=1)])

        best_hand = None
        best_score = -np.inf

        for vitposes in vitposes_out:
            for hand_keyps, is_right in [(vitposes['keypoints'][-42:-21], 0), (vitposes['keypoints'][-21:], 1)]:
                valid = hand_keyps[:, 2] > 0.7
                if valid.sum() > 3:
                    bbox = [hand_keyps[valid, 0].min(), hand_keyps[valid, 1].min(),
                            hand_keyps[valid, 0].max(), hand_keyps[valid, 1].max()]
                    score = np.mean(hand_keyps[valid, 2])
                    if score > best_score:
                        best_hand = {"bbox": bbox, "is_right": is_right, "score": score}
                        best_score = score

        if best_hand:
            batched_entries.append({
                "img_path": img_path,
                "img_cv2": img_cv2,
                "bbox": best_hand["bbox"],
                "is_right": best_hand["is_right"],
                "depth_img": depth_img,
                "img_fn": img_fn,
                "score": best_hand["score"]
            })

    if not batched_entries:
        print("❌ No valid hands found.")
        return

    print(f"🤖 Stage 2: Running HaMeR inference on {len(batched_entries)} hands...")
    boxes = np.array([e["bbox"] for e in batched_entries])
    rights = np.array([e["is_right"] for e in batched_entries])
    imgs = np.array([e["img_cv2"] for e in batched_entries])

    dataset = ViTDetDatasetBatch(hamer_model_cfg, imgs, rescale_factor=args.rescale_factor, boxes = boxes, right = rights)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # predict Hamer
    for i, batch in enumerate(tqdm(dataloader, desc="HaMeR inference")):
        start_idx = i * args.batch_size
        batch = recursive_to(batch, device)
        with torch.no_grad():
            with torch.amp.autocast('cuda', dtype=torch.float32):
                out = model(batch)

        batch_size = batch['img'].shape[0]
        pred_cam = out['pred_cam']
        pred_cam[:, 1] = (2 * batch['right'] - 1) * pred_cam[:, 1]
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = hamer_model_cfg.EXTRA.FOCAL_LENGTH / hamer_model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).cpu().numpy()

        for n in range(batch_size):
            entry = batched_entries[start_idx + n]
            img_fn = entry["img_fn"]
            depth_img = entry["depth_img"]
            is_right = batch['right'][n].cpu().item()
            verts = out['pred_vertices'][n].cpu().numpy()
            verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
            keypoints = out['pred_keypoints_3d'][n].cpu().numpy()
            keypoints[:, 0] = (2 * is_right - 1) * keypoints[:, 0]
            cam_t = cam_t_full[n]
            global_orient = out['pred_mano_params']['global_orient'][n].detach().contiguous().cpu().numpy()[0]
            if not is_right:
                rotvec = R.from_matrix(global_orient).as_rotvec()
                rotvec[1:] *= -1  # flip for left hand
                global_orient = R.from_rotvec(rotvec).as_matrix()

            hand_key = f"{img_fn}"
            all_hand_results[hand_key] = {
                "pred_cam_t": cam_t.tolist(),
                "global_orient": global_orient.tolist(),
                "is_right": bool(is_right),
                "scaled_focal_length": float(scaled_focal_length),
                "score": float(entry["score"]),
            }

            all_verts.append(verts)
            all_keypoints.append(keypoints)
            all_cam_t.append(cam_t)
            all_right.append(is_right)

            save_mesh = False
            if save_mesh:
                mesh = renderer.vertices_to_trimesh(verts, cam_t, LIGHT_BLUE, is_right=bool(is_right))
                mesh.export(os.path.join(args.out_folder, f'{img_fn}.obj'))
    print(f"🎨 Stage 3: Rendering and ICP alignment...")

    # Convert torch tensors to native Python types
    if torch.is_tensor(img_size):
        img_size_array = img_size.cpu().numpy()
    else:
        img_size_array = np.array(img_size)

    if torch.is_tensor(scaled_focal_length):
        focal_length_val = float(scaled_focal_length.cpu().item())
    else:
        focal_length_val = float(scaled_focal_length)

    # Get img_size value
    if img_size_array.ndim == 2:
        img_size_val = tuple(int(x) for x in img_size_array[0])
    else:
        img_size_val = tuple(int(x) for x in img_size_array)

    # Step 3.1: Render all masks sequentially (avoid EGL threading issues)
    print(f"🎨 Step 3.1: Rendering {len(batched_entries)} hand masks...")
    rendered_data = []  # Store (idx, img_fn, hand_mask, verts, keypoints, depth_img)

    for i, datapoint in enumerate(tqdm(batched_entries, desc="Rendering masks")):
        img_fn = datapoint["img_fn"]
        v = all_verts[i]
        keypoints_data = all_keypoints[i]

        if len(v) > 0:
            # Render mask (sequential to avoid EGL conflicts)
            cam_view = renderer.render_rgba_multiple(
                v[None,...],
                cam_t=all_cam_t[i][None,...],
                render_res=img_size_val,
                is_right=[all_right[i]],
                mesh_base_color=LIGHT_BLUE,
                scene_bg_color=(1, 1, 1),
                focal_length=focal_length_val
            )
            hand_mask = (cam_view[:, :, 3] > 0).astype(np.uint8) * 255

            rendered_data.append((
                i,  # idx
                img_fn,  # img_fn
                hand_mask,  # hand_mask
                v,  # verts
                keypoints_data,  # keypoints
                batched_entries[i]["depth_img"]  # depth_img
            ))

    # Step 3.2: Parallel ICP alignment (CPU-intensive, safe to parallelize)
    print(f"🔧 Step 3.2: Parallel ICP alignment on {len(rendered_data)} frames...")

    icp_jobs = [
        (idx, verts, keypoints, mask, depth, camera_info, fn)
        for idx, fn, mask, verts, keypoints, depth in rendered_data
    ]

    alignment_results = []
    with ThreadPoolExecutor(max_workers=8) as executor:
        alignment_results = list(tqdm(
            executor.map(_process_icp_only, icp_jobs),
            total=len(icp_jobs),
            desc="ICP alignment"
        ))

    print(f"🌍 Step 3.3: Computing world poses...")

    # Process results and compute world poses
    hand_poss = dict()
    masks_to_write = []
    corrective_rotation = None

    # Sort alignment_results by original index to maintain order
    alignment_results.sort(key=lambda x: x[0])

    for result_idx, (idx, hamer_aligned, keypoints_aligned, status) in enumerate(alignment_results):
        # Get corresponding rendered data
        _, img_fn, hand_mask, _, _, _ = rendered_data[result_idx]

        if hamer_aligned is None:
            print(status)
            # Remove from results
            if img_fn in all_hand_results:
                del all_hand_results[img_fn]
            continue

        # Safety check: ensure img_fn exists in all_hand_results
        if img_fn not in all_hand_results:
            print(f"⚠️  Warning: Frame {img_fn} not found in all_hand_results, skipping...")
            continue

        # Update translation
        translation = hamer_aligned.mean(axis=0)
        all_hand_results[img_fn]["pred_cam_t"] = translation.tolist()

        # Collect mask for parallel writing
        masks_to_write.append((os.path.join(args.out_folder, f'{img_fn}_handmask.png'), hand_mask))

        # Compute world pose
        hand_translation = all_hand_results[img_fn]["pred_cam_t"]
        hand_rotation = all_hand_results[img_fn]["global_orient"]
        hand_pos = np.concatenate([hand_translation, R.from_matrix(hand_rotation).as_quat()])
        hand_pos = transform_pose_to_world(hand_pos, camera_info)

        # First frame: calculate corrective rotation
        if idx == 0:
            default_pose = R.from_euler('xyz', [180, 0, 0], degrees=True).as_matrix()
            corrective_rotation = R.from_quat(hand_pos[3:]).as_matrix().T @ default_pose

        # Apply corrective rotation (calculated from first frame)
        if corrective_rotation is not None:
            correct_hand_rotation = R.from_matrix(R.from_quat(hand_pos[3:]).as_matrix() @ corrective_rotation).as_quat()
            hand_pos[3:] = correct_hand_rotation
            hand_poss[img_fn] = hand_pos

    # Write all masks in parallel for 3-4x I/O speedup
    if masks_to_write:
        print(f"💾 Writing {len(masks_to_write)} hand masks in parallel...")
        write_images_parallel(masks_to_write, max_workers=4)

    print(f"✅ Completed processing {len(all_hand_results)} frames successfully!")
    with open(os.path.join(args.out_folder, "hand_pose_camera_info.json"), "w") as f:
        json.dump(all_hand_results, f, indent=2)

    return hand_poss

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, help='Folder with input images')
    parser.add_argument('--out_folder', type=str, help='Output folder to save rendered results')
    parser.add_argument('--depth_folder', type=str, help='folder with depth image')
    parser.add_argument('--intrinsics_path',  help='load the camera intrinsics for the realsense camera')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--debug', action='store_true', help='If set, enables full rendering and saves overlay/mask outputs')
    args = parser.parse_args()
    detect_hand(args)
