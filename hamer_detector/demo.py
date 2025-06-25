from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import trimesh
import sys
sys.path.append("./")
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import HAMER, download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer_detector.icp_conversion import extract_hand_point_cloud, compute_aligned_hamer_translation
LIGHT_BLUE=(0.65098039,  0.74117647,  0.85882353)
import time
from vitpose_model import ViTPoseModel

import json
from typing import Dict, Optional

def detect_hand(args):
    min_score = 0.75
    with open(args.intrinsics_path, 'r') as f:
        camera_intrinsics = json.load(f)

    # Download and load checkpoints
    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)

    # Setup HaMeR model
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    torch.backends.cudnn.benchmark = True
    model = model.to(device)
    model.eval()

    # Load detector
    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent/'configs'/'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    elif args.body_detector == 'regnety':
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detectron2_cfg.model.roi_heads.box_predictor.test_nms_thresh   = 0.4
        detector       = DefaultPredictor_Lazy(detectron2_cfg)

    # keypoint detector
    cpm = ViTPoseModel(device)

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.mano.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Get all demo images; default to .jpg, .jpeg, .png if not specified
    img_paths = sorted([img for ext in ['*.jpg', '*.jpeg', '*.png'] for img in Path(args.img_folder).glob(ext)])
    
    # all_centroids = dict()
    # Iterate over all images in folder
    all_hand_results = {}

    for img_path in img_paths:
        img_timer = time.time()
        # Process image
        img_cv2 = cv2.imread(str(img_path))
        if img_cv2 is None:
            print(f"❌ Failed to read image: {img_path}")
            continue
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
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes, right, rescale_factor=args.rescale_factor)
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
                with torch.cuda.amp.autocast(dtype=torch.float32):
                    out = model(batch)
                    
            # Inference complete

            multiplier = (2*batch['right']-1)
            pred_cam = out['pred_cam']
            pred_cam[:,1] = multiplier*pred_cam[:,1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            multiplier = (2*batch['right']-1)
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
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
            hand_point_cloud = extract_hand_point_cloud(hand_mask, depth_img, camera_intrinsics)
            hamer_aligned = compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, hand_mask, camera_intrinsics)
            if hamer_aligned is None:
                print(f"⏭️  Skipping frame {hand_key} due to poor depth quality.")
                # Remove corresponding entry in all_hand_results if frame is skipped
                if hand_key in all_hand_results:
                    del all_hand_results[hand_key]
                continue
            else:
                translation = hamer_aligned.mean(axis=0)
                print(f"--> 1) original transition Z = {all_hand_results[hand_key]['pred_cam_t'][2]} flipped Z = {translation[2]}")
                
                all_hand_results[hand_key]["pred_cam_t"] = translation.tolist()
                
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_handmask.png'), hand_mask)
            # Overlay image
            if args.debug:
                # Save binary mask of rendered hand (from alpha channel)
                
                input_img = img_cv2.astype(np.float32)[:,:,::-1]/255.0
                input_img = np.concatenate([input_img, np.ones_like(input_img[:,:,:1])], axis=2) # Add alpha channel
                input_img_overlay = input_img[:,:,:3] * (1-cam_view[:,:,3:]) + cam_view[:,:,:3] * cam_view[:,:,3:]
                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_all.jpg'), 255*input_img_overlay[:, :, ::-1])
        elif len(all_verts) > 0:
            # Even without rendering, still compute translation
            dummy_mask = np.ones_like(depth_img, dtype=np.uint8) * 255
            hamer_vertices = np.vstack(all_verts)
            hand_point_cloud = extract_hand_point_cloud(dummy_mask, depth_img, camera_intrinsics)
            hamer_aligned = compute_aligned_hamer_translation(hamer_vertices, hand_point_cloud, dummy_mask, camera_intrinsics)
            if hamer_aligned is None:
                print(f"⏭️  Skipping frame {hand_key} due to poor depth quality.")
                # Remove corresponding entry in all_hand_results if frame is skipped
                if hand_key in all_hand_results:
                    del all_hand_results[hand_key]
                continue
            else:
                translation = hamer_aligned.mean(axis=0)
                print(f"--> 2) original transition Z = {all_hand_results[hand_key]['pred_cam_t'][2]} flipped Z = {translation[2]}")
                all_hand_results[hand_key]["pred_cam_t"] = translation.tolist()

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

        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='/home/xhe71/Desktop/robotool_data/Color/', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='/home/xhe71/Desktop/robotool_data/pipeline_output/', help='Output folder to save rendered results')
    parser.add_argument('--depth_folder', type=str, default='/home/xhe71/Desktop/robotool_data/Depth', help='folder with depth image')
    parser.add_argument('--intrinsics_path', default = 'camera_intrinsics_zed.json',  help='load the camera intrinsics for the realsense camera')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=48, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    parser.add_argument('--debug', action='store_true', help='If set, enables full rendering and saves overlay/mask outputs')
    args = parser.parse_args()
    detect_hand(args)
