from pathlib import Path
import torch
import argparse
import os
import cv2
import numpy as np
import trimesh
import time
import json
import sys
sys.path.append("./")
from vitpose_model import ViTPoseModel
from hamer.configs import CACHE_DIR_HAMER
from hamer.models import download_models, load_hamer, DEFAULT_CHECKPOINT
from hamer.utils import recursive_to
from hamer.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from hamer.utils.renderer import Renderer, cam_crop_to_full
from hamer_detector.icp_conversion import extract_hand_point_cloud, compute_aligned_hamer_translation

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)
torch.cuda.empty_cache() 

def detect_hand(args):
    with open(args.intrinsics_path, 'r') as f:
        camera_intrinsics = json.load(f)

    download_models(CACHE_DIR_HAMER)
    model, model_cfg = load_hamer(args.checkpoint)
    if torch.cuda.is_available():
        device = torch.device('cuda') 
        print("USING CUDA")
    else:
        device = torch.device('cpu') 
    torch.backends.cudnn.benchmark = True
    model = model.to(device).eval()

    from hamer.utils.utils_detectron2 import DefaultPredictor_Lazy
    if args.body_detector == 'vitdet':
        from detectron2.config import LazyConfig
        import hamer
        cfg_path = Path(hamer.__file__).parent / 'configs' / 'cascade_mask_rcnn_vitdet_h_75ep.py'
        detectron2_cfg = LazyConfig.load(str(cfg_path))
        detectron2_cfg.train.init_checkpoint = "https://dl.fbaipublicfiles.com/detectron2/ViTDet/COCO/cascade_mask_rcnn_vitdet_h/f328730692/model_final_f05665.pkl"
        for i in range(3):
            detectron2_cfg.model.roi_heads.box_predictors[i].test_score_thresh = 0.5
        detector = DefaultPredictor_Lazy(detectron2_cfg)
    else:
        from detectron2 import model_zoo
        from detectron2.config import get_cfg
        detectron2_cfg = model_zoo.get_config('new_baselines/mask_rcnn_regnety_4gf_dds_FPN_400ep_LSJ.py', trained=True)
        detectron2_cfg.model.roi_heads.box_predictor.test_score_thresh = 0.5
        detector = DefaultPredictor_Lazy(detectron2_cfg)

    cpm = ViTPoseModel(device)
    renderer = Renderer(model_cfg, faces=model.mano.faces)
    os.makedirs(args.out_folder, exist_ok=True)

    img_paths = [img for ext in args.file_type for img in Path(args.img_folder).glob(ext)]
    all_hand_results = {}
    all_verts, all_cam_t, all_right, batched_entries = [], [], [], []

    BATCH_IMAGE_COUNT = 5  # You can adjust this value
    for i in range(0, len(img_paths), BATCH_IMAGE_COUNT):
        batch_paths = img_paths[i:i+BATCH_IMAGE_COUNT]
        for img_path in batch_paths:
            print(f"\n🖼️ Processing image: {img_path.name}")
            img_cv2 = cv2.imread(str(img_path))
            h, w = img_cv2.shape[:2]
            img_fn = os.path.splitext(os.path.basename(img_path))[0]
            frame_id = int(img_fn.split('_')[0])

            depth_img = None
            for f in os.listdir(args.depth_folder):
                if f.endswith(f"{frame_id}.npy"):
                    depth_img = np.load(os.path.join(args.depth_folder, f))
                    break
                elif f.endswith(f"{frame_id}.png"):
                    depth_img = cv2.imread(os.path.join(args.depth_folder, f), cv2.IMREAD_UNCHANGED) / 1000
                    if depth_img.shape != (h, w):
                        raise ValueError("Depth and image shape mismatch")
                    break
            if depth_img is None:
                raise FileNotFoundError(f"No depth for {frame_id}")

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
                    "img_cv2": img_cv2.copy(),
                    "bbox": best_hand["bbox"],
                    "is_right": best_hand["is_right"],
                    "depth_img": depth_img,
                    "img_fn": img_fn,
                    "score": best_hand["score"]
                })

            if len(batched_entries) < BATCH_IMAGE_COUNT:
                continue

        boxes = np.array([e["bbox"] for e in batched_entries])
        rights = np.array([e["is_right"] for e in batched_entries])
        dataset = ViTDetDataset(model_cfg, batched_entries, rescale_factor=args.rescale_factor, boxes=boxes, right=rights)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

        for i, batch in enumerate(dataloader):
            start_idx = i * args.batch_size
            batch = recursive_to(batch, device)
            with torch.no_grad():
                with torch.cuda.amp.autocast():
                    out = model(batch)

            batch_size = batch['img'].shape[0]
            pred_cam = out['pred_cam']
            pred_cam[:, 1] = (2 * batch['right'] - 1) * pred_cam[:, 1]
            box_center = batch["box_center"].float()
            box_size = batch["box_size"].float()
            img_size = batch["img_size"].float()
            scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
            cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size, scaled_focal_length).cpu().numpy()

            for n in range(batch_size):
                entry = batched_entries[start_idx + n]
                img_fn = entry["img_fn"]
                depth_img = entry["depth_img"]
                is_right = batch['right'][n].cpu().item()
                verts = out['pred_vertices'][n].cpu().numpy()
                verts[:, 0] = (2 * is_right - 1) * verts[:, 0]
                cam_t = cam_t_full[n]
                global_orient = out['pred_mano_params']['global_orient'][n].cpu().numpy().tolist()
                if not is_right:
                    global_orient[0] *= -1

                hand_key = f"{img_fn}"
                all_hand_results[hand_key] = {
                    "pred_cam_t": cam_t.tolist(),
                    "global_orient": global_orient,
                    "is_right": bool(is_right),
                    "scaled_focal_length": float(scaled_focal_length),
                    "score": float(entry["score"]),
                }

                all_verts.append(verts)
                all_cam_t.append(cam_t)
                all_right.append(is_right)

                if args.save_mesh:
                    mesh = renderer.vertices_to_trimesh(verts, cam_t, LIGHT_BLUE, is_right=bool(is_right))
                    mesh.export(os.path.join(args.out_folder, f'{img_fn}.obj'))

        batched_entries.clear()

    # Rendering full image, hand mask, overlay, and updating translation
    if len(all_verts) > 0:
        # Render RGBA for all hands
        cam_view = renderer.render_rgba_multiple(
            all_verts, cam_t=all_cam_t,
            render_res=img_size[n], is_right=all_right,
            mesh_base_color=LIGHT_BLUE,
            scene_bg_color=(1, 1, 1),
            focal_length=scaled_focal_length
        )
        # Save hand mask as <img_fn>_handmask.png
        last_img_fn = batched_entries[-1]["img_fn"] if batched_entries else None
        if last_img_fn is None and len(all_hand_results) > 0:
            last_img_fn = list(all_hand_results.keys())[-1]
        hand_mask = (cam_view[:, :, 3] > 0).astype(np.uint8) * 255
        if last_img_fn is not None:
            mask_path = os.path.join(args.out_folder, f'{last_img_fn}_handmask.png')
            cv2.imwrite(mask_path, hand_mask)

            # Compute and update corrected translation using ICP
            hamer_vertices = np.vstack(all_verts)
            # Use the last image's depth and intrinsics
            if batched_entries:
                last_depth_img = batched_entries[-1]["depth_img"]
            else:
                last_depth_img = None
                # fallback: try to get from any entry
                for entry in all_hand_results:
                    last_depth_img = None
                    break
            if last_depth_img is not None:
                hand_pcd = extract_hand_point_cloud(hand_mask, last_depth_img, camera_intrinsics)
                hamer_aligned = compute_aligned_hamer_translation(hamer_vertices, hand_pcd, hand_mask, camera_intrinsics)
                translation = hamer_aligned.mean(axis=0)
                all_hand_results[last_img_fn]["pred_cam_t"] = translation.tolist()

            # Create overlay image and save as <img_fn>_all.jpg
            if batched_entries:
                bg = batched_entries[-1]["img_cv2"].astype(np.float32)[:, :, ::-1] / 255.0
            else:
                bg = None
            if bg is not None:
                bg = np.concatenate([bg, np.ones_like(bg[:, :, :1])], axis=2)
                overlay = bg[:, :, :3] * (1 - cam_view[:, :, 3:]) + cam_view[:, :, :3] * cam_view[:, :, 3:]
                overlay_path = os.path.join(args.out_folder, f'{last_img_fn}_all.jpg')
                cv2.imwrite(overlay_path, 255 * overlay[:, :, ::-1])

    with open(os.path.join(args.out_folder, "hand_pose_camera_info.json"), "w") as f:
        json.dump(all_hand_results, f, indent=2)

    print("✅ Done")
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='HaMeR demo code')
    parser.add_argument('--checkpoint', type=str, default=DEFAULT_CHECKPOINT, help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='/home/xhe71/Desktop/robotool_data/06232025/slow/cam3/rgb', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='/home/xhe71/Desktop/robotool_data/06232025/slow/cam3', help='Output folder to save rendered results')
    parser.add_argument('--depth_folder', type=str, default='/home/xhe71/Desktop/robotool_data/06232025/slow/cam3/depth', help='folder with depth image')
    parser.add_argument('--intrinsics_path', default = 'setup/intrinsics_cam3.json',  help='load the camera intrinsics for the realsense camera')
    # parser.add_argument('--side_view', dest='side_view', action='store_true', default=False, help='If set, render side view also')
    # parser.add_argument('--full_frame', dest='full_frame', action='store_true', default=True, help='If set, render all people together also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=True, help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size for inference/fitting')
    parser.add_argument('--rescale_factor', type=float, default=1.0, help='Factor for padding the bbox')
    parser.add_argument('--body_detector', type=str, default='regnety', choices=['vitdet', 'regnety'], help='Using regnety improves runtime and reduces memory')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png'], help='List of file extensions to consider')
    args = parser.parse_args()
    detect_hand(args)
