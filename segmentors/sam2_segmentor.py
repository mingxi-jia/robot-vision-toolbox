import os
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

import torch
from sam2.build_sam import build_sam2_camera_predictor
from sam2.sam2_video_predictor import SAM2VideoPredictor

from segmentors.utils import show_points, show_mask, visualize_prompt
from segmentors.configs import GRIPPER_ID, POINT_PROMPTS, POINT_PROMPT_LABELS

def blend_rgb_and_mask_for_visualization(current_frame, segmentation_mask):
    segmentation_vis = np.zeros_like(current_frame)
    segmentation_vis[segmentation_mask] = current_frame[segmentation_mask]
    segmentation_vis[~segmentation_mask] = current_frame[~segmentation_mask] // 2
    return segmentation_vis

class SamVideoSegmentor():
    def __init__(self, load_from_hugging_face=False):

        if load_from_hugging_face:
            self.predictor = SAM2VideoPredictor.from_pretrained("facebook/sam2-hiera-large")
        else:
            sam_path = "submodules/segment-anything-2-real-time"
            checkpoint = os.path.join(sam_path, "checkpoints/sam2.1_hiera_small.pt")
            model_cfg = os.path.join("configs/sam2.1/sam2.1_hiera_s.yaml")
            self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint)

        self.init = False
            

    def tracking(self, new_frame, gripper_id, point_prompts, prompt_labels):
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            if not self.init:
                self.predictor.load_first_frame(new_frame)
                self.init = True
                _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(frame_idx=0,
                                                                                obj_id=gripper_id,
                                                                                points=point_prompts,
                                                                                labels=prompt_labels,
                                                                            )
            else:
                out_obj_ids, out_mask_logits = self.predictor.track(new_frame)
            
            return (out_mask_logits[0][0] > 0.0).cpu().numpy()

    def segment_video(self, video_root: str, gripper_id:int, point_prompts: np.array, prompt_labels: np.array):
        video_dir = os.path.join(video_root, 'rgbs')
        assert os.path.exists(video_dir), f"A 'rgbs' folder is missing from {video_root}"

        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            inference_state = self.predictor.init_state(video_path=video_dir)
            self.predictor.reset_state(inference_state)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_points_or_box(
                inference_state=inference_state,
                frame_idx=0,
                obj_id=gripper_id,
                points=point_prompts,
                labels=prompt_labels,
            )

            first_frame = np.asarray(Image.open(os.path.join(video_dir, "0.jpg")))
            H, W, C = first_frame.shape
            visualize_prompt(first_frame, point_prompts, prompt_labels)

            # run propagation throughout the video and collect the results in a dict
            mask_vis_path = os.path.join(video_root, 'gripper_masks_vis')
            os.makedirs(mask_vis_path, exist_ok=True)
            video_segments = {}  # video_segments contains the per-frame segmentation results
            for out_frame_idx, out_obj_ids, out_mask_logits in self.predictor.propagate_in_video(inference_state):
                video_segments[out_frame_idx] = {
                    out_obj_id: (out_mask_logits[i] > 0.0).cpu().numpy()
                    for i, out_obj_id in enumerate(out_obj_ids)
                }
                # visualization
                segmentation_mask = (out_mask_logits[0][0] > 0.0).cpu().numpy()
                current_frame = np.asarray(Image.open(os.path.join(video_dir, f"{out_frame_idx}.jpg")))
                segmentation_vis = blend_rgb_and_mask_for_visualization(current_frame, segmentation_mask)
                segmentation_vis = Image.fromarray(segmentation_vis)
                segmentation_vis.save(os.path.join(mask_vis_path, f"{out_frame_idx}.jpg"), "JPEG")
            np.save(os.path.join(video_root, 'masks.npy'), video_segments)

def segment_entire_video():
    segmentor = SamVideoSegmentor()
    # ---------seting up prompts-----------
    # give a unique id to each object we interact with (it can be any integers)
    
    camera_name = 'dave'
    video_root = './raw_data/episode_0000'
    segmentor.segment_video(video_root, GRIPPER_ID, POINT_PROMPTS[camera_name], POINT_PROMPT_LABELS[camera_name])

def track_video_frames():
    segmentor = SamVideoSegmentor()

    camera_name = 'dave'
    video_path = './example_data/episode_0000/rgbs'
    gripper_id, point_prompts, prompt_labels = GRIPPER_ID, POINT_PROMPTS[camera_name], POINT_PROMPT_LABELS[camera_name]
    frame_names = os.listdir(video_path)
    frame_names.sort(key=lambda p: int(os.path.splitext(p)[0]))

    for i in range(len(frame_names)):
        current_frame = np.asarray(Image.open(os.path.join(video_path, f"{i}.jpg")))
        mask = segmentor.tracking(current_frame, gripper_id, point_prompts, prompt_labels)
        frame = blend_rgb_and_mask_for_visualization(current_frame, mask)
        cv2.imshow('Frame', frame[:,:,::-1])
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    # segment_entire_video()
    track_video_frames()