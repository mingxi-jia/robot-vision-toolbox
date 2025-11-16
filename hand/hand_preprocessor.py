"""Preprocessing pipeline for episodes."""

import os
import numpy as np
import torch
from time import perf_counter
from tqdm import tqdm
from hand.hamer_wrapper import HandPreprocessor as Hamer
from hand.hand_utils import generate_pcd_sequence



class HandPreprocessor:
    """Handles preprocessing of episodes."""

    def __init__(self, real_dataset_path: str, info_dict: dict, main_cam_idx: int):
        """Initialize preprocessor.

        Args:
            real_dataset_path: Path to real dataset
            info_dict: Camera information dictionary
        """
        self.main_cam_idx = main_cam_idx
        self.real_dataset_path = real_dataset_path
        self.process_path = os.path.join(real_dataset_path, "output")
        self.info_dict = info_dict
        self.hamer = Hamer(real_dataset_path)

    def preprocess_episode(self, episode_name: str) -> None:
        """Preprocess a single episode.

        Args:
            episode_name: Name of episode to process
        """
        done_indicator = os.path.join(
            self.process_path, episode_name, "hand_poses_wrt_world.npy"
        )
        if os.path.exists(done_indicator):
            print(f"Episode {episode_name} already processed. Skipping...")
            return

        starting_time = perf_counter()
        episode_path = os.path.join(self.real_dataset_path, episode_name)

        # Process each camera with HAMER
        for cam_id in [1, 2, 3]:
            print(f"\n========= Processing Camera {cam_id} =========")
            self.hamer.process(episode_path, cam_id)
            torch.cuda.empty_cache()

        # Generate PCD with rendered sphere
        pcd_gen_start = perf_counter()
        generate_pcd_sequence(
            episode_path, self.hamer.process_path, self.info_dict,
            sphere_cam=self.main_cam_idx, segment=False, visualize_coordinate_axis=True
        )
        elapsed_segment_false = perf_counter() - pcd_gen_start

        # Generate PCD with human segmentation
        pcd_gen_start = perf_counter()
        generate_pcd_sequence(
            episode_path, self.hamer.process_path, self.info_dict,
            sphere_cam=self.main_cam_idx, segment=True, visualize_coordinate_axis=False
        )
        elapsed_segment_true = perf_counter() - pcd_gen_start

        # Save poses
        pcd_gen_start = perf_counter()
        torch.cuda.empty_cache()
        elapsed_save = perf_counter() - pcd_gen_start

        print(f"pcd generation (segment=False) takes {elapsed_segment_false:.2f} seconds.")
        print(f"pcd generation (segment=True) takes {elapsed_segment_true:.2f} seconds.")
        print(f"pcd saving takes {elapsed_save:.2f} seconds.")
        print(f"Episode {episode_name} processed in {perf_counter() - starting_time:.2f} seconds.")

        # Create done marker
        with open(os.path.join(self.process_path, episode_name, 'DONE'), "a") as f:
            f.write("This is a marker file to indicate that the preprocessing is done for this episode.\n")

    def preprocess_all(self, episode_list: list[str]) -> None:
        """Preprocess all episodes.

        Args:
            episode_list: List of episode names to process
        """
        print(f"Extracting actions from real dataset using HAMER...")
        episode_list.sort() 
        preprocess_start_time = perf_counter()

        for episode_name in tqdm(episode_list, desc="Preprocessing episodes"):
            self.preprocess_episode(episode_name)

        print(f"Preprocessing done in {perf_counter() - preprocess_start_time:.2f} seconds.")
