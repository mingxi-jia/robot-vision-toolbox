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
        if self._is_episode_processed(episode_name):
            print(f"Episode {episode_name} already processed. Skipping...")
            return

        start_time = perf_counter()
        episode_path = os.path.join(self.real_dataset_path, episode_name)

        # Process cameras and generate point clouds
        self._process_cameras(episode_path)
        timings = self._generate_point_clouds(episode_path)

        # Report timings and mark complete
        self._report_timings(episode_name, start_time, timings)
        self._mark_episode_complete(episode_name)

    def _is_episode_processed(self, episode_name: str) -> bool:
        """Check if episode has already been processed."""
        done_marker = os.path.join(self.process_path, episode_name, 'DONE')
        return os.path.exists(done_marker)

    def _process_cameras(self, episode_path: str) -> None:
        """Process all cameras with HAMER."""
        for cam_id in [1, 2, 3]:
            print(f"\n========= Processing Camera {cam_id} =========")
            self.hamer.process(episode_path, cam_id)
            torch.cuda.empty_cache()

    def _generate_point_clouds(self, episode_path: str) -> dict:
        """Generate point clouds with and without segmentation.

        Returns:
            Dictionary with timing information
        """
        timings = {}

        # Generate PCD with rendered sphere
        start = perf_counter()
        generate_pcd_sequence(
            episode_path, self.hamer.process_path, self.info_dict,
            sphere_cam=self.main_cam_idx, segment=False, visualize_coordinate_axis=True
        )
        timings['pcd_no_segment'] = perf_counter() - start

        # Generate PCD with human segmentation
        start = perf_counter()
        generate_pcd_sequence(
            episode_path, self.hamer.process_path, self.info_dict,
            sphere_cam=self.main_cam_idx, segment=True, visualize_coordinate_axis=False
        )
        timings['pcd_with_segment'] = perf_counter() - start

        torch.cuda.empty_cache()
        return timings

    def _report_timings(self, episode_name: str, start_time: float, timings: dict) -> None:
        """Print timing information."""
        print(f"PCD generation (segment=False): {timings['pcd_no_segment']:.2f}s")
        print(f"PCD generation (segment=True): {timings['pcd_with_segment']:.2f}s")
        print(f"Episode {episode_name} processed in {perf_counter() - start_time:.2f}s")

    def _mark_episode_complete(self, episode_name: str) -> None:
        """Create marker file to indicate processing is complete."""
        done_marker = os.path.join(self.process_path, episode_name, 'DONE')
        with open(done_marker, "w") as f:
            f.write("Preprocessing complete\n")

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
