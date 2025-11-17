"""Entry point for dataset conversion."""

import argparse
import sys
sys.path.append("./")

"""Main converter class."""

import os
import h5py
import numpy as np
from tqdm import tqdm
from configs.workspace import WORKSPACE, MAX_POINT_NUM_HDF5
from hand.hand_utils import load_camera_info_dict
from hand.trajectory_loader import PointCloudProcessor
from hand.hand_preprocessor import HandPreprocessor
from hand.trajectory_loader import TrajectoryLoader


class RealToRobomimicConverter:
    """Converts real dataset to robomimic format."""

    def __init__(self, real_dataset_path: str, output_robomimic_path: str):
        """Initialize converter.

        Args:
            real_dataset_path: Path to real dataset
            output_robomimic_path: Output path for HDF5 file
        """
        self.real_dataset_path = real_dataset_path
        self.process_path = os.path.join(real_dataset_path, "output")
        self.robomimic_dataset_path = output_robomimic_path
        self.robomimic_center = np.array([0, 0, 0.7])

        # Load episode and camera lists
        self.episode_list = [
            f for f in os.listdir(real_dataset_path) if f.startswith("episode")
        ]
        cam_list = [
            f for f in os.listdir(os.path.join(real_dataset_path, self.episode_list[0]))
            if f.startswith("cam")
        ]

        self.cam_list = cam_list
        self.num_cams = len(cam_list)

        main_cam_idx = 3
        self.main_cam = f'cam{main_cam_idx}'
        self.workspace = WORKSPACE
        self.fix_point_num = MAX_POINT_NUM_HDF5
        obs_type = 'pcd'

        # Load camera info
        self.info_dict = load_camera_info_dict(os.path.join('configs', "camera_info.yaml"))

        # Initialize components
        self.pcd_processor = PointCloudProcessor(
            self.workspace, self.fix_point_num, self.robomimic_center
        )
        self.hand_preprocessor = HandPreprocessor(real_dataset_path, self.info_dict, main_cam_idx)
        self.trajectory_loader = TrajectoryLoader(
            real_dataset_path, self.process_path, obs_type,
            cam_list, self.main_cam, self.pcd_processor
        )

        # Run preprocessing
        print(f"Extracting actions from real dataset using HAMER...")
        self.hand_preprocessor.preprocess_all(self.episode_list)

    def convert(self) -> None:
        """Convert dataset to robomimic format."""
        print(f"Converting data to robomimic format...")

        with h5py.File(self.robomimic_dataset_path, "w") as f_out:
            data_grp = f_out.create_group("data")

            bar = tqdm(total=len(self.episode_list), desc="Converting episodes")
            for episode_idx, episode_name in enumerate(self.episode_list):
                traj = self.trajectory_loader.load_trajectory(episode_name)

                ep = f"demo_{episode_idx}"
                ep_data_grp = data_grp.create_group(ep)

                # Save trajectory data
                ep_data_grp.create_dataset("actions", data=np.array(traj["actions"]))
                ep_data_grp.create_dataset("states", data=np.array(traj["states"]))
                ep_data_grp.create_dataset("rewards", data=np.array(traj["rewards"]))
                ep_data_grp.create_dataset("dones", data=np.array(traj["dones"]))

                # Save observations
                for k in traj["obs"].keys():
                    data = np.array(traj["obs"][k])
                    assert data.dtype != np.dtype('O'), \
                        f"Data type should not be object, but got {data.dtype}"
                    ep_data_grp.create_dataset(f"obs/{k}", data=data, compression="gzip")

                ep_data_grp.attrs["num_samples"] = traj["actions"].shape[0]
                tqdm.write(f"ep {episode_idx}: wrote {ep_data_grp.attrs['num_samples']} "
                          f"transitions to group {ep}")
                bar.update(1)
            bar.close()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Convert a real dataset to robomimic format."
    )
    parser.add_argument(
        "--real_dataset_path",
        type=str,
        required=True,
        help="Path to the real dataset."
    )
    parser.add_argument(
        "--output_robomimic_path",
        type=str,
        required=True,
        help="Output path for the robomimic HDF5 file."
    )
    args = parser.parse_args()
    input("Ensure that all your hand trajectories always start with the default robot pose (Euler XYZ [180, 0, 0]) because this script is handling pose by enforcing this. Press Enter to continue...")
    converter = RealToRobomimicConverter(
        real_dataset_path=args.real_dataset_path,
        output_robomimic_path=args.output_robomimic_path
    )
    converter.convert()


if __name__ == "__main__":
    main()
