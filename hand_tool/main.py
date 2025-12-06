"""Entry point for dataset conversion."""

import argparse
import sys
sys.path.append("./")

"""Main converter class."""

import os
import h5py
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from configs.workspace import WORKSPACE, MAX_POINT_NUM_HDF5
from hand_tool.hand_utils import load_camera_info_dict
from hand_tool.trajectory_loader import ObservationProcessor
from hand_tool.trajectory_loader import TrajectoryLoader


def load_trajectory_worker(args):
    """Helper function for parallel trajectory loading.

    Args:
        args: Tuple of (episode_name, real_dataset_path, process_path, data_type,
              info_dict, cam_list, main_cam, workspace, fix_point_num)

    Returns:
        Tuple of (episode_name, trajectory_data)
    """
    (episode_name, real_dataset_path, process_path, data_type,
     info_dict, cam_list, main_cam, workspace, fix_point_num) = args

    # Create processor and loader in worker process
    pcd_processor = ObservationProcessor(workspace, fix_point_num, data_type)
    trajectory_loader = TrajectoryLoader(
        real_dataset_path, process_path, data_type, info_dict,
        cam_list, main_cam, pcd_processor
    )

    traj = trajectory_loader.load_trajectory(episode_name)
    return episode_name, traj


class RealToRobomimicConverter:
    """Converts real dataset to robomimic format."""

    def __init__(self, real_dataset_path: str, output_robomimic_path: str, data_type: str = "hand") -> None:
        """Initialize converter.

        Args:
            real_dataset_path: Path to real dataset
            output_robomimic_path: Output path for HDF5 file
        """
        self.real_dataset_path = real_dataset_path
        self.process_path = os.path.join(real_dataset_path, "output")
        self.robomimic_dataset_path = output_robomimic_path

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

        # Load camera info
        self.info_dict = load_camera_info_dict(os.path.join('configs', "camera_info.yaml"))

        # Initialize components
        self.pcd_processor = ObservationProcessor(
            self.workspace, self.fix_point_num, data_type
        )
        self.trajectory_loader = TrajectoryLoader(
            real_dataset_path, self.process_path, data_type, self.info_dict,
            cam_list, self.main_cam, self.pcd_processor
        )

        self.data_type = data_type
        # Run preprocessing
        if data_type == "hand":
            from hand_tool.hand_preprocessor import HandPreprocessor
            print(f"Extracting actions from real dataset using HAMER...")
            self.hand_preprocessor = HandPreprocessor(real_dataset_path, self.info_dict, main_cam_idx)
            self.hand_preprocessor.preprocess_all(self.episode_list)
        else:
            print(f"Skipping HAMER preprocessing for data type: {data_type}")

    def convert(self, num_workers=None) -> None:
        """Convert dataset to robomimic format.

        Args:
            num_workers: Number of parallel workers. If None, uses CPU count.
        """
        print(f"Converting data to robomimic format...")

        # Prepare arguments for parallel trajectory loading
        load_args = [
            (episode_name, self.real_dataset_path, self.process_path, self.data_type,
             self.info_dict, self.cam_list, self.main_cam, self.workspace, self.fix_point_num)
            for episode_name in self.episode_list
        ]

        # Load all trajectories in parallel
        print(f"Loading {len(self.episode_list)} trajectories in parallel...")
        trajectories = {}
        with ProcessPoolExecutor(max_workers=num_workers) as executor:
            futures = {executor.submit(load_trajectory_worker, args): args[0]
                      for args in load_args}

            bar = tqdm(total=len(futures), desc="Loading trajectories")
            for future in as_completed(futures):
                episode_name, traj = future.result()
                trajectories[episode_name] = traj
                bar.update(1)
            bar.close()

        # Write trajectories to HDF5 sequentially
        print(f"Writing trajectories to HDF5...")
        with h5py.File(self.robomimic_dataset_path, "w") as f_out:
            data_grp = f_out.create_group("data")

            bar = tqdm(total=len(self.episode_list), desc="Writing episodes")
            episode_idx = 0
            for episode_name in self.episode_list:
                traj = trajectories[episode_name]
                if len(traj["actions"]) < 16:
                    print(f"Skipping episode {episode_name} due to insufficient length "
                          f"({len(traj['actions'])} < 16)")
                    continue

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
                episode_idx += 1
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
    parser.add_argument(
        "--data_type",
        type=str,
        required=True,
        choices=["hand", "robot"],
        help="Data type to convert (hand or robot)."
    )
    args = parser.parse_args()
    print("Ensure that all your hand trajectories always start with the default robot pose (Euler XYZ [180, 0, 0]) because this script is handling pose by enforcing this. ")
    converter = RealToRobomimicConverter(
        real_dataset_path=args.real_dataset_path,
        output_robomimic_path=args.output_robomimic_path,
        data_type=args.data_type,
    )
    converter.convert()


if __name__ == "__main__":
    main()
