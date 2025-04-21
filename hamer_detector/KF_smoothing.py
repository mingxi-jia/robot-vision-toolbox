import json
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
import os

# === Global smoothing settings ===
SMOOTH_PARAMS = {
    "trans_noise": 5e-5,
    "obs_noise": 1e-4,
}

def make_rotation_vectors_continuous(rot_vecs):
    # rot_vecs: (N, 3)
    unwrapped = [rot_vecs[0]]
    for i in range(1, len(rot_vecs)):
        prev = unwrapped[-1]
        curr = rot_vecs[i]

        # Try both current and flipped version
        if np.linalg.norm(curr - prev) > np.linalg.norm(-curr - prev):
            curr = -curr  # flip to avoid sudden jump

        unwrapped.append(curr)
    return np.array(unwrapped)


def smooth_3d_sequence(data):
    kf = KalmanFilter(
        initial_state_mean=data[0],
        transition_matrices=np.eye(3),
        observation_matrices=np.eye(3),
        transition_covariance=SMOOTH_PARAMS["trans_noise"] * np.eye(3),
        observation_covariance=SMOOTH_PARAMS["obs_noise"] * np.eye(3)
    )
    smoothed, _ = kf.smooth(data)
    return smoothed

def plot_original_vs_smoothed(original, smoothed, labels, title=""):
    fig, axs = plt.subplots(len(labels), 1, figsize=(12, 8), sharex=True)
    for i in range(len(labels)):
        axs[i].plot(original[:, i], label="Original", alpha=0.5)
        axs[i].plot(smoothed[:, i], label="Smoothed", linewidth=2)
        axs[i].set_ylabel(labels[i])
        axs[i].legend()
    axs[-1].set_xlabel("Frame Index")
    fig.suptitle(title)
    plt.tight_layout()
    obs_noise = SMOOTH_PARAMS["obs_noise"]
    trans_noise = SMOOTH_PARAMS["trans_noise"]
    plt.savefig(f"smoothing_centroid_plot_ON{obs_noise}_TN{trans_noise}.png", dpi=300)  
    plt.close()

def smooth_KF_yml(yml_path):
    with open(yml_path, "r") as f:
        data = json.load(f)

    sorted_items = sorted(data.items(), key=lambda x: x[0])
    frame_ids = [k for k, _ in sorted_items]
    values = np.array([v for _, v in sorted_items])

    smoothed = smooth_3d_sequence(values)

    output_path = os.path.splitext(yml_path)[0] + "_smoothed.yml"
    smoothed_dict = {fid: smoothed[i].tolist() for i, fid in enumerate(frame_ids)}
    with open(output_path, "w") as f:
        json.dump(smoothed_dict, f, indent=2)

    plot_original_vs_smoothed(values, smoothed, labels=["X", "Y", "Z"], title="YML Smoothing")

def smooth_hand_pose_json_KF(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    sorted_items = sorted(data.items(), key=lambda x: x[0])
    frame_ids = [k for k, _ in sorted_items]
    cam_ts = np.array([v["pred_cam_t"] for _, v in sorted_items])
    rot_mats = np.array([v["global_orient"][0] for _, v in sorted_items])
    rot_vecs = R.from_matrix(rot_mats).as_rotvec()

    smoothed_cam_ts = smooth_3d_sequence(cam_ts)
    rot_vecs = make_rotation_vectors_continuous(rot_vecs)
    smoothed_rot_vecs = smooth_3d_sequence(rot_vecs)

    smoothed_rot_mats = R.from_rotvec(smoothed_rot_vecs).as_matrix()

    # Visualization
    fig, axs = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
    for i, label in enumerate(["X", "Y", "Z"]):
        axs[i][0].plot(cam_ts[:, i], label="Original", alpha=0.4)
        axs[i][0].plot(smoothed_cam_ts[:, i], label="Smoothed")
        axs[i][0].set_ylabel(f"{label} (cam_t)")
        axs[i][0].legend()

        axs[i][1].plot(rot_vecs[:, i], label="Original", alpha=0.4)
        axs[i][1].plot(smoothed_rot_vecs[:, i], label="Smoothed")
        axs[i][1].set_ylabel(f"{label} (rotvec)")
        axs[i][1].legend()
    axs[-1][0].set_xlabel("Frame Index")
    axs[-1][1].set_xlabel("Frame Index")
    fig.suptitle("Camera Translation and Orientation Smoothing")
    plt.tight_layout()
    obs_noise = SMOOTH_PARAMS["obs_noise"]
    trans_noise = SMOOTH_PARAMS["trans_noise"]
    plt.savefig(f"smoothing_handpose_plot_ON{obs_noise}_TN{trans_noise}.png", dpi=300)  # You can change filename/path
    plt.close()

    # Save updated JSON
    for i, fid in enumerate(frame_ids):
        data[fid]["pred_cam_t"] = smoothed_cam_ts[i].tolist()
        data[fid]["global_orient"] = [smoothed_rot_mats[i].tolist()]

    output_path = os.path.splitext(json_path)[0] + "_smoothed.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

# === Example use ===
if __name__ == "__main__":
    smooth_hand_pose_json_KF("/home/xhe71/Documents/GitHub/robot-vision-toolbox/hamer_detector/hamer_output/hand_pose_camera_info.json")
    smooth_KF_yml("/home/xhe71/Documents/GitHub/robot-vision-toolbox/hamer_detector/hamer_output/centroids.yml")
