import json
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
import os

# === Global smoothing settings ===
SMOOTH_PARAMS = {
    "trans_noise": 1e-5,
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


def smooth_quaternion_sequence(quats, window_size=7):
    smoothed = []
    for i in range(len(quats)):
        start = max(0, i - window_size // 2)
        end = min(len(quats), i + window_size // 2 + 1)
        avg = np.mean(quats[start:end], axis=0)
        avg /= np.linalg.norm(avg)
        smoothed.append(avg)
    smoothed = np.array(smoothed)

    # Prefer Z axis to point upward (not fixed, just to avoid 180° flipping)
    reference_z = np.array([0, 1, 0])  # Upward in world coordinates
    for i in range(len(smoothed)):
        z_axis = R.from_quat(smoothed[i]).apply(np.array([0, 0, 1]))
        if np.dot(z_axis, reference_z) < 0:
            smoothed[i] = -smoothed[i]

    return smoothed

def make_quaternion_sequence_continuous(quats):
    def rotation_distance(q1, q2):
        dot = np.clip(np.dot(q1, q2), -1.0, 1.0)
        return 2 * np.arccos(dot)

    aligned = [quats[0]]
    for i in range(1, len(quats)):
        prev = aligned[-1]
        candidates = [quats[i], -quats[i]]  # full flip

        # Also test flipping x, y, z individually
        for axis in range(3):
            flipped = quats[i].copy()
            flipped[axis] *= -1
            candidates.append(flipped)

        # Choose version with smallest angular difference from prev
        best = min(candidates, key=lambda q: rotation_distance(prev, q))
        aligned.append(best)
    return np.array(aligned)


def clamp_quaternion_velocity(quats, max_degrees=30):
    """Clamp quaternion angular velocity to limit frame-to-frame jumps."""
    max_angle = np.radians(max_degrees)
    clamped = [quats[0]]
    for i in range(1, len(quats)):
        prev = clamped[-1]
        curr = quats[i]
        dot = np.dot(prev, curr)
        if dot < 0:
            curr = -curr
            dot = -dot
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        if angle > np.pi:
            angle = 2 * np.pi - angle
        if angle > max_angle:
            ratio = max_angle / angle
            interp = (1 - ratio) * prev + ratio * curr
            interp /= np.linalg.norm(interp)
            clamped.append(interp)
        else:
            clamped.append(curr)
    return np.array(clamped)


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

def smooth_hand_pose_json_KF(json_path):
    with open(json_path, "r") as f:
        data = json.load(f)

    sorted_items = sorted(data.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0]))))

    # Remove leading frames with low score
    trimmed_items = []
    trim_threshold = 0.85
    found_valid_start = False
    for k, v in sorted_items:
        score = v.get("score", 0.0)
        if not found_valid_start and score < trim_threshold:
            continue
        found_valid_start = True
        trimmed_items.append((k, v))

    frame_ids = [k for k, _ in trimmed_items]
    cam_ts = np.array([v["pred_cam_t"] for _, v in trimmed_items])
    rot_mats = []
    valid_frame_ids = []
    scores = []
    for fid, v in trimmed_items:
        try:
            mat = np.array(v["global_orient"][0])
            if mat.shape == (3, 3):
                rot_mats.append(mat)
                valid_frame_ids.append(fid)
                scores.append(v.get("score", 0.0))  # default score if not present
            else:
                print(f"Skipping frame {fid} due to invalid shape: {mat.shape}")
        except Exception as e:
            print(f"Skipping frame {fid} due to error: {e}")

    rot_mats = np.array(rot_mats)
    cam_ts = np.array([data[fid]["pred_cam_t"] for fid in valid_frame_ids])
    frame_ids = valid_frame_ids
    frame_ids_numeric = [int(''.join(filter(str.isdigit, fid))) for fid in frame_ids]
    scores = np.array(scores)

    # Also smooth translation with KF
    smoothed_cam_ts = smooth_3d_sequence(cam_ts)

    # First apply KF to rotation vectors
    rot_vecs = R.from_matrix(rot_mats).as_rotvec()
    rot_vecs = make_rotation_vectors_continuous(rot_vecs)
    smoothed_rot_vecs = smooth_3d_sequence(rot_vecs)

    # Convert to quaternions and smooth
    smoothed_quats = R.from_rotvec(smoothed_rot_vecs).as_quat()
    smoothed_quats = make_quaternion_sequence_continuous(smoothed_quats)
    smoothed_quats = smooth_quaternion_sequence(smoothed_quats, window_size=15)
    smoothed_quats = clamp_quaternion_velocity(smoothed_quats, max_degrees=8)
    smoothed_rot_mats = R.from_quat(smoothed_quats).as_matrix()

    # 2D Visualization: Translation and SLERP-smoothed orientation
    original_quats = R.from_matrix(rot_mats).as_quat()
    smoothed_quats = R.from_matrix(smoothed_rot_mats).as_quat()

    # Compute angular velocity (frame-to-frame quaternion difference)
    angle_diff = []
    for i in range(1, len(smoothed_quats)):
        dot = np.dot(smoothed_quats[i - 1], smoothed_quats[i])
        dot = np.clip(dot, -1.0, 1.0)
        angle = 2 * np.arccos(dot)
        if angle > np.pi:
            angle = 2 * np.pi - angle
        angle_diff.append(np.degrees(angle))
    angle_diff = np.array([0] + angle_diff)  # pad first frame with 0

    score_threshold = trim_threshold
    replacement_flags = []
    for i, fid in enumerate(frame_ids):
        score = data[fid].get("score", 0.0)
        if score < score_threshold:
            # Get previous and next valid rotation matrices
            prev = None
            next_ = None
            for j in range(i - 1, -1, -1):
                if data[frame_ids[j]].get("score", 1.0) >= score_threshold:
                    prev = np.array(data[frame_ids[j]]["global_orient"][0])
                    break
            for j in range(i + 1, len(frame_ids)):
                if data[frame_ids[j]].get("score", 1.0) >= score_threshold:
                    next_ = np.array(data[frame_ids[j]]["global_orient"][0])
                    break
            if prev is not None and next_ is not None:
                avg = (prev + next_) / 2
                data[fid]["global_orient"] = [avg.tolist()]
                replacement_flags.append(1)
            else:
                replacement_flags.append(0)
        else:
            replacement_flags.append(0)

    fig, axs = plt.subplots(9, 1, figsize=(12, 18), sharex=True)
    trans_labels = ["X", "Y", "Z"]
    for i in range(3):
        axs[i].plot(frame_ids_numeric, cam_ts[:, i], label="Original", alpha=0.4)
        axs[i].plot(frame_ids_numeric, smoothed_cam_ts[:, i], label="KF Smoothed", linewidth=2)
        axs[i].set_ylabel(f"Trans {trans_labels[i]}")
        # axs[i].set_yscale('symlog')  # Use symmetric log scale to handle both + and - values
        axs[i].legend()

    quat_labels = ["X", "Y", "Z", "W"]
    for i in range(4):
        axs[i+3].plot(frame_ids_numeric, original_quats[:, i], label="Original", alpha=0.4)
        axs[i+3].plot(frame_ids_numeric, smoothed_quats[:, i], label="Smoothed", linewidth=2)
        axs[i+3].set_ylabel(f"Quat {quat_labels[i]}")
        axs[i+3].legend()

    # confidence
    axs[-2].plot(frame_ids_numeric, scores, color="black", label="Confidence Score")
    replaced_x = [i for i, flag in enumerate(replacement_flags) if flag]
    replaced_y = [scores[i] for i in replaced_x]
    axs[-2].scatter([frame_ids_numeric[i] for i in replaced_x], replaced_y, color="red", label="Replaced")
    axs[-2].legend()
    
    axs[-1].plot(frame_ids_numeric, angle_diff, color="purple", label="Quat Δθ (deg)")
    axs[-1].set_ylabel("Δ Angle")
    axs[-1].legend()

    axs[-1].set_xlabel("Frame Index")
    fig.suptitle("Translation (KF) and Orientation Smoothing")
    plt.tight_layout()
    obs_noise = SMOOTH_PARAMS["obs_noise"]
    trans_noise = SMOOTH_PARAMS["trans_noise"]
    plt.savefig(f"combined_pose_smoothing_plot.png", dpi=300)
    plt.close()

    # Save updated JSON
    for i, fid in enumerate(frame_ids):
        data[fid]["global_orient"] = [smoothed_rot_mats[i].tolist()]

    output_path = os.path.splitext(json_path)[0] + "_smoothed.json"
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)

# === Example use ===
if __name__ == "__main__":
    smooth_hand_pose_json_KF("/home/xhe71/Desktop/robotool_data/Color_hamer/hand_pose_camera_info.json")
