
import json
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os
from scipy.signal import medfilt
import trimesh
# === Global smoothing settings ===
# for sample rate less that 3
SMOOTH_PARAMS = {
    "trans_noise": 5e-6,
    "obs_noise": 1e-4,
}
SMOOTH_PARAMS = {
    "trans_noise": 1e-6,
    "obs_noise": 1e-3,
}
# SMOOTH_PARAMS = {
#     "trans_noise": 1e-4,
#     "obs_noise": 1e-4,
# }

# Separate smoothing for orientation
ORIENT_SMOOTH_PARAMS = {
    "trans_noise": 1e-4,
    "obs_noise": 1e-5,
}


def load_and_filter_handedness(json_path):
    with open(json_path, "r") as f:
        all_hand_results = json.load(f)
    right_count = sum(1 for v in all_hand_results.values() if v['is_right'])
    left_count = sum(1 for v in all_hand_results.values() if not v['is_right'])
    majority_is_right = right_count >= left_count
    filtered = {k: v for k, v in all_hand_results.items() if v['is_right'] == majority_is_right}
    return filtered, os.path.dirname(json_path)

def trim_low_score_frames(data, threshold=0.7):
    sorted_items = sorted(data.items(), key=lambda x: int(''.join(filter(str.isdigit, x[0]))))
    trimmed = []
    found_valid = False
    for k, v in sorted_items:
        if not found_valid and v.get("score", 0.0) < threshold:
            continue
        found_valid = True
        trimmed.append((k, v))
    return trimmed

def plot_raw_pose_changes(raw_cam_ts, raw_rot_mats, rootpath):
    trans_deltas, angle_deltas = [], []
    for i in range(1, len(raw_cam_ts)):
        trans_deltas.append(np.linalg.norm(raw_cam_ts[i] - raw_cam_ts[i-1]))
        delta_rot = R.from_matrix(raw_rot_mats[i-1]).inv() * R.from_matrix(raw_rot_mats[i])
        angle_deltas.append(np.degrees(np.linalg.norm(delta_rot.as_rotvec())))

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(trans_deltas, label="Δ Translation (m)")
    axs[1].plot(angle_deltas, label="Δ Orientation (deg)")
    plt.savefig(os.path.join(rootpath, "debug_raw_pose_jump_plot.png"), dpi=300)
    plt.close()


def filter_frames(trimmed_items, skip_rate):
    filtered_ids, filtered_ts, filtered_rots = [], [], []
    update_filtering_to_per_axis(filtered_ids, filtered_ts, filtered_rots, trimmed_items,
                                score_threshold=0.75, max_axis_jump=0.3*skip_rate, max_angle_jump=65)
    return filtered_ids, np.array(filtered_ts), np.array(filtered_rots)



# === Weighted Moving Average ===
def weighted_moving_average(data, window_size=3, weights=None):
    """
    Apply weighted moving average to a 2D array (N, D), returns same shape.
    """
    if weights is None:
        weights = np.linspace(0.5, 1, window_size)
    weights /= weights.sum()  # Normalize

    smoothed = []
    for i in range(len(data)):
        start = max(0, i - window_size + 1)
        window = data[start:i+1]
        w = weights[-len(window):]
        weighted_avg = np.average(window, axis=0, weights=w)
        smoothed.append(weighted_avg)
    return np.array(smoothed)


def smooth_3d_sequence(data):
    initial_state = np.hstack([data[0], [0, 0, 0]])  # [x, y, z, vx, vy, vz]

    transition_matrix = np.block([
        [np.eye(3), np.eye(3)],       # position update: x_new = x_old + v
        [np.zeros((3, 3)), np.eye(3)] # constant velocity model
    ])
    observation_matrix = np.hstack([np.eye(3), np.zeros((3, 3))])  # observe only position

    kf = KalmanFilter(
        initial_state_mean=initial_state,
        transition_matrices=transition_matrix,
        observation_matrices=observation_matrix,
        transition_covariance=SMOOTH_PARAMS["trans_noise"] * np.eye(6),
        observation_covariance=SMOOTH_PARAMS["obs_noise"] * np.eye(3)
    )

    smoothed, _ = kf.smooth(data)
    return smoothed[:, :3]  # Return only position

def smooth_orientation_sequence(rot_vecs):
    kf = KalmanFilter(
        initial_state_mean=rot_vecs[0],
        transition_matrices=np.eye(3),
        observation_matrices=np.eye(3),
        transition_covariance=ORIENT_SMOOTH_PARAMS["trans_noise"] * np.eye(3),
        observation_covariance=ORIENT_SMOOTH_PARAMS["obs_noise"] * np.eye(3)
    )
    smoothed, _ = kf.smooth(rot_vecs)
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

def update_filtering_to_per_axis(
    filtered_frame_ids, filtered_cam_ts, filtered_rot_mats,
    trimmed_items, score_threshold=0.8,
    max_axis_jump=0.2,  # meters per axis
    max_angle_jump=45   # degrees
):
    """
    Filters frames based on:
    - Low confidence score
    - Large translation delta in any axis
    - Large orientation change

    All changes are done in-place on filtered_* lists.
    """
    filtered_frame_ids.clear()
    filtered_cam_ts.clear()
    filtered_rot_mats.clear()

    last_valid_t = None
    last_valid_r = None

    for i, (fid, v) in enumerate(trimmed_items):
        score = v.get("score", 0.0)
        cam_t = np.array(v["pred_cam_t"])
        try:
            rot_mat = np.array(v["global_orient"])
            if rot_mat.shape != (3, 3):
                raise ValueError(f"Invalid rotation matrix shape for frame {fid}: {rot_mat.shape}")
        except Exception as e:
            print(f"[WARN] Skipping {fid} due to invalid rotation: {e}")
            continue

        keep = True

        if score < score_threshold:
            keep = False
            print(f"[DEBUG] Skipping {fid}: low score ({score:.2f})")

        if last_valid_t is not None:
            delta_axis = np.abs(cam_t - last_valid_t)
            if np.any(delta_axis > max_axis_jump):
                print(f"[DEBUG] Skipping {fid}: large axis translation jump {delta_axis}")
                keep = False

        if last_valid_r is not None:
            delta_rot = R.from_matrix(last_valid_r).inv() * R.from_matrix(rot_mat)
            angle_rad = np.linalg.norm(delta_rot.as_rotvec())
            angle = np.degrees(angle_rad)
            if angle > max_angle_jump:
                print(f"[DEBUG] Skipping {fid}: large orientation jump ({angle:.1f} deg)")
                keep = False

        if keep:
            filtered_frame_ids.append(fid)
            filtered_cam_ts.append(cam_t)
            filtered_rot_mats.append(rot_mat)
            last_valid_t = cam_t
            last_valid_r = rot_mat

    return filtered_frame_ids, filtered_cam_ts, filtered_rot_mats


def interpolate_dropped_frames(frame_ids, cam_ts, rot_mats, all_frame_ids):
    """
    Interpolate missing frames in the sequence using linear interpolation for translation
    and SLERP (spherical linear interpolation) for rotation.

    Args:
        frame_ids: list of filtered frame IDs (strings).
        cam_ts: np.ndarray of shape (N, 3) - filtered translation vectors.
        rot_mats: np.ndarray of shape (N, 3, 3) - filtered rotation matrices.
        all_frame_ids: list of all frame IDs (strings), sorted in order.

    Returns:
        interp_frame_ids: full list of frame IDs with interpolated data.
        interp_cam_ts: np.ndarray of shape (M, 3) - interpolated translation.
        interp_rot_mats: np.ndarray of shape (M, 3, 3) - interpolated rotation.
    """
    full_cam_ts = {fid: t for fid, t in zip(frame_ids, cam_ts)}
    full_rot_mats = {fid: r for fid, r in zip(frame_ids, rot_mats)}

    interp_cam_ts = []
    interp_rot_mats = []
    interp_frame_ids = []

    sorted_all = sorted(all_frame_ids, key=lambda fid: int(''.join(filter(str.isdigit, fid))))

    for i, fid in enumerate(sorted_all):
        if fid in full_cam_ts:
            interp_cam_ts.append(full_cam_ts[fid])
            interp_rot_mats.append(full_rot_mats[fid])
        else:
            # Skip interpolation at beginning or end
            prev_idx = next((j for j in range(i - 1, -1, -1) if sorted_all[j] in full_cam_ts), None)
            next_idx = next((j for j in range(i + 1, len(sorted_all)) if sorted_all[j] in full_cam_ts), None)

            if prev_idx is None or next_idx is None:
                continue  # cannot interpolate at edge

            fid_prev = sorted_all[prev_idx]
            fid_next = sorted_all[next_idx]

            t_prev = full_cam_ts[fid_prev]
            t_next = full_cam_ts[fid_next]
            r_prev = R.from_matrix(full_rot_mats[fid_prev])
            r_next = R.from_matrix(full_rot_mats[fid_next])

            # Ratio for interpolation
            ratio = (i - prev_idx) / (next_idx - prev_idx)

            # Interpolate translation
            t_interp = (1 - ratio) * t_prev + ratio * t_next

            # SLERP interpolation for rotation
            key_times = [0, 1]
            key_rots = R.from_quat([r_prev.as_quat(), r_next.as_quat()])
            slerp = Slerp(key_times, key_rots)
            r_interp = slerp(ratio).as_matrix()

            interp_cam_ts.append(t_interp)
            interp_rot_mats.append(r_interp)

        interp_frame_ids.append(fid)

    return interp_frame_ids, np.array(interp_cam_ts), np.array(interp_rot_mats)


def smooth_hand_pose_json_KF(json_path, skip_rate=1):
    data, rootpath = load_and_filter_handedness(json_path)
    trimmed_items = trim_low_score_frames(data)

    # Prepare raw pose for debug plot
    raw_cam_ts = [np.array(v["pred_cam_t"]) for _, v in trimmed_items]
    raw_rot_mats = [np.array(v["global_orient"]) for _, v in trimmed_items]
    plot_raw_pose_changes(raw_cam_ts, raw_rot_mats, rootpath)

    # Filter frames
    frame_ids, cam_ts, rot_mats = filter_frames(trimmed_items, skip_rate)

    # Apply weighted moving average smoothing to translation
    smoothed_cam_ts = weighted_moving_average(cam_ts, window_size=3)
    smoothed_rot_mats = rot_mats

    # Save results
    smoothed_data, hand_poss = {}, {}
    for i, fid in enumerate(frame_ids):
        if fid in data:
            smoothed_data[fid] = data[fid]
            smoothed_data[fid]["pred_cam_t"] = smoothed_cam_ts[i].tolist()
            smoothed_data[fid]["global_orient"] = [smoothed_rot_mats[i].tolist()]
            rot = R.from_matrix(smoothed_rot_mats[i]).as_quat()
            hand_poss[fid] = np.concatenate([smoothed_cam_ts[i], rot])

    output_path = os.path.splitext(json_path)[0] + "_smoothed.json"
    with open(output_path, "w") as f:
        json.dump(smoothed_data, f, indent=2)
    return hand_poss


def smooth_hand_pose_json(json_path, skip_rate=1):
    """
    Smooths the hand pose JSON using filtering and weighted moving average smoothing.
    Steps:
      1. Load and filter by majority handedness.
      2. Trim leading low-score frames.
      3. Plot raw pose changes.
      4. Filter frames (remove jumps, low-confidence).
      5. Smooth translation with weighted moving average.
      6. Save smoothed results.
    """
    # 1. Load and filter by handedness
    data, rootpath = load_and_filter_handedness(json_path)
    # 2. Trim low-score frames
    trimmed_items = trim_low_score_frames(data)
    # 3. Plot raw pose changes
    raw_cam_ts = [np.array(v["pred_cam_t"]) for _, v in trimmed_items]
    raw_rot_mats = [np.array(v["global_orient"]) for _, v in trimmed_items]
    plot_raw_pose_changes(raw_cam_ts, raw_rot_mats, rootpath)
    # 4. Filter frames (remove jumps, low-confidence)
    frame_ids, cam_ts, rot_mats = filter_frames(trimmed_items, skip_rate)
    # 5. Smooth translation with weighted moving average (small window)
    smoothed_cam_ts = weighted_moving_average(cam_ts, window_size=3)
    smoothed_rot_mats = rot_mats
    # 6. Save results
    smoothed_data = {}
    hand_poss = {}
    for i, fid in enumerate(frame_ids):
        if fid in data:
            smoothed_data[fid] = data[fid]
            smoothed_data[fid]["pred_cam_t"] = smoothed_cam_ts[i].tolist()
            smoothed_data[fid]["global_orient"] = [smoothed_rot_mats[i].tolist()]
            rot = R.from_matrix(smoothed_rot_mats[i]).as_quat()
            hand_poss[fid] = np.concatenate([smoothed_cam_ts[i], rot])
    output_path = os.path.splitext(json_path)[0] + "_smoothed.json"
    with open(output_path, "w") as f:
        json.dump(smoothed_data, f, indent=2)
    return hand_poss


# === Example use ===
if __name__ == "__main__":
    smooth_hand_pose_json_KF("/home/xhe71/Desktop/robotool_data/hand_rotation_test/output/episode_0/cam3/hamer_out/hand_pose_camera_info.json")
    from human_segmentor.sphere_pcd import generate_pcd_sequence
    from dataset_utils.real_to_robomimic_converter import load_camera_info_dict
    cam_info = load_camera_info_dict("configs/camera_info.yaml")
    generate_pcd_sequence(
        "/home/xhe71/Desktop/robotool_data/hand_rotation_test/",
        "/home/xhe71/Desktop/robotool_data/hand_rotation_test/output", cam_info
    )
