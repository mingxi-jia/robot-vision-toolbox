import json
import numpy as np
import matplotlib.pyplot as plt
from pykalman import KalmanFilter
from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp
import os

# === Global smoothing settings ===
# for sample rate less that 3
SMOOTH_PARAMS = {
    "trans_noise": 5e-6,
    "obs_noise": 1e-4,
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
            rot_mat = np.array(v["global_orient"][0])
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


def smooth_hand_pose_json_KF(json_path, skip_rate = 1):
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

    # Debug visualization on raw data before filtering
    raw_cam_ts = []
    raw_rot_mats = []
    raw_frame_ids = []
    frame_ids_numeric = []
    for fid, v in trimmed_items:
        try:
            cam_t = np.array(v["pred_cam_t"])
            rot_mat = np.array(v["global_orient"][0])
            if rot_mat.shape != (3, 3):
                raise ValueError(f"Invalid rotation matrix shape for frame {fid}: {rot_mat.shape}")
            raw_cam_ts.append(cam_t)
            raw_rot_mats.append(rot_mat)
            raw_frame_ids.append(fid)
            frame_ids_numeric.append(int(''.join(filter(str.isdigit, fid))))
        except Exception as e:
            print(f"[WARN] Skipping {fid} in raw debug due to error: {e}")

    trans_deltas = []
    angle_deltas = []
    for i in range(1, len(raw_cam_ts)):
        delta_t = np.linalg.norm(raw_cam_ts[i] - raw_cam_ts[i - 1])
        trans_deltas.append(delta_t)

        delta_rot = R.from_matrix(raw_rot_mats[i - 1]).inv() * R.from_matrix(raw_rot_mats[i])
        angle_rad = np.linalg.norm(delta_rot.as_rotvec())
        angle_deg = np.degrees(angle_rad)
        angle_deltas.append(angle_deg)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(range(1, len(raw_cam_ts)), trans_deltas, label="Δ Translation (m)", color="blue")
    axs[0].plot(range(1, len(raw_cam_ts)), np.ones(len(raw_cam_ts)-1)*0.2, '--',label="Δ Translation (m)=0.2", color="red")
    
    axs[0].set_ylabel("Translation Δ (m)")
    axs[0].set_yscale('log')
    axs[0].legend()

    axs[1].plot(range(1, len(raw_cam_ts)), angle_deltas, label="Δ Orientation (deg)", color="green")
    axs[1].set_ylabel("Rotation Δ (deg)")
    axs[1].set_xlabel("Frame Index")
    axs[1].plot(range(1, len(raw_cam_ts)), np.ones(len(raw_cam_ts)-1)*30, '--',label="Δ Orientation (deg)=35", color="red")
    axs[1].legend()

    plt.suptitle("Raw Frame-to-Frame Pose Change")
    plt.tight_layout()
    plt.savefig("debug_raw_pose_jump_plot.png", dpi=300)
    plt.close()

    print(f"[INFO] Avg Δ Translation: {np.mean(trans_deltas):.4f} m")
    print(f"[INFO] Avg Δ Orientation: {np.mean(angle_deltas):.2f} deg")

    # frame_ids = [k for k, _ in trimmed_items]
    # cam_ts = np.array([v["pred_cam_t"] for _, v in trimmed_items])
    print(f"[DEBUG] Total frames before filtering: {len(trimmed_items)}")
    filtered_frame_ids = []
    filtered_cam_ts = []
    filtered_rot_mats = []


    update_filtering_to_per_axis(filtered_frame_ids, filtered_cam_ts, filtered_rot_mats, trimmed_items,
                                score_threshold=0.85, max_axis_jump=0.3 * skip_rate, max_angle_jump=60)
    interp_frame_ids, interp_cam_ts, interp_rot_mats = interpolate_dropped_frames(
        filtered_frame_ids, filtered_cam_ts, filtered_rot_mats, [k for k, _ in trimmed_items]
    )

    last_valid_t = None
    last_valid_r = None
    score_threshold = 0.8
    max_trans_jump = 0.5 * skip_rate   # meters
    max_angle_jump = 60    # degrees

    for i, (fid, v) in enumerate(trimmed_items):
        score = v.get("score", 0.0)
        cam_t = np.array(v["pred_cam_t"])
        try:
            rot_mat = np.array(v["global_orient"][0])
            if rot_mat.shape != (3, 3):
                raise ValueError(f"Invalid rotation matrix shape for frame {fid}: {rot_mat.shape}")
        except Exception as e:
            print(f"[WARN] Skipping {fid} due to missing or invalid rotation: {e}")
            continue

        keep = True
        if score < score_threshold:
            keep = False
            print(f"[DEBUG] Skipping {fid}: low score ({score:.2f})")
        if last_valid_t is not None:
            delta_t = np.linalg.norm(cam_t - last_valid_t)
            if delta_t > max_trans_jump:
                print(f"[DEBUG] Skipping {fid}: large translation jump ({delta_t:.3f} m)")
                keep = False
        if last_valid_r is not None and len(rot_mat) > 0:
            # check rot_mat shape before delta_rot
            if rot_mat.shape != (3, 3):
                raise ValueError(f"Invalid rotation matrix shape for frame {fid}: {rot_mat.shape}")
            delta_rot = R.from_matrix(last_valid_r).inv() * R.from_matrix(rot_mat)
            angle_rad = np.linalg.norm(delta_rot.as_rotvec())
            angle = np.degrees(angle_rad)
            if angle > max_angle_jump:
                keep = False
                print(f"[DEBUG] Skipping {fid}: large orientation jump ({angle:.1f} deg)")

        if keep:
            filtered_frame_ids.append(fid)
            filtered_cam_ts.append(cam_t)
            filtered_rot_mats.append(rot_mat)
            last_valid_t = cam_t
            last_valid_r = rot_mat
            
    # rot_mats = []
    # valid_frame_ids = []
    # scores = []
    # for fid, v in trimmed_items:
    #     try:
    #         mat = np.array(v["global_orient"][0])
    #         if mat.shape == (3, 3):
    #             rot_mats.append(mat)
    #             valid_frame_ids.append(fid)
    #             scores.append(v.get("score", 0.0))  # default score if not present
    #         else:
    #             print(f"Skipping frame {fid} due to invalid shape: {mat.shape}")
    #     except Exception as e:
    #         print(f"Skipping frame {fid} due to error: {e}")

    # rot_mats = np.array(rot_mats)
    # cam_ts = np.array([data[fid]["pred_cam_t"] for fid in valid_frame_ids])
    # frame_ids = valid_frame_ids
    

    frame_ids = interp_frame_ids
    cam_ts = interp_cam_ts
    rot_mats = interp_rot_mats

    # Build frame_ids_numeric for plotting, matching length of frame_ids
    frame_ids_numeric = [int(''.join(filter(str.isdigit, fid))) for fid in frame_ids]
    scores = np.array([data[fid].get("score", 0.0) if fid in data else 0.0 for fid in frame_ids])

    # Also smooth translation with KF
    smoothed_cam_ts = smooth_3d_sequence(cam_ts)

    # First apply KF to rotation vectors
    rot_vecs = R.from_matrix(rot_mats).as_rotvec()
    # rot_vecs = make_rotation_vectors_continuous(rot_vecs)
    # smoothed_rot_vecs = smooth_orientation_sequence(rot_vecs)

    # # Convert to quaternions and smooth
    # smoothed_quats = R.from_rotvec(smoothed_rot_vecs).as_quat()
    # smoothed_quats = make_quaternion_sequence_continuous(smoothed_quats)
    # smoothed_quats = smooth_quaternion_sequence(smoothed_quats, window_size=15)
    # smoothed_quats = clamp_quaternion_velocity(smoothed_quats, max_degrees=8)
    # smoothed_rot_mats = R.from_quat(smoothed_quats).as_matrix()

    smoothed_rot_mats = rot_mats  # Use original orientations
    # Debug: Plot translation and orientation delta after filtering and smoothing
    trans_deltas_post = []
    angle_deltas_post = []
    for i in range(1, len(smoothed_cam_ts)):
        delta_t = np.linalg.norm(smoothed_cam_ts[i] - smoothed_cam_ts[i - 1])
        trans_deltas_post.append(delta_t)

        delta_rot = R.from_matrix(smoothed_rot_mats[i - 1]).inv() * R.from_matrix(smoothed_rot_mats[i])
        angle_rad = np.linalg.norm(delta_rot.as_rotvec())
        angle_deg = np.degrees(angle_rad)
        angle_deltas_post.append(angle_deg)

    fig, axs = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    axs[0].plot(range(1, len(smoothed_cam_ts)), trans_deltas_post, label="Δ Translation (m)", color="blue")
    axs[0].set_ylabel("Translation Δ (m)")
    axs[0].set_yscale("log")
    axs[0].legend()

    axs[1].plot(range(1, len(smoothed_cam_ts)), angle_deltas_post, label="Δ Orientation (deg)", color="green")
    axs[1].set_ylabel("Rotation Δ (deg)")
    axs[1].set_xlabel("Frame Index")
    axs[1].legend()

    plt.suptitle("Smoothed Frame-to-Frame Pose Change")
    plt.tight_layout()
    plt.savefig("debug_smoothed_pose_jump_plot.png", dpi=300)
    plt.close()

    print(f"[INFO] Smoothed Avg Δ Translation: {np.mean(trans_deltas_post):.4f} m")
    print(f"[INFO] Smoothed Avg Δ Orientation: {np.mean(angle_deltas_post):.2f} deg")
    # 2D Visualization: Translation and SLERP-smoothed orientation
    # Prepare raw_cam_ts_np and raw_quats for plotting
    raw_cam_ts_np = np.array(raw_cam_ts)
    raw_quats = R.from_matrix(raw_rot_mats).as_quat()
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
        score = data[fid].get("score", 0.0) if fid in data else 0.0
        if score < score_threshold:
            # Get previous and next valid rotation matrices
            prev = None
            next_ = None
            for j in range(i - 1, -1, -1):
                fid_j = frame_ids[j]
                if fid_j in data and data[fid_j].get("score", 1.0) >= score_threshold:
                    prev = np.array(data[fid_j]["global_orient"][0])
                    break
            for j in range(i + 1, len(frame_ids)):
                fid_j = frame_ids[j]
                if fid_j in data and data[fid_j].get("score", 1.0) >= score_threshold:
                    next_ = np.array(data[fid_j]["global_orient"][0])
                    break
            if prev is not None and next_ is not None:
                avg = (prev + next_) / 2
                if fid in data:
                    data[fid]["global_orient"] = [avg.tolist()]
                replacement_flags.append(1)
            else:
                replacement_flags.append(0)
        else:
            replacement_flags.append(0)

    fig, axs = plt.subplots(9, 1, figsize=(12, 18), sharex=True)
    trans_labels = ["X", "Y", "Z"]
    # Ensure dimensions match for plotting
    min_len = min(
        len(frame_ids_numeric),
        raw_cam_ts_np.shape[0],
        smoothed_cam_ts.shape[0],
        raw_quats.shape[0],
        smoothed_quats.shape[0],
        scores.shape[0],
        angle_diff.shape[0],
    )
    frame_ids_numeric_plot = frame_ids_numeric[:min_len]
    raw_cam_ts_np_plot = raw_cam_ts_np[:min_len]
    smoothed_cam_ts_plot = smoothed_cam_ts[:min_len]
    raw_quats_plot = raw_quats[:min_len]
    smoothed_quats_plot = smoothed_quats[:min_len]
    scores_plot = scores[:min_len]
    angle_diff_plot = angle_diff[:min_len]
    replacement_flags_plot = replacement_flags[:min_len]
    for i in range(3):
        axs[i].plot(frame_ids_numeric_plot, raw_cam_ts_np_plot[:, i], label="Original", alpha=0.4, linewidth=2)
        axs[i].plot(frame_ids_numeric_plot, smoothed_cam_ts_plot[:, i], label="KF Smoothed", linewidth=1)
        axs[i].set_ylabel(f"Trans {trans_labels[i]}")
        axs[i].legend()

    quat_labels = ["X", "Y", "Z", "W"]
    for i in range(4):
        axs[i+3].plot(frame_ids_numeric_plot, raw_quats_plot[:, i], label="Original", alpha=0.4, linewidth=2)
        axs[i+3].plot(frame_ids_numeric_plot, smoothed_quats_plot[:, i], label="Smoothed", linewidth=1)
        axs[i+3].set_ylabel(f"Quat {quat_labels[i]}")
        axs[i+3].legend()

    # confidence
    axs[-2].plot(frame_ids_numeric_plot, scores_plot, color="black", label="Confidence Score")
    replaced_x = [i for i, flag in enumerate(replacement_flags_plot) if flag]
    replaced_y = [scores_plot[i] for i in replaced_x]
    axs[-2].scatter([frame_ids_numeric_plot[i] for i in replaced_x], replaced_y, color="red", label="Replaced")
    axs[-2].legend()
    
    axs[-1].plot(frame_ids_numeric_plot, angle_diff_plot, color="purple", label="Quat Δθ (deg)")
    axs[-1].set_ylabel("Δ Angle")
    axs[-1].legend()

    axs[-1].set_xlabel("Frame Index")
    fig.suptitle("Translation (KF) and Orientation Smoothing")
    plt.tight_layout()
    obs_noise = SMOOTH_PARAMS["obs_noise"]
    trans_noise = SMOOTH_PARAMS["trans_noise"]
    plt.savefig(f"combined_pose_smoothing_plot.png", dpi=300)
    plt.close()

    # Save updated JSON: only filtered and smoothed frames
    smoothed_data = {}
    for i, fid in enumerate(frame_ids):
        if fid in data:
            smoothed_data[fid] = data[fid]
            smoothed_data[fid]["global_orient"] = [smoothed_rot_mats[i].tolist()]
            smoothed_data[fid]["pred_cam_t"] = smoothed_cam_ts[i].tolist()
            
    output_path = os.path.splitext(json_path)[0] + "_smoothed.json"
    with open(output_path, "w") as f:
        json.dump(smoothed_data, f, indent=2)

# === Example use ===
if __name__ == "__main__":
    smooth_hand_pose_json_KF("/home/xhe71/Desktop/robotool_data/Color_hamer/hand_pose_camera_info.json")
