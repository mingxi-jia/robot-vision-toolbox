import numpy as np
import sys
from scipy.spatial.transform import Rotation as R
import yaml
import os
def build_transform_matrix(pose):
    # Extract translation and quaternion
    trans = np.array(pose[:3])
    quat = np.array(pose[3:])  # [x, y, z, w]
    rot = R.from_quat(quat).as_matrix()

    T = np.eye(4)
    T[:3, :3] = rot
    T[:3, 3] = trans
    return T

def load_npy_dict(path, origin_frame=0):
    data = np.load(path, allow_pickle=True).item()  # important: use .item() for dict
    frame_key = f"{origin_frame:06d}"
    hand_matrix = data[frame_key]
    print(f'frame {frame_key} has hand transformation matrix of {hand_matrix}')
    return hand_matrix

def get_alignment_matrix(file_path, origin_frame_number):
    # Select the hand pose where thumb/index face cam3, palm faces down
    hamer_pose = load_npy_dict(file_path, origin_frame=origin_frame_number)
    hamer_matrix = build_transform_matrix(hamer_pose)

    # Define rotation from HAMER to table-aligned frame
    # HAMER (X: right, Y: up, Z: back) → TABLE (X: forward, Y: left, Z: down)
    R_desired = np.array([
        [1,  0,  0],  # +X (forward) ← HAMER Z
        [0,  -1,  0],  # +Y (left)    ← -HAMER X
        [ 0, 0,  -1]   # +Z (down)    ← -HAMER Y
    ])

    T_desired = np.eye(4)
    T_desired[:3, 3] = hamer_pose[:3].T
    T_desired[:3, :3] = R_desired
    T_desired[:3, 3] = R_desired @ hamer_pose[:3]  # ✅ Rotate translation

    # Compute the transform to align HAMER pose to table-aligned frame
    T_transform = np.linalg.inv(hamer_matrix) @ T_desired

    # Extract rotation matrix and quaternion
    R_transform = T_transform[:3, :3]
    quat_transform = R.from_matrix(R_transform).as_quat().tolist()

    print("✅ Transformation Matrix to convert from HAMER to Table-Aligned Frame:")
    print(R_transform)
    print("\n✅ Full transformed pose (T_aligned):")
    print(hamer_matrix @ T_transform)

    # Save to YAML
    save_dict = {
        'R_transform': R_transform.tolist(),
        'quat_transform': quat_transform,
        'description': 'Rotation from HAMER hand frame to table-aligned frame (X=forward, Y=left, Z=down)'
    }

    os.makedirs('./configs', exist_ok=True)
    with open('./configs/hamer_alignment_matrix.yaml', "w") as f:
        yaml.dump(save_dict, f)

    print('✅ Rotation matrix and quaternion saved to configs/hamer_alignment_matrix.yaml')
if __name__ == "__main__":
    path = '/home/xhe71/Desktop/robotool_data/hand_rotation_test/output/episode_0/hand_poses_wrt_world.npy'
    frame_num = 72
    get_alignment_matrix(path, frame_num)