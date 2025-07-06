
import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle
import numpy as np
from robotArmSegmentation import RobotArmSegmentation

dataset_path = 'example_data/robomimic/square_d1_rel.hdf5'
dataset = h5py.File(dataset_path, 'r')

demo = dataset['data/demo_0']
rgbs = demo['obs']['spaceview_image'][:]
depths = demo['obs']['spaceview_depth'][:]
robot_joints = demo['obs']['robot0_joint_pos'][:]
gripper_joints = demo['obs']['robot0_gripper_qpos'][:]


num_traj = len(rgbs)
print(f"Number of trajectories in the dataset: {num_traj}")

# initialize the RobotArmSegmentation class
robot_seg = RobotArmSegmentation("example_data/robomimic/camera_meta.json")
# Load the robot URDF file
robot_seg.load_urdf("panda_description/urdf/panda_arm_hand.urdf")
# Set the camera
robot_seg.set_camera("spaceview")
# Process each frame in the dataset

for i in range(num_traj): #just one for now
    rgb = rgbs[i]
    depth = depths[i]
    joints = np.array([*robot_joints[i], gripper_joints[i][0]])  

    # For example, you could save them to disk or perform some analysis
    print(f"Frame {i}: RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Joints: {joints}")

    # Perform segmentation using the robot arm segmentation class
    filtered_pcd, scene_pcd = robot_seg.segment(rgb, depth, joints)





dataset.close()