
import multiprocessing
import os
import shutil
import click
import pathlib
import h5py
from tqdm import tqdm
import collections
import pickle

dataset_path = 'example_data/robomimic/square_d1_rel.hdf5'
dataset = h5py.File(dataset_path, 'r')
num_traj = len(dataset['data'])

demo = dataset['data/demo_0']
rgbs = demo['obs']['spaceview_image'][:]
depths = demo['obs']['spaceview_depth'][:]
robot_joints = demo['obs']['robot0_joint_pos'][:]

for i in range(len(rgbs)):
    rgb = rgbs[i]
    depth = depths[i]
    joints = robot_joints[i]

    # Here you can process the rgb, depth, and joints as needed
    # For example, you could save them to disk or perform some analysis
    print(f"Frame {i}: RGB shape: {rgb.shape}, Depth shape: {depth.shape}, Joints: {joints}")

dataset.close()