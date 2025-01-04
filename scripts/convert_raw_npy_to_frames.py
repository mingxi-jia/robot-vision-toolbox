import os
import numpy as np
from datetime import datetime
from PIL import Image
import matplotlib.pyplot as plt

from segmentors.configs import DEPTH_MINMAXS
from segmentors.utils import normalize_depth

def main(data_root: str, npy_file_name: str):
    camera_name = 'dave'

    now = datetime.now()
    formatted_time = now.strftime("%Y-%m-%d %H:%M:%S")
    save_folder_path = os.path.join(data_root, npy_file_name.split('.')[0])
    rgb_path = os.path.join(save_folder_path, 'rgbs')
    depth_path = os.path.join(save_folder_path, 'depths')
    os.makedirs(rgb_path, exist_ok=True)
    os.makedirs(depth_path, exist_ok=True)

    raw_data: list = np.load(os.path.join(data_root, npy_file_name), allow_pickle=True)
    for i, step in enumerate(raw_data):
        rgb, depth = step[f'rgb_{camera_name}'], normalize_depth(step[f'depth_{camera_name}'], DEPTH_MINMAXS[camera_name])
        depth = np.asarray(depth* 255, dtype=np.uint8)

        rgb = Image.fromarray(rgb)
        rgb.save(os.path.join(rgb_path, f"{i}.jpg"), "JPEG")
        depth = Image.fromarray(depth)
        depth.save(os.path.join(depth_path, f"{i}.jpg"), "JPEG")

    print(1)

if __name__ == "__main__":
    data_root = "/home/mingxi/mingxi_ws/videopo/video_segmentor/raw_data"
    npy_file_name = "episode_0000.npy"
    main(data_root, npy_file_name)