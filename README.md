# 0. conda env
```
conda create --name robotool python=3.10
conda activate hamer

git clone git@github.com:SaulBatman/robot-vision-toolbox.git
cd robot-vision-toolbox
export PYTHONPATH=/YOURPATH/robot-vision-toolbox:$PYTHONPATH
```

# 1. real-time video segmentor
```
# install the cuda version torch (tested on cuda 12.2)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# install realtime sam
git clone https://github.com/Gy920/segment-anything-2-real-time.git ./submodules/segment-anything-2-real-time
cd segment-anything-2-real-time && pip install -e .
export LD_LIBRARY_PATH=/your/env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# install other necessary packages:
pip install mediapipe
pip install pykalman
pip install git+https://github.com/cansik/mesh-sequence-player.git@1.10.1
```

# 2. hamer hand detector
```
# install hamer
git clone --recursive https://github.com/geopavlakos/hamer.git ./submodules/hamer
cd ./submodules/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
bash fetch_demo_data.sh
```
Besides these files, you also need to download the MANO model. Please visit the MANO website and register to get access to the downloads section. We only require the right hand model. [MANO_RIGHT.pkl](https://mano.is.tue.mpg.de/) under the _DATA/data/mano folder.
```
# change hamer camera parameter (VERY IMPORTANT)
# in ./submodules/hamer/_DATA/hamer_ckpts/model_config.yaml
[Line 59] FOCAL_LENGTH: 5000 -> FOCAL_LENGTH: 389 # for example
```


# Pipeline: detect hand and replace with disco ball
```
python pipeline.py 
--video_path YOUR_VIDEO_PATH (or image folder path <-- preferred)
--depth_folder YOUR_DEPTH_PATH
-- cam_num 3 -> front;  2 -> left;  3-> right; 0 (otherwise, if 0 is used need to put in background_img adn intrinsics_path) 
--background_img: PATH_TO_ENVIRONMENT_IMAGE [If none, use the first frame of the video as the background]
--intrinsics_path CAMERA_INTRINSICS_JSON
--debug 

```
# visualize pcd output:

```
mesh-sequence-player  -p folder_to_meshes
```


```
side note: adjust the following parameters if needed
pipeline.py | [line 18] | SAMPLE_RATE = 3 by default 
human_pose_segmentor_mp_sam.py | [line 281] | top 2/5 is cropped
```

## Example: intrinsics json
{
    "width": 640,
    "height": 360,
    "fx": 261.2876281738281,
    "fy": 261.2876281738281,
    "cx": 316.0948791503906,
    "cy": 177.75343322753906,
    "distortion_model": "rational_polynomial",
    "distortion_coefficients": [
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0
    ]
}

## 📁 Output Folder Structure
After running the processing pipeline on a video (e.g., video.mp4), the following folder structure will be created in the same directory:

video.mp4
├── video_frames/       # Extracted frames from the video
├── video_hamer/        # 3D hand pose results (from HaMeR), includes meshes and pose JSON
├── video_segmented/    # Segmented hands or objects from each frame
├── video_final/        # Final visualizations with overlays (e.g., hand mesh, spheres, segmentation)


## substeps
### 1. preprocess video
```
python hamer_detector/video_preprocessor.py

# hamer detection
python hamer_detector/demo.py --img_folder hamer_detector/example_data/realsense-test --out_folder hamer_detector/example_data/realsense-test-hamer --batch_size=48 --save_mesh"

# render sphere based on hamer results
python hamer_detector/sphere_renderer.py
```

# 3. fix your PC
There are occasions when roboticists need to deal with computer problems. Here are some solutions for some cases.
1. docker is taking up a lot root space
```
# check your root
sudo ncdu -x /
docker info | grep 'Docker Root Dir'
# move your docker files to home
# reference: https://medium.com/@calvineotieno010/change-docker-default-root-data-directory-a1d9271056f4
sudo systemctl stop docker
sudo vim /etc/docker/daemon.json
# add this into /etc/docker/daemon.json
#{
#  "data-root": "/newpath"
#}
# Move the files
sudo rsync -axPS /var/lib/docker/ /home/mingxi/docker
# restart docker
sudo systemctl start docker
docker info | grep 'Docker Root Dir'
# delete the old folder (double check before you do this!)
sudo rm -r /var/lib/docker 
# clean builder cache just in case
docker builder prune
sudo docker rmi $(sudo docker images -f "dangling=true" -q)
```
2. clean your root
   ```
   sudo apt autoclean
   sudo apt autoremove
   sudo apt clean all
   # delete old snap archive
   snap list --all
   sudo snap remove firefox --revision=6159
   ```
# TODOs
- [x] sam segmentor
- [x] single-view video segmentation
- [x] point cloud projection
- [ ] realtime cloud visualization
- [ ] multi-view video sementation
- [ ] multi-view rgbd segmentation to segmented point clouds
