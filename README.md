 # 0. conda env
```
conda create --name robotool python=3.10
conda activate robotool
```

# 1. install real-time video segmentor
```
# install the cuda version torch (tested on cuda 12.4)
mamba install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.4 -c pytorch -c nvidia
pip install mediapipe==0.10.14 pykalman h5py open3d supervision

# install realtime sam
git clone https://github.com/Gy920/segment-anything-2-real-time.git ./submodules/segment-anything-2-real-time
cd ./submodules/segment-anything-2-real-time && pip install -e .
cd checkpoints
bash bash download_ckpts.sh
cd ../../..
#export LD_LIBRARY_PATH=/your_conda_env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
# install other necessary packages:
pip install git+https://github.com/cansik/mesh-sequence-player.git@1.10.1
```

# 2. install hamer hand detector
Hamer pipeline needs the video segmentor above.
```
# install hamer
git clone --recursive https://github.com/geopavlakos/hamer.git ./submodules/hamer
cd ./submodules/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
bash fetch_demo_data.sh
cd ../..

pip install numpy==1.26.4 opencv-python==4.11.0.86 opencv-contrib-python==4.11.0.86
```
* need to change one line in hamer
```
#change this
VIT_DIR = os.path.join(ROOT_DIR, "third-party/ViTPose")
# into
VIT_DIR = os.path.join(ROOT_DIR, "submodules/hamer/third-party/ViTPose")
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
```
### 2. hamer detection
```
python hamer_detector/demo.py \
  --img_folder path/to/your/images \
  --out_folder path/to/output_folder \
  --batch_size=48 
  --save_mesh 
```

### 3. Human Segmentation

run segmentation to remove human from the scene. 
```
python human_segmentor/human_pose_segmentor_mp_sam.py
```

### 4. Rendering with DiscoBall
```
python /human_segmentor/replace_hand_with_sphere.py
--hamer_out_dir PATH_TO_THE_HAMER_PROCESSED_DATA
--segmentation_out_dir PATH_TO_SEGMENTED_DATA
--sphere_out_dir OUTPUT_DIRECTORY
```

### 5. side note: convert a folder of images into a video
To turn image folder into videos, open terminal and run:
```
ls *.png | sort -V | awk '{print "file '\''" $0 "'\''"}' > list.txt


ffmpeg -f concat -safe 0 -r 30 -i list.txt -c:v libx264 -pix_fmt yuv420p output_video.mp4

```
# TODOs
- [x] sam segmentor
- [x] single-view video segmentation
- [x] point cloud projection
- [x] realtime cloud visualization
- [x] multi-view video sementation
- [x] multi-view rgbd segmentation to segmented point clouds
