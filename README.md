# 0. conda env
```
conda create --name robotool python=3.10
conda activate robotool
```

# 1. real-time video segmentor
```
# install the cuda version torch (tested on cuda 12.2)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# install realtime sam
git clone https://github.com/Gy920/segment-anything-2-real-time.git ./submodules/segment-anything-2-real-time
cd segment-anything-2-real-time && pip install -e .
export LD_LIBRARY_PATH=/your/env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH
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

# preprocess video
python hamer_detector/video_preprocessor.py

# hamer detection
[NOTE(not affecting running): iopath imcompactable, detectron requireing iopath<0.1.10, not compatable with SAM2's installation requirement of iopath>0.1.10]
python hamer_detector/demo.py --img_folder hamer_detector/example_data/realsense-test --out_folder hamer_detector/example_data/realsense-test-hamer --batch_size=48 --save_mesh

# render sphere based on hamer results
python hamer_detector/sphere_renderer.py
```

# 3. Human Segmentation
```
pip install mediapipe
```
run segmentation to remove human from the scene. 
```
python human_segmentor/human_pose_segmentor_mp_sam.py
```
To turn image folder into videos, open terminal and run:
```
ls *_final.png | sort -V | awk '{print "file '\''" $0 "'\''"}' > list.txt


ffmpeg -f concat -safe 0 -r 30 -i list.txt -c:v libx264 -pix_fmt yuv420p output_video.mp4
```

# 4. Rendering with DiscoBall
```
python hamer_detector/pipeline.py --video_path YOUR_VIDEO_PATH --handedness "right"
```

# 5. side note: convert a folder of images into a video
```
ls processed_results/*.png | sort -V | awk '{print "file '\''" $0 "'\''"}' > list.txt


ffmpeg -f concat -safe 0 -r 30 -i list.txt -c:v libx264 -pix_fmt yuv420p output_video.mp4

```
# TODOs
- [x] sam segmentor
- [x] single-view video segmentation
- [x] point cloud projection
- [ ] realtime cloud visualization
- [ ] multi-view video sementation
- [ ] multi-view rgbd segmentation to segmented point clouds
