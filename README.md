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

# change hamer camera parameter (VERY IMPORTANT)
# in ./submodules/hamer/_DATA/hamer_ckpts/model_config.yaml
[Line 59] FOCAL_LENGTH: 5000 -> FOCAL_LENGTH: 389 # for example

# preprocess video
python hamer_detector/video_preprocessor.py

# hamer detection
[NOTE(not affecting running): iopath imcompactable, detectron requireing iopath<0.1.10, not compatable with SAM2's installation requirement of iopath>0.1.10]
python hamer_detector/demo.py --img_folder hamer_detector/example_data/realsense-test --out_folder hamer_detector/example_data/realsense-test-hamer --batch_size=48 --save_mesh"

# render sphere based on hamer results
python hamer_detector/sphere_renderer.py
```
# TODOs
- [x] sam segmentor
- [x] single-view video segmentation
- [x] point cloud projection
- [ ] realtime cloud visualization
- [ ] multi-view video sementation
- [ ] multi-view rgbd segmentation to segmented point clouds
