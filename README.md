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
```

# 2. hamer hand detector
```
# install the cuda version torch (tested on cuda 12.2)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# install hamer
git clone --recursive https://github.com/geopavlakos/hamer.git ./submodules/hamer
cd ./submodules/hamer
pip install -e .[all]
pip install -v -e third-party/ViTPose
bash fetch_demo_data.sh

# change hamer camera parameter (VERY IMPORTANT)
# in ./submodules/hamer/_DATA/hamer_ckpts/model_config.yaml
FOCAL_LENGTH: 5000 -> FOCAL_LENGTH: 389 # for example

# preprocess video
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
# clean builder cache just in cache
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
