# video segmentor

git clone https://github.com/Gy920/segment-anything-2-real-time.git
cd segment-anything-2-real-time && pip install -e .
export LD_LIBRARY_PATH=/your/env/lib/python3.10/site-packages/nvidia/nvjitlink/lib:$LD_LIBRARY_PATH

# TODOs
- [x] sam segmentor
- [x] single-view video segmentation
- [x] point cloud projection
- [ ] multi-view video sementation
- [ ] multi-view rgbd segmentatio to segmented point clouds
