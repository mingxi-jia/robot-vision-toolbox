#!/usr/bin/env python3
"""Final test with all fixes applied"""
import sys
sys.path.insert(0, '/home/mingxi/mingxi_ws/videopo/robot-vision-toolbox/submodules/hamer')

from hamer.utils.renderer import Renderer, create_raymond_lights
import numpy as np
from yacs.config import CfgNode

print("Testing complete HaMer Renderer with all fixes...")

# Create minimal config
cfg = CfgNode()
cfg.EXTRA = CfgNode()
cfg.EXTRA.FOCAL_LENGTH = 5000
cfg.MODEL = CfgNode()
cfg.MODEL.IMAGE_SIZE = 256
cfg.MODEL.IMAGE_STD = [0.229, 0.224, 0.225]
cfg.MODEL.IMAGE_MEAN = [0.485, 0.456, 0.406]

# Create renderer (like HaMer does)
faces = np.random.randint(0, 100, (100, 3))  # Dummy faces
renderer = Renderer(cfg, faces)

# Test render_rgba_multiple (the failing function)
print("Testing render_rgba_multiple...")
try:
    vertices = [np.random.randn(778, 3) * 0.1]  # Typical hand mesh vertex count
    cam_t = [np.array([0.0, 0.0, 2.5])]

    result = renderer.render_rgba_multiple(
        vertices=vertices,
        cam_t=cam_t,
        render_res=[256, 256],
        is_right=[1],
        mesh_base_color=(0.53, 0.81, 0.92),  # Light blue
        scene_bg_color=(1, 1, 1),
        focal_length=5000
    )

    print(f"✓ SUCCESS! Rendered shape: {result.shape}")
    print(f"✓ Output range: [{result.min():.3f}, {result.max():.3f}]")
    print(f"✓ Alpha channel present: {result.shape[2] == 4}")
    print("\n🎉 All tests passed! HaMer rendering is now working.")

except Exception as e:
    print(f"✗ FAILED: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
