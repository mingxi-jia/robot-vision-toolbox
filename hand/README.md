# Refactored Dataset Converter

A minimalist refactor of the real-to-robomimic converter, breaking down the monolithic script into focused, modular components.

## Structure

```
refactor/
├── __init__.py              # Package exports
├── main.py                  # Entry point with CLI
├── converter.py             # Main converter orchestrator
├── preprocessing.py         # Episode preprocessing with HAMER
├── trajectory_loader.py     # Trajectory loading and assembly
└── utils.py                 # Shared utilities (camera, state conversion)
```

## Usage

```bash
python hand/main.py \
    --real_dataset_path /path/to/dataset \
    --output_robomimic_path /path/to/output.hdf5
```

1. Visualizing the pcds

```bash
python scripts/play_pcd_sequence.py /home/mingxi/data/realworld/hand_rotation_test_strict/output/episode_0/pcd --fps=20 --loop
# export DISPLAY=:1 if  GLFW Error: X11: Failed to open display :0
```


## Components

### `utils.py`
- `load_camera_info_dict()`: Load camera calibration from YAML
- `convert_state_to_action()`: Convert poses to actions

### `point_cloud_processor.py`
- `PointCloudProcessor`: Filters, downsamples, and processes point clouds

### `preprocessing.py`
- `Preprocessor`: Runs HAMER hand tracking and PCD generation

### `trajectory_loader.py`
- `TrajectoryLoader`: Loads RGB/depth/PCD data and assembles trajectories

### `converter.py`
- `RealToRobomimicConverter`: Main orchestrator that coordinates all components

### `main.py`
- Command-line interface entry point

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Separation of Concerns**: Data loading, processing, and conversion are isolated
3. **Composability**: Components can be used independently
4. **Clarity**: Small, focused functions instead of large methods
5. **Maintainability**: Easy to test and modify individual components
