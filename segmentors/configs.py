import os
import numpy as np
import yaml

def load_yaml(yaml_path: str):
    with open(yaml_path) as file:
        data = yaml.safe_load(file)
    return data

# ----------- DAVE CONFIGS --------
_DAVE_DEPTH_MINMAX = [0., 1.5]
# first_frame_ point prompts. FORMAT: [v, u]
_DAVE_POINT_PROMPTS = np.array([[255, 240], 
                            [300, 240], 
                            [290, 210], 
                            [237, 215], 
                            [267, 166],
                            [267, 116]], dtype=np.float32)
# for labels, `1` means positive click and `0` means negative click
_DAVE_PROMPT_LABELS = np.array([1, 1, 1, 1, 0, 0], np.int32)

# ----------- PUBLIC CONFIGS --------
config_file_path = os.path.join(os.path.dirname(__file__), 'config_files')
INTRINSICS = load_yaml(os.path.join(config_file_path, 'intrinsics.yaml'))
EXTRINSICS =  load_yaml(os.path.join(config_file_path, 'extrinsics.yaml'))
DEPTH_MINMAXS = {'dave': _DAVE_DEPTH_MINMAX}
GRIPPER_ID = 1  
POINT_PROMPTS = {'dave': _DAVE_POINT_PROMPTS}
POINT_PROMPT_LABELS = {'dave': _DAVE_PROMPT_LABELS}