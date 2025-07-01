import zipfile
import subprocess
import os
from human_segmentor.sphere_pcd import generate_pcd_sequence
from pipeline import main as pipeline_main
def unzip_folder(zip_path, extract_to='.'):
    """Unzips a folder and returns the root directory name."""
    if not zipfile.is_zipfile(zip_path):
        raise ValueError(f"{zip_path} is not a valid zip file")

    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(extract_to)
        print(f"Extracted to {extract_to}")

        # Get top-level folder name
        top_level = zip_ref.namelist()[0].split(os.sep)[0]
        unzipped_root = os.path.join(extract_to, top_level)
        return unzipped_root

def run_pipeline(root_path, cam_num):
    """Runs pipeline.py for a specific camera number."""
    video_path = os.path.join(root_path, f"cam{cam_num}", "rgb")
    depth_path = os.path.join(root_path, f"cam{cam_num}", "depth")

    cmd = [
        "python3", "pipeline.py",
        "--video_path", video_path,
        "--cam_num", str(cam_num),
        "--depth_folder", depth_path
    ]
    print(f"Running command: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

# === Main Execution ===
zip_path = "/path/to/your/file.zip"
root_path = "YOUR_ROOTPATH"

# # Step 1: Unzip and get root directory
# root_path = unzip_folder(zip_path, extract_to)

# Step 2: Run the pipeline on cam1, cam2, cam3
for cam in [1, 2, 3]:
    print(f"Processing camera {cam}...")
    run_pipeline(root_path, cam)
# Step 3: Generate point cloud sequence for cam3
generate_pcd_sequence(root_path, start_frame=0, sphere_cam=3)