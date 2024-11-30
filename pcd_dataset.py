import open3d as o3d
import numpy as np
import PIL.Image as Image
from datasets import Dataset
from pathlib import Path

_7_SCENE_SETTING = {
    'fx': 585,
    'fy': 585,
    'cx': 320,
    'cy': 240,
}

def rgbd_to_ply(rgb_path, depth_path, pose_path = '', ply_path = '', intrinsic_setting = _7_SCENE_SETTING):
    """
    Reads an RGB image and a depth image, and outputs a point cloud in PLY format.
    
    Args:
    - rgb_path (str): Path to the RGB image file.
    - depth_path (str): Path to the depth image file.
    - ply_path (str): Path to save the output PLY file.
    """
    # Load the RGB and depth images
    color_image = o3d.io.read_image(rgb_path)
    depth_image = o3d.io.read_image(depth_path)
    # Get the dimensions of the RGB image
    width = Image.open(rgb_path).size[0]
    height = Image.open(rgb_path).size[1]

    # Create an RGBD image
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_image,
        depth_image,
        convert_rgb_to_intensity=False
    )

    # Set the camera intrinsic parameters
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=int(width),
        height=int(height),
        fx=intrinsic_setting['fx'],
        fy=intrinsic_setting['fy'],
        cx=intrinsic_setting['cx'],
        cy=intrinsic_setting['cy'],
    )
    
    if pose_path != '':
        # Load the camera pose
        pose = np.loadtxt(pose_path)
        pose = np.linalg.inv(pose)
        extrinsic = pose
    else:
        extrinsic = np.eye(4)

    # Generate the point cloud from the RGBD image
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(rgbd_image, intrinsic, extrinsic)

    if ply_path != '':
        o3d.io.write_point_cloud(ply_path, pcd)
    return pcd


def create_7_Scenes_dataset(dir: str):
    result = {
        'scenes': [],
    }
    root_dir = Path(dir)
    for scene_dir in root_dir.iterdir():
        if not scene_dir.is_dir():
            continue

        for sequence in scene_dir.iterdir():
            if not sequence.is_dir():
                continue
            
            image_count = len(list(sequence.glob('*.color.png')))
            # Random choose 10 images
            indices = np.random.choice(image_count, 10, replace=False)
            indices = [str(i).zfill(6) for i in indices]
            scene = [ 
                (
                    str(sequence / f'frame-{index}.color.png'),
                    str(sequence / f'frame-{index}.depth.png'),
                    str(sequence / f'frame-{index}.pose.txt'),
                ) for index in indices
            ]
            result['scenes'].append(scene)
    dataset = Dataset.from_dict(result)
    
    def transform(batch_scene):
        scenes = [ 
            [ 
                rgbd_to_ply(rgb_path, depth_path, pose_path) 
                for rgb_path, depth_path, pose_path in scenes
            ] 
            for scenes in batch_scene['scenes']
        ]
        poses = [
            [
                np.loadtxt(pose_path)
                for _, _, pose_path in scenes
            ]
            for scenes in batch_scene['scenes']
        ]
        return {
            'scenes': scenes,
            'poses': poses,
        }

    dataset.set_transform(transform)
    return dataset
            