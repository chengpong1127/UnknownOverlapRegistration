from pcd_dataset import create_7_Scenes_dataset, rgbd_to_ply
from uor import UOR
import numpy as np
from scipy.spatial.transform import Rotation as R

dataset = create_7_Scenes_dataset('datasets/7-Scenes')

# generate test dataset by randomly transform each point cloud in the ground truth dataset
def get_random_transform():
    translation = np.random.uniform(-0.1, 0.1, size=3)
    
    rotation = R.random().as_matrix()
    
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    
    return transform

def RRE(estimated_transformation, ground_truth_transformation):
    rotation_error = np.trace(
        estimated_transformation[:3, :3].T @ ground_truth_transformation[:3, :3]
    ) - 1
    # Clipping to avoid numerical errors
    rotation_error = np.clip(rotation_error / 2, -1.0, 1.0)
    return np.arccos(rotation_error)
    
def RTE(estimated_transformation, ground_truth_transformation):
    return np.linalg.norm(estimated_transformation[:3, 3] - ground_truth_transformation[:3, 3])

for test_sample in dataset:
    
    transformed_scenes = [scene.transform(get_random_transform()) for scene in test_sample['scenes']]
    poses = test_sample['poses']
    
    registered = UOR(transformed_scenes, 0.001, 0)
    
    # find the center of the point cloud
    center_index = [index for index, transformation in registered.items() if (transformation == np.eye(4)).all()][0]
    estimated_transformations = [registered[i] if i in registered else np.eye(4) for i in range(len(registered))]
    
    ground_truth_transformations = [np.linalg.inv(poses[center_index]) @ poses[i] for i in range(len(poses))]

    RREs = [RRE(estimated_transformation, ground_truth_transformation) for estimated_transformation, ground_truth_transformation in zip(estimated_transformations, ground_truth_transformations)]
    RTEs = [RTE(estimated_transformation, ground_truth_transformation) for estimated_transformation, ground_truth_transformation in zip(estimated_transformations, ground_truth_transformations)]
    
    print(f'RRE: {np.mean(RREs)}, RTE: {np.mean(RTEs)}')
