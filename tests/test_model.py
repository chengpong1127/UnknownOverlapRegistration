from models.uor import extract_features, compute_dij, extract_distance_array
import torch
from torchsparse import SparseTensor
import open3d as o3d
import numpy as np


def test_extract_features():
    point_clouds = [o3d.geometry.PointCloud() for _ in range(10)]
    for pcd in point_clouds:
        pcd.points = o3d.utility.Vector3dVector(np.random.rand(100, 3) * 10)
        pcd.colors = o3d.utility.Vector3dVector(np.random.rand(100, 3))
    features = extract_features(point_clouds, 32)
    assert len(features) == 10
    assert all(f.F.shape == (100, 32) for f in features)

def test_compute_dij():
    distance_array = torch.tensor([0.1, 0.2, 0.4, 0.7, 1.0])
    dij, indices = compute_dij(distance_array, 0.2)
    assert isinstance(dij, float)
    assert isinstance(indices, torch.Tensor)
    

def test_extract_distance_array():
    coords1 = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2]], dtype=torch.float32)
    feats1 = torch.tensor([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.1, 2.1, 2.1]], dtype=torch.float32)
    feature1 = SparseTensor(feats=feats1, coords=coords1)
    
    coords2 = torch.tensor([[0, 0, 0], [1, 1, 1], [3, 3, 3]], dtype=torch.float32)
    feats2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=torch.float32)
    feature2 = SparseTensor(feats=feats2, coords=coords2)
    
    distance_array, p_index_array, q_index_array = extract_distance_array(feature1, feature2)
    
    expected_distances = torch.tensor([0.1732, 0.0, 1.5588], dtype=torch.float32)
    expected_q_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    assert torch.allclose(distance_array, expected_distances, atol=1e-2)
    assert torch.equal(q_index_array, expected_q_indices)
    assert torch.equal(p_index_array, torch.arange(len(distance_array)))