from uor.uor import compute_dij, extract_distance_array, compute_graph, compute_central_node, UOR
from uor.extract_features import extract_features
import torch
from torchsparse import SparseTensor
import open3d as o3d
import numpy as np
import networkx as nx
from .utils import create_test_point_clouds
import pytest


@pytest.mark.parametrize("feature_dim", [16, 32])
def test_extract_features(feature_dim):
    point_clouds = create_test_point_clouds(10, 100)
    features = extract_features(point_clouds, feature_dim)
    assert len(features) == 10
    assert all(f.shape[1] == feature_dim for f in features)

def test_compute_dij():
    distance_array = torch.tensor([0.1, 0.2, 0.4, 0.7, 1.0])
    dij, indices = compute_dij(distance_array, 0.2)
    assert isinstance(dij, float)
    assert isinstance(indices, torch.Tensor)
    

def test_extract_distance_array():
    feats1 = torch.tensor([[0.1, 0.1, 0.1], [1.0, 1.0, 1.0], [2.1, 2.1, 2.1]], dtype=torch.float32)
    feats2 = torch.tensor([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0], [3.0, 3.0, 3.0]], dtype=torch.float32)
    
    distance_array, p_index_array, q_index_array = extract_distance_array(feats1, feats2)
    
    expected_distances = torch.tensor([0.1732, 0.0, 1.5588], dtype=torch.float32)
    expected_q_indices = torch.tensor([0, 1, 2], dtype=torch.long)

    assert torch.allclose(distance_array, expected_distances, atol=1e-2)
    assert torch.equal(q_index_array, expected_q_indices)
    assert torch.equal(p_index_array, torch.arange(len(distance_array)))

def test_compute_graph():
    PQC_pairs = [
        (0, 1, [0, 1], [2, 3], 0.9),
        (1, 2, [1, 2], [3, 4], 0.8),
        (2, 3, [2, 3], [4, 5], 0.7),
        (3, 4, [0, 2], [3, 5], 0.4),
    ]
    point_cloud_len = 5
    threshold = 0.5

    g = compute_graph(PQC_pairs, point_cloud_len, threshold)

    assert nx.is_connected(g)

    assert len(g.nodes) == point_cloud_len - 1

    assert len(g.edges) == 3

    for (u, v) in g.edges:
        cij = next((c for (i, j, _, _, c) in PQC_pairs if (i == u and j == v) or (i == v and j == u)), None)
        assert cij is not None
        assert cij > threshold

def test_compute_central_node():
    star_graph = nx.star_graph(4)
    assert compute_central_node(star_graph) == 0

    path_graph = nx.path_graph(5)
    assert compute_central_node(path_graph) == 2

    complete_graph = nx.complete_graph(4)
    central_node = compute_central_node(complete_graph)
    assert central_node in complete_graph.nodes

    tree_graph = nx.balanced_tree(2, 2)
    assert compute_central_node(tree_graph) == 0


def test_UOR():
    point_clouds = create_test_point_clouds(5, 100)
    global_transforms = UOR(point_clouds, 0.5, 0.5, fine_registration_max_iter=10)
    assert len(global_transforms) <= 5
    assert all(t.shape == (4, 4) for t in global_transforms.values())