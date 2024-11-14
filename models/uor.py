from fcgf import ResUNet
from torchsparse import SparseTensor
import numpy
import open3d as o3d
import torch
from torch import Tensor


def extract_features(point_clouds: list, feature_dim = 32) -> list:
    fcgf_model = ResUNet(in_channels=3, out_channels=feature_dim, bn_momentum=0.1, normalize_feature=True, conv1_kernel_size=3)
    point_sparse_tensor = [SparseTensor(
        feats=torch.from_numpy(numpy.asarray(pcd.colors)).to(torch.float32), 
        coords=torch.concatenate([torch.from_numpy(numpy.asarray(pcd.points)), torch.zeros((numpy.asarray(pcd.points).shape[0], 1))], dim=1).to(torch.int32))
        for pcd in point_clouds]
    features = [fcgf_model(pst) for pst in point_sparse_tensor]
    return features

def extract_distance_array(feature1: SparseTensor, feature2: SparseTensor) -> Tensor:
    distance_array = torch.zeros(feature1.F.shape[0])
    p_index_array = torch.arange(feature1.F.shape[0])
    q_index_array = torch.zeros(feature1.F.shape[0], dtype=torch.long)
    for i in range(feature1.F.shape[0]):
        distances = torch.norm(feature2.F - feature1.F[i], dim=1)
        distance, index = distances.min(0)
        distance_array[i] = distance
        q_index_array[i] = index
    return distance_array, p_index_array, q_index_array

def compute_dij(distance_array: Tensor, threshold: float) -> tuple:
    sorted_distance_array, sorted_indices = torch.sort(distance_array)
    delta = sorted_distance_array[1:] - sorted_distance_array[:-1]
    mask = delta < threshold
    filtered_distances = sorted_distance_array[:-1][mask]
    filtered_indices = sorted_indices[:-1][mask]  
    if len(filtered_distances) == 0:
        return 1e6, torch.tensor([], dtype=torch.long)
    dij = filtered_distances.mean().item()
    return dij, filtered_indices
    
    


def UOR(point_clouds: list) -> list:
    features = extract_features(point_clouds)
    
    PQD_pairs = []
    max_dij = 0
    min_dij = 1e6
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            distance_array, p_indices, q_indices = extract_distance_array(features[i], features[j])
            dij, indices = compute_dij(distance_array, 0.5)
            p_indices = p_indices[indices]
            q_indices = q_indices[indices]
            PQD_pairs.append((i, j, p_indices, q_indices, dij))
            max_dij = max(max_dij, 1 / dij)
            min_dij = min(min_dij, 1 / dij)
    
    PQC_pairs = []
    for (i, j, p_indices, q_indices, dij) in PQD_pairs:
        if dij != 0:
            cij = (1 / dij - min_dij) / (max_dij - min_dij)
        else:
            cij = 1.0
        PQC_pairs.append((i, j, p_indices, q_indices, cij))

    
    
