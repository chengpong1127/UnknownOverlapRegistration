from fcgf import ResUNet
from torchsparse import SparseTensor
import numpy
import open3d as o3d
import torch
from torch import Tensor
import networkx as nx
from .fine_registration import fine_registration

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

def compute_graph(PQC_pairs: list, point_cloud_len: int, threshold: float) -> list:
    graph = nx.Graph()
    graph.add_nodes_from(range(point_cloud_len))
    sorted_PQC_pairs = sorted(PQC_pairs, key=lambda x: x[4], reverse=True)
    for (i, j, _, _, cij) in sorted_PQC_pairs:
        if not nx.is_connected(graph):
            if cij > threshold and not graph.has_edge(i, j):
                graph.add_edge(i, j)
            elif cij <= threshold:
                largest_cc = max(nx.connected_components(graph), key=len)
                nodes_to_remove = set(graph.nodes) - largest_cc
                graph.remove_nodes_from(nodes_to_remove)
        else:
            break
    
    return graph

def compute_central_node(graph: nx.Graph) -> int:
    closeness_centrality = nx.closeness_centrality(graph)
    central_node = max(closeness_centrality, key=closeness_centrality.get)
    return central_node



def UOR(point_clouds: list, eplison1: float, eplison2: float, fine_registration_max_iter: int = 20) -> list:
    features = extract_features(point_clouds)
    
    PQD_pairs = []
    max_dij = 0
    min_dij = 1e6
    for i in range(len(features)):
        for j in range(i+1, len(features)):
            distance_array, p_indices, q_indices = extract_distance_array(features[i], features[j])
            dij, indices = compute_dij(distance_array, eplison1)
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

    graph = compute_graph(PQC_pairs, len(features), eplison2)
    central_node = compute_central_node(graph)
    
    global_transforms = fine_registration(point_clouds, PQC_pairs, graph, central_node, max_iter=fine_registration_max_iter)
    return global_transforms