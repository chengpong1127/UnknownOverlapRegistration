import torch
from torchsparse import SparseTensor

def nearest_neighbor(query_feature: torch.Tensor, ref_features: SparseTensor):
    """Find the nearest neighbor in the reference features for each query

    Args:
        query_feature (torch.Tensor): (D) tensor of query features
        ref_features (SparseTensor): (M, D) tensor of reference features
    """
    
    dist = torch.norm(ref_features.F - query_feature, dim=1)
    min_dist, index = torch.min(dist, 0)
    return min_dist, index
    
    