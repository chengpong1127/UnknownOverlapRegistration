from models.utils import nearest_neighbor
import torch
from torchsparse import SparseTensor

def test_nearest_neighbor():
    query_feature = torch.tensor([1.0, 2.0, 3.0])
    ref_features = SparseTensor(feats=torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]), coords=torch.zeros((2, 4), dtype=torch.int32))
    distance, index = nearest_neighbor(query_feature, ref_features)
    assert distance == 0.0
    assert index == 0