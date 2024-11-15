import torch

from torchsparse import SparseTensor
from fcgf import ResUNet, ResidualBlock



def test_residual_block():
    block = ResidualBlock(32, 32).to('cuda')
    sparse_input = SparseTensor(coords=torch.zeros((2, 4), dtype=torch.int32), feats=torch.ones((2, 32), dtype=torch.float32)).to('cuda')
    output = block(sparse_input)
    assert output.F.shape == (2, 32)
    assert output.C.shape == (2, 4)


def test_resunet():
    model = ResUNet(normalize_feature=True).to('cuda')
    sparse_input = SparseTensor(coords=torch.randint(0, 100, (1000, 4), dtype=torch.int32), feats=torch.rand((1000, 3), dtype=torch.float32)).to('cuda')
    output = model(sparse_input)
    assert output.F.shape == (1000, 32)
    assert output.C.shape == (1000, 4)