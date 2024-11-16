import numpy
import open3d as o3d
import torch
import sys
from pathlib import Path
sys.path.append(str(Path.cwd() / 'FCGF'))
from model.resunet import ResUNetBN2C
from util.misc import extract_features as extract_features_fcgf

def extract_features(point_clouds: list, feature_dim: int = 16, voxel_size: float = 0.025) -> list:
    if feature_dim == 16:
        model_name = 'FCGF/ResUNetBN2C-16feat-5conv.pth'
    elif feature_dim == 32:
        model_name = 'FCGF/ResUNetBN2C-32feat-5conv.pth'
        
    device = 'cpu'
    checkpoint = torch.load(model_name)
    print(checkpoint.keys())
    model = ResUNetBN2C(1, feature_dim, normalize_feature=True, conv1_kernel_size=5, D=3)
    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    model = model.to(device)
    features = [
        extract_features_fcgf(
            model,
            xyz=numpy.asarray(pcd.points),
            voxel_size=voxel_size,
            device=device,
            skip_check=True
        )[1]
        for pcd in point_clouds
    ]
    
    return features
