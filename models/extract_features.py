import numpy
import open3d as o3d
import torch
import sys
from pathlib import Path
from urllib.request import urlretrieve
sys.path.append(str(Path.cwd() / 'FCGF'))
from model.resunet import ResUNetBN2C
from util.misc import extract_features as extract_features_fcgf

model_list = [
    'https://node1.chrischoy.org/data/publications/fcgf/KITTI-v0.3-ResUNetBN2C-conv1-5-nout16.pth',
    'https://node1.chrischoy.org/data/publications/fcgf/KITTI-v0.3-ResUNetBN2C-conv1-5-nout32.pth'
]

def check_model() -> None:
    if not all(Path(model.split('/')[-1]).is_file() for model in model_list):
        print('Downloading weights...')
        for model in model_list:
            urlretrieve(model, model.split('/')[-1])

def extract_features(point_clouds: list, feature_dim: int = 16, voxel_size: float = 0.025) -> list:
    check_model()
    if feature_dim == 16:
        model_name = model_list[0].split('/')[-1]
    elif feature_dim == 32:
        model_name = model_list[1].split('/')[-1]
        
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
