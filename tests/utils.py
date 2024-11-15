import numpy as np
import open3d as o3d

def create_test_point_clouds(num_clouds=3, num_points=10):
    point_clouds = []
    for _ in range(num_clouds):
        points = np.random.rand(num_points, 3)
        colors = np.random.rand(num_points, 3)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        point_clouds.append(pcd)
    return point_clouds