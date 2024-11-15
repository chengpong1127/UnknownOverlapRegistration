import numpy as np
from models.fine_registration import irls_optimize, fine_registration
import networkx as nx
from .utils import create_test_point_clouds

def test_irls_optimize():
    A = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 1, 0, 0, 0],
                  [1, 1, 0, 0, 0, 0],
                  [0, 1, 1, 0, 0, 0],
                  [1, 0, 1, 0, 0, 0]])
    true_xi = np.array([1, 2, 3, 0, 0, 0])
    noise = np.random.normal(0, 0.1, A.shape[0])
    b = A @ true_xi + noise

    xi_estimated = irls_optimize(A, b, max_iter=10, tol=1e-6)

    assert np.allclose(xi_estimated[:3], true_xi[:3], atol=0.2)

def create_test_graph_and_pairs(point_clouds, covered_points):
    num_clouds = len(point_clouds)
    graph = nx.Graph()
    PQC_pairs = []
    
    for i in range(num_clouds - 1):
        j = i + 1
        graph.add_edge(i, j)
        
        
        p_indices = list(range(covered_points))
        q_indices = list(range(covered_points))
        
        cij = np.random.rand()
        
        PQC_pairs.append((i, j, p_indices, q_indices, cij))
    
    central_node = 0
    return graph, PQC_pairs, central_node


def test_fine_registration():
    point_clouds = create_test_point_clouds(3, 10)
    
    graph, PQC_pairs, central_node = create_test_graph_and_pairs(point_clouds, 5)
    
    global_transforms = fine_registration(point_clouds, PQC_pairs, graph, central_node, max_iter=10)
    
    np.testing.assert_array_almost_equal(global_transforms[central_node], np.eye(4), decimal=6)

    for (i, j, _, _, _) in PQC_pairs:
        rel_transform = np.linalg.inv(global_transforms[i]) @ global_transforms[j]
        assert rel_transform.shape == (4, 4)