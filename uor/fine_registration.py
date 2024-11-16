import numpy as np
from scipy.linalg import expm
import networkx as nx

def hat(vector):
    if len(vector) == 3:
        return np.array([[0, -vector[2], vector[1]],
                         [vector[2], 0, -vector[0]],
                         [-vector[1], vector[0], 0]])
    else:
        return np.array([[0, -vector[5], vector[4], vector[0]],
                         [vector[5], 0, -vector[3], vector[1]],
                         [-vector[4], vector[3], 0, vector[2]],
                         [0, 0, 0, 0]])

def irls_optimize(A, b, max_iter=10, tol=1e-6):
    xi = np.zeros(6)
    W = np.eye(len(b))
    
    for _ in range(max_iter):
        residuals = A @ xi - b
        weights = 1.0 / (np.abs(residuals) + tol)
        W = np.diag(weights)
        
        WA = W @ A
        Wb = W @ b
        xi_new = np.linalg.lstsq(WA.T @ WA, WA.T @ Wb, rcond=None)[0]
        
        if np.linalg.norm(xi_new - xi) < tol:
            break
        xi = xi_new
    
    return xi

def fine_registration(point_clouds: list, PQC_pairs: list[tuple], graph: nx.Graph, central_node: int, max_iter=10):
    relative_transforms = {}

    for (i, j, p_indices, q_indices, _) in PQC_pairs:
        if graph.has_edge(i, j):
            p_points = np.asarray(point_clouds[i].points)[p_indices]
            q_points = np.asarray(point_clouds[j].points)[q_indices]
            T_k = np.eye(4)
            
            A_list = []
            b_list = []
            for p, q in zip(p_points, q_points):
                T_q = T_k[:3, :3] @ q + T_k[:3, 3]
                A_kl = np.hstack([np.eye(3), -hat(T_q)[:3, :3]])
                b_kl = p - T_q
                A_list.append(A_kl)
                b_list.append(b_kl)
            
            A = np.vstack(A_list)
            b = np.hstack(b_list)

            xi_k = irls_optimize(A, b, max_iter=max_iter)
            
            relative_transforms[(i, j)] = expm(hat(xi_k))
            relative_transforms[(j, i)] = expm(-hat(xi_k))
    
    global_transforms = compute_global_transforms_from_central(graph, relative_transforms, central_node)
    
    return global_transforms

def compute_global_transforms_from_central(graph, relative_transforms, central_node):
    global_transforms = {node: np.eye(4) for node in graph.nodes}
    visited = set()
    queue = [central_node]
    visited.add(central_node)

    while queue:
        current = queue.pop(0)
        for neighbor in graph.neighbors(current):
            if neighbor not in visited:
                global_transforms[neighbor] = global_transforms[current] @ relative_transforms[(current, neighbor)]
                visited.add(neighbor)
                queue.append(neighbor)

    return global_transforms