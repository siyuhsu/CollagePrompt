import torch
from torch_geometric.utils import *

def init_grid_adj():
    # 3x3 grid graph, each node is connected to its neighbors
    adjacency_matrix = torch.zeros((9, 9), dtype=torch.float32)
    ord_rand = torch.randperm(9)

    for i in range(3):
        for j in range(3):
            node_idx = i * 3 + j


            # Connect with the right neighbor
            if j < 2:
                st, ed = ord_rand[node_idx], ord_rand[node_idx + 1]
                adjacency_matrix[st, ed] = 1
                adjacency_matrix[ed, st] = 1

                # adjacency_matrix[node_idx, node_idx + 1] = 1

            # Connect with the bottom neighbor
            if i < 2:
                st, ed = ord_rand[node_idx], ord_rand[node_idx + 3]
                adjacency_matrix[st, ed] = 1
                adjacency_matrix[ed, st] = 1
                # adjacency_matrix[node_idx, node_idx + 3] = 1

            # Connect with the left neighbor
            if j > 0:
                st, ed = ord_rand[node_idx], ord_rand[node_idx - 1]
                adjacency_matrix[st, ed] = 1
                adjacency_matrix[ed, st] = 1

                # adjacency_matrix[node_idx, node_idx - 1] = 1

            # Connect with the top neighbor
            if i > 0:
                st, ed = ord_rand[node_idx], ord_rand[node_idx - 3]
                adjacency_matrix[st, ed] = 1
                adjacency_matrix[ed, st] = 1
                # adjacency_matrix[node_idx, node_idx - 3] = 1

    return adjacency_matrix, ord_rand

def ord_grid_adj(ord_rand):
    # 3x3 grid graph, each node is connected to its neighbors
    adjacency_matrix = torch.zeros((9, 9), dtype=torch.float32)
    ord_rand = ord_rand.astype(int)

    try:
        for i in range(3):
            for j in range(3):
                node_idx = i * 3 + j


                # Connect with the right neighbor
                if j < 2:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx + 1]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1

                    # adjacency_matrix[node_idx, node_idx + 1] = 1

                # Connect with the bottom neighbor
                if i < 2:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx + 3]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1
                    # adjacency_matrix[node_idx, node_idx + 3] = 1

                # Connect with the left neighbor
                if j > 0:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx - 1]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1

                    # adjacency_matrix[node_idx, node_idx - 1] = 1

                # Connect with the top neighbor
                if i > 0:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx - 3]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1
                    # adjacency_matrix[node_idx, node_idx - 3] = 1
    except Exception as e:
        print(e)
    return adjacency_matrix


def ord_grid_adj_2x2(ord_rand):
    # 2x2 grid graph, each node is connected to its neighbors
    adjacency_matrix = torch.zeros((4, 4), dtype=torch.float32)
    ord_rand = ord_rand.astype(int)

    try:
        for i in range(2):
            for j in range(2):
                node_idx = i * 2 + j

                # Connect with the right neighbor
                if j < 1:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx + 1]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1

                # Connect with the bottom neighbor
                if i < 1:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx + 2]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1

                # Connect with the left neighbor
                if j > 0:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx - 1]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1

                # Connect with the top neighbor
                if i > 0:
                    st, ed = ord_rand[node_idx], ord_rand[node_idx - 2]
                    adjacency_matrix[st, ed] = 1
                    adjacency_matrix[ed, st] = 1
    except Exception as e:
        print(e)
    return adjacency_matrix


def process_adj(adj_matrix):
    
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    edge_index, edge_attr = remove_self_loops(edge_index, edge_attr)
    # edge_index, edge_attr = to_undirected(edge_index, edge_attr)

    return edge_index, edge_attr

if __name__ == '__main__':

    # Example usage
    adj_matrix = init_grid_adj()
    # print(adj_matrix)
    # print(decode_adjacency_matrix(adj_matrix))
    edge_index, edge_attr = dense_to_sparse(adj_matrix)
    print("Grid Adjacency Matrix:")
    print(adj_matrix)