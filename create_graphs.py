import numpy as np
import torch

from torch_geometric.data import Data
from torch_geometric.utils import coalesce
from torch_geometric.loader import DataLoader

from generate_data import generate


def normalize_coordinates(coord):
    """
        Helper function to apply the normalization of 
        the coordinates i.e. the graph features to a normal gaussian

    
    """

    coords = np.asarray(coord, dtype = np.float32)

    return (coords - coords.mean(axis = 0))/ coords.std(axis = 0)


def to_graph(det_row, label, coords, k = 6):
    """
        helper function to define the graphs, one experiment = one graph
        where the nodes are the turned on detectors with normalized features (x, y, t)
        edges are defined using a k-neighbous approach and implement a weight
        defined as 1/distance 


    """

    #define the index of all the turned on detectors
    idx = np.flatnonzero(det_row)
    #define the number of turned on detector
    n = len(idx)
    if n == 0:
        return None

    #define the coordinates of turned on detector that will be our node in the graph
    x = torch.from_numpy(coords[idx])

    if n == 1:
        edge_index = torch.empty((2,0), dtype = torch.long)
        edge_weight = torch.empty(0)

    else:
        #compute all the distances resulting in a N x N matrix
        distance = torch.cdist(x, x)
        distance.fill_diagonal_(float("inf"))
        kk = min(k, n-1)
        neighbours = distance.topk(kk, largest = False).indices

        source = torch.arange(n).repeat_interleave(kk) 
        receiver = neighbours.reshape(-1)

        edge_index = torch.stack([
            torch.cat([source, receiver]),
            torch.cat([receiver, source])
        ])

        edge_weight = 1.0 / distance[edge_index[0], edge_index[1]]

        #drop the duplicates
        edge_index, edge_weight = coalesce(
            edge_index, edge_weight,
            num_nodes = n, reduce = 'mean'
        )

    return Data(x = x, edge_index = edge_index, edge_weight = edge_weight, y = torch.tensor([float(label)]))


def build_dataset(dets, labels, coords, k = 6):
    """
        Helper function to return a list of graphs

    """

    graphs = []
    for row, lab in zip(dets, labels):
        g = to_graph(row, lab, coords, k)
        if g is not None:
            graphs.append(g)

    return graphs

if __name__ == '__main__':

    dets, labels, coords = generate(d = 3, rounds = 3, p = 0.005, shots = 2000, seed = 1)
    coords = normalize_coordinates(coords)
    graphs = build_dataset(dets, labels, coords, k = 6)

    print(f"number of graphs {len(graphs)}, with number of detectors {len(dets)}")
    print("\n remember that one graph correspond to one experiment and one node correspond to one turned on detector in that experiment \n")

    g = next(g for g in graphs if g.num_nodes == 2)
    print("\n here is an example of a graphs with only two nodes")
    print(f"coordinates: \n {g.x} \n")
    print(f"edges: \n {g.edge_index} \n")
    print(f"edge weights \n {g.edge_weight} \n")

    batch = next(iter(DataLoader(graphs, batch_size = 32, shuffle = False)))
    print(f"\n batch: \n {batch}")



