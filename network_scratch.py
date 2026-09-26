"""GNN decoder written from scratch: message passing with plain tensor ops."""
import torch
import torch.nn as nn


class MessagePassingLayer(nn.Module):
    """One graph convolution, by hand.

        h_i' = W_self @ h_i  +  W_neigh @ ( sum_j  e_ij * h_j )  +  b

    Same formula as Eq. (7) of Lange et al. (and as torch_geometric GraphConv).
    """

    def __init__(self, in_features, out_features):
        super().__init__()
        self.lin_self = nn.Linear(in_features, out_features, bias=False)
        self.lin_neigh = nn.Linear(in_features, out_features, bias=True)

    def forward(self, h, edge_index, edge_weight):
        source, target = edge_index                       # each edge: source -> target
        messages = h[source] * edge_weight.unsqueeze(-1)  # [E, F]: neighbour feature x weight

        aggregated = torch.zeros_like(h)                  # [N, F]
        aggregated.index_add_(0, target, messages)        # sum the messages arriving at each node

        return self.lin_self(h) + self.lin_neigh(aggregated)


def mean_pool(h, batch, num_graphs):
    """Average the node features of each graph. Replaces global_mean_pool."""
    out = torch.zeros(num_graphs, h.size(1), dtype=h.dtype, device=h.device)
    out.index_add_(0, batch, h)
    counts = torch.zeros(num_graphs, dtype=h.dtype, device=h.device)
    counts.index_add_(0, batch, torch.ones_like(batch, dtype=h.dtype))
    return out / counts.clamp(min=1).unsqueeze(-1)


class GNNDecoderScratch(nn.Module):
    def __init__(self, in_features=3, hidden=64, layers=4):
        super().__init__()
        dims = [in_features] + [hidden] * layers
        self.convs = nn.ModuleList(
            MessagePassingLayer(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):
        h = data.x
        for conv in self.convs:
            h = torch.relu(conv(h, data.edge_index, data.edge_weight))
        g = mean_pool(h, data.batch, int(data.batch.max()) + 1)
        return self.head(g).squeeze(-1)


if __name__ == "__main__":
    from torch_geometric.nn import GraphConv, global_mean_pool
    from torch_geometric.loader import DataLoader
    from generate_data import generate
    from create_graphs import normalize_coordinates, build_dataset

    dets, labels, coords = generate(d=3, rounds=3, p=0.005, shots=500, seed=1)
    graphs = build_dataset(dets, labels, normalize_coordinates(coords))
    batch = next(iter(DataLoader(graphs, batch_size=64, shuffle=False)))

    # --- test 1: our layer must equal GraphConv when they share the same weights
    mine, theirs = MessagePassingLayer(3, 8), GraphConv(3, 8)
    with torch.no_grad():
        mine.lin_self.weight.copy_(theirs.lin_root.weight)
        mine.lin_neigh.weight.copy_(theirs.lin_rel.weight)
        mine.lin_neigh.bias.copy_(theirs.lin_rel.bias)
    a = mine(batch.x, batch.edge_index, batch.edge_weight)
    b = theirs(batch.x, batch.edge_index, batch.edge_weight)
    print("layer max difference:", (a - b).abs().max().item())

    # --- test 2: our pooling must equal global_mean_pool
    h = torch.randn(batch.num_nodes, 8)
    p1 = mean_pool(h, batch.batch, batch.num_graphs)
    p2 = global_mean_pool(h, batch.batch)
    print("pooling max difference:", (p1 - p2).abs().max().item())

    # --- test 3: shapes and overfitting on 100 graphs
    model = GNNDecoderScratch()
    print("parameters:", sum(p.numel() for p in model.parameters()))
    print("output shape:", model(batch).shape)

    small = next(iter(DataLoader(graphs[:100], batch_size=100, shuffle=False)))
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    loss_fn = nn.BCEWithLogitsLoss()
    for step in range(301):
        opt.zero_grad()
        loss = loss_fn(model(small), small.y)
        loss.backward()
        opt.step()
        if step % 150 == 0:
            acc = ((model(small) > 0).float() == small.y).float().mean()
            print(f"step {step:3d}  loss {loss.item():.4f}  accuracy {acc:.2%}")
