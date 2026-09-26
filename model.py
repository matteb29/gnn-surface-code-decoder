import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from generate_data import generate
from create_graphs import normalize_coordinates,  build_dataset


CODE_DIMENSION = 3
LATENT_DIMENSION = 128
NUMBER_OF_LINEAR_LAYERS = 4
NUMBER_OF_GRADIENT_DESCENT_STEPS = 300


class MessagePassingLayer(nn.Module):

    def __init__(self, input_features, output_features):

        super().__init__()

        self.linear_self = nn.Linear(
            input_features, output_features, bias = False
        )
        self.linear_neighbours = nn.Linear(
            input_features, output_features, bias = False
        )

    def forward(self, h, edge_index, edge_weight):

        sources, receivers = edge_index
        messages = h[sources] * edge_weight.unsqueeze(-1)

        #define the aggregated message
        aggregated = torch.zeros_like(h)
        aggregated.index_add_(0, receivers, messages)

        return self.linear_self(h) + self.linear_neighbours(aggregated) 



def mean_pool(h, batch, num_graphs):

    out = torch.zeros(num_graphs, h.size(1), dtype = h.dtype, device = h.device)
    out.index_add_(0, batch, h)

    counts = torch.zeros(num_graphs, dtype = h.dtype, device = h.device)
    counts.index_add(0, batch, torch.ones_like(batch, dtype = h.dtype))

    return out / counts.clamp(min=1).unsqueeze(-1)


class GNN(nn.Module):

    def __init__(
            self, 
            input_features = CODE_DIMENSION, 
            hidden = LATENT_DIMENSION,
            num_layers = NUMBER_OF_LINEAR_LAYERS
            ):

        super().__init__()

        dims = [input_features] + [hidden] * num_layers
        self.mp_layers = nn.ModuleList(
            MessagePassingLayer(a, b) for a, b in zip(dims[:-1], dims[1:])
        )
        self.head = nn.Sequential(
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1),
        )

    def forward(self, data):

        h = data.x
        for layer in self.mp_layers:
            h = torch.relu(layer(h, data.edge_index, data.edge_weight))

        g = mean_pool(h, data.batch, int(data.batch.max()) + 1)

        #output has shape [num_graphs]
        return self.head(g).squeeze(-1)



if __name__ == '__main__':



    data, labels, coords = generate(CODE_DIMENSION, 
                                    rounds = 3,
                                    p = 0.005,
                                    shots = 500,
                                    seed = 1)

    graphs = build_dataset(data, labels, coords)
    batch = next(iter(DataLoader(graphs, batch_size = 32, shuffle = False)))

    model = GNN()

    print(f"number of parameters of the GNN is {sum(p.numel() for p in model.parameters())}")
    print(f"output shape: {model(batch).shape}")

    optimizer = torch.optim.Adam(model.parameters(), lr = 1e-2)
    loss_function = nn.BCEWithLogitsLoss()
    small = next(iter(DataLoader(graphs[:100], batch_size = 100, shuffle = False)))

    for step in range(NUMBER_OF_GRADIENT_DESCENT_STEPS + 1):

        
        optimizer.zero_grad()
        loss = loss_function(model(small), small.y)
        loss.backward()
        optimizer.step()

        if step % 150 == 0:

            accuracy = ((model(small)>0).float() == small.y ).float().mean()
            print(f"step: {step}   loss: {loss.item():.4f}   accuracy {accuracy:.2%} ")

        

