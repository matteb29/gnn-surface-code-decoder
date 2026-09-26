"""Logical error rate of a decoder, comparable with the MWPM baseline."""
import numpy as np
import torch
from torch_geometric.loader import DataLoader

from create_graphs import to_graph


@torch.no_grad()
def logical_error_rate(model, dets, labels, coords, batch_size=4096):
    """Fraction of experiments where the decoder gets the logical flip wrong.

    Empty syndromes are kept and predicted as 'no flip', so the number is
    directly comparable with the MWPM baseline.
    """
    model.eval()
    predictions = np.zeros(len(labels), dtype=bool)

    non_empty = dets.any(axis=1)
    graphs = [to_graph(row, lab, coords)
              for row, lab in zip(dets[non_empty], labels[non_empty])]

    outputs = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        outputs.append(model(batch) > 0)
    predictions[non_empty] = torch.cat(outputs).numpy()

    rate = float((predictions != labels).mean())
    return rate, float(np.sqrt(rate * (1 - rate) / len(labels)))
