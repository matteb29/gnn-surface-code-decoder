import time
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import torch.nn as nn

from torch_geometric.loader import DataLoader

from generate_data import generate
from create_graphs import build_dataset, normalize_coordinates
from model import GNN
from evaluate import logical_error_rate

CODE_DIMENSION = 3
ROUND = 3
TRAIN_PROBABILITIES = [0.001, 0.002, 0.003, 0.004, 0.005]
SHOTS_PER_EPOCH = 1_000_000
TEST_P, TEST_SHOTS, TEST_SEED = 0.005, 2_000_000, 12345

MWPM_REFERENCE_VALUE = 0.0174
EPOCHS = 50
BATCH_SIZE = 256
LEARNING_RATE = 1e-3

def main():

    torch.manual_seed(0)
    np.random.seed(0)

    Path("results").mkdir(exist_ok=True)

    dets_test, label_test, coords = generate(
        CODE_DIMENSION,
        ROUND,
        TEST_P,
        TEST_SHOTS,
        seed=TEST_SEED
    )

    coords = normalize_coordinates(coords)

    model = GNN()
    optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_function = nn.BCEWithLogitsLoss()

    history, best = [], 1.0
    start = time.time()

    for epoch in range(EPOCHS):

        p = float(np.random.choice(TRAIN_PROBABILITIES))
        dets, labels, _ = generate(
            CODE_DIMENSION, ROUND, p, SHOTS_PER_EPOCH, seed = None
        )
        graphs = build_dataset(dets, labels, coords )

        model.train()
        total_loss = 0.0
        n = 0


        for batch in DataLoader(graphs, batch_size = BATCH_SIZE, shuffle = True):
            optimzer.zero_grad()
            loss = loss_function(model(batch), batch.y)
            loss.backward()
            optimzer.step()
            total_loss += loss.item() * batch.num_graphs
            n += batch.num_graphs


        rate, error = logical_error_rate(
            model, dets_test, label_test, coords
        )
        history.append(
            {
                "epoch": epoch,
                "train_p": p,
                "loss": total_loss / n,
                "logical_error_estimator": rate,
                "error_on_estimator": error

             }
        )

        print(f"epoch {epoch:3d}  p={p:.3f}  loss {total_loss/n:.4f}  "
                      f"logic error {rate:.4f} ± {error:.4f}  "
                      f"(MWPM {MWPM_REFERENCE_VALUE:.4f})  [{time.time()-start:.0f}s]")


        if rate < best:
            best = rate
            torch.save(model.state_dict(), "results/best_model.pt")

        pd.DataFrame(history).to_csv("results/training_history.csv", index = False)


if __name__ == '__main__':

    main()

