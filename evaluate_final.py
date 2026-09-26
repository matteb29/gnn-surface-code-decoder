"""Final comparison GNN vs MWPM on exactly the same test experiments.

Three things:
  1. MWPM re-measured on the SAME shots given to the network (not a separate sample).
  2. McNemar paired test: who is wrong on which experiment.
  3. Logical error rate vs physical error rate p, for both decoders.
"""
import math
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import torch

from torch_geometric.loader import DataLoader

from generate_data import generate
from create_graphs import to_graph, normalize_coordinates
from baseline import mwpm_decoder_predictions, binomial_error
from model import GNN

CODE_DIMENSION = 3
ROUND = 3

TEST_P, TEST_SHOTS, TEST_SEED = 0.005, 2_000_000, 12345
CURVE_PROBABILITIES = [0.002, 0.003, 0.005, 0.007, 0.010]
CURVE_SHOTS, CURVE_SEED = 500_000, 999

MODEL_PATH = "results/best_model.pt"


@torch.no_grad()
def gnn_predictions(model, dets, coords, batch_size=4096):
    """Predicted logical flip for every experiment.

    Empty syndromes get 0 without asking the network: with no detector lit
    there is no graph, and 'no flip' is the only sensible guess.
    """
    model.eval()
    predictions = np.zeros(len(dets), dtype=bool)

    non_empty = dets.any(axis=1)
    graphs = [to_graph(row, 0.0, coords) for row in dets[non_empty]]

    outputs = []
    for batch in DataLoader(graphs, batch_size=batch_size, shuffle=False):
        outputs.append(model(batch) > 0)
    predictions[non_empty] = torch.cat(outputs).numpy()

    return predictions


def mcnemar(wrong_a, wrong_b):
    """Paired test on the same experiments.

    n01 = experiments where A is right and B is wrong
    n10 = experiments where A is wrong and B is right
    Only the disagreements carry information: if the two decoders were
    equally good, n01 and n10 would be two halves of a coin flip.
    """
    n01 = int(np.sum(~wrong_a & wrong_b))
    n10 = int(np.sum(wrong_a & ~wrong_b))

    if n01 + n10 == 0:
        return n01, n10, 0.0, 1.0

    z = (n01 - n10) / math.sqrt(n01 + n10)
    p_value = math.erfc(abs(z) / math.sqrt(2))

    return n01, n10, z, p_value


def main():

    Path("results").mkdir(exist_ok=True)

    model = GNN()
    model.load_state_dict(torch.load(MODEL_PATH))

    # ---------- 1. same test set for both decoders ----------
    print(f"generating {TEST_SHOTS} test experiments at p={TEST_P} (seed {TEST_SEED})")
    dets, labels, raw_coords = generate(
        CODE_DIMENSION, ROUND, TEST_P, TEST_SHOTS, seed=TEST_SEED
    )
    coords = normalize_coordinates(raw_coords)

    gnn_pred = gnn_predictions(model, dets, coords)
    mwpm_pred = mwpm_decoder_predictions(CODE_DIMENSION, ROUND, TEST_P, dets)

    gnn_wrong = gnn_pred != labels
    mwpm_wrong = mwpm_pred != labels

    gnn_rate = gnn_wrong.mean()
    mwpm_rate = mwpm_wrong.mean()

    print("\n--- same experiments, both decoders ---")
    print(f"MWPM : {mwpm_rate:.5f} +- {binomial_error(mwpm_rate, TEST_SHOTS):.5f}")
    print(f"GNN  : {gnn_rate:.5f} +- {binomial_error(gnn_rate, TEST_SHOTS):.5f}")
    print(f"relative improvement: {(mwpm_rate - gnn_rate) / mwpm_rate:.2%}")

    # ---------- 2. paired test ----------
    n01, n10, z, p_value = mcnemar(gnn_wrong, mwpm_wrong)

    print("\n--- McNemar paired test ---")
    print(f"GNN right, MWPM wrong : {n01}")
    print(f"GNN wrong, MWPM right : {n10}")
    print(f"both wrong            : {int(np.sum(gnn_wrong & mwpm_wrong))}")
    print(f"z = {z:.2f}   p-value = {p_value:.2e}")
    if p_value < 1e-3:
        print("the difference is not a fluctuation of the sample.")

    # ---------- 3. curve over p ----------
    print(f"\n--- curve over p ({CURVE_SHOTS} experiments each) ---")
    rows = []
    for p in CURVE_PROBABILITIES:
        d_p, l_p, _ = generate(CODE_DIMENSION, ROUND, p, CURVE_SHOTS, seed=CURVE_SEED)

        g_rate = float((gnn_predictions(model, d_p, coords) != l_p).mean())
        m_rate = float(
            (mwpm_decoder_predictions(CODE_DIMENSION, ROUND, p, d_p) != l_p).mean()
        )

        rows.append({
            "p": p,
            "shots": CURVE_SHOTS,
            "mwpm": m_rate, "mwpm_err": binomial_error(m_rate, CURVE_SHOTS),
            "gnn": g_rate, "gnn_err": binomial_error(g_rate, CURVE_SHOTS),
        })
        flag = "  (outside training range)" if p > 0.005 else ""
        print(f"p={p:.3f}  mwpm={m_rate:.4f}  gnn={g_rate:.4f}{flag}")

    df = pd.DataFrame(rows)
    df.to_csv("results/final_comparison.csv", index=False)

    # ---------- plot ----------
    fig, ax = plt.subplots(figsize=(6, 4.5))
    ax.errorbar(df["p"], df["mwpm"], yerr=df["mwpm_err"],
                marker="o", capsize=3, label="MWPM (knows the noise model)")
    ax.errorbar(df["p"], df["gnn"], yerr=df["gnn_err"],
                marker="s", capsize=3, label="GNN (data only)")
    ax.axvspan(min(CURVE_PROBABILITIES), 0.005, alpha=0.08, color="green")
    ax.text(0.0035, ax.get_ylim()[1] * 0.05, "training range",
            ha="center", fontsize=8, color="green")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("physical error rate $p$")
    ax.set_ylabel("logical error rate")
    ax.set_title(f"Surface code d={CODE_DIMENSION}, {ROUND} rounds, circuit-level noise")
    ax.legend()
    ax.grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig("results/final_comparison.png", dpi=150)

    print("\nsaved results/final_comparison.csv and results/final_comparison.png")


if __name__ == "__main__":
    main()
