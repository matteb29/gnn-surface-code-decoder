import pymatching
import numpy as np
import pandas as pd
from pathlib import Path

from generate_data import make_circuit, generate



def mwpm_decoder_predictions(d, rounds, p, dets):
    """
        helper function to define the MWPM 
        decoder predictions that will be our baseline
    
        Args:
            - d: the surface code
            - rounds:
            - p: 
            - dets: 
    
    """

    circuit = make_circuit(d, rounds, p)
    dem = circuit.detector_error_model(decompose_errors = True)
    matching = pymatching.Matching.from_detector_error_model(dem)

    return matching.decode_batch(dets)[:, 0].astype(bool)


def binomial_error(rate, n):
    """
        helper function to define the binomial distribution error
        in order to estimate MWMP's efficiency
    
    """

    return np.sqrt((rate * (1-rate))/ n)


if __name__ == "__main__":

    d, rounds, shots = 3, 3, 200_000
    test_probabilty = [0.002, 0.003, 0.005, 0.007, 0.010]
    rows = []

    print(f"the surface code is {d}, the number of test experiments is {shots}")
    print("\n")

    for p in test_probabilty:
        dets, labels, _ = generate(d, rounds, p, shots, seed=12345)
        #mean number of error that occurs, 0 no error occurred in the qubit
        always0 = labels.mean()
        #applying the decoder the number of error gets lower than always 0
        mwpm_results = (mwpm_decoder_predictions(d, rounds, p, dets) != labels).mean()
        rows.append({"d": d, "rounds": rounds, "p": p, "shots": shots,
                     "always0": always0, "always0_err": binomial_error(always0, shots),
                     "mwpm": mwpm_results, "mwpm_err": binomial_error(mwpm_results, shots)})
        print(f"p={p}: mwpm={mwpm_results:.4f}")

    print("\n")

    Path("results").mkdir(exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv("results/baseline.csv", index = False)

    print(df)

    print("\n")


