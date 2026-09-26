import numpy as np
import stim

from pathlib import Path

def make_circuit(d, rounds, p):
    """
        Helper function to implement circuit generation
        using stim

        Args:
            - d: the distance of the code 
            - rounds: the number of detector
            - p: the probability of error


    """

    return stim.Circuit.generated(
        "surface_code:rotated_memory_z",
        distance = d,
        rounds = rounds,
        after_clifford_depolarization=p,     
        before_round_data_depolarization=p,  
        before_measure_flip_probability=p,   
        after_reset_flip_probability=p,      

    )


def generate(d, rounds, p, shots, seed = None):
    """

        Helper function to generate the circuit and 
        organize the data with labels and coordinates for the
        detector that will be encoded as nodes in the graph


    """

    circuit = make_circuit(d, rounds, p)
    sampler = circuit.compile_detector_sampler(seed = seed)
    dets, obs = sampler.sample(shots = shots, separate_observables = True)
    raw = circuit.get_detector_coordinates()
    coords = np.array([raw[i] for i in range(circuit.num_detectors) ], dtype = np.float32)


    return dets, obs[:, 0], coords



if __name__ == "__main__":

    d, rounds, p = 3, 3, 0.005

    Path("data").mkdir(exist_ok = True)

    dets, labels, coords = generate(d = 3, rounds = 3, p = 0.005, shots = 1_000_000 , seed = 1)
    #save the not void experiments in which at least one detector lights on because of one error occurred
    not_void = dets.any(axis=1)
    np.savez_compressed(f"data/train_d{d}_p{p}.npz",
                    dets=dets[not_void],       
                    labels=labels[not_void],   
                    coords=coords)             


    print(f"saved {not_void.sum()} simulated samples. ")