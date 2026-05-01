import os
import networkx as nx
import numpy as np
import pickle

# Parameters
n_values = np.arange(5, 70)   # n must be >= d+1 and n*d must be even
d = 3                         # degree
num_instances = 10
output_folder = "instances_d3"

# Create output directory
os.makedirs(output_folder, exist_ok=True)


for n in n_values:

    # Skip invalid combinations (required for regular graphs)
    if (n * d) % 2 != 0:
        continue

    for inst in range(1, num_instances + 1):

        seed = inst
        G = nx.random_regular_graph(d, n, seed=seed)

        filename = f"mis_d{d}_n{n}_{inst}.gpickle"
        filepath = os.path.join(output_folder, filename)

        with open(filepath, "wb") as f:
            pickle.dump(G, f)

        print(f"Saved: {filepath}")