# -*- coding: utf-8 -*-
"""
Gurobi-based solution to the MIS
"""

import os
import pickle
import csv
import time
import re
from gurobipy import Model, GRB

time_limit = 1800  # 2 seconds

input_dir = "instances_d3"
plot_dir = os.path.join(input_dir, "soln_plots_gurobi")
soln_dir = os.path.join(input_dir, "solns_gurobi_test")

os.makedirs(plot_dir, exist_ok=True)
os.makedirs(soln_dir, exist_ok=True)

csv_path = os.path.join(input_dir, "mis_solutions_gurobi.csv")
csv_header = ["filename", "n", "gurobi_time", "total_time", "gap", "solution_nodes"]

with open(csv_path, 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow(csv_header)

def extract_n(filename):
    match = re.search(r"mis_n(\d+)_p", filename)
    return int(match.group(1)) if match else float('inf')

def solve_mis_gurobi(G):
    model = Model()
    model.setParam('OutputFlag', 0)
    model.setParam('TimeLimit', time_limit)

    # Create binary variables for each node
    x = {}
    for v in G.nodes:
        x[v] = model.addVar(vtype=GRB.BINARY, name=f"x_{v}")

    # Add independence constraints
    for u, v in G.edges:
        model.addConstr(x[u] + x[v] <= 1)

    # Objective: maximize number of selected nodes
    model.setObjective(sum(x[v] for v in G.nodes), GRB.MAXIMIZE)

    # Solve the model
    model.optimize()

    # Extract solution
    soln = [v for v in G.nodes if x[v].X > 0.5]

    # Get Gurobi solve time and MIP gap
    gurobi_time = model.Runtime
    gap = model.MIPGap if model.SolCount > 0 and model.MIPGap is not None else float('nan')

    return soln, gurobi_time, gap

# Process each .gpickle file
for filename in sorted(os.listdir(input_dir), key=extract_n):
    if filename.endswith(".gpickle"):
        filepath = os.path.join(input_dir, filename)
        print(f"Processing {filename}...")

        with open(filepath, 'rb') as f:
            G = pickle.load(f)

        try:
            start = time.time()
            soln, gurobi_time, gap = solve_mis_gurobi(G)
            end = time.time()
            total_time = end - start

            if gurobi_time > time_limit:
                print(f"  Skipping {filename}: time exceeded {time_limit} seconds ({gurobi_time:.2f}s)")
                break

            print(f"  MIS size: {len(soln)}, Gurobi time: {gurobi_time:.6f}s, Total time: {total_time:.6f}s, Gap: {gap:.4f}")

            with open(csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([
                    filename,
                    G.number_of_nodes(),
                    round(gurobi_time, 6),
                    round(total_time, 6),
                    round(gap, 6) if gap == gap else 'nan',
                    " ".join(map(str, sorted(soln)))
                ])
        except Exception as e:
            print(f"  Error processing {filename}: {e}")
